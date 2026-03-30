from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from rich.live import Live
from rich.spinner import Spinner

from .agents.evaluator_agent import EvaluatorAgent
from .agents.meta_agent import MetaAgent
from .agents.reflector_agent import ReflectorAgent
from .core.config import HealthFlowConfig
from .core.llm_provider import create_llm_provider
from .ehr import detect_risk_findings, output_contract, profile_workspace_data
from .execution import ExecutionContext, create_executor_adapter
from .experience.experience_manager import ExperienceManager
from .tools import ToolBroker
from .verification import WorkspaceVerifier


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class HealthFlowSystem:
    """
    Backend-agnostic HealthFlow runtime with EHR-aware orchestration, verifier gating,
    and hierarchical long-term memory.
    """

    def __init__(self, config: HealthFlowConfig, experience_path: Path):
        self.config = config
        self.llm_provider = create_llm_provider(config.llm)
        self.experience_manager = ExperienceManager(experience_path, self.llm_provider)

        self.meta_agent = MetaAgent(self.llm_provider)
        self.evaluator = EvaluatorAgent(self.llm_provider)
        self.reflector = ReflectorAgent(self.llm_provider)
        self.executor = create_executor_adapter(config.active_executor_name, config.active_executor)
        self.tool_broker = ToolBroker()
        self.verifier = WorkspaceVerifier(config.verification.required_report_sections)

        self.workspace_dir = Path(config.system.workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    async def run_task(
        self,
        user_request: str,
        live: Optional[Live] = None,
        spinner: Optional[Spinner] = None,
        train_mode: bool = False,
        reference_answer: str = None,
        uploaded_files: Optional[Dict[str, bytes]] = None,
    ) -> dict:
        task_id = str(uuid.uuid4())
        task_workspace = self.workspace_dir / task_id
        task_workspace.mkdir(parents=True, exist_ok=True)

        if self.config.memory.mode == "reset":
            await self.experience_manager.reset()

        if uploaded_files:
            logger.info("[{}] Saving {} uploaded files to the workspace.", task_id, len(uploaded_files))
            for filename, content in uploaded_files.items():
                safe_filename = Path(filename).name
                file_path = task_workspace / safe_filename
                with open(file_path, "wb") as handle:
                    handle.write(content)

        start_time = time.time()
        result = await self._run_unified_flow(
            task_id,
            task_workspace,
            user_request,
            live,
            spinner,
            train_mode,
            reference_answer,
        )
        execution_time = round(time.time() - start_time, 2)
        result["execution_time"] = execution_time
        result["workspace_path"] = str(task_workspace)
        result["backend"] = self.config.active_executor_name
        result["reasoning_model"] = self.config.llm.model_name
        result["memory_mode"] = self.config.memory.mode
        result["run_result_path"] = str(task_workspace / "run_result.json")
        result["run_manifest_path"] = str(task_workspace / "run_manifest.json")
        self._write_json(
            task_workspace / "run_result.json",
            result,
        )
        self._write_json(
            task_workspace / "run_manifest.json",
            {
                "task_id": task_id,
                "user_request": user_request,
                "workspace_path": str(task_workspace),
                "backend": self.config.active_executor_name,
                "reasoning_model": self.config.llm.model_name,
                "memory_mode": self.config.memory.mode,
                "execution_time": execution_time,
                "verification_passed": result.get("verification_passed", False),
                "success": result.get("success", False),
                "log_path": result.get("log_path"),
                "prompt_path": result.get("prompt_path"),
            },
        )
        return result

    async def _run_unified_flow(
        self,
        task_id: str,
        task_workspace: Path,
        user_request: str,
        live: Optional[Live],
        spinner: Optional[Spinner],
        train_mode: bool = False,
        reference_answer: str = None,
    ) -> Dict[str, Any]:
        if spinner and live:
            spinner.text = "Profiling EHR inputs and retrieving memory..."

        data_profile = profile_workspace_data(
            task_workspace,
            user_request,
            max_preview_rows=self.config.ehr.max_preview_rows,
        )
        risk_findings = detect_risk_findings(user_request, data_profile)
        tool_bundle = self.tool_broker.select_bundle(data_profile.task_family, data_profile)
        expected_outputs = output_contract(data_profile.task_family)
        memory_budgets = {
            "strategy": self.config.memory.strategy_k,
            "failure": self.config.memory.failure_k,
            "dataset": self.config.memory.dataset_k,
            "artifact": max(1, self.config.memory.retrieve_k - self.config.memory.strategy_k - self.config.memory.failure_k - self.config.memory.dataset_k),
        }
        retrieved_experiences = await self.experience_manager.retrieve_experiences(
            user_request,
            task_family=data_profile.task_family,
            dataset_signature=data_profile.dataset_signature,
            budgets=memory_budgets,
        )
        memory_summary = self._summarize_retrieved_experiences(retrieved_experiences)
        self._write_json(
            task_workspace / "memory_context.json",
            {
                "task_family": data_profile.task_family,
                "dataset_signature": data_profile.dataset_signature,
                "budgets": memory_budgets,
                "selected_memories": [exp.model_dump(mode="json") for exp in retrieved_experiences],
                "summary": memory_summary,
            },
        )

        full_history: Dict[str, Any] = {
            "task_id": task_id,
            "user_request": user_request,
            "task_family": data_profile.task_family,
            "dataset_signature": data_profile.dataset_signature,
            "backend": self.config.active_executor_name,
            "reasoning_model": self.config.llm.model_name,
            "memory_mode": self.config.memory.mode,
            "data_profile": data_profile.to_markdown(),
            "risk_findings": [item.to_bullet() for item in risk_findings],
            "tool_bundle": tool_bundle,
            "output_contract": expected_outputs,
            "memory_context_path": str(task_workspace / "memory_context.json"),
            "retrieved_experiences": [exp.model_dump() for exp in retrieved_experiences],
            "attempts": [],
        }

        is_success = False
        verification_passed = False
        final_answer = "No answer generated."
        final_summary = "Task failed to complete within the allowed attempts."

        for attempt in range(self.config.system.max_retries + 1):
            attempt_num = attempt + 1
            previous_feedback = full_history["attempts"][-1]["evaluation"]["feedback"] if attempt > 0 else None
            if spinner and live:
                spinner.text = f"Attempt {attempt_num}: Planning..."

            task_list_md = await self.meta_agent.generate_plan(
                user_request=user_request,
                experiences=retrieved_experiences,
                task_family=data_profile.task_family,
                data_profile=data_profile.to_markdown(),
                risk_checks=[item.to_bullet() for item in risk_findings],
                tool_bundle=tool_bundle,
                output_contract=expected_outputs,
                previous_feedback=previous_feedback,
            )
            task_list_path = task_workspace / f"task_list_v{attempt_num}.md"
            task_list_path.write_text(task_list_md, encoding="utf-8")

            execution_context = ExecutionContext(
                user_request=user_request,
                task_family=data_profile.task_family,
                data_profile=data_profile.to_markdown(),
                risk_checks=[item.to_bullet() for item in risk_findings],
                tool_bundle=tool_bundle,
                output_contract=expected_outputs,
                plan_markdown=task_list_md,
                prior_feedback=previous_feedback,
                memory_summary=memory_summary,
                verification_requirements=expected_outputs,
            )

            if spinner and live:
                spinner.text = f"Attempt {attempt_num}: Executing with {self.config.active_executor_name}..."
            execution_result = await self.executor.execute(
                execution_context,
                task_workspace,
                self.config.executor.prompt_file_name,
            )

            verification = self.verifier.verify(
                task_workspace,
                data_profile.task_family,
                execution_result.log,
                data_profile,
            )

            if spinner and live:
                spinner.text = f"Attempt {attempt_num}: Evaluating..."
            evaluation = await self.evaluator.evaluate(
                user_request=user_request,
                task_list=task_list_md,
                execution_log=execution_result.log,
                verification_summary=verification.summary(),
                train_mode=train_mode,
                reference_answer=reference_answer,
            )

            attempt_history = {
                "attempt": attempt_num,
                "task_list": task_list_md,
                "execution": {
                    "success": execution_result.success,
                    "return_code": execution_result.return_code,
                    "log": execution_result.log,
                    "log_path": execution_result.log_path,
                    "prompt_path": execution_result.prompt_path,
                    "backend": execution_result.backend,
                    "command": execution_result.command,
                    "duration_seconds": execution_result.duration_seconds,
                    "timed_out": execution_result.timed_out,
                    "usage": execution_result.usage,
                },
                "verification": {
                    "passed": verification.passed,
                    "checks": verification.checks,
                    "artifact_paths": verification.artifact_paths,
                },
                "gate": {
                    "execution_ok": execution_result.success,
                    "evaluation_threshold_ok": evaluation["score"] >= self.config.evaluation.success_threshold,
                    "verifier_required": self.config.verification.require_verifier_pass,
                    "verifier_ok": verification.passed or not self.config.verification.require_verifier_pass,
                },
                "evaluation": evaluation,
            }
            full_history["attempts"].append(attempt_history)

            verification_passed = verification.passed
            passed_threshold = evaluation["score"] >= self.config.evaluation.success_threshold
            verifier_gate_ok = verification.passed or not self.config.verification.require_verifier_pass
            if execution_result.success and passed_threshold and verifier_gate_ok:
                is_success = True
                final_answer = self._extract_answer_from_workspace(task_workspace, execution_result.log, user_request)
                final_summary = (
                    "Task completed successfully. "
                    f"Verification: {verification.summary()} "
                    f"Evaluation: {evaluation.get('reasoning', 'N/A')}"
                )
                break

            final_summary = (
                f"Attempt {attempt_num} failed. Verification: {verification.summary()} "
                f"Feedback: {evaluation['feedback']}"
            )
            logger.warning("[{}] Attempt {} failed: {}", task_id, attempt_num, final_summary)

        should_write_memory = self.config.memory.mode != "frozen_train" and (
            is_success or self.config.memory.writeback_on_failure
        )
        if should_write_memory:
            if spinner and live:
                spinner.text = "Synthesizing memory..."
            new_experiences = await self.reflector.synthesize_experience(
                full_history,
                verified=is_success and verification_passed,
            )
            if new_experiences:
                await self.experience_manager.save_experiences(new_experiences)
                full_history["new_experiences"] = [exp.model_dump() for exp in new_experiences]

        history_path = task_workspace / "full_history.json"
        self._write_json(history_path, full_history)

        return {
            "success": is_success,
            "final_summary": final_summary,
            "answer": final_answer,
            "task_family": data_profile.task_family,
            "dataset_signature": data_profile.dataset_signature,
            "verification_passed": verification_passed,
            "log_path": full_history["attempts"][-1]["execution"]["log_path"] if full_history["attempts"] else None,
            "prompt_path": full_history["attempts"][-1]["execution"]["prompt_path"] if full_history["attempts"] else None,
            "memory_context_path": str(task_workspace / "memory_context.json"),
        }

    def _extract_answer_from_workspace(self, task_workspace: Path, execution_log: str, user_request: str) -> str:
        final_report = task_workspace / "final_report.md"
        if final_report.exists():
            return final_report.read_text(encoding="utf-8", errors="ignore")[:4000]

        metrics_file = next(iter(sorted(task_workspace.glob("metrics.*"))), None)
        if metrics_file:
            return metrics_file.read_text(encoding="utf-8", errors="ignore")[:2000]

        lines = execution_log.split("\n")
        answer_indicators = [
            "final answer:",
            "answer:",
            "result:",
            "conclusion:",
            "output:",
            "solution:",
            "the answer is",
            "priority:",
        ]
        for i in range(len(lines) - 1, -1, -1):
            line_content = lines[i].replace("STDOUT: ", "").replace("STDERR: ", "").strip()
            if any(indicator in line_content.lower() for indicator in answer_indicators):
                return line_content

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("STDOUT: "):
                content = lines[i].replace("STDOUT: ", "").strip()
                if content and len(content) > 10:
                    return content

        return (
            "Execution completed. Review the workspace artifacts for details. "
            f"The original request was: '{user_request}'"
        )

    def _summarize_retrieved_experiences(self, experiences: list[Any]) -> str:
        if not experiences:
            return "- No prior memory was retrieved for this task."

        lines = []
        for exp in experiences:
            prefix = "Avoid" if exp.layer.value == "failure" else "Use"
            lines.append(
                f"- {prefix}: [{exp.layer.value}/{exp.validation_status.value}] {exp.category} -> {exp.content}"
            )
        return "\n".join(lines)

    def _write_json(self, path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, cls=DateTimeEncoder)
