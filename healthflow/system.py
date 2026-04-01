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
from .core.contracts import EvaluationVerdict
from .core.llm_provider import create_llm_provider
from .execution import ExecutionContext, create_executor_adapter
from .experience.experience_manager import ExperienceManager
from .tools import ToolCatalog


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class HealthFlowSystem:
    """
    Backend-agnostic HealthFlow runtime organized around a four-stage
    Meta -> Executor -> Evaluator -> Reflector loop.
    """

    def __init__(self, config: HealthFlowConfig, experience_path: Path):
        self.config = config
        provider_cache: dict[tuple[str, str], Any] = {}

        def provider_for(role: str):
            role_config = config.llm_config_for_role(role)
            provider_key = (role_config.base_url, role_config.model_name)
            if provider_key not in provider_cache:
                provider_cache[provider_key] = create_llm_provider(role_config)
            return provider_cache[provider_key]

        self.llm_provider = provider_for("planner")
        self.experience_manager = ExperienceManager(experience_path, self.llm_provider)

        self.meta_agent = MetaAgent(provider_for("planner"))
        self.evaluator = EvaluatorAgent(provider_for("evaluator"))
        self.reflector = ReflectorAgent(provider_for("reflector"))
        self.executor = create_executor_adapter(config.active_executor_name, config.active_executor)
        self.tool_catalog = ToolCatalog.from_config(config.active_executor_name, config.tools)

        self.workspace_dir = Path(config.system.workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    async def run_task(
        self,
        user_request: str,
        live: Optional[Live] = None,
        spinner: Optional[Spinner] = None,
        uploaded_files: Optional[Dict[str, bytes]] = None,
    ) -> dict:
        task_id = str(uuid.uuid4())
        task_workspace = self.workspace_dir / task_id
        task_workspace.mkdir(parents=True, exist_ok=True)

        if self.config.memory.write_policy == "reset_before_run":
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
        )
        execution_time = round(time.time() - start_time, 2)
        result["execution_time"] = execution_time
        result["workspace_path"] = str(task_workspace)
        result["backend"] = self.config.active_executor_name
        result["reasoning_model"] = self.config.llm_config_for_role("planner").model_name
        result["llm_role_models"] = self._role_model_names()
        result["memory_write_policy"] = self.config.memory.write_policy
        result["run_result_path"] = str(task_workspace / "run_result.json")
        result["run_manifest_path"] = str(task_workspace / "run_manifest.json")
        result["cost_analysis_path"] = str(task_workspace / "cost_analysis.json")
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
                "reasoning_model": self.config.llm_config_for_role("planner").model_name,
                "llm_role_models": result.get("llm_role_models"),
                "memory_write_policy": self.config.memory.write_policy,
                "backend_version": result.get("backend_version"),
                "executor_metadata": result.get("executor_metadata"),
                "usage_summary": result.get("usage_summary"),
                "cost_summary": result.get("cost_summary"),
                "cost_analysis_path": result.get("cost_analysis_path"),
                "execution_time": execution_time,
                "evaluation_status": result.get("evaluation_status"),
                "success": result.get("success", False),
                "log_path": result.get("log_path"),
                "prompt_path": result.get("prompt_path"),
                "evaluation_path": result.get("evaluation_path"),
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
    ) -> Dict[str, Any]:
        if spinner and live:
            spinner.text = "Retrieving memory and preparing the planner..."

        retrieval_result = await self.experience_manager.retrieve_experiences(user_request)
        recommended_experiences = retrieval_result.recommended_experiences
        avoidance_experiences = retrieval_result.avoidance_experiences
        self._write_json(
            task_workspace / "memory_context.json",
            retrieval_result.audit.model_dump(mode="json"),
        )

        full_history: Dict[str, Any] = {
            "task_id": task_id,
            "user_request": user_request,
            "backend": self.config.active_executor_name,
            "reasoning_model": self.config.llm_config_for_role("planner").model_name,
            "llm_role_models": self._role_model_names(),
            "memory_write_policy": self.config.memory.write_policy,
            "available_tools": self.tool_catalog.names(),
            "memory_context_path": str(task_workspace / "memory_context.json"),
            "memory_retrieval": retrieval_result.audit.model_dump(mode="json"),
            "recommended_experiences": [exp.model_dump(mode="json") for exp in recommended_experiences],
            "avoidance_experiences": [exp.model_dump(mode="json") for exp in avoidance_experiences],
            "attempts": [],
        }

        is_success = False
        final_answer = "No answer generated."
        final_summary = "Task failed to complete within the allowed attempts."
        final_verdict = EvaluationVerdict(
            status="failed",
            score=0.0,
            failure_type="not_started",
            feedback="The task did not start.",
            repair_instructions=[],
            retry_recommended=False,
            memory_worthy_insights=[],
            reasoning="No attempt was executed.",
        )
        reflection_usage = None

        for attempt_num in range(1, self.config.system.max_attempts + 1):
            previous_feedback = self._feedback_from_attempt(full_history["attempts"][-1]) if attempt_num > 1 else None
            if spinner and live:
                spinner.text = f"Attempt {attempt_num}: Planning..."

            plan = await self.meta_agent.generate_plan(
                user_request=user_request,
                recommended_experiences=recommended_experiences,
                avoidance_experiences=avoidance_experiences,
                available_tools=self.tool_catalog.names(),
                previous_feedback=previous_feedback,
            )
            planning_usage = self._capture_agent_usage(self.meta_agent)
            plan_markdown = plan.to_markdown()
            task_list_path = task_workspace / f"task_list_v{attempt_num}.md"
            task_list_path.write_text(plan_markdown, encoding="utf-8")

            execution_context = ExecutionContext(
                user_request=user_request,
                plan=plan,
                available_tools=self.tool_catalog,
                recommended_memory=self._format_memory_lines(recommended_experiences),
                avoidance_memory=self._format_memory_lines(avoidance_experiences),
                prior_feedback=previous_feedback,
            )

            if spinner and live:
                spinner.text = f"Attempt {attempt_num}: Executing with {self.config.active_executor_name}..."
            execution_result = await self.executor.execute(
                execution_context,
                task_workspace,
            )

            generated_answer = self._extract_answer_from_workspace(task_workspace, execution_result.log, user_request)
            workspace_artifacts = self._workspace_artifact_paths(task_workspace)

            if spinner and live:
                spinner.text = f"Attempt {attempt_num}: Evaluating..."
            verdict = await self.evaluator.evaluate(
                user_request=user_request,
                plan=plan,
                execution_log=execution_result.log,
                workspace_artifacts=workspace_artifacts,
                generated_answer=generated_answer,
            )
            evaluation_usage = self._capture_agent_usage(self.evaluator)
            final_verdict = verdict

            attempt_history = {
                "attempt": attempt_num,
                "plan": plan.model_dump(mode="json"),
                "plan_markdown": plan_markdown,
                "planning": planning_usage,
                "execution": {
                    "success": execution_result.success,
                    "return_code": execution_result.return_code,
                    "log": execution_result.log,
                    "log_path": execution_result.log_path,
                    "prompt_path": execution_result.prompt_path,
                    "backend": execution_result.backend,
                    "backend_version": execution_result.backend_version,
                    "executor_metadata": execution_result.executor_metadata,
                    "command": execution_result.command,
                    "duration_seconds": execution_result.duration_seconds,
                    "timed_out": execution_result.timed_out,
                    "usage": execution_result.usage,
                    "telemetry": execution_result.telemetry,
                },
                "artifacts": {
                    "workspace_paths": workspace_artifacts,
                    "generated_answer": generated_answer,
                },
                "gate": {
                    "execution_ok": execution_result.success,
                    "evaluation_threshold_ok": verdict.score >= self.config.evaluation.success_threshold,
                    "verdict_success": verdict.status == "success",
                    "retry_recommended": verdict.retry_recommended,
                },
                "evaluation": verdict.model_dump(mode="json"),
                "evaluation_meta": evaluation_usage,
            }
            full_history["attempts"].append(attempt_history)

            if execution_result.success and verdict.is_success(self.config.evaluation.success_threshold):
                is_success = True
                final_answer = generated_answer
                final_summary = (
                    "Task completed successfully. "
                    f"Evaluator verdict: {verdict.feedback} "
                    f"Reasoning: {verdict.reasoning or 'N/A'}"
                )
                break

            final_answer = generated_answer
            final_summary = (
                f"Attempt {attempt_num} failed. "
                f"Failure type: {verdict.failure_type}. "
                f"Feedback: {verdict.feedback}"
            )
            logger.warning("[{}] Attempt {} failed: {}", task_id, attempt_num, final_summary)
            if not verdict.retry_recommended:
                break

        should_write_memory = self.config.memory.write_policy != "freeze"
        if should_write_memory:
            if spinner and live:
                spinner.text = "Synthesizing memory..."
            new_experiences = await self.reflector.synthesize_experience(
                full_history,
                final_verdict=final_verdict,
            )
            reflection_usage = self._capture_agent_usage(self.reflector)
            if new_experiences:
                await self.experience_manager.save_experiences(new_experiences)
                full_history["new_experiences"] = [exp.model_dump(mode="json") for exp in new_experiences]
        if reflection_usage:
            full_history["reflection"] = reflection_usage

        history_path = task_workspace / "full_history.json"
        self._write_json(history_path, full_history)
        if full_history["attempts"]:
            self._write_json(
                task_workspace / "evaluation.json",
                full_history["attempts"][-1]["evaluation"],
            )
        last_execution = full_history["attempts"][-1]["execution"] if full_history["attempts"] else {}
        usage_summary = self._summarize_run_usage(full_history["attempts"], reflection_usage)
        cost_summary = self._summarize_run_costs(usage_summary)
        cost_analysis = self._summarize_cost_analysis(full_history["attempts"], usage_summary, cost_summary, reflection_usage)
        self._write_json(task_workspace / "cost_analysis.json", cost_analysis)

        return {
            "success": is_success,
            "final_summary": final_summary,
            "answer": final_answer,
            "evaluation_status": final_verdict.status,
            "evaluation_score": final_verdict.score,
            "memory_write_policy": self.config.memory.write_policy,
            "backend_version": last_execution.get("backend_version"),
            "executor_metadata": last_execution.get("executor_metadata"),
            "usage_summary": usage_summary,
            "cost_summary": cost_summary,
            "cost_analysis": cost_analysis,
            "log_path": full_history["attempts"][-1]["execution"]["log_path"] if full_history["attempts"] else None,
            "prompt_path": full_history["attempts"][-1]["execution"]["prompt_path"] if full_history["attempts"] else None,
            "evaluation_path": str(task_workspace / "evaluation.json"),
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

    def _workspace_artifact_paths(self, task_workspace: Path) -> list[str]:
        artifact_paths: list[str] = []
        for path in sorted(task_workspace.rglob("*")):
            if not path.is_file():
                continue
            try:
                artifact_paths.append(str(path.relative_to(task_workspace)))
            except ValueError:
                artifact_paths.append(str(path))
        return artifact_paths

    def _format_memory_lines(self, experiences: list[Any]) -> list[str]:
        lines: list[str] = []
        for exp in experiences:
            layer = exp.layer.value if hasattr(exp.layer, "value") else str(exp.layer)
            validation_status = (
                exp.validation_status.value
                if hasattr(exp.validation_status, "value")
                else str(exp.validation_status)
            )
            lines.append(f"[{layer}/{validation_status}] {exp.category}: {exp.content}")
        return lines

    def _feedback_from_attempt(self, attempt: dict[str, Any]) -> str | None:
        if not attempt:
            return None
        evaluation = attempt.get("evaluation", {})
        feedback = evaluation.get("feedback")
        repairs = evaluation.get("repair_instructions", [])
        parts = [feedback] if feedback else []
        if repairs:
            parts.append("Repair instructions: " + "; ".join(repairs))
        return "\n".join(parts) if parts else None

    def _summarize_retrieval_audit(self, audit: dict | Any) -> str:
        selected_entries = getattr(audit, "selected", None)
        if selected_entries is None and isinstance(audit, dict):
            selected_entries = audit.get("selected", [])
        if not selected_entries:
            return "- No prior memory was retrieved for this task."

        lines = []
        for entry in selected_entries:
            layer = entry.layer.value if hasattr(entry, "layer") else entry["layer"]
            validation_status = (
                entry.validation_status.value if hasattr(entry, "validation_status") else entry["validation_status"]
            )
            category = entry.category if hasattr(entry, "category") else entry["category"]
            content_preview = (
                entry.content_preview if hasattr(entry, "content_preview") else entry["content_preview"]
            )
            safety_critical = (
                entry.safety_critical if hasattr(entry, "safety_critical") else entry.get("safety_critical", False)
            )
            verifier_supported = (
                entry.verifier_supported if hasattr(entry, "verifier_supported") else entry.get("verifier_supported", False)
            )
            prefix = "Avoid" if layer == "failure" else "Use"
            flags = []
            if safety_critical:
                flags.append("safety")
            if verifier_supported:
                flags.append("verifier")
            flag_text = f" [{'|'.join(flags)}]" if flags else ""
            lines.append(
                f"- {prefix}: [{layer}/{validation_status}]{flag_text} {category} -> {content_preview}"
            )
        safety_overrides = getattr(audit, "safety_overrides", None)
        if safety_overrides is None and isinstance(audit, dict):
            safety_overrides = audit.get("safety_overrides", [])
        if safety_overrides:
            lines.append(f"- Safety override count: {len(safety_overrides)} conflicting memories were suppressed.")
        return "\n".join(lines)

    def _write_json(self, path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, cls=DateTimeEncoder)

    def _role_model_names(self) -> dict[str, str]:
        return {
            "planner": getattr(self.meta_agent, "last_model_name", self.config.llm_config_for_role("planner").model_name),
            "evaluator": getattr(self.evaluator, "last_model_name", self.config.llm_config_for_role("evaluator").model_name),
            "reflector": getattr(self.reflector, "last_model_name", self.config.llm_config_for_role("reflector").model_name),
        }

    def _capture_agent_usage(self, agent: Any) -> dict:
        usage = getattr(agent, "last_usage", {}) or {}
        return {
            "model_name": getattr(agent, "last_model_name", self.config.llm.model_name),
            "usage": usage,
            "estimated_cost_usd": getattr(agent, "last_estimated_cost_usd", None),
        }

    def _aggregate_component_records(self, records: list[dict[str, Any]]) -> dict:
        if not records:
            return {}
        numeric_totals: dict[str, float] = {}
        models: list[str] = []
        for record in records:
            model_name = record.get("model_name")
            if model_name and model_name not in models:
                models.append(model_name)
            usage = record.get("usage", {})
            for key, value in usage.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    numeric_totals[key] = numeric_totals.get(key, 0.0) + float(value)
            estimated_cost = record.get("estimated_cost_usd")
            if isinstance(estimated_cost, (int, float)):
                numeric_totals["estimated_cost_usd"] = numeric_totals.get("estimated_cost_usd", 0.0) + float(estimated_cost)
        aggregated = {
            key: round(value, 8) if "cost" in key else round(value, 4)
            for key, value in numeric_totals.items()
        }
        aggregated["calls"] = len(records)
        if models:
            aggregated["models"] = models
        return aggregated

    def _aggregate_execution_records(self, attempts: list[dict[str, Any]]) -> dict:
        if not attempts:
            return {}
        totals: dict[str, float] = {}
        versions: list[str] = []
        session_ids: list[str] = []
        models: list[str] = []
        tool_names: list[str] = []
        step_reasons: dict[str, int] = {}
        for attempt in attempts:
            execution = attempt.get("execution", {})
            version = execution.get("backend_version")
            if version and version not in versions:
                versions.append(version)
            for key, value in execution.get("usage", {}).items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0.0) + float(value)
            telemetry = execution.get("telemetry", {})
            session_id = telemetry.get("session_id")
            if session_id and session_id not in session_ids:
                session_ids.append(session_id)
            for model_name in telemetry.get("models", []):
                if model_name and model_name not in models:
                    models.append(model_name)
            for tool_name in telemetry.get("tool_names", []):
                if tool_name and tool_name not in tool_names:
                    tool_names.append(tool_name)
            for reason, count in telemetry.get("step_reasons", {}).items():
                if isinstance(count, int):
                    step_reasons[reason] = step_reasons.get(reason, 0) + count
        aggregated = {
            key: round(value, 8) if "cost" in key else round(value, 4)
            for key, value in totals.items()
        }
        aggregated["calls"] = len(attempts)
        if versions:
            aggregated["backend_versions"] = versions
        if session_ids:
            aggregated["session_ids"] = session_ids
        if models:
            aggregated["models"] = models
        if tool_names:
            aggregated["tool_names"] = tool_names
        if step_reasons:
            aggregated["step_reasons"] = step_reasons
        return aggregated

    def _summarize_run_usage(self, attempts: list[dict[str, Any]], reflection_usage: dict | None) -> dict:
        planning_records = [attempt.get("planning", {}) for attempt in attempts if attempt.get("planning")]
        evaluation_records = [attempt.get("evaluation_meta", {}) for attempt in attempts if attempt.get("evaluation_meta")]
        summary = {
            "planning": self._aggregate_component_records(planning_records),
            "evaluation": self._aggregate_component_records(evaluation_records),
            "execution": self._aggregate_execution_records(attempts),
        }
        if reflection_usage:
            summary["reflection"] = self._aggregate_component_records([reflection_usage])
        return summary

    def _summarize_run_costs(self, usage_summary: dict[str, Any]) -> dict:
        llm_estimated_cost_usd = 0.0
        has_llm_cost = False
        for component in ["planning", "evaluation", "reflection"]:
            component_cost = usage_summary.get(component, {}).get("estimated_cost_usd")
            if isinstance(component_cost, (int, float)):
                llm_estimated_cost_usd += float(component_cost)
                has_llm_cost = True
        executor_estimated_cost_usd = usage_summary.get("execution", {}).get("estimated_cost_usd")
        has_executor_cost = isinstance(executor_estimated_cost_usd, (int, float))
        total_estimated_cost_usd = (
            (llm_estimated_cost_usd if has_llm_cost else 0.0)
            + (float(executor_estimated_cost_usd) if has_executor_cost else 0.0)
        )
        has_total_cost = has_llm_cost or has_executor_cost
        return {
            "llm_estimated_cost_usd": round(llm_estimated_cost_usd, 8) if has_llm_cost else None,
            "executor_estimated_cost_usd": round(float(executor_estimated_cost_usd), 8) if has_executor_cost else None,
            "total_estimated_cost_usd": round(total_estimated_cost_usd, 8) if has_total_cost else None,
        }

    def _summarize_cost_analysis(
        self,
        attempts: list[dict[str, Any]],
        usage_summary: dict[str, Any],
        cost_summary: dict[str, Any],
        reflection_usage: dict | None,
    ) -> dict[str, Any]:
        attempt_summaries: list[dict[str, Any]] = []
        for attempt in attempts:
            planning = self._aggregate_component_records([attempt.get("planning", {})]) if attempt.get("planning") else {}
            evaluation = (
                self._aggregate_component_records([attempt.get("evaluation_meta", {})])
                if attempt.get("evaluation_meta")
                else {}
            )
            execution = self._aggregate_execution_records([{"execution": attempt.get("execution", {})}])

            attempt_total = 0.0
            has_cost = False
            for component in [planning, evaluation, execution]:
                component_cost = component.get("estimated_cost_usd")
                if isinstance(component_cost, (int, float)):
                    attempt_total += float(component_cost)
                    has_cost = True

            attempt_summaries.append(
                {
                    "attempt": attempt.get("attempt"),
                    "planning": planning,
                    "execution": execution,
                    "evaluation": evaluation,
                    "attempt_total_estimated_cost_usd": round(attempt_total, 8) if has_cost else None,
                }
            )

        run_total = {
            "attempts": len(attempts),
            "planning": usage_summary.get("planning", {}),
            "execution": usage_summary.get("execution", {}),
            "evaluation": usage_summary.get("evaluation", {}),
            "reflection": usage_summary.get("reflection", {}) if reflection_usage else {},
            "llm_estimated_cost_usd": cost_summary.get("llm_estimated_cost_usd"),
            "executor_estimated_cost_usd": cost_summary.get("executor_estimated_cost_usd"),
            "total_estimated_cost_usd": cost_summary.get("total_estimated_cost_usd"),
        }
        return {
            "attempts": attempt_summaries,
            "run_total": run_total,
        }
