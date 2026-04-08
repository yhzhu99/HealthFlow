from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger
from rich.live import Live
from rich.spinner import Spinner

from .agents.evaluator_agent import EvaluatorAgent
from .agents.meta_agent import MetaAgent
from .agents.reflector_agent import ReflectorAgent
from .core.config import HealthFlowConfig
from .core.contracts import EvaluationVerdict
from .core.direct_responses import DirectResponse, DirectResponseRouter
from .core.llm_provider import create_llm_provider
from .ehr import detect_risk_findings, profile_workspace_data
from .execution import ExecutionCancelledError, ExecutionContext, WorkflowRecommendationBroker, create_executor_adapter
from .experience.experience_manager import ExperienceManager
from .experience.experience_models import RetrievalContext
from .reporting import generate_task_report
from .runtime_artifacts import AttemptPaths, TaskRuntimePaths
from .session import HealthFlowProgressEvent, SessionPromptContext, TaskSessionState, TaskTurnRecord

_STEP_FINISH_REASON_RE = re.compile(r"reason=([^\s]+)")
_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9']+")
_ANSWER_INTRO_MARKERS = (
    "final answer",
    "here's the information you requested",
    "here is the information you requested",
    "here's the answer",
    "here is the answer",
    "corrected final answer",
    "provide the corrected final answer",
)
_INLINE_ANSWER_PATTERNS = (
    re.compile(r"(?i)\bfinal answer:\s*(.+)$"),
    re.compile(r"(?i)\banswer:\s*(.+)$"),
    re.compile(r"(?i)\bresult:\s*(.+)$"),
    re.compile(r"(?i)\bconclusion:\s*(.+)$"),
    re.compile(r"(?i)\boutput:\s*(.+)$"),
    re.compile(r"(?i)\bsolution:\s*(.+)$"),
    re.compile(r"(?i)\bthe answer is\s+(.+)$"),
)
_REQUEST_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "tell",
    "the",
    "to",
    "we",
    "what",
    "whats",
    "with",
}
_TOKEN_EQUIVALENTS = {
    "hk": {"hk", "hong", "kong"},
}
_RUNTIME_SCHEMA_VERSION = "2.0"


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
        provider_cache: dict[tuple[str, str, str], Any] = {}

        def provider_for(role: str):
            role_config = config.llm_config_for_role(role)
            provider_key = (
                role_config.base_url,
                role_config.model_name,
                role_config.reasoning_effort or "",
            )
            if provider_key not in provider_cache:
                provider_cache[provider_key] = create_llm_provider(role_config)
            return provider_cache[provider_key]

        self.experience_manager = ExperienceManager(experience_path)

        self.meta_agent = MetaAgent(provider_for("planner"))
        self.evaluator = EvaluatorAgent(provider_for("evaluator"))
        self.reflector = ReflectorAgent(provider_for("reflector"))
        self.direct_response_router = DirectResponseRouter(provider_for("planner"))
        self.executor = create_executor_adapter(config.active_executor_name, config.active_executor)
        self.workflow_broker = WorkflowRecommendationBroker()

        self.workspace_dir = Path(config.system.workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def create_task_session(self, task_id: str | None = None, *, original_goal: str = "") -> TaskSessionState:
        resolved_task_id = task_id or str(uuid.uuid4())
        task_workspace = self.workspace_dir / resolved_task_id
        paths = TaskRuntimePaths.build(task_workspace)
        paths.ensure_base_dirs()
        session_path = self._session_state_path(task_workspace)
        if session_path.exists():
            return self.load_task_session(resolved_task_id)

        now = self._utc_now()
        state = TaskSessionState(
            task_id=resolved_task_id,
            task_root=str(task_workspace),
            created_at_utc=now,
            updated_at_utc=now,
            original_goal=original_goal.strip(),
            turn_count=0,
            latest_turn_number=0,
            latest_turn_status=None,
        )
        self._write_json(session_path, state.to_dict())
        history_path = self._session_history_path(task_workspace)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.touch(exist_ok=True)
        return state

    def load_task_session(self, task_id: str) -> TaskSessionState:
        task_workspace = self.workspace_dir / task_id
        session_path = self._session_state_path(task_workspace)
        if not session_path.exists():
            raise FileNotFoundError(f"Task session '{task_id}' does not exist.")
        payload = self._read_json(session_path)
        return TaskSessionState.from_dict(payload)

    def load_task_history(self, task_id: str) -> list[TaskTurnRecord]:
        task_workspace = self.workspace_dir / task_id
        return self._read_session_history(task_workspace)

    async def run_task_turn(
        self,
        task_id: str,
        user_message: str,
        uploaded_files: Optional[Dict[str, bytes]] = None,
        report_requested: bool = False,
        progress_callback: Callable[[HealthFlowProgressEvent], None] | None = None,
        live: Optional[Live] = None,
        spinner: Optional[Spinner] = None,
    ) -> dict[str, Any]:
        task_workspace = self.workspace_dir / task_id
        root_paths = TaskRuntimePaths.build(task_workspace)
        root_paths.ensure_base_dirs()
        session_state = self.create_task_session(task_id, original_goal=user_message)
        turn_number = session_state.turn_count + 1
        turn_runtime_dir = self._turn_runtime_dir(root_paths, turn_number)
        turn_paths = TaskRuntimePaths.build(
            task_workspace,
            sandbox_dir=root_paths.sandbox_dir,
            runtime_dir=turn_runtime_dir,
        )
        turn_paths.ensure_base_dirs()

        uploaded_file_records = self._store_turn_uploads(
            task_workspace=task_workspace,
            sandbox_dir=root_paths.sandbox_dir,
            turn_number=turn_number,
            uploaded_files=uploaded_files or {},
        )
        session_context = self._build_session_prompt_context(
            task_workspace=task_workspace,
            session_state=session_state,
            turn_number=turn_number,
            user_message=user_message,
            uploaded_file_records=uploaded_file_records,
        )
        allow_direct_response = turn_number == 1 and not uploaded_file_records
        result = await self._execute_task_run(
            task_id=task_id,
            task_workspace=task_workspace,
            paths=turn_paths,
            user_request=user_message,
            live=live,
            spinner=spinner,
            uploaded_files=None,
            report_requested=report_requested,
            has_uploaded_files=bool(uploaded_file_records),
            allow_direct_response=allow_direct_response,
            session_context=session_context,
            progress_callback=progress_callback,
        )
        self._mirror_latest_runtime(root_paths=root_paths, turn_paths=turn_paths, report_generated=result.get("report_generated", False))
        root_index_result = dict(result)
        root_index_result["report_path"] = str(root_paths.report_path) if result.get("report_generated") else None
        self._write_runtime_index(
            root_paths,
            task_id,
            user_message,
            float(result.get("execution_time") or 0.0),
            root_index_result,
        )
        self._append_aggregate_events(root_paths.events_path, turn_paths.events_path)
        turn_record = TaskTurnRecord(
            turn_number=turn_number,
            user_message=user_message,
            answer=str(result.get("answer") or ""),
            status=str(result.get("evaluation_status") or ("cancelled" if result.get("cancelled") else "unknown")),
            runtime_dir=str(turn_paths.runtime_dir),
            report_path=result.get("report_path"),
            evaluation_feedback=result.get("final_summary"),
            uploaded_files=uploaded_file_records,
            created_at_utc=self._utc_now(),
        )
        self._append_session_history(task_workspace, turn_record)
        self._update_task_session_state(
            task_workspace=task_workspace,
            session_state=session_state,
            turn_number=turn_number,
            turn_status=turn_record.status,
        )
        result["task_id"] = task_id
        result["turn_number"] = turn_number
        result["session_path"] = str(self._session_state_path(task_workspace))
        result["session_history_path"] = str(self._session_history_path(task_workspace))
        result["uploaded_files"] = uploaded_file_records
        return result

    async def run_task(
        self,
        user_request: str,
        live: Optional[Live] = None,
        spinner: Optional[Spinner] = None,
        uploaded_files: Optional[Dict[str, bytes]] = None,
        report_requested: bool = False,
    ) -> dict:
        task_id = str(uuid.uuid4())
        task_workspace = self.workspace_dir / task_id
        paths = TaskRuntimePaths.build(task_workspace)
        paths.ensure_base_dirs()
        self.create_task_session(task_id, original_goal=user_request)
        return await self._execute_task_run(
            task_id=task_id,
            task_workspace=task_workspace,
            paths=paths,
            user_request=user_request,
            live=live,
            spinner=spinner,
            uploaded_files=uploaded_files,
            report_requested=report_requested,
            has_uploaded_files=bool(uploaded_files),
            allow_direct_response=True,
            session_context=None,
            progress_callback=None,
        )

    async def _execute_task_run(
        self,
        *,
        task_id: str,
        task_workspace: Path,
        paths: TaskRuntimePaths,
        user_request: str,
        live: Optional[Live],
        spinner: Optional[Spinner],
        uploaded_files: Optional[Dict[str, bytes]],
        report_requested: bool,
        has_uploaded_files: bool,
        allow_direct_response: bool,
        session_context: SessionPromptContext | None,
        progress_callback: Callable[[HealthFlowProgressEvent], None] | None,
    ) -> dict[str, Any]:
        if self.config.memory.write_policy == "reset_before_run":
            await self.experience_manager.reset()

        if uploaded_files:
            logger.info("[{}] Saving {} uploaded files to the workspace.", task_id, len(uploaded_files))
            for filename, content in uploaded_files.items():
                safe_filename = Path(filename).name
                file_path = paths.sandbox_dir / safe_filename
                with open(file_path, "wb") as handle:
                    handle.write(content)

        start_time = time.time()
        try:
            result = await self._run_task_flow(
                task_id,
                paths,
                user_request,
                live,
                spinner,
                report_requested,
                uploaded_files=uploaded_files,
                has_uploaded_files=has_uploaded_files,
                allow_direct_response=allow_direct_response,
                session_context=session_context,
                progress_callback=progress_callback,
            )
        except ExecutionCancelledError as exc:
            logger.warning("[{}] Task cancelled during execution.", task_id)
            result = self._build_cancelled_task_result(
                task_id=task_id,
                paths=paths,
                user_request=user_request,
                execution_result=exc.result,
                cancel_reason=exc.cancel_reason,
            )
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="turn_cancelled",
                    stage="run",
                    status="cancelled",
                    message=exc.cancel_reason,
                ),
            )
        except asyncio.CancelledError:
            logger.warning("[{}] Task cancelled before completion.", task_id)
            result = self._build_cancelled_task_result(
                task_id=task_id,
                paths=paths,
                user_request=user_request,
            )
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="turn_cancelled",
                    stage="run",
                    status="cancelled",
                    message="Task cancelled before completion.",
                ),
            )
        execution_time = round(time.time() - start_time, 2)
        result["execution_time"] = execution_time
        result["workspace_path"] = str(task_workspace)
        result["sandbox_path"] = str(paths.sandbox_dir)
        result["runtime_path"] = str(paths.runtime_dir)
        result["backend"] = self.config.active_executor_name
        result["executor_model"] = self.config.active_executor.model
        result["executor_provider"] = self.config.active_executor.provider
        result["planner_model"] = self.config.llm_config_for_role("planner").model_name
        result["llm_role_models"] = self._role_model_names()
        result["runtime_llm_keys"] = self.config.runtime_llm_keys
        result["memory_write_policy"] = self.config.memory.write_policy
        result["execution_environment"] = self.config.environment.model_dump(mode="json")
        result["workflow_recommendations"] = result.get("workflow_recommendations", [])
        result["available_project_cli_tools"] = result.get("available_project_cli_tools", [])
        result["runtime_index_path"] = str(paths.index_path)
        result["run_summary_path"] = str(paths.run_summary_path)
        result["run_trajectory_path"] = str(paths.run_trajectory_path)
        result["run_costs_path"] = str(paths.run_costs_path)
        result["final_evaluation_path"] = str(paths.final_evaluation_path)
        result["report_requested"] = report_requested
        result["report_generated"] = False
        result["report_path"] = None
        result["report_error"] = None
        result["response_mode"] = result.get("response_mode", "orchestrated")
        result["cancelled"] = result.get("cancelled", False)
        result["cancel_reason"] = result.get("cancel_reason")
        self._write_runtime_index(paths, task_id, user_request, execution_time, result)

        if report_requested and not result.get("cancelled", False):
            try:
                report_path = generate_task_report(
                    task_workspace,
                    runtime_dir=paths.runtime_dir,
                    report_path=paths.report_path,
                )
                result["report_generated"] = True
                result["report_path"] = str(report_path)
            except Exception as exc:
                logger.exception("[{}] Failed to generate report.md: {}", task_id, exc)
                result["report_generated"] = False
                result["report_path"] = None
                result["report_error"] = str(exc)
            self._write_runtime_index(paths, task_id, user_request, execution_time, result)

        self._emit_progress(
            progress_callback,
            HealthFlowProgressEvent(
                kind="turn_finished",
                stage="run",
                status="completed" if result.get("success") else ("cancelled" if result.get("cancelled") else "failed"),
                message=str(result.get("final_summary") or ""),
                metadata={
                    "runtime_path": str(paths.runtime_dir),
                    "report_path": result.get("report_path"),
                },
            ),
        )
        return result

    async def _run_task_flow(
        self,
        task_id: str,
        paths: TaskRuntimePaths,
        user_request: str,
        live: Optional[Live],
        spinner: Optional[Spinner],
        report_requested: bool,
        uploaded_files: Optional[Dict[str, bytes]] = None,
        has_uploaded_files: bool = False,
        allow_direct_response: bool = True,
        session_context: SessionPromptContext | None = None,
        progress_callback: Callable[[HealthFlowProgressEvent], None] | None = None,
    ) -> Dict[str, Any]:
        direct_response = None
        if allow_direct_response:
            direct_response = await self.direct_response_router.maybe_build_direct_response(
                user_request,
                has_uploaded_files=has_uploaded_files or bool(uploaded_files),
            )
        if direct_response is not None:
            return await self._run_direct_response_flow(
                task_id=task_id,
                paths=paths,
                user_request=user_request,
                direct_response=direct_response,
            )

        return await self._run_unified_flow(
            task_id,
            paths,
            user_request,
            live,
            spinner,
            report_requested,
            session_context=session_context,
            progress_callback=progress_callback,
        )

    async def _run_direct_response_flow(
        self,
        task_id: str,
        paths: TaskRuntimePaths,
        user_request: str,
        direct_response: DirectResponse,
    ) -> Dict[str, Any]:
        planning_usage = dict(direct_response.usage)
        total_estimated_cost_usd = direct_response.estimated_cost_usd
        data_profile = profile_workspace_data(paths.sandbox_dir, user_request)
        risk_findings = detect_risk_findings(user_request, data_profile)
        task_state = {
            "data_profile": asdict(data_profile),
            "risk_findings": [asdict(item) for item in risk_findings],
        }
        self._write_json(paths.task_state_path, task_state)
        self._append_runtime_event(
            paths,
            stage="run",
            event="task_state_profiled",
            status="completed",
            metadata={
                "task_family": data_profile.task_family,
                "domain_focus": data_profile.domain_focus,
            },
            path_map={"task_state": paths.relative_path(paths.task_state_path)},
        )

        memory_context = {
            "query": user_request,
            "task_family": data_profile.task_family,
            "domain_focus": data_profile.domain_focus,
            "dataset_signature": data_profile.dataset_signature,
            "capacity": 0,
            "selection_policy": ["Memory retrieval was skipped because the request used the direct-response path."],
            "selected": [],
            "safeguard_overrides": [],
            "suppressed_duplicates": [],
            "suppressed_competitors": [],
            "suppressed": [],
            "skipped": True,
            "skip_reason": direct_response.reason,
        }

        evaluation = {
            "status": "success",
            "score": 1.0,
            "failure_type": "none",
            "feedback": "Returned a built-in direct response for a lightweight conversational prompt.",
            "repair_instructions": [],
            "violated_constraints": [],
            "repair_hypotheses": [],
            "retry_recommended": False,
            "memory_worthy_insights": [],
            "reasoning": direct_response.reason,
        }
        direct_response_payload = {
            "mode": direct_response.mode,
            "category": direct_response.category,
            "answer": direct_response.answer,
            "reason": direct_response.reason,
            "model_name": direct_response.model_name,
            "usage": planning_usage,
            "estimated_cost_usd": direct_response.estimated_cost_usd,
            "memory_context": memory_context,
        }
        self._write_json(paths.direct_response_path, direct_response_payload)
        self._write_json(paths.final_evaluation_path, evaluation)

        trajectory = {
            "schema_version": _RUNTIME_SCHEMA_VERSION,
            "task_id": task_id,
            "user_request": user_request,
            "backend": self.config.active_executor_name,
            "executor_model": self.config.active_executor.model,
            "executor_provider": self.config.active_executor.provider,
            "planner_model": self.config.llm_config_for_role("planner").model_name,
            "llm_role_models": self._role_model_names(),
            "runtime_llm_keys": self.config.runtime_llm_keys,
            "memory_write_policy": self.config.memory.write_policy,
            "response_mode": direct_response.mode,
            "direct_response_category": direct_response.category,
            "direct_response_reason": direct_response.reason,
            "direct_response_model": direct_response.model_name,
            "direct_response_usage": planning_usage,
            "direct_response_estimated_cost_usd": direct_response.estimated_cost_usd,
            "available_project_cli_tools": [],
            "workflow_recommendations": [],
            "task_state_path": paths.relative_path(paths.task_state_path),
            "data_profile": task_state["data_profile"],
            "risk_findings": task_state["risk_findings"],
            "memory_context_path": paths.relative_path(paths.direct_response_path),
            "memory_retrieval": memory_context,
            "attempts": [],
        }
        self._write_json(paths.run_trajectory_path, trajectory)

        costs_payload = {
            "attempts": [],
            "run_total": {
                "attempts": 0,
                "planning": planning_usage,
                "execution": {},
                "evaluation": {},
                "reflection": {},
                "llm_estimated_cost_usd": total_estimated_cost_usd,
                "executor_estimated_cost_usd": None,
                "total_estimated_cost_usd": total_estimated_cost_usd,
            },
        }
        self._write_json(paths.run_costs_path, costs_payload)
        self._write_json(
            paths.run_summary_path,
            {
                "task_id": task_id,
                "mode": direct_response.mode,
                "success": True,
                "cancelled": False,
                "final_summary": "Returned a built-in direct response for a lightweight conversational prompt.",
                "answer": direct_response.answer,
                "evaluation_status": "success",
                "evaluation_score": 1.0,
                "attempt_count": 0,
                "available_project_cli_tools": [],
                "workflow_recommendations": [],
                "paths": {
                    "task_state": paths.relative_path(paths.task_state_path),
                    "direct_response": paths.relative_path(paths.direct_response_path),
                    "trajectory": paths.relative_path(paths.run_trajectory_path),
                    "costs": paths.relative_path(paths.run_costs_path),
                    "final_evaluation": paths.relative_path(paths.final_evaluation_path),
                },
            },
        )
        self._append_runtime_event(
            paths,
            stage="planner",
            event="direct_response_completed",
            status="completed",
            agent_name="direct_response_router",
            model_name=direct_response.model_name,
            usage=planning_usage,
            estimated_cost_usd=total_estimated_cost_usd,
            path_map={"direct_response": paths.relative_path(paths.direct_response_path)},
            metadata={"category": direct_response.category},
        )

        return {
            "success": True,
            "final_summary": "Returned a built-in direct response for a lightweight conversational prompt.",
            "answer": direct_response.answer,
            "evaluation_status": "success",
            "evaluation_score": 1.0,
            "memory_write_policy": self.config.memory.write_policy,
            "available_project_cli_tools": [],
            "workflow_recommendations": [],
            "backend_version": None,
            "executor_metadata": {},
            "usage_summary": {
                "planning": planning_usage,
                "execution": {},
                "evaluation": {},
            },
            "cost_summary": {
                "llm_estimated_cost_usd": total_estimated_cost_usd,
                "executor_estimated_cost_usd": None,
                "total_estimated_cost_usd": total_estimated_cost_usd,
            },
            "cost_analysis": costs_payload,
            "log_path": None,
            "prompt_path": None,
            "last_executor_log_path": None,
            "last_executor_prompt_path": None,
            "task_state_path": str(paths.task_state_path),
            "attempts_index": [],
            "response_mode": direct_response.mode,
            "cancelled": False,
            "cancel_reason": None,
        }

    async def _run_unified_flow(
        self,
        task_id: str,
        paths: TaskRuntimePaths,
        user_request: str,
        live: Optional[Live],
        spinner: Optional[Spinner],
        report_requested: bool,
        session_context: SessionPromptContext | None = None,
        progress_callback: Callable[[HealthFlowProgressEvent], None] | None = None,
    ) -> Dict[str, Any]:
        if spinner and live:
            spinner.text = "Profiling task state and preparing memory retrieval..."
        self._emit_progress(
            progress_callback,
            HealthFlowProgressEvent(
                kind="stage_started",
                stage="run",
                status="running",
                message="Profiling task state and preparing memory retrieval.",
            ),
        )

        data_profile = profile_workspace_data(paths.sandbox_dir, user_request)
        risk_findings = detect_risk_findings(user_request, data_profile)
        workflow_recommendations = self._workflow_recommendations(user_request, data_profile)
        available_project_cli_tools = self._available_project_cli_tools(user_request, data_profile)
        task_state = {
            "data_profile": asdict(data_profile),
            "risk_findings": [asdict(item) for item in risk_findings],
        }
        self._write_json(paths.task_state_path, task_state)
        self._append_runtime_event(
            paths,
            stage="run",
            event="task_state_profiled",
            status="completed",
            metadata={
                "task_family": data_profile.task_family,
                "domain_focus": data_profile.domain_focus,
            },
            path_map={"task_state": paths.relative_path(paths.task_state_path)},
        )

        trajectory: Dict[str, Any] = {
            "schema_version": _RUNTIME_SCHEMA_VERSION,
            "task_id": task_id,
            "user_request": user_request,
            "backend": self.config.active_executor_name,
            "executor_model": self.config.active_executor.model,
            "executor_provider": self.config.active_executor.provider,
            "planner_model": self.config.llm_config_for_role("planner").model_name,
            "llm_role_models": self._role_model_names(),
            "runtime_llm_keys": self.config.runtime_llm_keys,
            "memory_write_policy": self.config.memory.write_policy,
            "report_requested": report_requested,
            "execution_environment": self.config.environment.model_dump(mode="json"),
            "available_project_cli_tools": available_project_cli_tools,
            "workflow_recommendations": workflow_recommendations,
            "task_state_path": paths.relative_path(paths.task_state_path),
            "data_profile": task_state["data_profile"],
            "risk_findings": task_state["risk_findings"],
            "attempts": [],
        }
        if session_context is not None:
            trajectory["session_context"] = {
                "task_id": session_context.task_id,
                "turn_number": session_context.turn_number,
                "original_goal": session_context.original_goal,
                "previous_answer": session_context.previous_answer,
                "previous_feedback": session_context.previous_feedback,
                "recent_turns": session_context.recent_turns,
                "sandbox_artifacts": session_context.sandbox_artifacts,
                "current_uploads": session_context.current_uploads,
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
            attempt_paths = paths.attempt(attempt_num)
            attempt_paths.ensure_dirs()
            previous_feedback = self._feedback_from_attempt(trajectory["attempts"][-1]) if attempt_num > 1 else None
            if previous_feedback is None and session_context is not None:
                previous_feedback = session_context.previous_feedback
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="stage_started",
                    stage="memory",
                    status="running",
                    attempt=attempt_num,
                    message=f"Attempt {attempt_num}: retrieving memory.",
                ),
            )
            retrieval_context = self._build_retrieval_context(
                data_profile=data_profile,
                risk_findings=risk_findings,
                prior_failure_modes=self._prior_failure_modes(trajectory["attempts"]),
            )
            self._write_json(attempt_paths.retrieval_context_path, retrieval_context.model_dump(mode="json"))
            retrieval_result = await self.experience_manager.retrieve_experiences(
                user_request,
                retrieval_context=retrieval_context,
            )
            safeguard_experiences = retrieval_result.safeguard_experiences
            workflow_experiences = retrieval_result.workflow_experiences
            dataset_anchor_experiences = retrieval_result.dataset_anchor_experiences
            code_snippet_experiences = retrieval_result.code_snippet_experiences
            retrieval_audit = retrieval_result.audit.model_dump(mode="json")
            self._write_json(attempt_paths.retrieval_result_path, retrieval_audit)
            trajectory["memory_context_path"] = paths.relative_path(attempt_paths.retrieval_result_path)
            trajectory["memory_retrieval"] = retrieval_audit
            self._append_runtime_event(
                paths,
                attempt=attempt_num,
                stage="memory",
                event="retrieval_completed",
                status="completed",
                path_map={
                    "retrieval_context": paths.relative_path(attempt_paths.retrieval_context_path),
                    "retrieval_result": paths.relative_path(attempt_paths.retrieval_result_path),
                },
                metadata={
                    "selected_count": len(retrieval_audit.get("selected", [])),
                    "skipped": retrieval_audit.get("skipped", False),
                },
            )
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="stage_finished",
                    stage="memory",
                    status="completed",
                    attempt=attempt_num,
                    metadata={"selected_count": len(retrieval_audit.get("selected", []))},
                ),
            )

            if spinner and live:
                spinner.text = f"Attempt {attempt_num}: Planning..."
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="stage_started",
                    stage="planner",
                    status="running",
                    attempt=attempt_num,
                    message=f"Attempt {attempt_num}: planning.",
                ),
            )

            plan = await self.meta_agent.generate_plan(
                user_request=user_request,
                safeguard_experiences=safeguard_experiences,
                workflow_experiences=workflow_experiences,
                dataset_anchor_experiences=dataset_anchor_experiences,
                code_snippet_experiences=code_snippet_experiences,
                execution_environment=self.config.environment.summary_lines(),
                available_project_cli_tools=available_project_cli_tools,
                workflow_recommendations=workflow_recommendations,
                previous_feedback=previous_feedback,
                session_context=session_context,
            )
            planning_usage = self._capture_agent_usage(self.meta_agent, "planner")
            plan_markdown = plan.to_markdown()
            self._write_text(attempt_paths.planner_plan_markdown_path, plan_markdown)
            self._write_agent_trace_artifacts(
                input_messages_path=attempt_paths.planner_input_messages_path,
                output_raw_path=attempt_paths.planner_output_raw_path,
                output_parsed_path=attempt_paths.planner_output_parsed_path,
                call_path=attempt_paths.planner_call_path,
                repair_trace_path=attempt_paths.planner_repair_trace_path,
                trace=getattr(self.meta_agent, "last_trace", {}),
            )
            self._append_runtime_event(
                paths,
                attempt=attempt_num,
                stage="planner",
                event="plan_generated",
                status="completed",
                agent_name="planner",
                model_name=planning_usage.get("model_name"),
                usage=planning_usage.get("usage"),
                estimated_cost_usd=planning_usage.get("estimated_cost_usd"),
                path_map={
                    "plan": paths.relative_path(attempt_paths.planner_plan_markdown_path),
                    "planner_call": paths.relative_path(attempt_paths.planner_call_path),
                },
            )
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="stage_finished",
                    stage="planner",
                    status="completed",
                    attempt=attempt_num,
                    message=plan.objective,
                    metadata={"plan_path": str(attempt_paths.planner_plan_markdown_path)},
                ),
            )

            execution_context = ExecutionContext(
                user_request=user_request,
                plan=plan,
                execution_environment=self.config.environment,
                workflow_recommendations=workflow_recommendations,
                available_project_cli_tools=available_project_cli_tools,
                report_requested=report_requested,
                safeguard_memory=self._format_memory_lines(safeguard_experiences),
                workflow_memory=self._format_memory_lines(workflow_experiences),
                dataset_anchor_memory=self._format_memory_lines(dataset_anchor_experiences),
                code_snippet_memory=self._format_memory_lines(code_snippet_experiences),
                prior_feedback=previous_feedback,
                executor_artifact_dir=attempt_paths.executor_dir,
                session_context=session_context.execution_block() if session_context is not None else None,
                progress_callback=progress_callback,
                turn_number=session_context.turn_number if session_context is not None else None,
            )

            if spinner and live:
                spinner.text = f"Attempt {attempt_num}: Executing with {self.config.active_executor_name}..."
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="stage_started",
                    stage="executor",
                    status="running",
                    attempt=attempt_num,
                    message=f"Attempt {attempt_num}: executing with {self.config.active_executor_name}.",
                ),
            )
            execution_result = await self.executor.execute(
                execution_context,
                paths.sandbox_dir,
            )

            generated_answer = self._extract_answer_from_workspace(paths.sandbox_dir, execution_result.log, user_request)
            workspace_artifacts = self._workspace_artifact_paths(paths.sandbox_dir)
            self._write_json(
                attempt_paths.executor_command_path,
                {
                    "backend": execution_result.backend,
                    "command": execution_result.command,
                    "backend_version": execution_result.backend_version,
                    "executor_metadata": execution_result.executor_metadata,
                    "duration_seconds": execution_result.duration_seconds,
                    "timed_out": execution_result.timed_out,
                    "cancelled": execution_result.cancelled,
                    "cancel_reason": execution_result.cancel_reason,
                },
            )
            self._write_json(attempt_paths.executor_usage_path, execution_result.usage)
            self._write_json(attempt_paths.executor_telemetry_path, execution_result.telemetry)
            self._write_json(
                attempt_paths.executor_artifacts_index_path,
                {
                    "sandbox_artifacts": workspace_artifacts,
                    "generated_answer": generated_answer,
                },
            )
            self._append_runtime_event(
                paths,
                attempt=attempt_num,
                stage="executor",
                event="execution_completed",
                status="completed" if execution_result.success else "failed",
                backend=execution_result.backend,
                usage=execution_result.usage,
                estimated_cost_usd=execution_result.usage.get("estimated_cost_usd"),
                path_map={
                    "prompt": paths.relative_path(attempt_paths.executor_prompt_path),
                    "combined_log": paths.relative_path(attempt_paths.executor_combined_log_path),
                    "stdout": paths.relative_path(attempt_paths.executor_stdout_path),
                    "stderr": paths.relative_path(attempt_paths.executor_stderr_path),
                    "telemetry": paths.relative_path(attempt_paths.executor_telemetry_path),
                    "usage": paths.relative_path(attempt_paths.executor_usage_path),
                    "artifacts": paths.relative_path(attempt_paths.executor_artifacts_index_path),
                },
                metadata={
                    "return_code": execution_result.return_code,
                    "timed_out": execution_result.timed_out,
                    "cancelled": execution_result.cancelled,
                },
            )
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="log_chunk",
                    stage="executor",
                    status="completed" if execution_result.success else "failed",
                    attempt=attempt_num,
                    message=execution_result.log[-4000:],
                    metadata={"backend": execution_result.backend},
                ),
            )
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="artifact_delta",
                    stage="executor",
                    status="completed" if execution_result.success else "failed",
                    attempt=attempt_num,
                    metadata={
                        "artifacts": workspace_artifacts,
                        "generated_answer": generated_answer,
                    },
                ),
            )

            if spinner and live:
                spinner.text = f"Attempt {attempt_num}: Evaluating..."
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="stage_started",
                    stage="evaluator",
                    status="running",
                    attempt=attempt_num,
                    message=f"Attempt {attempt_num}: evaluating.",
                ),
            )
            verdict = await self.evaluator.evaluate(
                user_request=user_request,
                plan=plan,
                execution_log=execution_result.log,
                workspace_artifacts=workspace_artifacts,
                generated_answer=generated_answer,
                session_context=session_context,
            )
            verdict = self._normalize_evaluation_verdict(verdict)
            evaluation_usage = self._capture_agent_usage(self.evaluator, "evaluator")
            final_verdict = verdict
            self._write_agent_trace_artifacts(
                input_messages_path=attempt_paths.evaluator_input_messages_path,
                output_raw_path=attempt_paths.evaluator_output_raw_path,
                output_parsed_path=attempt_paths.evaluator_output_parsed_path,
                call_path=attempt_paths.evaluator_call_path,
                repair_trace_path=attempt_paths.evaluator_repair_trace_path,
                trace=getattr(self.evaluator, "last_trace", {}),
            )
            self._append_runtime_event(
                paths,
                attempt=attempt_num,
                stage="evaluator",
                event="evaluation_completed",
                status=verdict.status,
                agent_name="evaluator",
                model_name=evaluation_usage.get("model_name"),
                usage=evaluation_usage.get("usage"),
                estimated_cost_usd=evaluation_usage.get("estimated_cost_usd"),
                path_map={"evaluator_call": paths.relative_path(attempt_paths.evaluator_call_path)},
                metadata={
                    "score": verdict.score,
                    "failure_type": verdict.failure_type,
                    "retry_recommended": verdict.retry_recommended,
                },
            )
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="stage_finished",
                    stage="evaluator",
                    status=verdict.status,
                    attempt=attempt_num,
                    message=verdict.feedback,
                    metadata={
                        "score": verdict.score,
                        "failure_type": verdict.failure_type,
                    },
                ),
            )

            attempt_history = {
                "attempt": attempt_num,
                "memory_context_path": paths.relative_path(attempt_paths.retrieval_result_path),
                "memory": {
                    "retrieval": retrieval_audit,
                    "safeguards": [exp.model_dump(mode="json") for exp in safeguard_experiences],
                    "workflows": [exp.model_dump(mode="json") for exp in workflow_experiences],
                    "dataset_anchors": [exp.model_dump(mode="json") for exp in dataset_anchor_experiences],
                    "code_snippets": [exp.model_dump(mode="json") for exp in code_snippet_experiences],
                },
                "plan": plan.model_dump(mode="json"),
                "plan_markdown": plan_markdown,
                "planning": planning_usage,
                "execution": {
                    "success": execution_result.success,
                    "return_code": execution_result.return_code,
                    "log": execution_result.log,
                    "log_path": paths.relative_path(attempt_paths.executor_combined_log_path),
                    "prompt_path": paths.relative_path(attempt_paths.executor_prompt_path),
                    "stdout_path": paths.relative_path(attempt_paths.executor_stdout_path),
                    "stderr_path": paths.relative_path(attempt_paths.executor_stderr_path),
                    "backend": execution_result.backend,
                    "backend_version": execution_result.backend_version,
                    "executor_metadata": execution_result.executor_metadata,
                    "command": execution_result.command,
                    "duration_seconds": execution_result.duration_seconds,
                    "timed_out": execution_result.timed_out,
                    "usage": execution_result.usage,
                    "telemetry": execution_result.telemetry,
                    "cancelled": execution_result.cancelled,
                    "cancel_reason": execution_result.cancel_reason,
                },
                "artifacts": {
                    "sandbox_paths": workspace_artifacts,
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
                "paths": {
                    "attempt_dir": paths.relative_path(attempt_paths.attempt_dir),
                    "memory": {
                        "retrieval_context": paths.relative_path(attempt_paths.retrieval_context_path),
                        "retrieval_result": paths.relative_path(attempt_paths.retrieval_result_path),
                    },
                    "planner": {
                        "plan": paths.relative_path(attempt_paths.planner_plan_markdown_path),
                        "input_messages": paths.relative_path(attempt_paths.planner_input_messages_path),
                        "output_raw": paths.relative_path(attempt_paths.planner_output_raw_path),
                        "output_parsed": paths.relative_path(attempt_paths.planner_output_parsed_path),
                        "call": paths.relative_path(attempt_paths.planner_call_path),
                        "repair_trace": paths.relative_path(attempt_paths.planner_repair_trace_path),
                    },
                    "executor": {
                        "prompt": paths.relative_path(attempt_paths.executor_prompt_path),
                        "command": paths.relative_path(attempt_paths.executor_command_path),
                        "stdout": paths.relative_path(attempt_paths.executor_stdout_path),
                        "stderr": paths.relative_path(attempt_paths.executor_stderr_path),
                        "combined_log": paths.relative_path(attempt_paths.executor_combined_log_path),
                        "telemetry": paths.relative_path(attempt_paths.executor_telemetry_path),
                        "usage": paths.relative_path(attempt_paths.executor_usage_path),
                        "artifacts_index": paths.relative_path(attempt_paths.executor_artifacts_index_path),
                    },
                    "evaluator": {
                        "input_messages": paths.relative_path(attempt_paths.evaluator_input_messages_path),
                        "output_raw": paths.relative_path(attempt_paths.evaluator_output_raw_path),
                        "output_parsed": paths.relative_path(attempt_paths.evaluator_output_parsed_path),
                        "call": paths.relative_path(attempt_paths.evaluator_call_path),
                        "repair_trace": paths.relative_path(attempt_paths.evaluator_repair_trace_path),
                    },
                },
            }
            trajectory["attempts"].append(attempt_history)
            self._write_json(attempt_paths.summary_path, self._attempt_summary(attempt_history))

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

        should_write_memory = (
            self.config.memory.write_policy != "freeze"
            and self._should_synthesize_memory(trajectory)
        )
        if should_write_memory:
            if spinner and live:
                spinner.text = "Synthesizing memory..."
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="stage_started",
                    stage="reflection",
                    status="running",
                    message="Synthesizing memory.",
                ),
            )
            reflection_output = await self.reflector.synthesize_experience(
                trajectory,
                final_verdict=final_verdict,
            )
            reflection_usage = self._capture_agent_usage(self.reflector, "reflector")
            if isinstance(reflection_output, list):
                new_experiences = reflection_output
                memory_updates = []
            else:
                new_experiences = reflection_output.experiences
                memory_updates = reflection_output.memory_updates
            if memory_updates:
                applied_memory_update_ids = await self.experience_manager.apply_memory_updates(memory_updates)
                if applied_memory_update_ids:
                    trajectory["memory_updates"] = [
                        update.model_dump(mode="json")
                        for update in memory_updates
                        if update.experience_id in set(applied_memory_update_ids)
                    ]
            if new_experiences:
                await self.experience_manager.save_experiences(new_experiences)
                trajectory["new_experiences"] = [exp.model_dump(mode="json") for exp in new_experiences]
            self._write_agent_trace_artifacts(
                input_messages_path=paths.reflection_input_path,
                output_raw_path=paths.reflection_output_raw_path,
                output_parsed_path=paths.reflection_output_parsed_path,
                call_path=paths.reflection_call_path,
                repair_trace_path=paths.reflection_repair_trace_path,
                trace=getattr(self.reflector, "last_trace", {}),
            )
            self._append_runtime_event(
                paths,
                stage="reflection",
                event="reflection_completed",
                status="completed",
                agent_name="reflector",
                model_name=reflection_usage.get("model_name") if reflection_usage else None,
                usage=reflection_usage.get("usage") if reflection_usage else None,
                estimated_cost_usd=reflection_usage.get("estimated_cost_usd") if reflection_usage else None,
                path_map={"reflection_call": paths.relative_path(paths.reflection_call_path)},
                metadata={
                    "new_experiences": len(new_experiences),
                    "memory_updates": len(memory_updates),
                },
            )
            self._emit_progress(
                progress_callback,
                HealthFlowProgressEvent(
                    kind="stage_finished",
                    stage="reflection",
                    status="completed",
                    metadata={
                        "new_experiences": len(new_experiences),
                        "memory_updates": len(memory_updates),
                    },
                ),
            )
        if reflection_usage:
            trajectory["reflection"] = reflection_usage

        self._write_json(paths.run_trajectory_path, trajectory)
        self._write_json(paths.final_evaluation_path, final_verdict.model_dump(mode="json"))
        last_execution = trajectory["attempts"][-1]["execution"] if trajectory["attempts"] else {}
        usage_summary = self._summarize_run_usage(trajectory["attempts"], reflection_usage)
        cost_summary = self._summarize_run_costs(usage_summary)
        cost_analysis = self._summarize_cost_analysis(trajectory["attempts"], usage_summary, cost_summary, reflection_usage)
        self._write_json(paths.run_costs_path, cost_analysis)
        attempts_index = self._attempts_index(trajectory["attempts"])
        self._write_json(
            paths.run_summary_path,
            {
                "task_id": task_id,
                "mode": "orchestrated",
                "success": is_success,
                "cancelled": False,
                "final_summary": final_summary,
                "answer": final_answer,
                "evaluation_status": final_verdict.status,
                "evaluation_score": final_verdict.score,
                "attempt_count": len(trajectory["attempts"]),
                "attempts": attempts_index,
                "available_project_cli_tools": available_project_cli_tools,
                "workflow_recommendations": workflow_recommendations,
                "paths": {
                    "task_state": paths.relative_path(paths.task_state_path),
                    "trajectory": paths.relative_path(paths.run_trajectory_path),
                    "costs": paths.relative_path(paths.run_costs_path),
                    "final_evaluation": paths.relative_path(paths.final_evaluation_path),
                },
            },
        )

        return {
            "success": is_success,
            "final_summary": final_summary,
            "answer": final_answer,
            "evaluation_status": final_verdict.status,
            "evaluation_score": final_verdict.score,
            "memory_write_policy": self.config.memory.write_policy,
            "available_project_cli_tools": available_project_cli_tools,
            "workflow_recommendations": workflow_recommendations,
            "backend_version": last_execution.get("backend_version"),
            "executor_metadata": last_execution.get("executor_metadata"),
            "usage_summary": usage_summary,
            "cost_summary": cost_summary,
            "cost_analysis": cost_analysis,
            "log_path": str(attempt_paths.executor_combined_log_path) if trajectory["attempts"] else None,
            "prompt_path": str(attempt_paths.executor_prompt_path) if trajectory["attempts"] else None,
            "last_executor_log_path": str(attempt_paths.executor_combined_log_path) if trajectory["attempts"] else None,
            "last_executor_prompt_path": str(attempt_paths.executor_prompt_path) if trajectory["attempts"] else None,
            "task_state_path": str(paths.task_state_path),
            "attempts_index": attempts_index,
            "response_mode": "orchestrated",
            "cancelled": False,
            "cancel_reason": None,
        }

    def _build_cancelled_task_result(
        self,
        *,
        task_id: str,
        paths: TaskRuntimePaths,
        user_request: str,
        execution_result=None,
        cancel_reason: str = "Task cancelled by user.",
    ) -> dict[str, Any]:
        attempt_paths = paths.attempt(1)
        generated_answer = (
            self._extract_answer_from_workspace(paths.sandbox_dir, execution_result.log, user_request)
            if execution_result is not None
            else "Task cancelled before completion."
        )
        data_profile = profile_workspace_data(paths.sandbox_dir, user_request)
        workflow_recommendations = self._workflow_recommendations(user_request, data_profile)
        available_project_cli_tools = self._available_project_cli_tools(user_request, data_profile)
        if not paths.task_state_path.exists():
            self._write_json(
                paths.task_state_path,
                {
                    "data_profile": asdict(data_profile),
                    "risk_findings": [asdict(item) for item in detect_risk_findings(user_request, data_profile)],
                },
            )
        evaluation = self._cancelled_evaluation_payload(cancel_reason)
        self._write_json(paths.final_evaluation_path, evaluation)
        trajectory: dict[str, Any] = {
            "schema_version": _RUNTIME_SCHEMA_VERSION,
            "task_id": task_id,
            "user_request": user_request,
            "backend": self.config.active_executor_name,
            "executor_model": self.config.active_executor.model,
            "executor_provider": self.config.active_executor.provider,
            "planner_model": self.config.llm_config_for_role("planner").model_name,
            "llm_role_models": self._role_model_names(),
            "runtime_llm_keys": self.config.runtime_llm_keys,
            "memory_write_policy": self.config.memory.write_policy,
            "available_project_cli_tools": available_project_cli_tools,
            "workflow_recommendations": workflow_recommendations,
            "task_state_path": paths.relative_path(paths.task_state_path),
            "memory_context_path": None,
            "memory_retrieval": self._minimal_memory_context(user_request, cancel_reason),
            "attempts": [],
            "cancelled": True,
            "cancel_reason": cancel_reason,
        }
        if execution_result is not None:
            attempt_history = self._cancelled_attempt_history(
                execution_result=execution_result,
                paths=paths,
                generated_answer=generated_answer,
                user_request=user_request,
            )
            trajectory["attempts"].append(attempt_history)
            self._write_json(paths.attempt(1).summary_path, self._attempt_summary(attempt_history))
        self._write_json(paths.run_trajectory_path, trajectory)

        usage_summary = {
            "planning": {},
            "evaluation": {},
            "execution": self._aggregate_execution_records(
                [{"execution": self._execution_record_from_result(execution_result)}]
            )
            if execution_result is not None
            else {},
        }
        cost_summary = self._summarize_run_costs(usage_summary)
        cost_analysis = {
            "attempts": [],
            "run_total": {
                "attempts": 1 if execution_result is not None else 0,
                "planning": {},
                "execution": usage_summary["execution"],
                "evaluation": {},
                "reflection": {},
                "llm_estimated_cost_usd": cost_summary["llm_estimated_cost_usd"],
                "executor_estimated_cost_usd": cost_summary["executor_estimated_cost_usd"],
                "total_estimated_cost_usd": cost_summary["total_estimated_cost_usd"],
            },
        }
        self._write_json(paths.run_costs_path, cost_analysis)
        attempts_index = self._attempts_index(trajectory["attempts"])
        self._write_json(
            paths.run_summary_path,
            {
                "task_id": task_id,
                "mode": "orchestrated",
                "success": False,
                "cancelled": True,
                "cancel_reason": cancel_reason,
                "final_summary": cancel_reason,
                "answer": generated_answer,
                "evaluation_status": "cancelled",
                "evaluation_score": 0.0,
                "attempt_count": len(trajectory["attempts"]),
                "attempts": attempts_index,
                "available_project_cli_tools": available_project_cli_tools,
                "workflow_recommendations": workflow_recommendations,
                "paths": {
                    "task_state": paths.relative_path(paths.task_state_path),
                    "trajectory": paths.relative_path(paths.run_trajectory_path),
                    "costs": paths.relative_path(paths.run_costs_path),
                    "final_evaluation": paths.relative_path(paths.final_evaluation_path),
                },
            },
        )
        self._append_runtime_event(
            paths,
            stage="run",
            event="task_cancelled",
            status="cancelled",
            backend=execution_result.backend if execution_result is not None else None,
            usage=execution_result.usage if execution_result is not None else None,
            estimated_cost_usd=execution_result.usage.get("estimated_cost_usd") if execution_result is not None else None,
            metadata={"cancel_reason": cancel_reason},
        )

        return {
            "success": False,
            "final_summary": cancel_reason,
            "answer": generated_answer,
            "evaluation_status": "cancelled",
            "evaluation_score": 0.0,
            "memory_write_policy": self.config.memory.write_policy,
            "available_project_cli_tools": available_project_cli_tools,
            "workflow_recommendations": workflow_recommendations,
            "backend_version": execution_result.backend_version if execution_result is not None else None,
            "executor_metadata": execution_result.executor_metadata if execution_result is not None else {},
            "usage_summary": usage_summary,
            "cost_summary": cost_summary,
            "cost_analysis": cost_analysis,
            "log_path": str(attempt_paths.executor_combined_log_path) if execution_result is not None else None,
            "prompt_path": str(attempt_paths.executor_prompt_path) if execution_result is not None else None,
            "last_executor_log_path": str(attempt_paths.executor_combined_log_path) if execution_result is not None else None,
            "last_executor_prompt_path": str(attempt_paths.executor_prompt_path) if execution_result is not None else None,
            "task_state_path": str(paths.task_state_path),
            "attempts_index": attempts_index,
            "response_mode": "orchestrated",
            "cancelled": True,
            "cancel_reason": cancel_reason,
        }

    def _cancelled_attempt_history(
        self,
        *,
        execution_result,
        paths: TaskRuntimePaths,
        generated_answer: str,
        user_request: str,
    ) -> dict[str, Any]:
        attempt_paths = paths.attempt(1)
        attempt_paths.ensure_dirs()
        if execution_result.prompt_path and Path(execution_result.prompt_path).exists() and not attempt_paths.executor_prompt_path.exists():
            self._write_text(
                attempt_paths.executor_prompt_path,
                Path(execution_result.prompt_path).read_text(encoding="utf-8"),
            )
        if execution_result.stdout and not attempt_paths.executor_stdout_path.exists():
            self._write_text(attempt_paths.executor_stdout_path, execution_result.stdout)
        if execution_result.stderr and not attempt_paths.executor_stderr_path.exists():
            self._write_text(attempt_paths.executor_stderr_path, execution_result.stderr)
        if execution_result.log and not attempt_paths.executor_combined_log_path.exists():
            self._write_text(attempt_paths.executor_combined_log_path, execution_result.log)
        execution_record = self._execution_record_from_result(execution_result, paths=paths, attempt_paths=attempt_paths)
        workspace_artifacts = self._workspace_artifact_paths(paths.sandbox_dir)
        minimal_memory = self._minimal_memory_context(
            user_request,
            execution_result.cancel_reason or "Execution cancelled by user.",
        )
        self._write_json(attempt_paths.retrieval_result_path, minimal_memory)
        self._write_json(
            attempt_paths.executor_command_path,
            {
                "backend": execution_result.backend,
                "command": execution_result.command,
                "backend_version": execution_result.backend_version,
                "executor_metadata": execution_result.executor_metadata,
                "duration_seconds": execution_result.duration_seconds,
                "timed_out": execution_result.timed_out,
                "cancelled": execution_result.cancelled,
                "cancel_reason": execution_result.cancel_reason,
            },
        )
        self._write_json(attempt_paths.executor_usage_path, execution_result.usage)
        self._write_json(attempt_paths.executor_telemetry_path, execution_result.telemetry)
        self._write_json(
            attempt_paths.executor_artifacts_index_path,
            {
                "sandbox_artifacts": workspace_artifacts,
                "generated_answer": generated_answer,
            },
        )
        return {
            "attempt": 1,
            "memory_context_path": paths.relative_path(attempt_paths.retrieval_result_path),
            "memory": {
                "retrieval": minimal_memory,
                "safeguards": [],
                "workflows": [],
                "dataset_anchors": [],
                "code_snippets": [],
            },
            "plan": {},
            "plan_markdown": "",
            "planning": {},
            "execution": execution_record,
            "artifacts": {
                "sandbox_paths": workspace_artifacts,
                "generated_answer": generated_answer,
            },
            "gate": {
                "execution_ok": False,
                "evaluation_threshold_ok": False,
                "verdict_success": False,
                "retry_recommended": False,
            },
            "evaluation": self._cancelled_evaluation_payload(execution_result.cancel_reason or "Execution cancelled by user."),
            "evaluation_meta": {},
            "paths": {
                "attempt_dir": paths.relative_path(attempt_paths.attempt_dir),
                "memory": {
                    "retrieval_context": paths.relative_path(attempt_paths.retrieval_context_path),
                    "retrieval_result": paths.relative_path(attempt_paths.retrieval_result_path),
                },
                "planner": {
                    "plan": paths.relative_path(attempt_paths.planner_plan_markdown_path),
                    "input_messages": paths.relative_path(attempt_paths.planner_input_messages_path),
                    "output_raw": paths.relative_path(attempt_paths.planner_output_raw_path),
                    "output_parsed": paths.relative_path(attempt_paths.planner_output_parsed_path),
                    "call": paths.relative_path(attempt_paths.planner_call_path),
                    "repair_trace": paths.relative_path(attempt_paths.planner_repair_trace_path),
                },
                "executor": {
                    "prompt": paths.relative_path(attempt_paths.executor_prompt_path),
                    "command": paths.relative_path(attempt_paths.executor_command_path),
                    "stdout": paths.relative_path(attempt_paths.executor_stdout_path),
                    "stderr": paths.relative_path(attempt_paths.executor_stderr_path),
                    "combined_log": paths.relative_path(attempt_paths.executor_combined_log_path),
                    "telemetry": paths.relative_path(attempt_paths.executor_telemetry_path),
                    "usage": paths.relative_path(attempt_paths.executor_usage_path),
                    "artifacts_index": paths.relative_path(attempt_paths.executor_artifacts_index_path),
                },
                "evaluator": {
                    "input_messages": paths.relative_path(attempt_paths.evaluator_input_messages_path),
                    "output_raw": paths.relative_path(attempt_paths.evaluator_output_raw_path),
                    "output_parsed": paths.relative_path(attempt_paths.evaluator_output_parsed_path),
                    "call": paths.relative_path(attempt_paths.evaluator_call_path),
                    "repair_trace": paths.relative_path(attempt_paths.evaluator_repair_trace_path),
                },
            },
        }

    def _execution_record_from_result(
        self,
        execution_result,
        *,
        paths: TaskRuntimePaths | None = None,
        attempt_paths: AttemptPaths | None = None,
    ) -> dict[str, Any]:
        if execution_result is None:
            return {}
        log_path = execution_result.log_path
        prompt_path = execution_result.prompt_path
        stdout_path = None
        stderr_path = None
        if paths is not None and attempt_paths is not None:
            log_path = paths.relative_path(attempt_paths.executor_combined_log_path)
            prompt_path = paths.relative_path(attempt_paths.executor_prompt_path)
            stdout_path = paths.relative_path(attempt_paths.executor_stdout_path)
            stderr_path = paths.relative_path(attempt_paths.executor_stderr_path)
        return {
            "success": execution_result.success,
            "return_code": execution_result.return_code,
            "log": execution_result.log,
            "log_path": log_path,
            "prompt_path": prompt_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "backend": execution_result.backend,
            "backend_version": execution_result.backend_version,
            "executor_metadata": execution_result.executor_metadata,
            "command": execution_result.command,
            "duration_seconds": execution_result.duration_seconds,
            "timed_out": execution_result.timed_out,
            "usage": execution_result.usage,
            "telemetry": execution_result.telemetry,
            "cancelled": execution_result.cancelled,
            "cancel_reason": execution_result.cancel_reason,
        }

    def _cancelled_evaluation_payload(self, cancel_reason: str) -> dict[str, Any]:
        return {
            "status": "cancelled",
            "score": 0.0,
            "failure_type": "cancelled_by_user",
            "feedback": cancel_reason,
            "repair_instructions": [],
            "violated_constraints": [],
            "repair_hypotheses": [],
            "retry_recommended": False,
            "memory_worthy_insights": [],
            "reasoning": cancel_reason,
        }

    def _minimal_memory_context(self, query: str, cancel_reason: str) -> dict[str, Any]:
        return {
            "query": query,
            "task_family": "unknown",
            "domain_focus": "unknown",
            "dataset_signature": "unknown",
            "capacity": 0,
            "selection_policy": ["Memory retrieval did not complete before cancellation."],
            "selected": [],
            "safeguard_overrides": [],
            "suppressed_duplicates": [],
            "suppressed_competitors": [],
            "suppressed": [],
            "skipped": True,
            "skip_reason": cancel_reason,
        }

    def _normalize_evaluation_verdict(self, verdict: EvaluationVerdict) -> EvaluationVerdict:
        return verdict

    def _extract_answer_from_workspace(self, task_workspace: Path, execution_log: str, user_request: str) -> str:
        final_report = task_workspace / "final_report.md"
        if final_report.exists():
            return final_report.read_text(encoding="utf-8", errors="ignore")[:4000]

        metrics_file = next(iter(sorted(task_workspace.glob("metrics.*"))), None)
        if metrics_file:
            return metrics_file.read_text(encoding="utf-8", errors="ignore")[:2000]

        stdout_blocks = self._stdout_blocks_from_log(execution_log)
        best_stdout_answer = self._best_stdout_answer(stdout_blocks, user_request)
        if best_stdout_answer:
            return best_stdout_answer

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
                return line_content[:2000]

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("STDOUT: "):
                content = lines[i].replace("STDOUT: ", "").strip()
                if content and len(content) > 10:
                    return content

        return (
            "Execution completed. Review the workspace artifacts for details. "
            f"The original request was: '{user_request}'"
        )

    def _stdout_blocks_from_log(self, execution_log: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        current_lines: list[str] = []
        saw_step_events = False

        def flush_current_block(step_reason: str | None = None) -> None:
            if not current_lines:
                return

            text = "\n".join(current_lines).strip()
            current_lines.clear()
            if not text:
                return

            body, answer_marker_used = self._extract_answer_block_body(text)
            blocks.append(
                {
                    "index": len(blocks),
                    "text": text,
                    "body": body,
                    "step_reason": step_reason,
                    "answer_marker_used": answer_marker_used,
                }
            )

        for line in execution_log.splitlines():
            if line.startswith("EVENT: step_start"):
                saw_step_events = True
                flush_current_block()
                continue
            if line.startswith("EVENT: step_finish"):
                saw_step_events = True
                flush_current_block(self._step_reason_from_log_line(line))
                continue
            if line.startswith("STDOUT: "):
                current_lines.append(line.replace("STDOUT: ", "", 1))
                continue
            if not saw_step_events:
                flush_current_block()

        flush_current_block()
        return blocks

    def _best_stdout_answer(self, stdout_blocks: list[dict[str, Any]], user_request: str) -> str | None:
        if not stdout_blocks:
            return None

        stop_blocks = [block for block in stdout_blocks if block.get("step_reason") == "stop"]
        candidate_blocks = stop_blocks or stdout_blocks
        request_tokens = self._content_tokens(user_request)
        scored_blocks: list[tuple[float, int, str]] = []

        for index, block in enumerate(candidate_blocks):
            body = str(block.get("body") or block.get("text") or "").strip()
            if not body:
                continue

            score = 0.0
            if block.get("step_reason") == "stop":
                score += 6.0
            if not self._looks_like_process_narration(body):
                score += 2.0
            if self._looks_like_direct_answer(body):
                score += 2.0
            if block.get("answer_marker_used"):
                score += 3.0
            if len(body) >= 40:
                score += 1.0

            overlap = len(self._content_tokens(body) & request_tokens)
            score += overlap * 3.0
            if request_tokens:
                score += overlap / len(request_tokens)

            scored_blocks.append((score, index, body[:2000]))

        if not scored_blocks:
            return None

        best_score, _, best_body = max(scored_blocks, key=lambda item: (item[0], item[1]))
        if best_score <= 0:
            return None
        return best_body

    def _extract_answer_block_body(self, text: str) -> tuple[str, bool]:
        stripped_text = text.strip()
        if not stripped_text:
            return "", False

        lines = stripped_text.splitlines()
        marker_index: int | None = None
        for index, line in enumerate(lines):
            normalized = " ".join(line.lower().split())
            if any(marker in normalized for marker in _ANSWER_INTRO_MARKERS):
                marker_index = index

        if marker_index is None:
            return stripped_text[:2000], False

        inline_answer = self._inline_answer_from_line(lines[marker_index])
        trailing_text = "\n".join(lines[marker_index + 1 :]).strip()
        if inline_answer and trailing_text:
            return f"{inline_answer}\n{trailing_text}"[:2000], True
        if inline_answer:
            return inline_answer[:2000], True
        if trailing_text:
            return trailing_text[:2000], True
        return stripped_text[:2000], False

    def _inline_answer_from_line(self, line: str) -> str | None:
        stripped_line = line.strip()
        for pattern in _INLINE_ANSWER_PATTERNS:
            match = pattern.search(stripped_line)
            if match:
                answer = match.group(1).strip()
                if answer:
                    return answer
        return None

    def _step_reason_from_log_line(self, line: str) -> str | None:
        match = _STEP_FINISH_REASON_RE.search(line)
        if match:
            return match.group(1)
        return None

    def _content_tokens(self, text: str) -> set[str]:
        tokens: set[str] = set()
        for raw_token in _CONTENT_TOKEN_RE.findall(text.lower()):
            token = raw_token.strip("'")
            if len(token) < 2 or token in _REQUEST_TOKEN_STOPWORDS:
                continue
            tokens.add(token)
            tokens.update(_TOKEN_EQUIVALENTS.get(token, {token}))
        return tokens

    def _looks_like_process_narration(self, text: str) -> bool:
        normalized = " ".join(text.lower().split())
        markers = (
            "i'll start by",
            "i will start by",
            "i'll inspect",
            "i will inspect",
            "let me check",
            "let me inspect",
            "let me look",
            "now i understand",
            "based on the execution plan",
            "to understand the context",
            "i need to inspect",
            "i'm going to inspect",
        )
        return normalized.startswith(markers)

    def _looks_like_direct_answer(self, text: str) -> bool:
        normalized = text.lower()
        if normalized.startswith(("hello", "hi", "hey")):
            return True
        answer_markers = (
            "i'm ",
            "i am ",
            "here to help",
            "i can help",
            "i can assist",
            "healthflow",
        )
        return any(marker in normalized for marker in answer_markers)

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

    def _should_synthesize_memory(self, full_history: dict[str, Any]) -> bool:
        attempts = full_history.get("attempts", [])
        if not attempts:
            return False

        data_profile = full_history.get("data_profile", {})
        if data_profile.get("domain_focus") == "ehr":
            return True
        if data_profile.get("task_family") != "general_analysis":
            return True
        if len(attempts) > 1:
            return True

        workspace_paths = attempts[-1].get("artifacts", {}).get("sandbox_paths", [])
        non_runtime_paths = [path for path in workspace_paths if not self._is_runtime_artifact_path(path)]
        if non_runtime_paths:
            return True

        return len(str(full_history.get("user_request", "")).split()) > 20

    def _is_runtime_artifact_path(self, relative_path: str) -> bool:
        return relative_path.startswith(".healthflow_pi_agent/")

    def _format_memory_lines(self, experiences: list[Any]) -> list[str]:
        lines: list[str] = []
        for exp in experiences:
            kind = exp.kind.value if hasattr(exp.kind, "value") else str(exp.kind)
            source_outcome = (
                exp.source_outcome.value
                if hasattr(exp.source_outcome, "value")
                else str(exp.source_outcome)
            )
            lines.append(f"[{kind}/{source_outcome}] {exp.category}: {exp.content}")
        return lines

    def _feedback_from_attempt(self, attempt: dict[str, Any]) -> str | None:
        if not attempt:
            return None
        evaluation = attempt.get("evaluation", {})
        feedback = evaluation.get("feedback")
        repairs = evaluation.get("repair_instructions", [])
        violated_constraints = evaluation.get("violated_constraints", [])
        repair_hypotheses = evaluation.get("repair_hypotheses", [])
        parts = [feedback] if feedback else []
        if violated_constraints:
            parts.append("Violated constraints: " + "; ".join(violated_constraints))
        if repairs:
            parts.append("Repair instructions: " + "; ".join(repairs))
        if repair_hypotheses:
            parts.append("Repair hypotheses: " + "; ".join(repair_hypotheses))
        return "\n".join(parts) if parts else None

    def _prior_failure_modes(self, attempts: list[dict[str, Any]]) -> list[str]:
        failure_modes: list[str] = []
        for attempt in attempts:
            evaluation = attempt.get("evaluation", {})
            failure_type = evaluation.get("failure_type")
            if failure_type and failure_type != "none":
                failure_modes.append(str(failure_type))
        return list(dict.fromkeys(failure_modes))

    def _workflow_recommendations(self, user_request: str, data_profile) -> list[str]:
        return self.workflow_broker.recommend(user_request, data_profile)

    def _available_project_cli_tools(self, user_request: str, data_profile) -> list[str]:
        return self.workflow_broker.available_project_cli_tools(user_request, data_profile)

    def _build_retrieval_context(
        self,
        data_profile,
        risk_findings: list[Any],
        prior_failure_modes: list[str],
    ) -> RetrievalContext:
        return RetrievalContext(
            task_family=data_profile.task_family,
            domain_focus=data_profile.domain_focus,
            dataset_signature=data_profile.dataset_signature,
            schema_tags=self._schema_tags_for_profile(data_profile),
            risk_tags=[
                finding.category
                for finding in risk_findings
                if getattr(finding, "severity", None) not in {"info", "low"}
            ],
            prior_failure_modes=prior_failure_modes,
        )

    def _schema_tags_for_profile(self, data_profile) -> list[str]:
        tags: list[str] = [f"modality:{item}" for item in data_profile.modalities]
        if data_profile.domain_focus == "ehr":
            tags.append("domain:ehr")
        if data_profile.group_id_columns:
            tags.append("group_id")
        if data_profile.patient_id_columns:
            tags.append("patient_id")
        if data_profile.target_columns:
            tags.append("target_column")
        if data_profile.time_columns:
            tags.append("time_column")
        tags.extend(f"column:{item}" for item in data_profile.patient_id_columns[:3])
        tags.extend(f"column:{item}" for item in data_profile.target_columns[:3])
        tags.extend(f"column:{item}" for item in data_profile.time_columns[:3])
        return list(dict.fromkeys(tags))

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _read_json(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, cls=DateTimeEncoder)

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _write_agent_trace_artifacts(
        self,
        *,
        input_messages_path: Path,
        output_raw_path: Path,
        output_parsed_path: Path,
        call_path: Path,
        repair_trace_path: Path,
        trace: dict[str, Any],
    ) -> None:
        if not trace:
            self._write_json(input_messages_path, [])
            self._write_text(output_raw_path, "")
            self._write_json(output_parsed_path, {})
            self._write_json(call_path, {})
            self._write_json(repair_trace_path, {})
            return
        self._write_json(input_messages_path, trace.get("input_messages", []))
        self._write_text(output_raw_path, str(trace.get("output_raw", "")))
        self._write_json(output_parsed_path, trace.get("output_parsed", {}))
        self._write_json(call_path, trace.get("call", {}))
        self._write_json(repair_trace_path, trace.get("repair_trace", {}))

    def _emit_progress(
        self,
        progress_callback: Callable[[HealthFlowProgressEvent], None] | None,
        event: HealthFlowProgressEvent,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(event)
        except Exception as exc:
            logger.debug("HealthFlow progress callback failed: {}", exc)

    def _session_state_path(self, task_workspace: Path) -> Path:
        return task_workspace / "runtime" / "session.json"

    def _session_history_path(self, task_workspace: Path) -> Path:
        return task_workspace / "runtime" / "history.jsonl"

    def _turn_runtime_dir(self, root_paths: TaskRuntimePaths, turn_number: int) -> Path:
        return root_paths.runtime_dir / "turns" / f"turn_{turn_number:03d}"

    def _turn_upload_dir(self, task_workspace: Path, turn_number: int) -> Path:
        return task_workspace / "uploads" / f"turn_{turn_number:03d}"

    def _store_turn_uploads(
        self,
        *,
        task_workspace: Path,
        sandbox_dir: Path,
        turn_number: int,
        uploaded_files: dict[str, bytes],
    ) -> list[dict[str, str]]:
        if not uploaded_files:
            return []
        upload_dir = self._turn_upload_dir(task_workspace, turn_number)
        upload_dir.mkdir(parents=True, exist_ok=True)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, str]] = []
        for fallback_index, (filename, content) in enumerate(uploaded_files.items(), start=1):
            safe_name = Path(filename).name.strip() or f"upload_{fallback_index}"
            upload_path = self._unique_child_path(upload_dir, safe_name)
            upload_path.write_bytes(content)
            sandbox_path = self._unique_child_path(sandbox_dir, safe_name)
            sandbox_path.write_bytes(content)
            records.append(
                {
                    "original_name": filename,
                    "upload_path": str(upload_path.relative_to(task_workspace)),
                    "sandbox_path": str(sandbox_path.relative_to(task_workspace)),
                    "sandbox_name": sandbox_path.name,
                }
            )
        return records

    def _unique_child_path(self, directory: Path, filename: str) -> Path:
        candidate = directory / filename
        if not candidate.exists():
            return candidate
        stem = candidate.stem or "upload"
        suffix = candidate.suffix
        counter = 2
        while True:
            candidate = directory / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _read_session_history(self, task_workspace: Path) -> list[TaskTurnRecord]:
        history_path = self._session_history_path(task_workspace)
        if not history_path.exists():
            return []
        records: list[TaskTurnRecord] = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(TaskTurnRecord.from_dict(json.loads(line)))
        return records

    def _append_session_history(self, task_workspace: Path, turn_record: TaskTurnRecord) -> None:
        history_path = self._session_history_path(task_workspace)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(turn_record.to_dict(), cls=DateTimeEncoder) + "\n")

    def _update_task_session_state(
        self,
        *,
        task_workspace: Path,
        session_state: TaskSessionState,
        turn_number: int,
        turn_status: str,
    ) -> TaskSessionState:
        updated_state = TaskSessionState(
            task_id=session_state.task_id,
            task_root=session_state.task_root,
            created_at_utc=session_state.created_at_utc,
            updated_at_utc=self._utc_now(),
            original_goal=session_state.original_goal,
            turn_count=turn_number,
            latest_turn_number=turn_number,
            latest_turn_status=turn_status,
        )
        self._write_json(self._session_state_path(task_workspace), updated_state.to_dict())
        return updated_state

    def _build_session_prompt_context(
        self,
        *,
        task_workspace: Path,
        session_state: TaskSessionState,
        turn_number: int,
        user_message: str,
        uploaded_file_records: list[dict[str, str]],
    ) -> SessionPromptContext:
        history = self._read_session_history(task_workspace)
        recent_turns = [{"user": item.user_message, "assistant": item.answer} for item in history[-3:]]
        previous_answer = history[-1].answer if history else None
        previous_feedback = history[-1].evaluation_feedback if history else None
        original_goal = session_state.original_goal.strip() or (history[0].user_message if history else user_message)
        return SessionPromptContext(
            task_id=session_state.task_id,
            turn_number=turn_number,
            original_goal=original_goal,
            latest_user_message=user_message,
            previous_answer=previous_answer,
            previous_feedback=previous_feedback,
            recent_turns=recent_turns,
            sandbox_artifacts=self._workspace_artifact_paths(task_workspace / "sandbox"),
            current_uploads=uploaded_file_records,
        )

    def _mirror_latest_runtime(
        self,
        *,
        root_paths: TaskRuntimePaths,
        turn_paths: TaskRuntimePaths,
        report_generated: bool,
    ) -> None:
        self._copy_tree_contents(turn_paths.run_dir, root_paths.run_dir)
        self._copy_tree_contents(turn_paths.attempts_dir, root_paths.attempts_dir)
        self._copy_tree_contents(turn_paths.reflection_dir, root_paths.reflection_dir)
        if turn_paths.final_evaluation_path.exists():
            root_paths.final_evaluation_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(turn_paths.final_evaluation_path, root_paths.final_evaluation_path)
        if report_generated and turn_paths.report_path.exists():
            root_paths.report_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(turn_paths.report_path, root_paths.report_path)
        elif root_paths.report_path.exists():
            root_paths.report_path.unlink()

    def _copy_tree_contents(self, source_dir: Path, target_dir: Path) -> None:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        if source_dir.exists():
            shutil.copytree(source_dir, target_dir)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)

    def _append_aggregate_events(self, aggregate_path: Path, turn_events_path: Path) -> None:
        if not turn_events_path.exists():
            return
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        with open(aggregate_path, "a", encoding="utf-8") as aggregate_handle:
            aggregate_handle.write(turn_events_path.read_text(encoding="utf-8"))

    def _append_runtime_event(
        self,
        paths: TaskRuntimePaths,
        *,
        stage: str,
        event: str,
        status: str,
        attempt: int | None = None,
        agent_name: str | None = None,
        backend: str | None = None,
        model_name: str | None = None,
        usage: dict[str, Any] | None = None,
        estimated_cost_usd: float | None = None,
        path_map: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "task_id": paths.task_root.name,
            "attempt": attempt,
            "stage": stage,
            "event": event,
            "status": status,
            "agent_name": agent_name,
            "backend": backend,
            "model_name": model_name,
            "usage": usage or {},
            "estimated_cost_usd": estimated_cost_usd,
            "paths": path_map or {},
            "metadata": metadata or {},
        }
        paths.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, cls=DateTimeEncoder) + "\n")

    def _attempt_summary(self, attempt: dict[str, Any]) -> dict[str, Any]:
        execution = attempt.get("execution", {})
        evaluation = attempt.get("evaluation", {})
        return {
            "attempt": attempt.get("attempt"),
            "status": evaluation.get("status"),
            "score": evaluation.get("score"),
            "retry_recommended": evaluation.get("retry_recommended"),
            "execution": {
                "success": execution.get("success"),
                "return_code": execution.get("return_code"),
                "duration_seconds": execution.get("duration_seconds"),
                "timed_out": execution.get("timed_out"),
                "cancelled": execution.get("cancelled"),
            },
            "paths": attempt.get("paths", {}),
        }

    def _attempts_index(self, attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        index: list[dict[str, Any]] = []
        for attempt in attempts:
            evaluation = attempt.get("evaluation", {})
            gate = attempt.get("gate", {})
            index.append(
                {
                    "attempt": attempt.get("attempt"),
                    "status": evaluation.get("status"),
                    "score": evaluation.get("score"),
                    "failure_type": evaluation.get("failure_type"),
                    "retry_recommended": gate.get("retry_recommended"),
                    "paths": attempt.get("paths", {}),
                }
            )
        return index

    def _write_runtime_index(
        self,
        paths: TaskRuntimePaths,
        task_id: str,
        user_request: str,
        execution_time: float,
        result: dict[str, Any],
    ) -> None:
        self._write_json(
            paths.index_path,
            {
                "schema_version": _RUNTIME_SCHEMA_VERSION,
                "task_id": task_id,
                "user_request": user_request,
                "mode": result.get("response_mode"),
                "success": result.get("success", False),
                "cancelled": result.get("cancelled", False),
                "cancel_reason": result.get("cancel_reason"),
                "execution_time": execution_time,
                "models": {
                    "planner": result.get("planner_model"),
                    "executor": result.get("executor_model"),
                    "evaluator": (result.get("llm_role_models") or {}).get("evaluator"),
                    "reflector": (result.get("llm_role_models") or {}).get("reflector"),
                },
                "paths": {
                    "task_root": str(paths.task_root),
                    "sandbox": str(paths.sandbox_dir),
                    "runtime": str(paths.runtime_dir),
                    "events": str(paths.events_path),
                    "summary": str(paths.run_summary_path),
                    "trajectory": str(paths.run_trajectory_path),
                    "costs": str(paths.run_costs_path),
                    "final_evaluation": str(paths.final_evaluation_path),
                    "report": result.get("report_path"),
                },
                "attempts": result.get("attempts_index", []),
                "usage_summary": result.get("usage_summary", {}),
                "cost_summary": result.get("cost_summary", {}),
                "backend": result.get("backend"),
                "executor_provider": result.get("executor_provider"),
                "runtime_llm_keys": result.get("runtime_llm_keys"),
                "memory_write_policy": result.get("memory_write_policy"),
                "available_project_cli_tools": result.get("available_project_cli_tools", []),
                "workflow_recommendations": result.get("workflow_recommendations", []),
            },
        )

    def _role_model_names(self) -> dict[str, str]:
        return {
            "planner": getattr(self.meta_agent, "last_model_name", self.config.llm_config_for_role("planner").model_name),
            "evaluator": getattr(self.evaluator, "last_model_name", self.config.llm_config_for_role("evaluator").model_name),
            "reflector": getattr(self.reflector, "last_model_name", self.config.llm_config_for_role("reflector").model_name),
        }

    def _capture_agent_usage(self, agent: Any, role: str) -> dict:
        usage = getattr(agent, "last_usage", {}) or {}
        return {
            "model_name": getattr(agent, "last_model_name", self.config.llm_config_for_role(role).model_name),
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
