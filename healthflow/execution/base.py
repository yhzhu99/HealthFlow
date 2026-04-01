from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import EnvironmentConfig
from ..core.contracts import ExecutionPlan


@dataclass
class ExecutionContext:
    user_request: str
    plan: ExecutionPlan
    execution_environment: EnvironmentConfig
    workflow_recommendations: List[str] = field(default_factory=list)
    report_requested: bool = False
    safeguard_memory: List[str] = field(default_factory=list)
    workflow_memory: List[str] = field(default_factory=list)
    dataset_memory: List[str] = field(default_factory=list)
    execution_memory: List[str] = field(default_factory=list)
    prior_feedback: Optional[str] = None

    def render_prompt(self) -> str:
        safeguard_block = (
            "\n".join(f"- {item}" for item in self.safeguard_memory)
            or "- No safeguard memory was retrieved for this task."
        )
        retrieved_workflow_block = (
            "\n".join(f"- {item}" for item in self.workflow_memory)
            or "- No workflow memory was retrieved for this task."
        )
        dataset_block = (
            "\n".join(f"- {item}" for item in self.dataset_memory)
            or "- No dataset memory was retrieved for this task."
        )
        execution_block = (
            "\n".join(f"- {item}" for item in self.execution_memory)
            or "- No execution memory was retrieved for this task."
        )
        environment_block = "\n".join(
            f"- {item}" for item in self.execution_environment.summary_lines()
        ) or "- Use the default executor environment."
        workflow_recommendation_block = (
            "\n".join(f"- {item}" for item in self.workflow_recommendations)
            or "- No workflow recommendations were provided for this run."
        )

        prompt = [
            "# HealthFlow Executor Brief",
            "",
            "You are the active external execution backend selected by HealthFlow.",
            "Operate as a CodeAct-style executor: think in explicit actions, choose the next best action, run it, inspect the result, and continue.",
            "Work inside the current workspace and keep your process reproducible and inspectable.",
            "Do not rely on repository-level executor-specific instruction files; use only the shared instructions in this prompt.",
            "HealthFlow does not manage MCP servers, plugin registries, or local CLI tool catalogs.",
            "",
            "## Original Task",
            self.user_request.strip(),
            "",
            "## Execution Environment",
            environment_block,
            "",
            "## Workflow Recommendations",
            workflow_recommendation_block,
            "",
            "## EHR Safeguards",
            safeguard_block,
            "",
            "## Workflow Memory",
            retrieved_workflow_block,
            "",
            "## Dataset Anchors",
            dataset_block,
            "",
            "## Execution Hints",
            execution_block,
            "",
            "## Execution Plan",
            self.plan.to_markdown(),
        ]
        if self.prior_feedback:
            prompt.extend(["", "## Feedback from Previous Attempt", self.prior_feedback.strip()])
        prompt.extend(
            [
                "",
                "## Execution Rules",
                "- Inspect the workspace and any task inputs before committing to an implementation path.",
                "- Save every artifact inside the current workspace. Do not write files outside it.",
                "- Prefer reproducible Python and CLI workflows, usually through the configured run prefix when it fits.",
                "- Use the planner's recommended workflows when they fit, but adapt if execution reality requires a better path.",
                "- Only rely on CLI tools or services that are already available in your executor environment; verify availability before depending on them.",
                "- Treat safeguard memories as high-priority guardrails when choosing and validating your approach.",
                "- Record meaningful intermediate artifacts when they make the final result easier to inspect or reuse.",
                "- Keep execution narration out of the final user-facing reply unless the task explicitly asks for process detail.",
                "- End with a concise final answer that references any produced artifacts.",
            ]
        )
        if self.report_requested:
            prompt.extend(
                [
                    "- Keep filenames, headings, and comments clear enough for a later system-generated report to describe them accurately.",
                    "- When you create figures, code, or analysis notes, keep them inspectable and self-describing inside the workspace.",
                    "- Make the final answer explicitly reference the key artifacts you produced and what each artifact contains.",
                ]
            )
        return "\n".join(prompt).strip()


@dataclass
class ExecutionResult:
    success: bool
    return_code: int
    log: str
    log_path: str
    prompt_path: str | None
    backend: str
    command: List[str]
    backend_version: Optional[str] = None
    executor_metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timed_out: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    cancel_reason: str | None = None


class ExecutionCancelledError(asyncio.CancelledError):
    def __init__(self, result: ExecutionResult, cancel_reason: str = "Execution cancelled by user."):
        super().__init__(cancel_reason)
        self.result = result
        self.cancel_reason = cancel_reason


class ExecutorAdapter(ABC):
    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    @abstractmethod
    async def execute(self, context: ExecutionContext, working_dir: Path) -> ExecutionResult:
        raise NotImplementedError
