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
    available_project_cli_tools: List[str] = field(default_factory=list)
    report_requested: bool = False
    safeguard_memory: List[str] = field(default_factory=list)
    workflow_memory: List[str] = field(default_factory=list)
    dataset_memory: List[str] = field(default_factory=list)
    prior_feedback: Optional[str] = None
    executor_artifact_dir: Path | None = None

    def render_prompt(self) -> str:
        prompt = [
            "# HealthFlow Execution Task",
            "",
            "## Original Task",
            self.user_request.strip(),
            "",
            "## Execution Environment",
            self._render_bullet_block(
                self.execution_environment.summary_lines(),
                fallback="- Use the default executor environment.",
            ),
        ]
        self._append_optional_section(prompt, "## Available Project CLI Tools", self.available_project_cli_tools)
        self._append_optional_section(prompt, "## Workflow Recommendations", self.workflow_recommendations)
        self._append_optional_section(prompt, "## EHR Safeguards", self.safeguard_memory)
        self._append_optional_section(prompt, "## Workflow Memories", self.workflow_memory)
        self._append_optional_section(prompt, "## Dataset Anchors", self.dataset_memory)
        has_project_cli_tools = self._has_content(self.available_project_cli_tools)
        has_workflow_recommendations = self._has_content(self.workflow_recommendations)
        has_safeguards = self._has_content(self.safeguard_memory)
        has_prior_feedback = bool(self.prior_feedback and self.prior_feedback.strip())
        if has_prior_feedback:
            prompt.extend(["", "## Feedback from Previous Attempt", self.prior_feedback.strip()])
        prompt.extend(["", self.plan.to_markdown(title_level=2)])
        rules = [
            "Inspect the workspace and any task inputs before committing to an implementation path.",
            "Save every artifact inside the current workspace. Do not write files outside it.",
            "Prefer reproducible Python and CLI workflows, usually through the configured run prefix when it fits.",
            "Only rely on CLI tools or services that are already available in your executor environment.",
            "Verify external tools before depending on them for the core path.",
            "Record meaningful intermediate artifacts when they make the final result easier to inspect or reuse.",
            "End with a concise final answer that references any produced artifacts.",
        ]
        if has_project_cli_tools:
            rules.append("Treat surfaced project CLI tools as approved project-local workflows for this run.")
            rules.append(
                "When a surfaced project CLI directly fits the task, prefer it over reimplementing the same workflow and validate it early with a lightweight help or status command."
            )
        if has_workflow_recommendations:
            rules.append("Use planner workflow recommendations when they fit, but adapt if execution reality requires a better path.")
        if has_safeguards:
            rules.append("Treat safeguard memories as task-bounded constraints that override softer workflow preferences.")
            rules.append("Safeguards do not authorize extra deliverables or extra data transformations beyond what the user explicitly requested.")
        if has_prior_feedback:
            rules.append("Address the previous attempt feedback explicitly when choosing the next path.")
        prompt.extend(["", "## Execution Rules", *[f"- {rule}" for rule in rules]])
        if self.report_requested:
            prompt.extend(
                [
                    "- Keep filenames, headings, and comments clear enough for a later system-generated report to describe them accurately.",
                    "- When you create figures, code, or analysis notes, keep them inspectable and self-describing inside the workspace.",
                    "- Make the final answer explicitly reference the key artifacts you produced and what each artifact contains.",
                ]
            )
        return "\n".join(prompt).strip()

    def _append_optional_section(self, prompt: list[str], heading: str, items: list[str]) -> None:
        block = self._render_bullet_block(items)
        if not block:
            return
        prompt.extend(["", heading, block])

    def _render_bullet_block(self, items: list[str], fallback: str | None = None) -> str:
        cleaned_items = [item.strip() for item in items if item.strip()]
        if cleaned_items:
            return "\n".join(f"- {item}" for item in cleaned_items)
        return fallback or ""

    def _has_content(self, items: list[str]) -> bool:
        return any(item.strip() for item in items)


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
    stdout: str = ""
    stderr: str = ""


class ExecutionCancelledError(Exception):
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
