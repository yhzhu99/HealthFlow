from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.contracts import ExecutionPlan
from ..tools import ToolCatalog


@dataclass
class ExecutionContext:
    user_request: str
    plan: ExecutionPlan
    available_tools: ToolCatalog
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
        workflow_block = (
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
        tool_block = "\n".join(self.available_tools.prompt_lines()) or "- No tools were advertised for this run."

        prompt = [
            "# HealthFlow Executor Brief",
            "",
            "You are the active external execution backend selected by HealthFlow.",
            "Operate as a CodeAct-style executor: think in explicit actions, choose the next best action, run it, inspect the result, and continue.",
            "Work inside the current workspace and keep your process reproducible and inspectable.",
            "Do not rely on repository-level executor-specific instruction files; use only the shared instructions in this prompt.",
            "",
            "## Original Task",
            self.user_request.strip(),
            "",
            "## Available Tools",
            tool_block,
            "",
            "## EHR Safeguards",
            safeguard_block,
            "",
            "## Recommended Workflows",
            workflow_block,
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
                "- Prefer Python, reproducible CLI workflows, and explicit tool calls when possible.",
                "- Use the planner's preferred tools when they fit, but adapt if execution reality requires a better path.",
                "- Treat safeguard memories as high-priority guardrails when choosing and validating your approach.",
                "- Record meaningful intermediate artifacts when they make the final result easier to inspect or reuse.",
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


class ExecutorAdapter(ABC):
    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    @abstractmethod
    async def execute(self, context: ExecutionContext, working_dir: Path) -> ExecutionResult:
        raise NotImplementedError
