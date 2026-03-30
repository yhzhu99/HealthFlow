from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionContext:
    user_request: str
    task_family: str
    data_profile: str
    risk_checks: List[str] = field(default_factory=list)
    tool_bundle: List[str] = field(default_factory=list)
    output_contract: List[str] = field(default_factory=list)
    plan_markdown: str = ""
    prior_feedback: Optional[str] = None

    def render_prompt(self) -> str:
        risk_block = "\n".join(f"- {item}" for item in self.risk_checks) or "- No high-risk issues detected."
        tool_block = "\n".join(f"- {item}" for item in self.tool_bundle) or "- Use the minimum required command-line tooling."
        output_block = "\n".join(f"- {item}" for item in self.output_contract) or "- Provide a final answer in stdout."

        prompt = [
            "# HealthFlow Executor Brief",
            "",
            "You are the execution backend for HealthFlow, an EHR-focused analysis harness.",
            "Work inside the current workspace and keep your process reproducible and auditable.",
            "",
            "## Original Task",
            self.user_request.strip(),
            "",
            "## Task Family",
            self.task_family,
            "",
            "## Data Profile",
            self.data_profile.strip(),
            "",
            "## Risk Checks",
            risk_block,
            "",
            "## Preferred Tool Bundle",
            tool_block,
            "",
            "## Output Contract",
            output_block,
            "",
            "## Execution Plan",
            self.plan_markdown.strip() or "No explicit plan provided.",
        ]
        if self.prior_feedback:
            prompt.extend(["", "## Feedback from Previous Attempt", self.prior_feedback.strip()])
        prompt.extend(
            [
                "",
                "## Execution Rules",
                "- Inspect data before choosing a method.",
                "- Avoid printing raw sensitive rows unless strictly necessary for debugging.",
                "- Prefer small, reproducible scripts and save important artifacts to files.",
                "- End with a concise final answer that references the produced artifacts.",
            ]
        )
        return "\n".join(prompt).strip()


@dataclass
class ExecutionResult:
    success: bool
    return_code: int
    log: str
    log_path: str
    backend: str
    command: List[str]
    usage: Dict[str, Any] = field(default_factory=dict)


class ExecutorAdapter(ABC):
    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    @abstractmethod
    async def execute(self, context: ExecutionContext, working_dir: Path, prompt_file_name: str) -> ExecutionResult:
        raise NotImplementedError
