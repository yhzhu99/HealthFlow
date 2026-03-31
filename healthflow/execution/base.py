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
    domain_focus: str = "general"
    risk_checks: List[str] = field(default_factory=list)
    tool_bundle: List[str] = field(default_factory=list)
    deliverable_guidance: List[str] = field(default_factory=list)
    plan_markdown: str = ""
    prior_feedback: Optional[str] = None
    memory_summary: str = ""
    verification_guidance: List[str] = field(default_factory=list)

    def render_prompt(self) -> str:
        risk_block = "\n".join(f"- {item}" for item in self.risk_checks) or "- No high-risk issues detected."
        tool_block = "\n".join(f"- {item}" for item in self.tool_bundle) or "- Use the minimum required command-line tooling."
        guidance_block = (
            "\n".join(f"- {item}" for item in self.deliverable_guidance)
            or "- Provide a concise final answer, and save artifacts only when they materially support the result."
        )
        memory_block = self.memory_summary.strip() or "- No prior memory was retrieved for this task."
        verification_block = (
            "\n".join(f"- {item}" for item in self.verification_guidance)
            or "- Prefer auditable artifacts that make important claims easy to verify."
        )

        prompt = [
            "# HealthFlow Executor Brief",
            "",
            "You are `healthflow_agent`, the integrated execution backend for HealthFlow.",
            "Work inside the current workspace and keep your process reproducible and auditable.",
            "Do not rely on repository-level executor-specific instruction files; use only the shared instructions in this prompt.",
            "",
            "## Original Task",
            self.user_request.strip(),
            "",
            "## Task Profile",
            f"- Task family: {self.task_family}",
            f"- Domain focus: {self.domain_focus}",
            "",
            "## Data Profile",
            self.data_profile.strip(),
            "",
            "## Retrieved Memory",
            memory_block,
            "",
            "## Risk Checks",
            risk_block,
            "",
            "## Preferred Tool Bundle",
            tool_block,
            "",
            "## Deliverable Guidance",
            "These are suggested deliverables, not fixed file-level contracts unless the task explicitly asks for them.",
            guidance_block,
            "",
            "## Verification Focus",
            "Prefer auditable evidence for the following points when they are relevant to the task.",
            verification_block,
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
                "- Save every artifact inside the current workspace. Do not write files outside it.",
                "- Prefer Python and reproducible CLI workflows when possible.",
                "- Avoid printing raw sensitive rows unless strictly necessary for debugging.",
                "- Prefer small, reproducible scripts and save important artifacts to files.",
                "- Treat failure memories as avoidance guidance and do not repeat the same mistake.",
                "- Keep healthcare-specific safeguards only when the request or data actually warrants them.",
                "- End with a concise final answer that references any produced artifacts.",
            ]
        )
        return "\n".join(prompt).strip()


@dataclass
class ExecutionResult:
    success: bool
    return_code: int
    log: str
    log_path: str
    prompt_path: str
    backend: str
    command: List[str]
    backend_version: Optional[str] = None
    executor_metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timed_out: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)


class ExecutorAdapter(ABC):
    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    @abstractmethod
    async def execute(self, context: ExecutionContext, working_dir: Path, prompt_file_name: str) -> ExecutionResult:
        raise NotImplementedError
