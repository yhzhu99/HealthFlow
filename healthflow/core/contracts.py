from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExecutionPlan(BaseModel):
    objective: str = Field(..., description="Primary objective for the current attempt.")
    assumptions_to_check: list[str] = Field(
        default_factory=list,
        description="Open assumptions the executor should verify before committing to an approach.",
    )
    recommended_steps: list[str] = Field(
        default_factory=list,
        description="Ordered high-level steps for the executor to follow.",
    )
    preferred_tools: list[str] = Field(
        default_factory=list,
        description="Preferred tools or action surfaces suggested by the planner.",
    )
    avoidances: list[str] = Field(
        default_factory=list,
        description="Specific anti-patterns or mistakes to avoid on this attempt.",
    )
    success_signals: list[str] = Field(
        default_factory=list,
        description="Observable signs that the attempt is on track to succeed.",
    )
    executor_brief: str = Field(
        default="",
        description="Concise guidance for the executor about how to interpret and apply the plan.",
    )

    def to_markdown(self) -> str:
        sections = [
            "# Execution Plan",
            "",
            "## Objective",
            self.objective.strip() or "No objective provided.",
            "",
            "## Assumptions To Check",
            *self._render_bullets(self.assumptions_to_check, "No explicit assumptions were listed."),
            "",
            "## Recommended Steps",
            *self._render_numbered(self.recommended_steps, "Inspect the environment and act directly."),
            "",
            "## Preferred Tools",
            *self._render_bullets(self.preferred_tools, "No preferred tools were specified."),
            "",
            "## Avoidances",
            *self._render_bullets(self.avoidances, "No explicit avoidances were listed."),
            "",
            "## Success Signals",
            *self._render_bullets(self.success_signals, "No explicit success signals were listed."),
            "",
            "## Executor Brief",
            self.executor_brief.strip() or "Use the most direct reproducible path to satisfy the task.",
        ]
        return "\n".join(sections).strip()

    def _render_bullets(self, items: list[str], fallback: str) -> list[str]:
        if not items:
            return [f"- {fallback}"]
        return [f"- {item}" for item in items]

    def _render_numbered(self, items: list[str], fallback: str) -> list[str]:
        if not items:
            return [f"1. {fallback}"]
        return [f"{index}. {item}" for index, item in enumerate(items, start=1)]


class EvaluationVerdict(BaseModel):
    status: Literal["success", "needs_retry", "failed"] = Field(
        ...,
        description="Top-level outcome classification for the attempt.",
    )
    score: float = Field(..., ge=0.0, le=10.0, description="Overall quality score for the attempt.")
    failure_type: str = Field(
        default="none",
        description="Structured failure category. Use 'none' on success.",
    )
    feedback: str = Field(..., description="Short actionable explanation of the verdict.")
    repair_instructions: list[str] = Field(
        default_factory=list,
        description="Concrete steps for the planner/executor to address next.",
    )
    retry_recommended: bool = Field(
        default=False,
        description="Whether another attempt is likely to help.",
    )
    memory_worthy_insights: list[str] = Field(
        default_factory=list,
        description="Reusable insights worth passing to reflection.",
    )
    reasoning: str = Field(default="", description="Brief supporting reasoning for the verdict.")
    usage: dict[str, Any] = Field(default_factory=dict)
    judge_model: str | None = None
    estimated_cost_usd: float | None = None

    def is_success(self, threshold: float) -> bool:
        return self.status == "success" and self.score >= threshold
