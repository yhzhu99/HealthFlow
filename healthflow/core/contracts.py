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
    recommended_workflows: list[str] = Field(
        default_factory=list,
        description="Recommended execution workflows suggested by the planner.",
    )
    avoidances: list[str] = Field(
        default_factory=list,
        description="Specific anti-patterns or mistakes to avoid on this attempt.",
    )
    success_signals: list[str] = Field(
        default_factory=list,
        description="Observable signs that the attempt is on track to succeed.",
    )

    def to_markdown(self, title_level: int = 1) -> str:
        title_level = max(1, title_level)
        title_prefix = "#" * title_level
        section_prefix = "#" * (title_level + 1)
        sections = [
            f"{title_prefix} Execution Plan",
            "",
            f"{section_prefix} Objective",
            self.objective.strip() or "No objective provided.",
        ]
        self._append_list_section(
            sections,
            heading=f"{section_prefix} Assumptions To Check",
            items=self.assumptions_to_check,
            ordered=False,
        )
        self._append_list_section(
            sections,
            heading=f"{section_prefix} Recommended Steps",
            items=self.recommended_steps,
            ordered=True,
        )
        self._append_list_section(
            sections,
            heading=f"{section_prefix} Recommended Workflows",
            items=self.recommended_workflows,
            ordered=False,
        )
        self._append_list_section(
            sections,
            heading=f"{section_prefix} Avoidances",
            items=self.avoidances,
            ordered=False,
        )
        self._append_list_section(
            sections,
            heading=f"{section_prefix} Success Signals",
            items=self.success_signals,
            ordered=False,
        )
        return "\n".join(sections).strip()

    def _append_list_section(self, sections: list[str], heading: str, items: list[str], *, ordered: bool) -> None:
        cleaned_items = [item.strip() for item in items if item.strip()]
        if not cleaned_items:
            return
        sections.extend(["", heading])
        if ordered:
            sections.extend(f"{index}. {item}" for index, item in enumerate(cleaned_items, start=1))
            return
        sections.extend(f"- {item}" for item in cleaned_items)


class EvaluationVerdict(BaseModel):
    status: Literal["success", "needs_retry", "failed"] = Field(
        ...,
        description="Top-level outcome classification for the attempt.",
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score for the attempt on a 0-1 scale.")
    failure_type: str = Field(
        default="none",
        description="Structured failure category. Use 'none' on success.",
    )
    feedback: str = Field(..., description="Short actionable explanation of the verdict.")
    repair_instructions: list[str] = Field(
        default_factory=list,
        description="Concrete steps for the planner/executor to address next.",
    )
    violated_constraints: list[str] = Field(
        default_factory=list,
        description="Structured constraints, assumptions, or contracts that the attempt violated.",
    )
    repair_hypotheses: list[str] = Field(
        default_factory=list,
        description="Candidate hypotheses about what strategic change is most likely to improve the next attempt.",
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
