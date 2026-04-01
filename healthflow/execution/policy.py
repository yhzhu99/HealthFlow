from __future__ import annotations

from dataclasses import dataclass

from ..ehr.models import DataProfile
from ..ehr.tasking import default_workflow_recommendations


_ONEEHR_TASK_FAMILIES = {
    "predictive_modeling",
    "survival_analysis",
    "time_series_modeling",
}

_TOOLUNIVERSE_SIGNALS = (
    "tooluniverse",
    "tool universe",
    "tu find",
    "tu run",
    "find a tool",
    "discover a tool",
    "tool lookup",
    "biomedical tool",
    "biomedical knowledge",
    "literature lookup",
    "knowledge lookup",
)


@dataclass
class WorkflowRecommendationBroker:
    max_recommendations: int = 5

    def recommend(self, user_request: str, data_profile: DataProfile) -> list[str]:
        recommendations = list(default_workflow_recommendations(data_profile.task_family, data_profile.domain_focus))

        if self._should_recommend_oneehr(data_profile):
            recommendations.append(
                "If `oneehr` is already available in the executor environment, it can cover preprocess/train/test/analyze/plot stages for EHR modeling pipelines."
            )

        if self._should_recommend_tooluniverse(user_request):
            recommendations.append(
                "If `tu` is already available in the executor environment, use ToolUniverse CLI for targeted biomedical tool discovery or execution."
            )

        recommendations.extend(
            [
                "Prefer workspace-local Python entrypoints via `uv run` for reproducible analysis steps.",
                "If new dependencies are genuinely required, keep additions minimal and use `uv add` so the environment stays auditable.",
            ]
        )

        deduped = list(dict.fromkeys(recommendations))
        return deduped[: self.max_recommendations]

    def _should_recommend_oneehr(self, data_profile: DataProfile) -> bool:
        return data_profile.domain_focus == "ehr" and data_profile.task_family in _ONEEHR_TASK_FAMILIES

    def _should_recommend_tooluniverse(self, user_request: str) -> bool:
        request = user_request.lower()
        return any(signal in request for signal in _TOOLUNIVERSE_SIGNALS)
