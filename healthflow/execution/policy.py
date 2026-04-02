from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TOOLUNIVERSE_BINARIES = ("tu", "tooluniverse")


@dataclass
class WorkflowRecommendationBroker:
    max_recommendations: int = 5

    def recommend(self, user_request: str, data_profile: DataProfile) -> list[str]:
        recommendations = list(default_workflow_recommendations(data_profile.task_family, data_profile.domain_focus))

        if self._should_recommend_oneehr(data_profile):
            recommendations.append(
                "If the surfaced `oneehr` CLI directly fits the task, prefer it over rebuilding the EHR pipeline from scratch."
            )

        if self._should_recommend_tooluniverse(user_request):
            recommendations.append(
                "If the surfaced `tu` / `tooluniverse` CLI directly fits the task, use it for biomedical tool discovery or execution."
            )

        recommendations.extend(
            [
                "Prefer workspace-local Python entrypoints via `uv run` for reproducible analysis steps.",
                "If new dependencies are genuinely required, keep additions minimal and use `uv add` so the environment stays auditable.",
            ]
        )

        deduped = list(dict.fromkeys(recommendations))
        return deduped[: self.max_recommendations]

    def available_project_cli_tools(self, user_request: str, data_profile: DataProfile) -> list[str]:
        discovered_tools = self.discover_supported_project_cli_tools()
        tool_contracts: list[str] = []

        if "oneehr" in discovered_tools:
            tool_contracts.append(
                "`oneehr`: Available in the project environment for EHR modeling pipelines. "
                "Best fit for EHR preprocess/train/test/analyze/plot workflows. "
                "Prefer `uv run oneehr <subcommand>` from the repo root. "
                "Validate with `uv run oneehr --help` before depending on it. "
                "Supported stages: `preprocess`, `train`, `test`, `analyze`, `plot`, `convert`."
            )

        if "tooluniverse" in discovered_tools:
            tool_contracts.append(
                "`tu` / `tooluniverse`: Available in the project environment for biomedical tool discovery or execution. "
                "Best fit for biomedical tool lookup, inspection, execution, or MCP serving. "
                "Prefer `uv run tu <command>` from the repo root. "
                "Validate with `uv run tu --help` or `uv run tu status` before depending on it. "
                "Useful commands: `list`, `find`, `info`, `run`, `status`, `serve`."
            )

        return tool_contracts

    def discover_supported_project_cli_tools(self) -> list[str]:
        lookup_path = self._tool_lookup_path()
        discovered_tools: list[str] = []

        if shutil.which("oneehr", path=lookup_path):
            discovered_tools.append("oneehr")
        if any(shutil.which(binary, path=lookup_path) for binary in _TOOLUNIVERSE_BINARIES):
            discovered_tools.append("tooluniverse")

        return discovered_tools

    def _should_recommend_oneehr(self, data_profile: DataProfile) -> bool:
        return data_profile.domain_focus == "ehr" and data_profile.task_family in _ONEEHR_TASK_FAMILIES

    def _should_recommend_tooluniverse(self, user_request: str) -> bool:
        request = user_request.lower()
        return any(signal in request for signal in _TOOLUNIVERSE_SIGNALS)

    def _tool_lookup_path(self) -> str | None:
        path_entries = []
        project_venv_bin = _PROJECT_ROOT / ".venv" / "bin"
        if project_venv_bin.exists():
            path_entries.append(str(project_venv_bin))

        current_path = os.environ.get("PATH", "")
        if current_path:
            path_entries.extend(entry for entry in current_path.split(os.pathsep) if entry)

        deduped_entries = list(dict.fromkeys(path_entries))
        return os.pathsep.join(deduped_entries) if deduped_entries else None
