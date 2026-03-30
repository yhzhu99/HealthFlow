from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ..ehr.models import DataProfile


@dataclass
class VerificationResult:
    passed: bool
    checks: List[str] = field(default_factory=list)
    artifact_paths: List[str] = field(default_factory=list)

    def summary(self) -> str:
        prefix = "passed" if self.passed else "failed"
        checks = "; ".join(self.checks) if self.checks else "no checks recorded"
        return f"Verification {prefix}: {checks}"


class WorkspaceVerifier:
    def __init__(self, required_report_sections: list[str]):
        self.required_report_sections = required_report_sections

    def verify(self, workspace_dir: Path, task_family: str, execution_log: str, data_profile: DataProfile) -> VerificationResult:
        checks: list[str] = []
        artifact_paths: list[str] = []
        passed = True

        if "traceback" in execution_log.lower():
            passed = False
            checks.append("Detected Python traceback in execution log.")

        report_files = sorted(workspace_dir.glob("*.md"))
        if report_files:
            artifact_paths.extend(str(path) for path in report_files)
            report_text = report_files[0].read_text(encoding="utf-8", errors="ignore").lower()
            missing_sections = [
                section for section in self.required_report_sections if section.lower() not in report_text
            ]
            if missing_sections:
                passed = False
                checks.append(f"Missing report sections: {', '.join(missing_sections)}.")
            else:
                checks.append(f"Report sections present in {report_files[0].name}.")
        else:
            checks.append("No markdown report file was found.")
            if task_family != "general_ehr_analysis":
                passed = False

        if task_family in {"predictive_modeling", "survival_analysis", "time_series_modeling"}:
            metric_files = list(workspace_dir.glob("metrics.*"))
            if metric_files:
                artifact_paths.extend(str(path) for path in metric_files)
                checks.append("Metric artifact found.")
            else:
                passed = False
                checks.append("Expected a metrics artifact for the modeling task family.")

        figure_files = list(workspace_dir.glob("*.png")) + list(workspace_dir.glob("*.pdf"))
        if task_family == "visualization":
            if figure_files:
                artifact_paths.extend(str(path) for path in figure_files)
                checks.append("Visualization artifact found.")
            else:
                passed = False
                checks.append("Expected at least one figure artifact for the visualization task family.")

        if not data_profile.schemas:
            checks.append("No uploaded structured inputs were profiled; verification relied on workspace artifacts only.")

        return VerificationResult(
            passed=passed,
            checks=checks,
            artifact_paths=sorted(set(artifact_paths)),
        )
