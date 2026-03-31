from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from ..ehr.models import DataProfile
from ..ehr.tasking import required_report_sections


MODELING_FAMILIES = {"predictive_modeling", "survival_analysis", "time_series_modeling"}


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    details: str
    severity: str = "error"
    artifact_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "severity": self.severity,
            "artifact_paths": self.artifact_paths,
        }


@dataclass
class VerificationResult:
    passed: bool
    checks: List[VerificationCheck] = field(default_factory=list)
    artifact_paths: List[str] = field(default_factory=list)

    def summary(self) -> str:
        prefix = "passed" if self.passed else "failed"
        check_text = "; ".join(
            f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.details}"
            for check in self.checks
        ) or "no checks recorded"
        return f"Verification {prefix}: {check_text}"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "artifact_paths": self.artifact_paths,
        }


class WorkspaceVerifier:
    def __init__(self, required_report_sections: list[str]):
        self.required_report_sections = required_report_sections

    def verify(self, workspace_dir: Path, task_family: str, execution_log: str, data_profile: DataProfile) -> VerificationResult:
        checks: list[VerificationCheck] = []
        artifact_paths: set[str] = set()

        self._add_check(
            checks,
            passed="traceback" not in execution_log.lower(),
            name="execution_log",
            details="No Python traceback detected in execution log."
            if "traceback" not in execution_log.lower()
            else "Detected Python traceback in execution log.",
        )

        report_files = sorted(workspace_dir.glob("*.md"))
        preferred_report = workspace_dir / "final_report.md"
        report_text = ""
        if report_files:
            artifact_paths.update(str(path) for path in report_files)
            report_file = preferred_report if preferred_report.exists() else report_files[0]
            report_text = report_file.read_text(encoding="utf-8", errors="ignore").lower()
            required_sections = list(
                dict.fromkeys(self.required_report_sections + required_report_sections(task_family, data_profile.domain_focus))
            )
            missing_sections = [section for section in required_sections if section.lower() not in report_text]
            self._add_check(
                checks,
                passed=not missing_sections,
                name="report_sections",
                details=(
                    f"Report sections present in {report_file.name}."
                    if not missing_sections
                    else f"Missing report sections: {', '.join(missing_sections)}."
                ),
                severity="error" if task_family == "report_generation" else "info",
                artifact_paths=[str(report_file)],
            )
        else:
            self._add_check(
                checks,
                passed=task_family != "report_generation",
                name="report_sections",
                details=(
                    "No markdown report file was found."
                    if task_family == "report_generation"
                    else "No markdown report file was found; verification relied on saved artifacts and execution logs."
                ),
                severity="error" if task_family == "report_generation" else "info",
            )

        cohort_paths = self._collect_paths(
            workspace_dir,
            ["cohort_definition.*", "cohort.*", "manifest.json"],
            relative_ok=False,
        )
        if task_family == "cohort_extraction" or (
            data_profile.domain_focus == "ehr" and task_family in MODELING_FAMILIES
        ):
            report_mentions_cohort = "cohort definition" in report_text
            cohort_passed = bool(cohort_paths) or report_mentions_cohort
            artifact_paths.update(cohort_paths)
            self._add_check(
                checks,
                passed=cohort_passed,
                name="cohort_definition",
                details=(
                    "Cohort definition evidence found."
                    if cohort_passed
                    else "Missing cohort definition artifact or report section."
                ),
                artifact_paths=cohort_paths,
            )

        if task_family in MODELING_FAMILIES:
            split_paths = self._collect_paths(
                workspace_dir,
                ["split_evidence.*", "split.json", "data_split.*", "preprocess/split.json"],
                relative_ok=True,
            )
            artifact_paths.update(split_paths)
            self._add_check(
                checks,
                passed=bool(split_paths) or any(
                    token in report_text for token in ["split evidence", "validation strategy", "train/test split", "train/validation/test"]
                ),
                name="split_evidence",
                details=(
                    "Train/validation/test split evidence found."
                    if split_paths or any(
                        token in report_text for token in ["split evidence", "validation strategy", "train/test split", "train/validation/test"]
                    )
                    else "Missing validation split evidence for the modeling task."
                ),
                artifact_paths=split_paths,
            )

            audit_label = "temporal validation" if task_family in {"survival_analysis", "time_series_modeling"} else "validation audit"
            audit_paths = self._collect_paths(
                workspace_dir,
                ["leakage_audit.*", "temporal_audit.*", "validation_audit.*", "causality_audit.*"],
                relative_ok=False,
            )
            artifact_paths.update(audit_paths)
            audit_in_report = audit_label in report_text or "leakage audit" in report_text
            strict_audit = data_profile.domain_focus == "ehr" or task_family in {"survival_analysis", "time_series_modeling"}
            self._add_check(
                checks,
                passed=bool(audit_paths) or audit_in_report or not strict_audit,
                name="audit_evidence",
                details=(
                    f"{audit_label.title()} evidence found."
                    if audit_paths or audit_in_report
                    else f"Missing {audit_label} evidence for the modeling task."
                ),
                severity="error" if strict_audit else "info",
                artifact_paths=audit_paths,
            )

            metric_files = self._collect_paths(workspace_dir, ["metrics.*", "test/metrics.json"], relative_ok=True)
            artifact_paths.update(metric_files)
            self._add_check(
                checks,
                passed=bool(metric_files),
                name="metrics_artifact",
                details="Metric artifact found." if metric_files else "Expected a metrics artifact for the modeling task family.",
                artifact_paths=metric_files,
            )

        figure_files = self._collect_paths(workspace_dir, ["*.png", "*.pdf"], relative_ok=False)
        if task_family == "visualization":
            artifact_paths.update(figure_files)
            self._add_check(
                checks,
                passed=bool(figure_files),
                name="visualization_artifact",
                details=(
                    "Visualization artifact found."
                    if figure_files
                    else "Expected at least one figure artifact for the visualization task family."
                ),
                artifact_paths=figure_files,
            )

        if not data_profile.schemas:
            self._add_check(
                checks,
                passed=True,
                name="profile_context",
                details="No uploaded structured inputs were profiled; verification relied on workspace artifacts only.",
                severity="info",
            )

        passed = all(check.passed or check.severity == "info" for check in checks)
        return VerificationResult(
            passed=passed,
            checks=checks,
            artifact_paths=sorted(artifact_paths),
        )

    def _collect_paths(self, workspace_dir: Path, patterns: Iterable[str], relative_ok: bool) -> list[str]:
        found: list[str] = []
        for pattern in patterns:
            if relative_ok and "/" in pattern:
                matches = [workspace_dir / pattern]
                matches = [path for path in matches if path.exists()]
            else:
                matches = list(workspace_dir.glob(pattern))
            found.extend(str(path) for path in matches if path.exists())
        return sorted(dict.fromkeys(found))

    def _add_check(
        self,
        checks: list[VerificationCheck],
        passed: bool,
        name: str,
        details: str,
        severity: str = "error",
        artifact_paths: list[str] | None = None,
    ) -> None:
        checks.append(
            VerificationCheck(
                name=name,
                passed=passed,
                details=details,
                severity=severity,
                artifact_paths=artifact_paths or [],
            )
        )
