from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from ..ehr.models import DataProfile
from .contracts import resolve_verification_contract


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    details: str
    blocking: bool = True
    artifact_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "blocking": self.blocking,
            "artifact_paths": self.artifact_paths,
        }


@dataclass
class VerificationResult:
    passed: bool
    blocking_passed: bool
    checks: List[VerificationCheck] = field(default_factory=list)
    artifact_paths: List[str] = field(default_factory=list)

    def summary(self) -> str:
        prefix = "passed" if self.blocking_passed else "failed"
        check_text = "; ".join(
            f"{'PASS' if check.passed else 'FAIL'} "
            f"{'BLOCK' if check.blocking else 'ADVISORY'} {check.name}: {check.details}"
            for check in self.checks
        ) or "no checks recorded"
        return f"Verification {prefix}: {check_text}"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "blocking_passed": self.blocking_passed,
            "checks": [check.to_dict() for check in self.checks],
            "artifact_paths": self.artifact_paths,
        }


class WorkspaceVerifier:
    def verify(self, workspace_dir: Path, task_family: str, execution_log: str, data_profile: DataProfile) -> VerificationResult:
        checks: list[VerificationCheck] = []
        artifact_paths: set[str] = set()
        contract = resolve_verification_contract(task_family, data_profile.domain_focus)

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
            missing_sections = [section for section in contract.report_sections if section.lower() not in report_text]
            self._add_check(
                checks,
                passed=not missing_sections,
                name="report_sections",
                details=(
                    f"Report sections present in {report_file.name}."
                    if not missing_sections
                    else f"Missing report sections: {', '.join(missing_sections)}."
                ),
                blocking=task_family == "report_generation",
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
                blocking=task_family == "report_generation",
            )

        for expectation in contract.artifact_expectations:
            matched_paths = self._collect_paths(workspace_dir, list(expectation.glob_patterns))
            artifact_paths.update(matched_paths)
            report_mentions = any(token in report_text for token in expectation.report_tokens)
            self._add_check(
                checks,
                passed=bool(matched_paths) or report_mentions,
                name=expectation.name,
                details=expectation.details_present if matched_paths or report_mentions else expectation.details_missing,
                blocking=expectation.blocking,
                artifact_paths=matched_paths,
            )

        if not data_profile.schemas:
            self._add_check(
                checks,
                passed=True,
                name="profile_context",
                details="No uploaded structured inputs were profiled; verification relied on workspace artifacts only.",
                blocking=False,
            )

        blocking_passed = all(check.passed or not check.blocking for check in checks)
        return VerificationResult(
            passed=blocking_passed,
            blocking_passed=blocking_passed,
            checks=checks,
            artifact_paths=sorted(artifact_paths),
        )

    def _collect_paths(self, workspace_dir: Path, patterns: Iterable[str]) -> list[str]:
        found: list[str] = []
        for pattern in patterns:
            matches = list(workspace_dir.glob(pattern))
            found.extend(str(path) for path in matches if path.exists())
        return sorted(dict.fromkeys(found))

    def _add_check(
        self,
        checks: list[VerificationCheck],
        passed: bool,
        name: str,
        details: str,
        blocking: bool = True,
        artifact_paths: list[str] | None = None,
    ) -> None:
        checks.append(
            VerificationCheck(
                name=name,
                passed=passed,
                details=details,
                blocking=blocking,
                artifact_paths=artifact_paths or [],
            )
        )
