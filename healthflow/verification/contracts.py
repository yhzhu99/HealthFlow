from __future__ import annotations

from dataclasses import dataclass


MODELING_FAMILIES = {"predictive_modeling", "survival_analysis", "time_series_modeling"}
BASE_REPORT_SECTIONS = ("Task Summary", "Data Profile", "Method", "Verification", "Limitations")


@dataclass(frozen=True)
class ArtifactExpectation:
    name: str
    glob_patterns: tuple[str, ...]
    report_tokens: tuple[str, ...] = ()
    blocking: bool = True
    details_present: str = "Required evidence found."
    details_missing: str = "Required evidence is missing."


@dataclass(frozen=True)
class VerificationContract:
    report_sections: tuple[str, ...]
    artifact_expectations: tuple[ArtifactExpectation, ...]

    @property
    def verification_targets(self) -> list[str]:
        targets = list(self.report_sections)
        targets.extend(expectation.name for expectation in self.artifact_expectations)
        return targets


def resolve_verification_contract(task_family: str, domain_focus: str = "general") -> VerificationContract:
    report_sections = list(BASE_REPORT_SECTIONS)
    artifact_expectations: list[ArtifactExpectation] = []

    if task_family == "report_generation":
        report_sections.append("Results")

    if task_family == "cohort_extraction":
        report_sections.append("Selection Logic")
        if domain_focus == "ehr":
            report_sections.append("Cohort Definition")
            artifact_expectations.append(
                ArtifactExpectation(
                    name="cohort_definition",
                    glob_patterns=("cohort_definition.*", "**/cohort_definition.*", "cohort.*", "**/cohort.*"),
                    report_tokens=("cohort definition",),
                    details_present="Cohort definition evidence found.",
                    details_missing="Missing cohort definition artifact or report section.",
                )
            )

    if task_family == "predictive_modeling":
        report_sections.extend(["Validation Strategy", "Metrics Summary"])
        if domain_focus == "ehr":
            report_sections.extend(["Cohort Definition", "Leakage Audit"])

    if task_family == "survival_analysis":
        report_sections.extend(["Temporal Validation", "Metrics Summary"])
        if domain_focus == "ehr":
            report_sections.append("Cohort Definition")

    if task_family == "time_series_modeling":
        report_sections.extend(["Temporal Validation", "Metrics Summary"])
        if domain_focus == "ehr":
            report_sections.append("Cohort Definition")

    if task_family in MODELING_FAMILIES:
        if domain_focus == "ehr":
            artifact_expectations.append(
                ArtifactExpectation(
                    name="cohort_definition",
                    glob_patterns=("cohort_definition.*", "**/cohort_definition.*", "cohort.*", "**/cohort.*"),
                    report_tokens=("cohort definition",),
                    details_present="Cohort definition evidence found.",
                    details_missing="Missing cohort definition artifact or report section.",
                )
            )
        artifact_expectations.append(
            ArtifactExpectation(
                name="split_evidence",
                glob_patterns=("split_evidence.*", "**/split_evidence.*", "split.json", "**/split.json", "data_split.*", "**/data_split.*"),
                report_tokens=("split evidence", "validation strategy", "train/test split", "train/validation/test"),
                details_present="Train/validation/test split evidence found.",
                details_missing="Missing validation split evidence for the modeling task.",
            )
        )
        audit_name = "temporal_validation" if task_family in {"survival_analysis", "time_series_modeling"} else "audit_evidence"
        audit_tokens = ("temporal validation", "leakage audit") if task_family in {"survival_analysis", "time_series_modeling"} else ("validation audit", "leakage audit")
        artifact_expectations.append(
            ArtifactExpectation(
                name=audit_name,
                glob_patterns=(
                    "leakage_audit.*",
                    "**/leakage_audit.*",
                    "temporal_audit.*",
                    "**/temporal_audit.*",
                    "validation_audit.*",
                    "**/validation_audit.*",
                    "causality_audit.*",
                    "**/causality_audit.*",
                ),
                report_tokens=audit_tokens,
                blocking=domain_focus == "ehr" or task_family in {"survival_analysis", "time_series_modeling"},
                details_present=(
                    "Temporal validation evidence found."
                    if task_family in {"survival_analysis", "time_series_modeling"}
                    else "Validation audit evidence found."
                ),
                details_missing=(
                    "Missing temporal validation evidence for the modeling task."
                    if task_family in {"survival_analysis", "time_series_modeling"}
                    else "Missing validation audit evidence for the modeling task."
                ),
            )
        )
        artifact_expectations.append(
            ArtifactExpectation(
                name="metrics_artifact",
                glob_patterns=("metrics.*", "**/metrics.*"),
                details_present="Metric artifact found.",
                details_missing="Expected a metrics artifact for the modeling task family.",
            )
        )

    if task_family == "visualization":
        artifact_expectations.append(
            ArtifactExpectation(
                name="visualization_artifact",
                glob_patterns=("*.png", "*.pdf", "**/*.png", "**/*.pdf"),
                details_present="Visualization artifact found.",
                details_missing="Expected at least one figure artifact for the visualization task family.",
            )
        )

    unique_sections = tuple(dict.fromkeys(report_sections))
    unique_expectations = tuple({expectation.name: expectation for expectation in artifact_expectations}.values())
    return VerificationContract(
        report_sections=unique_sections,
        artifact_expectations=unique_expectations,
    )
