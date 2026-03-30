from __future__ import annotations

from .models import DataProfile, RiskFinding


LEAKAGE_TERMS = {"mortality", "death", "outcome", "readmission", "label", "target"}
TEMPORAL_TERMS = {"discharge", "post", "future", "after", "follow-up"}
IDENTIFIER_COLUMNS = {"subject_id", "patient_id", "hadm_id", "stay_id", "encounter_id"}
TARGET_COLUMNS = {"label", "outcome", "target", "mortality", "readmission"}


def detect_risk_findings(user_request: str, data_profile: DataProfile) -> list[RiskFinding]:
    request = user_request.lower()
    findings: list[RiskFinding] = []

    if any(term in request for term in LEAKAGE_TERMS) and any(term in request for term in TEMPORAL_TERMS):
        findings.append(
            RiskFinding(
                severity="high",
                category="temporal_leakage",
                message="The task may involve post-outcome or future information; validate the prediction timeline explicitly.",
            )
        )

    all_columns = {column.lower() for schema in data_profile.schemas for column in schema.columns}
    if IDENTIFIER_COLUMNS.intersection(all_columns):
        findings.append(
            RiskFinding(
                severity="medium",
                category="patient_grouping",
                message="Patient-level identifiers detected. Use patient-aware splitting and avoid duplicate leakage across folds.",
            )
        )

    if TARGET_COLUMNS.intersection(all_columns):
        findings.append(
            RiskFinding(
                severity="medium",
                category="target_leakage",
                message="Target-like columns detected in inputs. Confirm they are not used as predictors except where explicitly intended.",
            )
        )

    if "survival" in data_profile.task_family and "time" not in request:
        findings.append(
            RiskFinding(
                severity="medium",
                category="time_origin",
                message="Survival tasks require a clearly defined time origin, censoring rule, and follow-up window.",
            )
        )

    if not findings:
        findings.append(
            RiskFinding(
                severity="info",
                category="audit",
                message="No obvious EHR-specific risk detected from the request and profiled inputs; continue with standard validation.",
            )
        )
    return findings
