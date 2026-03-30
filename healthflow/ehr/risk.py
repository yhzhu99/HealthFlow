from __future__ import annotations

from .models import DataProfile, RiskFinding


LEAKAGE_TERMS = {"mortality", "death", "outcome", "readmission", "label", "target"}
TEMPORAL_TERMS = {"discharge", "post", "future", "after", "follow-up", "followup", "later"}
IDENTIFIER_COLUMNS = {"subject_id", "patient_id", "hadm_id", "stay_id", "encounter_id", "visit_id"}
TARGET_COLUMNS = {"label", "outcome", "target", "mortality", "readmission", "death"}
TIME_COLUMNS = {"event_time", "charttime", "admit_time", "admission_time", "dischtime", "discharge_time", "label_time", "index_time", "time", "timestamp", "date"}
COHORT_TERMS = {"cohort", "inclusion", "exclusion", "eligibility", "index date", "population"}
MODELING_FAMILIES = {"predictive_modeling", "survival_analysis", "time_series_modeling"}


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

    if data_profile.patient_id_columns:
        findings.append(
            RiskFinding(
                severity="medium",
                category="patient_grouping",
                message="Patient-level identifiers detected. Use patient-aware splitting and avoid duplicate leakage across folds.",
            )
        )

    if data_profile.target_columns:
        severity = "high" if data_profile.task_family in MODELING_FAMILIES else "medium"
        findings.append(
            RiskFinding(
                severity=severity,
                category="target_leakage",
                message="Target-like columns detected in inputs. Confirm they are not used as predictors except where explicitly intended.",
            )
        )

    if data_profile.task_family in MODELING_FAMILIES:
        findings.append(
            RiskFinding(
                severity="medium",
                category="split_evidence",
                message="Modeling tasks should provide patient-level train/validation/test split evidence and preserve temporal causality.",
            )
        )
        if not data_profile.patient_id_columns:
            findings.append(
                RiskFinding(
                    severity="medium",
                    category="patient_linkage",
                    message="No obvious patient identifier columns were detected. Explicitly justify how patient-level splitting is enforced.",
                )
            )

    if data_profile.task_family in {"survival_analysis", "time_series_modeling"}:
        findings.append(
            RiskFinding(
                severity="medium",
                category="temporal_contract",
                message="Temporal tasks require a clearly defined time origin, censoring or forecast horizon, and no post-index leakage.",
            )
        )

    if data_profile.task_family == "cohort_extraction" and not any(term in request for term in COHORT_TERMS):
        findings.append(
            RiskFinding(
                severity="medium",
                category="cohort_definition",
                message="Cohort tasks should state inclusion and exclusion logic explicitly, including index date and outcome window.",
            )
        )

    if "oneehr_manifest" in data_profile.artifact_hints:
        findings.append(
            RiskFinding(
                severity="info",
                category="workflow_contract",
                message="OneEHR-style manifest artifacts detected. Reuse the saved split and metrics contracts rather than recreating them ad hoc.",
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
