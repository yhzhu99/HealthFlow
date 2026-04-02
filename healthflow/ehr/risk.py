from __future__ import annotations

from .models import DataProfile, RiskFinding


LEAKAGE_TERMS = {"mortality", "death", "outcome", "readmission", "label", "target", "future"}
TEMPORAL_TERMS = {"discharge", "post", "future", "after", "follow-up", "followup", "later"}
GROUP_ID_COLUMNS = {
    "id",
    "sample_id",
    "entity_id",
    "record_id",
    "group_id",
    "subject_id",
    "patient_id",
    "hadm_id",
    "stay_id",
    "encounter_id",
    "visit_id",
    "user_id",
    "customer_id",
    "account_id",
}
PATIENT_ID_COLUMNS = {"subject_id", "patient_id", "hadm_id", "stay_id", "encounter_id", "visit_id"}
TARGET_COLUMNS = {"label", "outcome", "target", "mortality", "readmission", "death", "response", "class", "y"}
TIME_COLUMNS = {
    "event_time",
    "charttime",
    "admit_time",
    "admission_time",
    "dischtime",
    "discharge_time",
    "label_time",
    "index_time",
    "start_time",
    "end_time",
    "time",
    "timestamp",
    "date",
    "datetime",
    "created_at",
}
COHORT_TERMS = {"cohort", "inclusion", "exclusion", "eligibility", "index date", "population"}
MODELING_FAMILIES = {"predictive_modeling", "survival_analysis", "time_series_modeling"}


def detect_risk_findings(user_request: str, data_profile: DataProfile) -> list[RiskFinding]:
    request = user_request.lower()
    findings: list[RiskFinding] = []
    is_modeling_task = data_profile.task_family in MODELING_FAMILIES

    if any(term in request for term in LEAKAGE_TERMS) and any(term in request for term in TEMPORAL_TERMS):
        findings.append(
            RiskFinding(
                severity="high",
                category="temporal_leakage",
                message="The task may involve post-outcome or future information; validate the prediction timeline explicitly.",
            )
        )

    if is_modeling_task and data_profile.group_id_columns:
        findings.append(
            RiskFinding(
                severity="medium",
                category="grouping",
                message=(
                    "Patient-level identifiers detected. Use patient-aware splitting and avoid duplicate leakage across folds."
                    if data_profile.domain_focus == "ehr" and data_profile.patient_id_columns
                    else "Group/entity identifiers detected. Use entity-aware splitting and avoid duplicated entities across folds."
                ),
            )
        )

    if is_modeling_task and data_profile.target_columns:
        findings.append(
            RiskFinding(
                severity="high",
                category="target_leakage",
                message="Target-like columns detected in inputs. Confirm they are not used as predictors except where explicitly intended.",
            )
        )

    if is_modeling_task:
        findings.append(
            RiskFinding(
                severity="medium",
                category="validation_strategy",
                message=(
                    "Modeling tasks should provide patient-level train/validation/test split evidence and preserve temporal causality."
                    if data_profile.domain_focus == "ehr"
                    else "Modeling tasks should document their train/validation/test strategy and preserve entity or temporal independence where relevant."
                ),
            )
        )
        if data_profile.domain_focus == "ehr" and not data_profile.patient_id_columns:
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

    if (
        data_profile.domain_focus == "ehr"
        and data_profile.task_family == "cohort_extraction"
        and not any(term in request for term in COHORT_TERMS)
    ):
        findings.append(
            RiskFinding(
                severity="medium",
                category="cohort_definition",
                message="Cohort tasks should state inclusion and exclusion logic explicitly, including index date and outcome window.",
                )
            )

    if data_profile.domain_focus == "ehr":
        findings.append(
            RiskFinding(
                severity="info",
                category="domain_overlay",
                message="EHR signals were detected. Apply only the healthcare-specific checks that are directly relevant to the requested task.",
            )
        )

    if not findings:
        findings.append(
            RiskFinding(
                severity="info",
                category="audit",
                message="No elevated domain-specific risk was detected from the request and profiled inputs; continue with standard validation.",
            )
        )
    return findings
