from __future__ import annotations

from typing import Dict, List


TASK_FAMILY_KEYWORDS: Dict[str, List[str]] = {
    "cohort_extraction": ["cohort", "inclusion", "exclusion", "eligibility", "population", "index date", "criteria", "subset", "filter"],
    "descriptive_analysis": ["describe", "distribution", "incidence", "prevalence", "summary statistics", "baseline characteristics", "eda", "exploratory analysis", "summarize"],
    "predictive_modeling": ["predict", "prediction", "predictive", "classification", "regression", "model", "auc", "auroc", "xgboost", "logistic", "random forest", "f1 score", "accuracy", "rmse", "mse"],
    "survival_analysis": ["survival", "hazard", "cox", "time-to-event", "kaplan", "censoring", "c-index"],
    "time_series_modeling": ["time series", "timeseries", "forecast", "longitudinal", "sequence", "trajectory", "temporal model", "gru", "lstm"],
    "phenotyping": ["phenotype", "subtype", "cluster", "clustering", "embedding"],
    "causal_or_statistical_analysis": ["causal", "effect", "odds ratio", "association", "hypothesis", "hazard ratio", "p-value", "significance", "correlation", "ab test", "statistical test"],
    "visualization": ["plot", "visualize", "chart", "figure", "heatmap", "roc curve"],
    "report_generation": ["report", "write-up", "interpretation", "manuscript"],
}

EHR_DOMAIN_TERMS: Dict[str, List[str]] = {
    "request": [
        "ehr",
        "electronic health record",
        "clinical",
        "hospital",
        "icu",
        "discharge summary",
        "readmission",
        "mortality",
        "diagnosis",
        "medication",
        "mimic",
        "eicu",
        "physionet",
        "phenotype",
        "icd",
    ],
    "file": [
        "mimic",
        "ehr",
        "patient",
        "clinical",
        "admission",
        "discharge",
        "icu",
        "medication",
        "diagnosis",
    ],
    "column": [
        "subject_id",
        "patient_id",
        "hadm_id",
        "stay_id",
        "encounter_id",
        "visit_id",
        "mortality",
        "readmission",
        "admission_time",
        "discharge_time",
        "charttime",
        "icd_code",
        "diagnosis_code",
        "medication",
        "lab",
    ],
}


def classify_task_family(user_request: str) -> str:
    request = user_request.lower()
    scored = []
    for family, keywords in TASK_FAMILY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in request)
        if score:
            scored.append((score, family))
    scored.sort(reverse=True)
    return scored[0][1] if scored else "general_analysis"


def detect_domain_focus(
    user_request: str,
    file_names: list[str] | None = None,
    columns: list[str] | None = None,
) -> tuple[str, list[str]]:
    request = user_request.lower()
    file_names = [name.lower() for name in file_names or []]
    columns = [column.lower() for column in columns or []]

    signals: list[str] = []
    if any(term in request for term in EHR_DOMAIN_TERMS["request"]):
        signals.append("request")
    if any(any(term in name for term in EHR_DOMAIN_TERMS["file"]) for name in file_names):
        signals.append("file")
    if any(column in set(EHR_DOMAIN_TERMS["column"]) for column in columns):
        signals.append("schema")

    is_ehr = len(signals) >= 2
    return ("ehr" if is_ehr else "general", signals)


def default_workflow_recommendations(task_family: str, domain_focus: str = "general") -> list[str]:
    family_recommendations = {
        "cohort_extraction": ["Use reproducible cohort logic", "Persist filtering criteria as an auditable artifact"],
        "descriptive_analysis": ["Save summary tables or figures when they support the answer"],
        "predictive_modeling": ["Save validation evidence and metrics artifacts"],
        "survival_analysis": ["Document time origin, censoring, and validation choices"],
        "time_series_modeling": ["Document chronology constraints and evaluation artifacts"],
        "phenotyping": ["Persist clustering or phenotyping outputs in reusable artifacts"],
        "causal_or_statistical_analysis": ["Record statistical assumptions, tests, and effect-size outputs"],
        "visualization": ["Save figures and a concise interpretation artifact"],
        "report_generation": ["Write a structured report artifact that answers the task directly"],
        "general_analysis": ["Prefer reproducible workspace-local scripts and saved artifacts"],
    }
    recommendations = list(family_recommendations.get(task_family, family_recommendations["general_analysis"]))
    if domain_focus == "ehr" and task_family in {"predictive_modeling", "survival_analysis", "time_series_modeling"}:
        recommendations.extend(["Preserve patient-aware split evidence", "Check temporal leakage explicitly"])
    if domain_focus == "ehr" and task_family == "cohort_extraction":
        recommendations.append("Persist the cohort definition in a machine-readable artifact")
    return list(dict.fromkeys(recommendations))


def deliverable_guidance(task_family: str, domain_focus: str = "general") -> list[str]:
    guidance = {
        "cohort_extraction": [
            "Persist the selection logic in a script, notebook, or reproducible artifact when reproducibility matters.",
            "If the final output is a cohort/table, save the resulting subset and briefly explain the criteria you applied.",
        ],
        "predictive_modeling": [
            "Save the training/evaluation code or commands when they are non-trivial.",
            "Save metrics and enough validation context to reproduce the result.",
            "If data splitting choices matter, record the split strategy in an artifact or concise report.",
        ],
        "survival_analysis": [
            "Save metrics and document the time origin, censoring, or event-horizon assumptions.",
            "Keep the analysis reproducible with scripts or concise notes describing temporal validation.",
        ],
        "time_series_modeling": [
            "Save metrics and document how chronology or forecasting constraints were handled.",
            "Keep scripts or notes that make feature extraction and evaluation reproducible.",
        ],
        "visualization": [
            "Save figures to the workspace as PNG or PDF files.",
            "Describe the figure and main takeaway in a concise note or final answer.",
        ],
        "report_generation": [
            "Write a structured report artifact that directly answers the task.",
        ],
    }
    suggestions = list(
        guidance.get(
            task_family,
            [
                "Prefer reproducible artifacts when they materially support the answer.",
                "A concise final answer is acceptable when the task does not require saved files.",
            ],
        )
    )
    if domain_focus == "ehr":
        if task_family in {"predictive_modeling", "survival_analysis", "time_series_modeling"}:
            suggestions.extend(
                [
                    "When relevant, save cohort, split, and leakage evidence so the EHR workflow can be inspected and reproduced.",
                    "Prefer patient-aware or time-aware validation artifacts over implicit assumptions.",
                ]
            )
        if task_family == "cohort_extraction":
            suggestions.append("If this is an EHR cohort task, save the cohort definition in a machine-readable artifact.")
    return list(dict.fromkeys(suggestions))
