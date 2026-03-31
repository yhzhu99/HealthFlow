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


def default_tool_bundle(task_family: str, domain_focus: str = "general") -> list[str]:
    family_tools = {
        "cohort_extraction": ["python", "tabular inspection", "filtering logic audit"],
        "descriptive_analysis": ["python", "summary tables", "plots"],
        "predictive_modeling": ["python", "train/validation split audit", "saved metrics"],
        "survival_analysis": ["python", "time-aware validation", "saved metrics"],
        "time_series_modeling": ["python", "time-aware validation", "saved metrics"],
        "phenotyping": ["python", "clustering/classification workflows"],
        "causal_or_statistical_analysis": ["python", "statistical testing", "effect-size reporting"],
        "visualization": ["python", "matplotlib/seaborn style plotting"],
        "report_generation": ["markdown", "structured summaries", "saved report"],
        "general_analysis": ["python", "bash", "saved artifacts"],
    }
    bundle = list(family_tools.get(task_family, family_tools["general_analysis"]))
    if domain_focus == "ehr" and task_family in {"predictive_modeling", "survival_analysis", "time_series_modeling"}:
        bundle.extend(["patient-level split audit", "temporal leakage check", "external cli workflows"])
    if domain_focus == "ehr" and task_family == "cohort_extraction":
        bundle.append("cohort definition artifact")
    return list(dict.fromkeys(bundle))


def deliverable_guidance(task_family: str, domain_focus: str = "general") -> list[str]:
    guidance = {
        "cohort_extraction": [
            "Persist the selection logic in a script, notebook, or auditable artifact when reproducibility matters.",
            "If the final output is a cohort/table, save the resulting subset and briefly explain the criteria you applied.",
        ],
        "predictive_modeling": [
            "Save the training/evaluation code or commands when they are non-trivial.",
            "Save metrics and enough validation context to reproduce the result.",
            "If data splitting choices matter, record the split strategy in an artifact or concise report.",
        ],
        "survival_analysis": [
            "Save metrics and document the time origin, censoring, or event-horizon assumptions.",
            "Keep the analysis auditable with scripts or concise notes describing temporal validation.",
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
                "Prefer auditable artifacts when they materially support the answer.",
                "A concise final answer is acceptable when the task does not require saved files.",
            ],
        )
    )
    if domain_focus == "ehr":
        if task_family in {"predictive_modeling", "survival_analysis", "time_series_modeling"}:
            suggestions.extend(
                [
                    "When relevant, save cohort, split, and leakage evidence so the EHR workflow can be audited.",
                    "Prefer patient-aware or time-aware validation artifacts over implicit assumptions.",
                ]
            )
        if task_family == "cohort_extraction":
            suggestions.append("If this is an EHR cohort task, save the cohort definition in a machine-readable artifact.")
    return list(dict.fromkeys(suggestions))


def verification_guidance(task_family: str, domain_focus: str = "general") -> list[str]:
    guidance = {
        "predictive_modeling": [
            "Keep metrics and validation strategy auditable.",
            "Record split logic when it materially affects the result.",
        ],
        "survival_analysis": [
            "Keep metrics, censoring assumptions, and temporal validation auditable.",
        ],
        "time_series_modeling": [
            "Keep metrics and chronological validation auditable.",
        ],
        "visualization": [
            "Save the generated figure so it can be inspected directly.",
        ],
        "report_generation": [
            "Produce a report artifact rather than relying only on stdout.",
        ],
    }
    suggestions = list(
        guidance.get(
            task_family,
            [
                "Prefer reproducible artifacts when they materially support deterministic review.",
            ],
        )
    )
    if domain_focus == "ehr" and task_family in {"cohort_extraction", "predictive_modeling", "survival_analysis", "time_series_modeling"}:
        suggestions.append("For EHR-style workflows, keep cohort, split, and leakage evidence auditable when applicable.")
    return list(dict.fromkeys(suggestions))


def required_report_sections(task_family: str, domain_focus: str = "general") -> list[str]:
    task_sections = {
        "report_generation": ["Task Summary", "Method", "Results"],
        "cohort_extraction": ["Selection Logic"],
        "predictive_modeling": ["Validation Strategy", "Metrics Summary"],
        "survival_analysis": ["Temporal Validation", "Metrics Summary"],
        "time_series_modeling": ["Temporal Validation", "Metrics Summary"],
    }
    sections = list(task_sections.get(task_family, []))
    if domain_focus == "ehr":
        if task_family == "cohort_extraction":
            sections.insert(0, "Cohort Definition")
        if task_family == "predictive_modeling":
            sections.extend(["Cohort Definition", "Leakage Audit"])
        if task_family in {"survival_analysis", "time_series_modeling"}:
            sections.append("Cohort Definition")
    return list(dict.fromkeys(sections))
