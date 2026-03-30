from __future__ import annotations

from typing import Dict, List


TASK_FAMILY_KEYWORDS: Dict[str, List[str]] = {
    "cohort_extraction": ["cohort", "inclusion", "exclusion", "eligibility", "population", "index date", "criteria"],
    "descriptive_analysis": ["describe", "distribution", "incidence", "prevalence", "summary statistics", "baseline characteristics"],
    "predictive_modeling": ["predict", "prediction", "predictive", "classification", "model", "auc", "auroc", "xgboost", "logistic", "mortality", "readmission", "risk score"],
    "survival_analysis": ["survival", "hazard", "cox", "time-to-event", "kaplan", "censoring", "c-index"],
    "time_series_modeling": ["time series", "forecast", "longitudinal", "sequence", "trajectory", "temporal model"],
    "phenotyping": ["phenotype", "subtype", "cluster", "coding phenotype"],
    "causal_or_statistical_analysis": ["causal", "effect", "odds ratio", "regression", "association", "hypothesis", "hazard ratio"],
    "visualization": ["plot", "visualize", "chart", "figure", "heatmap", "roc curve"],
    "report_generation": ["report", "summarize", "write-up", "interpretation", "manuscript"],
}


def classify_task_family(user_request: str) -> str:
    request = user_request.lower()
    scored = []
    for family, keywords in TASK_FAMILY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in request)
        if score:
            scored.append((score, family))
    scored.sort(reverse=True)
    return scored[0][1] if scored else "general_ehr_analysis"


def default_tool_bundle(task_family: str) -> list[str]:
    family_tools = {
        "cohort_extraction": ["python", "csv inspection", "cohort logic audit"],
        "descriptive_analysis": ["python", "summary tables", "plots"],
        "predictive_modeling": ["python", "patient-level split audit", "saved metrics", "oneehr preprocess/train/test/analyze"],
        "survival_analysis": ["python", "temporal validation", "saved metrics", "oneehr preprocess/train/test/analyze"],
        "time_series_modeling": ["python", "temporal validation", "saved metrics", "oneehr preprocess/train/test/analyze"],
        "phenotyping": ["python", "clustering/classification workflows"],
        "causal_or_statistical_analysis": ["python", "statistical testing", "effect-size reporting"],
        "visualization": ["python", "matplotlib/seaborn style plotting"],
        "report_generation": ["markdown", "structured summaries", "saved report"],
        "general_ehr_analysis": ["python", "bash", "saved artifacts"],
    }
    return family_tools.get(task_family, family_tools["general_ehr_analysis"])


def output_contract(task_family: str) -> list[str]:
    contracts = {
        "cohort_extraction": [
            "Save the cohort definition to `cohort_definition.json`, `cohort_definition.md`, or `cohort.csv`.",
            "Write `final_report.md` with cohort criteria and verification details.",
        ],
        "predictive_modeling": [
            "Save the cohort definition to `cohort_definition.json`, `cohort_definition.md`, or `manifest.json`.",
            "Save split evidence to `split_evidence.json`, `split.json`, or `preprocess/split.json`.",
            "Save a leakage or validation audit to `leakage_audit.md`, `temporal_audit.md`, or include those sections in `final_report.md`.",
            "Save metrics to `metrics.json`, `metrics.txt`, or `test/metrics.json`.",
            "Save a concise report to `final_report.md`.",
        ],
        "survival_analysis": [
            "Save the cohort definition to `cohort_definition.json`, `cohort_definition.md`, or `manifest.json`.",
            "Save split evidence to `split_evidence.json`, `split.json`, or `preprocess/split.json`.",
            "Save temporal validation evidence to `temporal_audit.md` or include it in `final_report.md`.",
            "Save metrics to `metrics.json`, `metrics.txt`, or `test/metrics.json`.",
            "Save a concise report to `final_report.md`.",
        ],
        "time_series_modeling": [
            "Save the cohort definition to `cohort_definition.json`, `cohort_definition.md`, or `manifest.json`.",
            "Save split evidence to `split_evidence.json`, `split.json`, or `preprocess/split.json`.",
            "Save temporal validation evidence to `temporal_audit.md` or include it in `final_report.md`.",
            "Save metrics to `metrics.json`, `metrics.txt`, or `test/metrics.json`.",
            "Save a concise report to `final_report.md`.",
        ],
        "visualization": [
            "Save figures to the workspace as PNG or PDF files.",
            "Describe the figure and main takeaway in `final_report.md`.",
        ],
        "report_generation": [
            "Write `final_report.md` with the required section headers.",
        ],
    }
    return contracts.get(
        task_family,
        [
            "Write `final_report.md` with the required section headers when the task is analysis-heavy.",
            "End stdout with a concise final answer and mention saved artifacts.",
        ],
    )


def required_report_sections(task_family: str) -> list[str]:
    task_sections = {
        "cohort_extraction": ["Cohort Definition", "Eligibility Logic"],
        "predictive_modeling": ["Cohort Definition", "Split Evidence", "Leakage Audit", "Metrics Summary"],
        "survival_analysis": ["Cohort Definition", "Split Evidence", "Temporal Validation", "Metrics Summary"],
        "time_series_modeling": ["Cohort Definition", "Split Evidence", "Temporal Validation", "Metrics Summary"],
    }
    return task_sections.get(task_family, [])
