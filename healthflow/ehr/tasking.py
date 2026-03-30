from __future__ import annotations

from typing import Dict, List


TASK_FAMILY_KEYWORDS: Dict[str, List[str]] = {
    "cohort_extraction": ["cohort", "inclusion", "exclusion", "eligibility", "population"],
    "descriptive_analysis": ["describe", "distribution", "incidence", "prevalence", "summary statistics"],
    "predictive_modeling": ["predict", "classification", "model", "auc", "auroc", "xgboost", "logistic"],
    "survival_analysis": ["survival", "hazard", "cox", "time-to-event", "kaplan"],
    "time_series_modeling": ["time series", "forecast", "longitudinal", "sequence", "trajectory"],
    "phenotyping": ["phenotype", "subtype", "cluster", "label", "coding"],
    "causal_or_statistical_analysis": ["causal", "effect", "odds ratio", "regression", "association", "hypothesis"],
    "visualization": ["plot", "visualize", "chart", "figure", "heatmap"],
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
        "cohort_extraction": ["python", "bash", "csv inspection"],
        "descriptive_analysis": ["python", "summary tables", "plots"],
        "predictive_modeling": ["python", "scikit-learn style workflows", "saved metrics"],
        "survival_analysis": ["python", "statistical packages", "saved survival plots"],
        "time_series_modeling": ["python", "temporal validation", "saved metrics"],
        "phenotyping": ["python", "clustering/classification workflows"],
        "causal_or_statistical_analysis": ["python", "statistical testing", "effect-size reporting"],
        "visualization": ["python", "matplotlib/seaborn style plotting"],
        "report_generation": ["markdown", "structured summaries", "saved report"],
        "general_ehr_analysis": ["python", "bash", "saved artifacts"],
    }
    return family_tools.get(task_family, family_tools["general_ehr_analysis"])


def output_contract(task_family: str) -> list[str]:
    contracts = {
        "predictive_modeling": [
            "Save metrics to `metrics.json` or `metrics.txt`.",
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
