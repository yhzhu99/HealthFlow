from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "config.toml").exists() and (parent / "data").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")


PROJECT_ROOT = find_project_root()
DATASET_ROOT = PROJECT_ROOT / "data" / "ehrflowbench"
PAPERS_ROOT = DATASET_ROOT / "processed" / "papers"
DEFAULT_INPUT_PATH = PAPERS_ROOT / "final_220_tasks.json"
DEFAULT_OUTPUT_PATH = PAPERS_ROOT / "focus_areas.md"
DEFAULT_CSV_OUTPUT_PATH = PAPERS_ROOT / "focus_areas.csv"

PRIMARY_CATEGORY_META = {
    "temporal_predictive_modeling_and_early_warning": {
        "label": "Temporal Predictive Modeling & Early Warning",
        "description": "Outcome or mortality modeling centered on temporal aggregation, longitudinal signal use, alerting, or dynamic risk scoring.",
    },
    "graph_similarity_retrieval_and_structure_aware_modeling": {
        "label": "Graph, Similarity, Retrieval & Structure-Aware Modeling",
        "description": "Graph-derived, patient-similarity, prototype, memory, or retrieval-based task designs.",
    },
    "representation_learning_and_feature_engineering": {
        "label": "Representation Learning & Feature Engineering",
        "description": "Tasks whose main novelty is representation construction, feature hierarchy, fusion, alignment, or structured feature engineering.",
    },
    "phenotyping_clustering_and_subgroup_discovery": {
        "label": "Phenotyping, Clustering & Subgroup Discovery",
        "description": "Clustering, latent states, phenotyping, subgroup routing, or patient subtyping tasks.",
    },
    "robustness_missingness_and_resource_constrained_learning": {
        "label": "Robustness, Missingness & Resource-Constrained Learning",
        "description": "Few-shot, low-label, missing-data, active acquisition, cost-aware, or latency-aware tasks.",
    },
    "multi_task_transfer_and_distillation": {
        "label": "Multi-Task Learning, Transfer & Distillation",
        "description": "Joint-task learning, transfer, auxiliary tasks, task routing, or distillation-focused tasks.",
    },
    "synthetic_data_privacy_and_linkage": {
        "label": "Synthetic Data, Privacy & Record Linkage",
        "description": "Synthetic EHR generation, privacy-preserving modeling, de-identification, distributed simulation, or linkage tasks.",
    },
    "causal_effect_and_counterfactual": {
        "label": "Causal Effect, Counterfactual & Mediation Analysis",
        "description": "Treatment-effect, mediation, counterfactual, or propensity-style analysis tasks.",
    },
    "forecasting_and_future_state_modeling": {
        "label": "Forecasting & Future-State Modeling",
        "description": "Next-step, next-record, lab-response, or future clinical state forecasting tasks.",
    },
    "nl_querying_and_structured_reporting": {
        "label": "Natural-Language Querying & Structured Report Generation",
        "description": "Natural-language parsing, cohort query answering, summarization, or report auto-completion tasks.",
    },
    "anomaly_detection_and_data_centric_analysis": {
        "label": "Anomaly Detection & Data-Centric Analysis",
        "description": "Anomaly detection, outlier auditing, or hard-negative/data-centric analysis tasks.",
    },
}

OVERLAP_THEMES = {
    "temporal_longitudinal_modeling": {
        "label": "Temporal / Longitudinal Modeling",
        "keywords": [
            "temporal",
            "longitudinal",
            "trajectory",
            "time-aware",
            "irregular",
            "multi-horizon",
            "sequence",
            "dynamic risk",
            "early warning",
        ],
    },
    "prediction_and_risk_scoring": {
        "label": "Prediction / Risk Scoring",
        "keywords": [
            "outcome prediction",
            "mortality prediction",
            "risk prediction",
            "outcome modeling",
            "risk modeling",
            "mortality modeling",
            "clinical prediction",
        ],
    },
    "interpretability_and_explanation": {
        "label": "Interpretability / Explanation",
        "keywords": [
            "interpretable",
            "interpretability",
            "explainable",
            "explanation",
            "interpretation",
            "clinical interpretability",
            "model interpretability",
            "counterfactual",
            "reasoning-aware",
        ],
    },
    "graph_similarity_retrieval": {
        "label": "Graph / Similarity / Retrieval",
        "keywords": ["graph", "similarity", "retrieval", "prototype", "memory", "neighbor", "cohort-aware"],
    },
    "representation_feature_learning": {
        "label": "Representation / Feature Learning",
        "keywords": [
            "representation",
            "embedding",
            "feature engineering",
            "fusion",
            "alignment",
            "concept-aware",
            "hierarchical",
            "knowledge-guided",
            "frequency features",
            "compositional",
        ],
    },
    "phenotyping_clustering_subgroups": {
        "label": "Phenotyping / Clustering / Subgroups",
        "keywords": [
            "phenotyping",
            "clustering",
            "subtyping",
            "subgroup",
            "latent stage",
            "latent state",
            "cohort discovery",
            "progression",
        ],
    },
    "robustness_missingness_efficiency": {
        "label": "Robustness / Missingness / Efficiency",
        "keywords": [
            "robust",
            "missing",
            "few-shot",
            "low-label",
            "label scarcity",
            "data efficiency",
            "cost-aware",
            "latency-aware",
            "feature acquisition",
            "active learning",
        ],
    },
    "multitask_transfer_distillation": {
        "label": "Multi-Task / Transfer / Distillation",
        "keywords": [
            "multi-task",
            "multitask",
            "transfer",
            "distillation",
            "auxiliary",
            "task selection",
            "task grouping",
            "consensus modeling",
        ],
    },
    "synthetic_privacy_linkage": {
        "label": "Synthetic / Privacy / Linkage",
        "keywords": ["synthetic", "privacy", "de-identification", "linkage", "fidelity", "distributed simulation"],
    },
    "causal_counterfactual_mediation": {
        "label": "Causal / Counterfactual / Mediation",
        "keywords": ["causal", "propensity", "counterfactual", "mediation", "doubly robust", "g-computation"],
    },
    "forecasting_future_state": {
        "label": "Forecasting / Future State",
        "keywords": ["forecasting", "future state", "next-step", "next-record", "trajectory forecasting", "temporal lab forecasting"],
    },
    "nl_querying_reporting": {
        "label": "NL Querying / Reporting",
        "keywords": ["question answering", "natural-language querying", "query parsing", "ehr summarization", "structured report generation", "report generation"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize focus areas for final_220_tasks.json.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-output-path", type=Path, default=DEFAULT_CSV_OUTPUT_PATH)
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%"


def classify_primary(task: dict) -> str:
    labels = [str(value).strip().lower() for value in task.get("focus_areas", [])]
    brief = str(task.get("task_brief", "")).lower()

    def any_kw(*keywords: str) -> bool:
        return any(any(keyword in label for keyword in keywords) for label in labels)

    def brief_has(*keywords: str) -> bool:
        return any(keyword in brief for keyword in keywords)

    if any_kw("causal", "propensity", "counterfactual", "mediation", "g-computation", "doubly robust", "effect decomposition"):
        return "causal_effect_and_counterfactual"
    if any_kw("synthetic", "privacy", "de-identification", "linkage", "fidelity", "distributed simulation"):
        return "synthetic_data_privacy_and_linkage"
    if any_kw("question answering", "natural-language querying", "query parsing", "ehr summarization", "structured report generation") or brief_has(
        "natural-language",
        "parser",
        "auto-completion",
        "summary generator",
        "case-summary",
        "structured patient risk reports",
        "cohort-question",
    ):
        return "nl_querying_and_structured_reporting"
    if any_kw("forecasting", "future state", "trajectory forecasting", "temporal lab forecasting") or brief_has(
        "forecasting",
        "future clinical state",
        "glucose prediction",
        "lab-response",
    ):
        return "forecasting_and_future_state_modeling"
    if any_kw(
        "phenotyping",
        "clustering",
        "subtyping",
        "subphenotypes",
        "subgroup",
        "latent stage",
        "latent state",
        "state inference",
        "state discovery",
        "health states",
        "progression",
        "cohort discovery",
    ):
        return "phenotyping_clustering_and_subgroup_discovery"
    if any_kw("graph", "similarity", "retrieval", "prototype", "memory", "neighbor", "cohort-aware"):
        return "graph_similarity_retrieval_and_structure_aware_modeling"
    if any_kw("multi-task", "multitask", "transfer", "distillation", "auxiliary", "task selection", "task grouping", "consensus modeling"):
        return "multi_task_transfer_and_distillation"
    if any_kw("few-shot", "low-label", "limited-data", "data efficiency", "missing", "robustness", "cost-aware", "latency-aware", "feature acquisition", "active learning", "label scarcity"):
        return "robustness_missingness_and_resource_constrained_learning"
    if any_kw(
        "representation",
        "embedding",
        "fusion",
        "alignment",
        "concept-aware",
        "hierarchical",
        "knowledge-guided",
        "feature engineering",
        "frequency features",
        "compositional",
        "reasoning-aware",
        "coarse-to-fine",
        "semantic feature",
        "time-aware summarization",
    ):
        return "representation_learning_and_feature_engineering"
    if any_kw("anomaly", "outlier", "negative sample"):
        return "anomaly_detection_and_data_centric_analysis"
    return "temporal_predictive_modeling_and_early_warning"


def compute_overlap_counts(tasks: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for task in tasks:
        labels = " | ".join(str(value).strip().lower() for value in task.get("focus_areas", []))
        brief = str(task.get("task_brief", "")).lower()
        text = f"{labels} | {brief}"
        for theme_key, meta in OVERLAP_THEMES.items():
            if any(keyword in text for keyword in meta["keywords"]):
                counts[theme_key] += 1
    return counts


def matched_overlap_themes(task: dict) -> list[str]:
    labels = " | ".join(str(value).strip().lower() for value in task.get("focus_areas", []))
    brief = str(task.get("task_brief", "")).lower()
    text = f"{labels} | {brief}"
    matched: list[str] = []
    for theme_key, meta in OVERLAP_THEMES.items():
        if any(keyword in text for keyword in meta["keywords"]):
            matched.append(theme_key)
    return matched


def analyze_tasks(tasks: list[dict]) -> tuple[list[dict], Counter[str], Counter[str], Counter[str]]:
    primary_counts: Counter[str] = Counter()
    focus_area_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    enriched_tasks: list[dict] = []

    for index, task in enumerate(tasks, start=1):
        primary_category = classify_primary(task)
        primary_counts[primary_category] += 1
        normalized_focus_areas = [str(value).strip() for value in task.get("focus_areas", [])]
        focus_area_counts.update(normalized_focus_areas)
        overlap_theme_keys = matched_overlap_themes(task)
        overlap_counts.update(overlap_theme_keys)
        enriched_tasks.append(
            {
                **task,
                "task_idx": index,
                "primary_category": primary_category,
                "focus_areas": normalized_focus_areas,
                "overlap_theme_keys": overlap_theme_keys,
            }
        )

    return enriched_tasks, primary_counts, focus_area_counts, overlap_counts


def build_markdown(source_path: Path, tasks: list[dict]) -> str:
    enriched_tasks, primary_counts, focus_area_counts, overlap_counts = analyze_tasks(tasks)
    total_focus_area_mentions = sum(focus_area_counts.values())

    lines: list[str] = []
    lines.append("# EHRFlowBench Focus Areas")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Source: `{source_path.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Tasks: `{len(tasks)}`")
    lines.append(f"- Total focus-area mentions: `{total_focus_area_mentions}`")
    lines.append(f"- Unique exact focus-area strings: `{len(focus_area_counts)}`")
    lines.append("- Note: the primary categories below are normalized analytical buckets derived from `focus_areas` and `task_brief`, not original dataset labels.")
    lines.append("")
    lines.append("## Primary Category Distribution")
    lines.append("")
    lines.append("| Category | Count | Share | Description |")
    lines.append("| --- | ---: | ---: | --- |")
    for category_key, count in primary_counts.most_common():
        meta = PRIMARY_CATEGORY_META[category_key]
        lines.append(f"| {meta['label']} | {count} | {percent(count, len(tasks))} | {meta['description']} |")
    lines.append("")
    lines.append("## Cross-Cutting Theme Distribution")
    lines.append("")
    lines.append("| Theme | Count | Share |")
    lines.append("| --- | ---: | ---: |")
    for theme_key, count in overlap_counts.most_common():
        meta = OVERLAP_THEMES[theme_key]
        lines.append(f"| {meta['label']} | {count} | {percent(count, len(tasks))} |")
    lines.append("")
    lines.append("## Top Exact Focus Areas")
    lines.append("")
    lines.append("| Rank | Focus Area | Count |")
    lines.append("| ---: | --- | ---: |")
    for rank, (label, count) in enumerate(focus_area_counts.most_common(50), start=1):
        lines.append(f"| {rank} | `{label}` | {count} |")
    lines.append("")
    lines.append("## All Exact Focus Areas")
    lines.append("")
    for label, count in focus_area_counts.most_common():
        lines.append(f"- `{label}`: {count}")
    lines.append("")
    lines.append("## Task-Level Focus Areas")
    lines.append("")
    for task in enriched_tasks:
        category_label = PRIMARY_CATEGORY_META[task["primary_category"]]["label"]
        lines.append(f"### Task {task['task_idx']:03d} | paper_id={task['paper_id']} | source_task_idx={task['source_task_idx']}")
        lines.append(f"- Task brief: {task['task_brief']}")
        lines.append(f"- Primary category: {category_label}")
        lines.append("- Focus areas:")
        for focus_area in task["focus_areas"]:
            lines.append(f"  - `{focus_area}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_csv(path: Path, enriched_tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_idx",
        "paper_id",
        "paper_title",
        "source_task_idx",
        "task_brief",
        "task_type",
        "primary_category_key",
        "primary_category_label",
        "focus_areas_json",
        "focus_areas_pipe",
        "overlap_theme_keys_json",
        "overlap_theme_labels_pipe",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in enriched_tasks:
            writer.writerow(
                {
                    "task_idx": task["task_idx"],
                    "paper_id": task["paper_id"],
                    "paper_title": task["paper_title"],
                    "source_task_idx": task["source_task_idx"],
                    "task_brief": task["task_brief"],
                    "task_type": task.get("task_type", ""),
                    "primary_category_key": task["primary_category"],
                    "primary_category_label": PRIMARY_CATEGORY_META[task["primary_category"]]["label"],
                    "focus_areas_json": json.dumps(task["focus_areas"], ensure_ascii=False),
                    "focus_areas_pipe": " | ".join(task["focus_areas"]),
                    "overlap_theme_keys_json": json.dumps(task["overlap_theme_keys"], ensure_ascii=False),
                    "overlap_theme_labels_pipe": " | ".join(OVERLAP_THEMES[key]["label"] for key in task["overlap_theme_keys"]),
                }
            )


def main() -> None:
    args = parse_args()
    payload = load_payload(args.input_path)
    tasks = payload["tasks"]
    enriched_tasks, _, _, _ = analyze_tasks(tasks)
    content = build_markdown(args.input_path, tasks)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(content, encoding="utf-8")
    write_csv(args.csv_output_path, enriched_tasks)
    print(
        json.dumps(
            {
                "input_path": str(args.input_path.relative_to(PROJECT_ROOT)),
                "output_path": str(args.output_path.relative_to(PROJECT_ROOT)),
                "csv_output_path": str(args.csv_output_path.relative_to(PROJECT_ROOT)),
                "task_count": len(tasks),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
