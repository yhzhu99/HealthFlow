from __future__ import annotations

import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from prompt import (
    PROMPT_VARIANTS,
    SYSTEM_PROMPT,
    TASK_TYPE_DATA_EXTRACTION,
    TASK_TYPE_PREDICTIVE_MODELING,
    TASK_TYPE_REPORT_GENERATION,
    TASK_TYPE_VISUALIZATION,
    TASK_TYPES,
    build_prompt,
)


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = BENCHMARK_ROOT / "processed"
TRAIN_OUTPUT_PATH = PROCESSED_ROOT / "train.jsonl"
TEST_OUTPUT_PATH = PROCESSED_ROOT / "test.jsonl"
COMBINED_OUTPUT_PATH = PROCESSED_ROOT / "medagentboard.jsonl"
MANIFEST_PATH = PROCESSED_ROOT / "subset_manifest.json"

SEED = 42
PROMPT_SAMPLE_PATIENTS = 3

TOTAL_TASK_COUNTS = {
    TASK_TYPE_DATA_EXTRACTION: 16,
    TASK_TYPE_PREDICTIVE_MODELING: 13,
    TASK_TYPE_VISUALIZATION: 13,
    TASK_TYPE_REPORT_GENERATION: 13,
}
TRAIN_TASK_COUNTS = {
    TASK_TYPE_DATA_EXTRACTION: 2,
    TASK_TYPE_PREDICTIVE_MODELING: 1,
    TASK_TYPE_VISUALIZATION: 1,
    TASK_TYPE_REPORT_GENERATION: 1,
}

IDENTIFIER_COLUMNS = {"PatientID", "AdmissionID", "StayID", "RecordID"}
TIME_COLUMNS = {"RecordTime", "AdmissionTime", "DischargeTime"}
TARGET_COLUMNS = {"Outcome", "LOS", "Readmission"}
SPECIAL_FEATURE_COLUMNS = {"Age", "Sex"}


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    source_path: Path
    task_data_path: str
    sample_key: str
    analysis_key: str
    unit_label: str
    unit_label_plural: str


@dataclass(frozen=True)
class FeatureStats:
    p25: float
    median: float
    p75: float


@dataclass
class DatasetContext:
    config: DatasetConfig
    frame: pd.DataFrame
    continuous_features: list[str]
    auxiliary_features: list[str]
    binary_targets: list[str]
    regression_targets: list[str]
    feature_stats: dict[str, FeatureStats]


DATASET_CONFIGS = (
    DatasetConfig(
        key="tjh",
        display_name="TJH",
        source_path=PROCESSED_ROOT / "tjh" / "tjh_formatted_ehr.parquet",
        task_data_path="data/medagentboard/processed/tjh/tjh_formatted_ehr.parquet",
        sample_key="PatientID",
        analysis_key="PatientID",
        unit_label="patient",
        unit_label_plural="patients",
    ),
    DatasetConfig(
        key="mimic_iv_demo",
        display_name="MIMIC-IV",
        source_path=PROCESSED_ROOT / "mimic_iv_demo" / "mimic_iv_demo_formatted_ehr.parquet",
        task_data_path="data/medagentboard/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet",
        sample_key="PatientID",
        analysis_key="AdmissionID",
        unit_label="admission",
        unit_label_plural="admissions",
    ),
)


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_number(value: float) -> str:
    rounded = round(float(value), 2)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def compute_feature_stats(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, FeatureStats]:
    stats: dict[str, FeatureStats] = {}
    for column in feature_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            stats[column] = FeatureStats(p25=0.0, median=0.0, p75=0.0)
            continue
        stats[column] = FeatureStats(
            p25=float(numeric.quantile(0.25)),
            median=float(numeric.quantile(0.50)),
            p75=float(numeric.quantile(0.75)),
        )
    return stats


def build_context(config: DatasetConfig) -> DatasetContext:
    frame = load_frame(config.source_path)

    continuous_features: list[str] = []
    auxiliary_features: list[str] = []
    for column in frame.columns:
        if column in IDENTIFIER_COLUMNS or column in TIME_COLUMNS or column in TARGET_COLUMNS:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(numeric) < 10:
            continue
        if column in SPECIAL_FEATURE_COLUMNS:
            auxiliary_features.append(column)
            continue
        if numeric.nunique() >= 8:
            continuous_features.append(column)
        else:
            auxiliary_features.append(column)

    if not continuous_features:
        fallback = [
            column
            for column in frame.columns
            if column not in IDENTIFIER_COLUMNS and column not in TIME_COLUMNS and column not in TARGET_COLUMNS
        ]
        continuous_features = fallback[:]

    auxiliary_features = auxiliary_features + [
        column for column in SPECIAL_FEATURE_COLUMNS if column in frame.columns and column not in auxiliary_features
    ]
    feature_stats = compute_feature_stats(frame, continuous_features + auxiliary_features)

    binary_targets = [column for column in ["Outcome", "Readmission"] if column in frame.columns]
    regression_targets = [column for column in ["LOS"] if column in frame.columns]

    return DatasetContext(
        config=config,
        frame=frame,
        continuous_features=continuous_features,
        auxiliary_features=auxiliary_features,
        binary_targets=binary_targets,
        regression_targets=regression_targets,
        feature_stats=feature_stats,
    )


def select_distinct_features(
    context: DatasetContext,
    *,
    index: int,
    count: int,
    include_auxiliary: bool = False,
) -> list[str]:
    pool = context.continuous_features[:]
    if include_auxiliary:
        pool.extend([feature for feature in context.auxiliary_features if feature not in pool])
    if not pool:
        raise ValueError(f"no feature columns available for {context.config.display_name}")

    selected: list[str] = []
    cursor = index * 3 + 1
    while len(selected) < count:
        feature = pool[cursor % len(pool)]
        if feature not in selected:
            selected.append(feature)
        cursor += index + 2
    return selected


def select_binary_target(context: DatasetContext, index: int) -> str:
    if not context.binary_targets:
        raise ValueError(f"no binary target available for {context.config.display_name}")
    return context.binary_targets[index % len(context.binary_targets)]


def select_model_target(context: DatasetContext, index: int) -> tuple[str, str]:
    if context.binary_targets and index % 2 == 0:
        return select_binary_target(context, index), "classification"
    if context.regression_targets:
        return context.regression_targets[index % len(context.regression_targets)], "regression"
    return select_binary_target(context, index), "classification"


def get_threshold(context: DatasetContext, feature: str, *, upper: bool) -> str:
    stats = context.feature_stats[feature]
    return format_number(stats.p75 if upper else stats.p25)


def row_to_sample_text(row: pd.Series) -> str:
    pairs: list[str] = []
    for column, value in row.items():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        pairs.append(f"{column}:{text}")
    return "[" + ", ".join(pairs) + "]"


def build_data_examples(context: DatasetContext, rng: random.Random) -> str:
    sample_values = context.frame[context.config.sample_key].dropna().unique().tolist()
    if not sample_values:
        raise ValueError(f"no sample keys available for {context.config.display_name}")

    sample_count = min(PROMPT_SAMPLE_PATIENTS, len(sample_values))
    selected_values = rng.sample(sample_values, sample_count)

    sample_rows: list[str] = []
    for sample_value in selected_values:
        subset = context.frame.loc[context.frame[context.config.sample_key] == sample_value]
        for _, row in subset.iterrows():
            sample_rows.append(row_to_sample_text(row))
    return "[\n" + ",\n".join(sample_rows) + "\n]"


def build_data_wrangling_brief(context: DatasetContext, index: int) -> str:
    first_feature, last_feature, mean_feature = select_distinct_features(
        context,
        index=index,
        count=3,
        include_auxiliary=True,
    )
    return (
        f"Create a cleaned {context.config.unit_label}-level table keyed by `{context.config.analysis_key}`. "
        f"For each {context.config.unit_label}, keep the first non-null `{first_feature}`, the last non-null "
        f"`{last_feature}`, and the mean `{mean_feature}` across all available records. Also carry the final "
        f"available `Outcome` and `LOS` into the same table, save it as `entity_features.csv`, and report the "
        f"row count and column count in `answer.json`."
    )


def build_data_querying_brief(context: DatasetContext, index: int) -> str:
    feature = select_distinct_features(context, index=index, count=1)[0]
    target = select_binary_target(context, index)
    threshold = get_threshold(context, feature, upper=True)
    return (
        f"List the unique `{context.config.analysis_key}` values for {context.config.unit_label_plural} where "
        f"`{feature}` is greater than {threshold} in at least one record and `{target}` equals 1. Exclude rows "
        f"with missing `{feature}`. Save the sorted identifiers to `cohort.csv` and the final count to "
        f"`answer.json`."
    )


def build_data_statistics_brief(context: DatasetContext, index: int) -> str:
    mean_feature, max_feature = select_distinct_features(context, index=index, count=2)
    group_target = select_binary_target(context, index + 1)
    return (
        f"At the `{context.config.analysis_key}` level, compute the mean `{mean_feature}` and maximum "
        f"`{max_feature}` for each {context.config.unit_label}. Then summarize both derived features by "
        f"`{group_target}` using count, mean, standard deviation, and median. Save the summary table to "
        f"`summary.csv` and the main numeric findings to `answer.json`."
    )


def build_data_preprocessing_brief(context: DatasetContext, index: int) -> str:
    mean_feature, max_feature, min_feature = select_distinct_features(
        context,
        index=index + 1,
        count=3,
        include_auxiliary=True,
    )
    return (
        f"Build a model-ready {context.config.unit_label}-level feature table keyed by `{context.config.analysis_key}`. "
        f"For each {context.config.unit_label}, compute the mean `{mean_feature}`, max `{max_feature}`, min "
        f"`{min_feature}`, and a per-row missing-value count across these three features. Keep `Outcome` and `LOS` "
        f"in the output, save the table to `patient_features.csv`, and report the missing-value rate for each "
        f"derived feature in `answer.json`."
    )


def build_modeling_brief(context: DatasetContext, index: int) -> str:
    feature_one, feature_two, feature_three = select_distinct_features(
        context,
        index=index + 2,
        count=3,
        include_auxiliary=True,
    )
    target, task_kind = select_model_target(context, index)
    pattern = index % 5

    if task_kind == "classification":
        metrics_text = "AUROC, AUPRC, accuracy, and F1"
        prediction_note = f"predicted probabilities and labels for `{target}`"
    else:
        metrics_text = "MAE, RMSE, and R2"
        prediction_note = f"predicted numeric values for `{target}`"

    if pattern == 0:
        return (
            f"Build a deterministic {task_kind} pipeline to predict `{target}` at the "
            f"{context.config.unit_label}-level. Use the mean, max, and last available values of "
            f"`{feature_one}`, `{feature_two}`, and `{feature_three}` together with `Age` and `Sex` when "
            f"available. Save `metrics.json` with {metrics_text}, save `predictions.csv` with {prediction_note}, "
            f"and summarize the feature set in `answer.json`."
        )
    if pattern == 1:
        return (
            f"Build a deterministic {task_kind} model to predict `{target}` using first-record values, last-record "
            f"values, and first-to-last deltas for `{feature_one}`, `{feature_two}`, and `{feature_three}` for each "
            f"`{context.config.analysis_key}`. Save `metrics.json`, `predictions.csv`, and a short model summary in "
            f"`answer.json`."
        )
    if pattern == 2:
        return (
            f"Train two reproducible {task_kind} baselines, logistic/linear regression and random forest, to predict "
            f"`{target}` using aggregated `{feature_one}`, `{feature_two}`, `{feature_three}`, `Age`, and `Sex`. "
            f"Compare the two models in `metrics.json`, save the best model's outputs to `predictions.csv`, and write "
            f"the winning model name to `answer.json`."
        )
    if pattern == 3:
        return (
            f"Construct a missingness-aware {task_kind} model for `{target}`. Use the mean values of "
            f"`{feature_one}`, `{feature_two}`, and `{feature_three}` plus binary indicators showing whether each "
            f"feature is ever missing for a given `{context.config.analysis_key}`. Save evaluation results to "
            f"`metrics.json`, save predictions to `predictions.csv`, and report the strongest missingness signal in "
            f"`answer.json`."
        )
    return (
        f"Build a deterministic {task_kind} model for `{target}` that uses three feature blocks for each "
        f"`{context.config.analysis_key}`: the earliest available values of `{feature_one}` and `{feature_two}`, the "
        f"latest available value of `{feature_three}`, and the total record count per {context.config.unit_label}. "
        f"Save `metrics.json`, `predictions.csv`, and the final feature list in `answer.json`."
    )


def build_visualization_data_direct_brief(context: DatasetContext, index: int) -> str:
    feature_one, feature_two = select_distinct_features(
        context,
        index=index,
        count=2,
        include_auxiliary=True,
    )
    group_target = select_binary_target(context, index)
    pattern = index % 5

    if pattern == 0:
        return (
            f"Create a line plot showing the average temporal trend of `{feature_one}` over `RecordTime`, grouped by "
            f"`{group_target}`. Save the aggregated plotting table to `plot_summary.csv`, save the figure to "
            f"`trend_plot.png`, and summarize the largest between-group difference in `answer.json`."
        )
    if pattern == 1:
        return (
            f"Create box plots comparing the distribution of `{feature_one}` across `{group_target}` groups. Save the "
            f"summary table used for plotting to `plot_summary.csv`, save the figure to `boxplot.png`, and report the "
            f"group medians in `answer.json`."
        )
    if pattern == 2:
        return (
            f"Create a scatter plot of `Age` versus `LOS`, with points colored by `{group_target}` and point sizes "
            f"scaled by `{feature_one}`. Save the plotting table to `plot_summary.csv`, save the figure to "
            f"`age_los_scatter.png`, and report the per-group sample counts together with the median `{feature_one}` "
            f"in `answer.json`."
        )
    if pattern == 3:
        return (
            f"Create side-by-side histograms of `{feature_one}` for each `{group_target}` group. Save the underlying "
            f"plotting table to `plot_summary.csv`, save the figure to `histogram.png`, and report which group shows "
            f"the higher mean `{feature_one}` in `answer.json`."
        )
    return (
        f"Create a correlation heatmap for `{feature_one}`, `{feature_two}`, `Age`, and `LOS` after aggregating the "
        f"data to one row per `{context.config.analysis_key}`. Save the patient-level or admission-level table to "
        f"`plot_summary.csv`, save the figure to `correlation_heatmap.png`, and report the strongest absolute "
        f"correlation in `answer.json`."
    )


def build_visualization_model_analysis_brief(context: DatasetContext, index: int) -> str:
    feature_one, feature_two, feature_three = select_distinct_features(
        context,
        index=index + 3,
        count=3,
        include_auxiliary=True,
    )
    target, task_kind = select_model_target(context, index + 1)
    pattern = index % 4

    if task_kind == "classification":
        if pattern == 0:
            return (
                f"Train a reproducible classifier for `{target}` using aggregated `{feature_one}`, `{feature_two}`, "
                f"`{feature_three}`, `Age`, and `Sex`. Visualize the ROC curve and the top 10 feature importances. "
                f"Save the plotting data to `plot_summary.csv`, save the figures to `roc_curve.png` and "
                f"`feature_importance.png`, and write the AUROC to `answer.json`."
            )
        if pattern == 1:
            return (
                f"Train a reproducible classifier for `{target}` and create a precision-recall curve plus a "
                f"calibration plot using aggregated `{feature_one}`, `{feature_two}`, and `{feature_three}`. Save the "
                f"plotting data to `plot_summary.csv`, save the figures to `pr_curve.png` and `calibration.png`, and "
                f"report the average predicted probability by outcome group in `answer.json`."
            )
        if pattern == 2:
            return (
                f"Train a reproducible classifier for `{target}` and visualize a confusion matrix together with the "
                f"distribution of prediction scores, using aggregated `{feature_one}`, `{feature_two}`, and "
                f"`{feature_three}`. Save the plotting data to `plot_summary.csv`, save the figures to "
                f"`confusion_matrix.png` and `score_distribution.png`, and report the accuracy in `answer.json`."
            )
        return (
            f"Train two reproducible classifiers for `{target}` and visualize their ROC curves on the same chart "
            f"using aggregated `{feature_one}`, `{feature_two}`, and `{feature_three}`. Save the plotting data to "
            f"`plot_summary.csv`, save the comparison figure to `model_comparison.png`, and report the better model in "
            f"`answer.json`."
        )

    if pattern in {0, 1}:
        return (
            f"Train a reproducible regression model for `{target}` using aggregated `{feature_one}`, `{feature_two}`, "
            f"`{feature_three}`, `Age`, and `Sex`. Visualize predicted versus actual `{target}` values and the "
            f"residual distribution. Save the plotting data to `plot_summary.csv`, save the figures to "
            f"`predicted_vs_actual.png` and `residuals.png`, and report the RMSE in `answer.json`."
        )
    if pattern == 2:
        return (
            f"Train two reproducible regression baselines for `{target}` and compare them with a bar chart of MAE and "
            f"RMSE. Use aggregated `{feature_one}`, `{feature_two}`, and `{feature_three}`. Save the plotting data to "
            f"`plot_summary.csv`, save the comparison figure to `regression_comparison.png`, and report the better "
            f"model in `answer.json`."
        )
    return (
        f"Train a reproducible regression model for `{target}` and visualize the top feature importances together with "
        f"the residual trend over predicted values. Use aggregated `{feature_one}`, `{feature_two}`, and "
        f"`{feature_three}`. Save the plotting data to `plot_summary.csv`, save the figures to "
        f"`feature_importance.png` and `residual_trend.png`, and report the strongest predictor in `answer.json`."
    )


def build_report_brief(context: DatasetContext, index: int) -> str:
    feature_one, feature_two, feature_three = select_distinct_features(
        context,
        index=index + 4,
        count=3,
        include_auxiliary=True,
    )
    group_target = select_binary_target(context, index)
    pattern = index % 6

    if pattern == 0:
        return (
            f"Investigate whether early abnormalities in `{feature_one}`, `{feature_two}`, and `{feature_three}` are "
            f"associated with `{group_target}`. Aggregate the data to one row per `{context.config.analysis_key}`, "
            f"summarize the relevant descriptive statistics, and write a concise analytical report to `report.md`. "
            f"Record the key numerical takeaway in `answer.json`."
        )
    if pattern == 1:
        return (
            f"Analyze the relationship between `{feature_one}` variability and `LOS` at the "
            f"{context.config.unit_label}-level. Compare the results with `{feature_two}` and `{feature_three}`, write "
            f"a structured findings report to `report.md`, and store the strongest association in `answer.json`."
        )
    if pattern == 2:
        return (
            f"Prepare a subgroup report that compares `{feature_one}` and `{feature_two}` across age and sex strata, "
            f"then relates those subgroup differences to `{group_target}`. Save the narrative findings to `report.md` "
            f"and the highest-risk subgroup label to `answer.json`."
        )
    if pattern == 3:
        return (
            f"Create a data-quality report focused on completeness and missingness for `{feature_one}`, `{feature_two}`, "
            f"and `{feature_three}`. Summarize missing-value patterns by `{group_target}` in `report.md` and record "
            f"the feature with the highest missingness rate in `answer.json`."
        )
    if pattern == 4:
        return (
            f"Investigate longitudinal change by comparing the earliest and latest available values of `{feature_one}`, "
            f"`{feature_two}`, and `{feature_three}` for each `{context.config.analysis_key}`. Summarize how those "
            f"changes relate to `{group_target}` in `report.md` and write the most directionally consistent feature to "
            f"`answer.json`."
        )
    return (
        f"Write a short clinical-style analytics report on whether a combined severity profile built from "
        f"`{feature_one}`, `{feature_two}`, and `{feature_three}` is associated with `{group_target}` and `LOS`. "
        f"Save the report to `report.md` and record the dominant severity indicator in `answer.json`."
    )


MOCK_BUILDERS = {
    "data_wrangling": build_data_wrangling_brief,
    "data_querying": build_data_querying_brief,
    "data_statistics": build_data_statistics_brief,
    "data_preprocessing": build_data_preprocessing_brief,
    "modeling": build_modeling_brief,
    "plotting_or_visualization_DATA_DIRECT": build_visualization_data_direct_brief,
    "plotting_or_visualization_MODEL_ANALYSIS": build_visualization_model_analysis_brief,
    "report": build_report_brief,
}


def format_task_text(dataset_name: str, data_path: str, task_brief: str) -> str:
    return (
        "As an expert AI agent, your goal is to accurately perform the requested task.\n"
        "Please use the dataset information below to understand the format and structure of the data you will "
        "be working with.\n\n"
        "--- Dataset Information ---\n"
        f'Dataset Name: "{dataset_name}"\n'
        f"Data Path: `{data_path}`\n\n"
        "--- Specific Task to Perform ---\n"
        f'"{task_brief}"'
    )


def generate_question_mock(
    prompt: str,
    system_prompt: str | None = None,
    *,
    context: DatasetContext,
    variant_name: str,
    task_index: int,
) -> str:
    del prompt, system_prompt
    task_brief = MOCK_BUILDERS[variant_name](context, task_index)
    payload = {
        "task": format_task_text(
            context.config.display_name,
            context.config.task_data_path,
            task_brief,
        ),
        "task_brief": task_brief,
    }
    return "```json\n" + json.dumps(payload, indent=2, ensure_ascii=False) + "\n```"


def parse_generated_block(block: str) -> dict[str, str]:
    start = block.find("{")
    end = block.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("failed to locate JSON object in generated block")

    payload = json.loads(block[start:end])
    task = str(payload["task"]).strip()
    task_brief = str(payload["task_brief"]).strip()
    if not task or not task_brief:
        raise ValueError("generated payload is missing task or task_brief")
    return {
        "task": task,
        "task_brief": task_brief,
    }


def assign_qids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned_rows: list[dict[str, Any]] = []
    for qid, row in enumerate(rows, start=1):
        assigned_rows.append(
            {
                "qid": qid,
                "task": row["task"],
                "task_brief": row["task_brief"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "answer": row["answer"],
            }
        )
    return assigned_rows


def generate_rows_for_dataset(
    context: DatasetContext,
    *,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    seen_task_briefs: set[str] = set()

    dataset_row_index = 0
    for task_type in TASK_TYPES:
        total_count = TOTAL_TASK_COUNTS[task_type]
        train_count = TRAIN_TASK_COUNTS[task_type]
        variant_names = [variant["name"] for variant in PROMPT_VARIANTS[task_type]]

        for task_index in range(total_count):
            variant_name = variant_names[task_index % len(variant_names)]
            prompt = build_prompt(
                task_type=task_type,
                variant_name=variant_name,
                data_examples=build_data_examples(context, rng),
                exclusion_tasks=sorted(seen_task_briefs),
            )
            block = generate_question_mock(
                prompt,
                SYSTEM_PROMPT,
                context=context,
                variant_name=variant_name,
                task_index=task_index,
            )
            generated = parse_generated_block(block)
            if generated["task_brief"] in seen_task_briefs:
                raise ValueError(
                    f"duplicate task generated for {context.config.display_name}: {generated['task_brief']}"
                )

            dataset_row_index += 1
            split = "train" if task_index < train_count else "test"
            row = {
                "task": generated["task"],
                "task_brief": generated["task_brief"],
                "dataset": context.config.display_name,
                "task_type": task_type,
                "answer": "",
            }
            all_rows.append(row)
            if split == "train":
                train_rows.append(row)
            else:
                test_rows.append(row)
            manifest_rows.append(
                {
                    "dataset": context.config.display_name,
                    "dataset_row_index": dataset_row_index,
                    "split": split,
                    "task_type": task_type,
                    "prompt_variant": variant_name,
                    "task_brief": generated["task_brief"],
                }
            )
            seen_task_briefs.add(generated["task_brief"])

    return train_rows, test_rows, all_rows, manifest_rows


def build_manifest(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    per_dataset_counts: dict[str, dict[str, Any]] = {}
    split_rows = {
        "train": train_rows,
        "test": test_rows,
        "combined": combined_rows,
    }
    for dataset_name in sorted({row["dataset"] for row in combined_rows}):
        dataset_payload: dict[str, Any] = {}
        for split_name, rows in split_rows.items():
            dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
            dataset_payload[split_name] = {
                "count": len(dataset_rows),
                "task_type_counts": dict(Counter(row["task_type"] for row in dataset_rows)),
            }
        per_dataset_counts[dataset_name] = dataset_payload

    split_task_type_counts = {
        split_name: dict(Counter(row["task_type"] for row in rows))
        for split_name, rows in split_rows.items()
    }

    return {
        "dataset": "medagentboard",
        "seed": SEED,
        "source_files": {
            config.display_name: str(config.source_path.relative_to(BENCHMARK_ROOT))
            for config in DATASET_CONFIGS
        },
        "task_type_definitions": {
            TASK_TYPE_DATA_EXTRACTION: "merged from the original data wrangling, querying, statistics, and preprocessing prompts",
            TASK_TYPE_PREDICTIVE_MODELING: "predictive modeling tasks",
            TASK_TYPE_VISUALIZATION: "merged from the original direct-data and model-analysis visualization prompts",
            TASK_TYPE_REPORT_GENERATION: "report-generation tasks",
        },
        "total_task_counts_per_dataset": TOTAL_TASK_COUNTS,
        "train_task_counts_per_dataset": TRAIN_TASK_COUNTS,
        "split_counts": {
            "train": len(train_rows),
            "test": len(test_rows),
            "combined": len(combined_rows),
        },
        "split_task_type_counts": split_task_type_counts,
        "per_dataset": per_dataset_counts,
        "selected_rows": manifest_rows,
    }


def main() -> None:
    random.seed(SEED)
    dataset_rng = random.Random(SEED)

    all_train_rows: list[dict[str, Any]] = []
    all_test_rows: list[dict[str, Any]] = []
    all_combined_rows: list[dict[str, Any]] = []
    all_manifest_rows: list[dict[str, Any]] = []

    for config in DATASET_CONFIGS:
        context = build_context(config)
        dataset_seed = dataset_rng.randint(0, 10_000_000)
        train_rows, test_rows, combined_rows, manifest_rows = generate_rows_for_dataset(
            context,
            rng=random.Random(dataset_seed),
        )
        all_train_rows.extend(train_rows)
        all_test_rows.extend(test_rows)
        all_combined_rows.extend(combined_rows)
        all_manifest_rows.extend(manifest_rows)

    train_output_rows = assign_qids(all_train_rows)
    test_output_rows = assign_qids(all_test_rows)
    combined_output_rows = assign_qids(all_combined_rows)

    manifest = build_manifest(
        train_rows=all_train_rows,
        test_rows=all_test_rows,
        combined_rows=all_combined_rows,
        manifest_rows=all_manifest_rows,
    )

    write_jsonl(TRAIN_OUTPUT_PATH, train_output_rows)
    write_jsonl(TEST_OUTPUT_PATH, test_output_rows)
    write_jsonl(COMBINED_OUTPUT_PATH, combined_output_rows)
    write_json(MANIFEST_PATH, manifest)

    summary = {
        "train_rows": len(train_output_rows),
        "test_rows": len(test_output_rows),
        "combined_rows": len(combined_output_rows),
        "train_task_type_counts": dict(Counter(row["task_type"] for row in train_output_rows)),
        "test_task_type_counts": dict(Counter(row["task_type"] for row in test_output_rows)),
        "per_dataset_train_counts": {
            dataset_name: len([row for row in train_output_rows if row["dataset"] == dataset_name])
            for dataset_name in sorted({row["dataset"] for row in combined_output_rows})
        },
        "per_dataset_test_counts": {
            dataset_name: len([row for row in test_output_rows if row["dataset"] == dataset_name])
            for dataset_name in sorted({row["dataset"] for row in combined_output_rows})
        },
        "outputs": {
            "train": str(TRAIN_OUTPUT_PATH),
            "test": str(TEST_OUTPUT_PATH),
            "combined": str(COMBINED_OUTPUT_PATH),
            "manifest": str(MANIFEST_PATH),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
