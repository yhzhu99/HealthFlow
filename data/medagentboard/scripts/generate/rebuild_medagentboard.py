from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from ast import literal_eval
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
CONTRACT_VERSION = "medagentboard_llm_v2"
JUDGE_PROMPT_TYPE = {
    "data_extraction": "data_extraction",
    "predictive_modeling": "predictive_modeling",
    "visualization": "visualization",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SCORE_SCALE_MAX = 10.0
PASS_THRESHOLD = 7.0

TASK_TYPE_DATA_EXTRACTION = "data_extraction"
TASK_TYPE_PREDICTIVE_MODELING = "predictive_modeling"
TASK_TYPE_VISUALIZATION = "visualization"
TASK_TYPE_REPORT_GENERATION = "report_generation"

IDENTIFIER_COLUMNS = {"PatientID", "AdmissionID", "StayID", "RecordID"}
TIME_COLUMNS = {"RecordTime", "AdmissionTime", "DischargeTime"}
TARGET_COLUMNS = {"Outcome", "LOS", "Readmission"}
SPECIAL_FEATURE_COLUMNS = {"Age", "Sex"}

SCRIPT_PATH = Path(__file__).resolve()


def find_project_root() -> Path:
    for parent in SCRIPT_PATH.parents:
        if (parent / "pyproject.toml").exists() and (parent / "data").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")


PROJECT_ROOT = find_project_root()
DATASET_ROOT = PROJECT_ROOT / "data" / "medagentboard"
PROCESSED_ROOT = DATASET_ROOT / "processed"
CURRENT_DATASET_PATH = PROCESSED_ROOT / "medagentboard.jsonl"
TRAIN_PATH = PROCESSED_ROOT / "train.jsonl"
TEST_PATH = PROCESSED_ROOT / "test.jsonl"
MANIFEST_PATH = PROCESSED_ROOT / "subset_manifest.json"
REFERENCE_ROOT = PROCESSED_ROOT / "reference_answers"
BACKUP_ROOT = PROCESSED_ROOT / "_backups"
STAGING_ROOT = PROCESSED_ROOT / "_rebuild_staging"


@dataclass(frozen=True)
class DatasetConfig:
    display_name: str
    dataset_key: str
    frame_path: Path
    split_metadata_path: Path
    task_data_path: str
    entity_key: str
    split_key: str
    unit_label: str
    unit_label_plural: str


@dataclass(frozen=True)
class FeatureStats:
    p25: float
    median: float
    p75: float


@dataclass
class DatasetResources:
    config: DatasetConfig
    frame: pd.DataFrame
    continuous_features: list[str]
    auxiliary_features: list[str]
    binary_targets: list[str]
    regression_targets: list[str]
    feature_stats: dict[str, FeatureStats]
    split_key_to_split: dict[str, str]
    train_like_splits: set[str]
    test_splits: set[str]


@dataclass
class TaskSpec:
    dataset: str
    task_type: str
    task: str
    task_brief: str
    origin: str
    source_qid: int | None = None
    builder_kind: str | None = None
    builder_config: dict[str, Any] = field(default_factory=dict)
    protocol_notes: list[str] = field(default_factory=list)
    required_outputs: list[str] = field(default_factory=list)
    judge_prompt_type: str | None = None
    qid: int | None = None
    split: str | None = None
    split_qid: int | None = None

    @property
    def reference_answer(self) -> str:
        if self.split is None or self.split_qid is None:
            raise ValueError("split qid is not assigned yet")
        return f"reference_answers/{self.split}/{self.split_qid}/answer_manifest.json"

    def to_row(self, *, use_split_qid: bool = False) -> dict[str, Any]:
        qid = self.split_qid if use_split_qid else self.qid
        if qid is None:
            raise ValueError("requested qid is not assigned yet")
        return {
            "qid": qid,
            "task": self.task,
            "task_brief": self.task_brief,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "reference_answer": self.reference_answer,
        }


DATASET_CONFIGS = {
    "TJH": DatasetConfig(
        display_name="TJH",
        dataset_key="tjh",
        frame_path=PROCESSED_ROOT / "tjh" / "tjh_formatted_ehr.parquet",
        split_metadata_path=PROCESSED_ROOT / "tjh" / "split_metadata.json",
        task_data_path="data/medagentboard/processed/tjh/tjh_formatted_ehr.parquet",
        entity_key="PatientID",
        split_key="PatientID",
        unit_label="patient",
        unit_label_plural="patients",
    ),
    "MIMIC-IV": DatasetConfig(
        display_name="MIMIC-IV",
        dataset_key="mimic_iv_demo",
        frame_path=PROCESSED_ROOT / "mimic_iv_demo" / "mimic_iv_demo_formatted_ehr.parquet",
        split_metadata_path=PROCESSED_ROOT / "mimic_iv_demo" / "split_metadata.json",
        task_data_path="data/medagentboard/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet",
        entity_key="AdmissionID",
        split_key="RecordID",
        unit_label="admission",
        unit_label_plural="admissions",
    ),
}


REFRESHED_TASK_OVERRIDES: dict[int, dict[str, Any]] = {
    21: {
        "builder_kind": "refresh_q21",
        "task_brief": (
            "Build deterministic outcome prediction models for fixed time points "
            "`T = {1, 3, 5, 7, 10, 14}` days after `AdmissionTime`. For each `PatientID`, "
            "use only records observed up to `AdmissionTime + T` and summarize "
            "`Hypersensitive c-reactive protein`, `White blood cell count`, `hemoglobin`, and "
            "`Platelet count` into mean, last, and record-count features, together with `Age` and "
            "`Sex`. Train a logistic regression model on `train + val`, evaluate on `test` with "
            "AUROC, AUPRC, accuracy, and F1, and visualize the AUROC trajectory."
        ),
        "protocol_notes": [
            "Fixed time points are 1, 3, 5, 7, 10, and 14 days after admission.",
            "Feature set is fixed to Hypersensitive c-reactive protein, White blood cell count, hemoglobin, Platelet count, Age, and Sex.",
            "Training uses train + val splits; evaluation uses test split from split_metadata.json.",
        ],
    },
    54: {
        "builder_kind": "refresh_q54",
        "task_brief": (
            "Visualize the `Heart Rate` trajectory for the specific `RecordID` "
            "`10002428_23473524`. Filter the dataset to that record and plot `Heart Rate` "
            "against `RecordTime`."
        ),
        "protocol_notes": [
            "The entity is fixed to RecordID 10002428_23473524.",
            "Rows are ordered by RecordTime ascending before plotting.",
        ],
    },
    68: {
        "builder_kind": "refresh_q68",
        "task_brief": (
            "Evaluate how early-window information changes outcome predictability for MIMIC-IV "
            "admissions. For each `AdmissionID`, aggregate `Heart Rate`, `Mean blood pressure`, "
            "`Respiratory rate`, `Oxygen saturation`, `Temperature`, `Glucose`, and `pH` within "
            "the first `6h`, `12h`, `24h`, `48h`, and `full_stay` windows after the admission "
            "start, together with `Age` and `Sex`. Train logistic regression on `train + val`, "
            "evaluate on `test`, and visualize AUROC by window."
        ),
        "protocol_notes": [
            "Evaluation windows are fixed to 6h, 12h, 24h, 48h, and full_stay.",
            "Window features are mean, last, and record-count summaries.",
            "Training uses train + val splits; evaluation uses test split from split_metadata.json.",
        ],
    },
    71: {
        "builder_kind": "refresh_q71",
        "task_brief": (
            "Cluster admissions using `KMeans(n_clusters=3, random_state=42)` on first-24-hour "
            "clinical state summaries. For each `AdmissionID`, aggregate `Heart Rate`, `Mean blood "
            "pressure`, `Respiratory rate`, `Oxygen saturation`, `Temperature`, `Glucose`, and "
            "`pH` within the first 24 hours, combine them with `Age` and `Sex`, and visualize "
            "`LOS` by cluster."
        ),
        "protocol_notes": [
            "The clustering algorithm is fixed to KMeans with n_clusters=3 and random_state=42.",
            "Only first-24-hour observations are used to build cluster features.",
        ],
    },
    91: {
        "builder_kind": "refresh_q91",
        "task_brief": (
            "Develop a deterministic model to estimate `Glascow coma scale total` at 24 hours "
            "after admission. For each `AdmissionID`, use `Age`, `Sex`, and first-6-hour summaries "
            "of `Heart Rate`, `Mean blood pressure`, `Respiratory rate`, `Oxygen saturation`, "
            "`Glucose`, and `Temperature`. Define the target as the non-null `Glascow coma scale "
            "total` measurement closest to 24 hours after admission, breaking ties by choosing the "
            "later record. Train a random forest regressor on `train + val` and evaluate on `test`."
        ),
        "protocol_notes": [
            "Input window is fixed to the first 6 hours after admission.",
            "The target is the non-null GCS total value closest to 24 hours after admission; ties use the later record.",
            "Training uses train + val splits; evaluation uses test split from split_metadata.json.",
        ],
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dataset_uses_reference_answer_schema(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            return "reference_answer" in json.loads(line)
    return False


def resolve_source_artifacts() -> tuple[Path, Path]:
    if not dataset_uses_reference_answer_schema(CURRENT_DATASET_PATH):
        return CURRENT_DATASET_PATH, REFERENCE_ROOT

    if not BACKUP_ROOT.exists():
        raise FileNotFoundError("current dataset is rebuilt and no backup source is available")

    for backup_dir in sorted(BACKUP_ROOT.iterdir(), reverse=True):
        candidate_dataset = backup_dir / "medagentboard.jsonl"
        candidate_reference_root = backup_dir / "reference_answers"
        if candidate_dataset.exists() and candidate_reference_root.exists():
            if not dataset_uses_reference_answer_schema(candidate_dataset):
                return candidate_dataset, candidate_reference_root

    raise FileNotFoundError("failed to locate a pre-rebuild MedAgentBoard backup")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return sanitized or "artifact"


def unique_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def first_non_null(series: pd.Series) -> Any:
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    return cleaned.iloc[0]


def last_non_null(series: pd.Series) -> Any:
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    return cleaned.iloc[-1]


def range_or_nan(series: pd.Series) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return math.nan
    return float(cleaned.max() - cleaned.min())


def format_number(value: float) -> str:
    rounded = round(float(value), 2)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def maybe_python_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def format_task_text(
    dataset_name: str,
    data_path: str,
    task_brief: str,
    required_outputs: list[str],
    task_type: str,
) -> str:
    if task_type == TASK_TYPE_VISUALIZATION:
        output_header = "Required Final Image Files"
        output_intro = "Save only the following final image files with exactly these names:"
    else:
        output_header = "Required Output Files"
        output_intro = "Save the following final output files with exactly these names:"
    output_lines = "\n".join(f"- `{file_name}`" for file_name in required_outputs)
    return (
        "As an expert AI agent, your goal is to accurately perform the requested task.\n"
        "Please use the dataset information below to understand the format and structure of the "
        "data you will be working with.\n\n"
        "--- Dataset Information ---\n"
        f'Dataset Name: "{dataset_name}"\n'
        f"Data Path: `{data_path}`\n\n"
        "--- Specific Task to Perform ---\n"
        "Please **read the file from the Data Path** and complete the following task:\n"
        f"\"{task_brief}\"\n\n"
        f"--- {output_header} ---\n"
        f"{output_intro}\n"
        f"{output_lines}\n"
    )


def is_image_filename(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in IMAGE_SUFFIXES


def guess_media_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix == ".txt":
        return "text"
    if suffix == ".parquet":
        return "parquet"
    if is_image_filename(file_name):
        return "image"
    return "binary"


def build_visualization_output_names(builder_kind: str | None, builder_config: dict[str, Any]) -> list[str]:
    if builder_kind == "generated_visualization_data_direct":
        return {
            0: ["trend_plot.png"],
            1: ["boxplot.png"],
            2: ["age_los_scatter.png"],
            3: ["histogram.png"],
            4: ["correlation_heatmap.png"],
        }[builder_config["pattern"]]
    if builder_kind == "generated_visualization_model_analysis":
        if builder_config["task_kind"] == "classification":
            return {
                0: ["roc_curve.png", "feature_importance.png"],
                1: ["pr_curve.png", "calibration.png"],
                2: ["confusion_matrix.png", "score_distribution.png"],
                3: ["model_comparison.png"],
            }[builder_config["pattern"]]
        return {
            0: ["predicted_vs_actual.png", "residuals.png"],
            1: ["predicted_vs_actual.png", "residuals.png"],
            2: ["regression_comparison.png"],
            3: ["feature_importance.png", "residual_trend.png"],
        }[builder_config["pattern"]]
    if builder_kind == "refresh_q21":
        return ["performance_trajectory.png"]
    if builder_kind == "refresh_q54":
        return ["heart_rate_trend.png"]
    if builder_kind == "refresh_q68":
        return ["performance_by_window.png"]
    if builder_kind == "refresh_q71":
        return ["los_distribution_by_cluster.png"]
    raise ValueError(f"cannot resolve visualization outputs for builder {builder_kind!r}")


def load_source_visualization_outputs(source_dir: Path) -> list[str]:
    manifest_path = source_dir / "answer_manifest.json"
    ordered: list[str] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for path_text in manifest.get("primary_outputs", []):
            file_name = Path(path_text).name
            if is_image_filename(file_name) and (source_dir / file_name).exists() and file_name not in ordered:
                ordered.append(file_name)
    if ordered:
        return ordered
    return sorted(path.name for path in source_dir.iterdir() if path.is_file() and is_image_filename(path.name))


def required_outputs_for_task(
    *,
    task_type: str,
    builder_kind: str | None,
    builder_config: dict[str, Any],
    source_reference_root: Path | None = None,
    source_qid: int | None = None,
) -> list[str]:
    if task_type == TASK_TYPE_DATA_EXTRACTION:
        return ["result.csv"]
    if task_type == TASK_TYPE_PREDICTIVE_MODELING:
        return ["metrics.json", "predictions.csv", "answer.json"]
    if builder_kind is not None:
        return build_visualization_output_names(builder_kind, builder_config)
    if source_reference_root is None or source_qid is None:
        raise ValueError("source reference root is required to infer legacy visualization outputs")
    output_names = load_source_visualization_outputs(source_reference_root / str(source_qid))
    if not output_names:
        raise ValueError(f"source visualization task {source_qid} does not expose image outputs")
    return output_names


def compute_feature_stats(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, FeatureStats]:
    stats: dict[str, FeatureStats] = {}
    for column in feature_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            stats[column] = FeatureStats(0.0, 0.0, 0.0)
            continue
        stats[column] = FeatureStats(
            p25=float(numeric.quantile(0.25)),
            median=float(numeric.quantile(0.50)),
            p75=float(numeric.quantile(0.75)),
        )
    return stats


def load_resources() -> dict[str, DatasetResources]:
    resources_by_name: dict[str, DatasetResources] = {}
    for dataset_name, config in DATASET_CONFIGS.items():
        frame = pd.read_parquet(config.frame_path).copy()
        for column in TIME_COLUMNS:
            if column in frame.columns:
                frame[f"__{column}_dt"] = pd.to_datetime(frame[column], errors="coerce")

        continuous_features: list[str] = []
        auxiliary_features: list[str] = []
        for column in frame.columns:
            if column in IDENTIFIER_COLUMNS or column in TIME_COLUMNS or column in TARGET_COLUMNS:
                continue
            if column.startswith("__"):
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

        feature_stats = compute_feature_stats(frame, continuous_features + auxiliary_features)
        binary_targets = [column for column in ["Outcome", "Readmission"] if column in frame.columns]
        regression_targets = [column for column in ["LOS"] if column in frame.columns]

        split_metadata = json.loads(config.split_metadata_path.read_text(encoding="utf-8"))
        split_key_to_split: dict[str, str] = {}
        for split_name, keys in split_metadata["split_keys"].items():
            for key in keys:
                split_key_to_split[str(key)] = split_name

        resources_by_name[dataset_name] = DatasetResources(
            config=config,
            frame=frame,
            continuous_features=continuous_features,
            auxiliary_features=auxiliary_features,
            binary_targets=binary_targets,
            regression_targets=regression_targets,
            feature_stats=feature_stats,
            split_key_to_split=split_key_to_split,
            train_like_splits={"train", "val"},
            test_splits={"test"},
        )
    return resources_by_name


def entity_sort_columns(resources: DatasetResources) -> list[str]:
    sort_columns = []
    for column in ["RecordTime", "AdmissionTime", "DischargeTime"]:
        dt_column = f"__{column}_dt"
        if dt_column in resources.frame.columns:
            sort_columns.append(dt_column)
    if not sort_columns:
        sort_columns.append(resources.config.entity_key)
    return sort_columns


def sorted_groups(resources: DatasetResources) -> list[tuple[Any, pd.DataFrame]]:
    sorted_frame = resources.frame.sort_values(entity_sort_columns(resources))
    return list(sorted_frame.groupby(resources.config.entity_key, sort=False))


def select_distinct_features(
    resources: DatasetResources,
    *,
    index: int,
    count: int,
    include_auxiliary: bool = False,
) -> list[str]:
    pool = resources.continuous_features[:]
    if include_auxiliary:
        pool.extend(feature for feature in resources.auxiliary_features if feature not in pool)
    if not pool:
        raise ValueError(f"no feature columns available for {resources.config.display_name}")
    unique_pool = list(dict.fromkeys(pool))
    if len(unique_pool) < count:
        raise ValueError(
            f"{resources.config.display_name} requires {count} distinct features but only "
            f"{len(unique_pool)} are available"
        )

    selected: list[str] = []
    cursor = (index * 3 + 1) % len(unique_pool)
    step = (index + 2) % len(unique_pool) or 1
    visited: list[int] = []
    seen_positions: set[int] = set()
    while cursor not in seen_positions:
        visited.append(cursor)
        seen_positions.add(cursor)
        cursor = (cursor + step) % len(unique_pool)
    visited.extend(position for position in range(len(unique_pool)) if position not in seen_positions)

    for position in visited:
        feature = unique_pool[position]
        if feature not in selected:
            selected.append(feature)
        if len(selected) == count:
            break
    return selected


def select_binary_target(resources: DatasetResources, index: int) -> str:
    if not resources.binary_targets:
        raise ValueError(f"no binary targets for {resources.config.display_name}")
    return resources.binary_targets[index % len(resources.binary_targets)]


def select_model_target(resources: DatasetResources, index: int) -> tuple[str, str]:
    if resources.binary_targets and index % 2 == 0:
        return select_binary_target(resources, index), "classification"
    if resources.regression_targets:
        return resources.regression_targets[index % len(resources.regression_targets)], "regression"
    return select_binary_target(resources, index), "classification"


def get_threshold(resources: DatasetResources, feature: str, *, upper: bool) -> str:
    stats = resources.feature_stats[feature]
    return format_number(stats.p75 if upper else stats.p25)


def build_data_wrangling_spec(resources: DatasetResources, index: int) -> tuple[str, dict[str, Any]]:
    first_feature, last_feature, mean_feature = select_distinct_features(
        resources,
        index=index,
        count=3,
        include_auxiliary=False,
    )
    task_brief = (
        f"Create a cleaned {resources.config.unit_label}-level table keyed by "
        f"`{resources.config.entity_key}`. For each {resources.config.unit_label}, keep the first "
        f"non-null `{first_feature}`, the last non-null `{last_feature}`, and the mean "
        f"`{mean_feature}` across all available records. Also carry the final available `Outcome` "
        f"and `LOS` into the same table."
    )
    return task_brief, {
        "kind": "generated_data_wrangling",
        "first_feature": first_feature,
        "last_feature": last_feature,
        "mean_feature": mean_feature,
    }


def build_data_querying_spec(resources: DatasetResources, index: int) -> tuple[str, dict[str, Any]]:
    feature = select_distinct_features(resources, index=index, count=1)[0]
    target = select_binary_target(resources, index)
    threshold = get_threshold(resources, feature, upper=True)
    task_brief = (
        f"List the unique `{resources.config.entity_key}` values for {resources.config.unit_label_plural} "
        f"where `{feature}` is greater than {threshold} in at least one record and `{target}` equals "
        "1. Exclude rows with missing values for the feature."
    )
    return task_brief, {
        "kind": "generated_data_querying",
        "feature": feature,
        "target": target,
        "threshold": float(threshold),
    }


def build_data_statistics_spec(resources: DatasetResources, index: int) -> tuple[str, dict[str, Any]]:
    mean_feature, max_feature = select_distinct_features(resources, index=index, count=2)
    group_target = select_binary_target(resources, index + 1)
    task_brief = (
        f"At the `{resources.config.entity_key}` level, compute the mean `{mean_feature}` and "
        f"maximum `{max_feature}` for each {resources.config.unit_label}. Then summarize both "
        f"derived features by `{group_target}` using count, mean, standard deviation, and median."
    )
    return task_brief, {
        "kind": "generated_data_statistics",
        "mean_feature": mean_feature,
        "max_feature": max_feature,
        "group_target": group_target,
    }


def build_data_preprocessing_spec(resources: DatasetResources, index: int) -> tuple[str, dict[str, Any]]:
    mean_feature, max_feature, min_feature = select_distinct_features(
        resources,
        index=index + 1,
        count=3,
        include_auxiliary=True,
    )
    task_brief = (
        f"Build a model-ready {resources.config.unit_label}-level feature table keyed by "
        f"`{resources.config.entity_key}`. For each {resources.config.unit_label}, compute the mean "
        f"`{mean_feature}`, max `{max_feature}`, min `{min_feature}`, and a per-row missing-value "
        "count across these three features. Keep `Outcome` and `LOS` in the output."
    )
    return task_brief, {
        "kind": "generated_data_preprocessing",
        "mean_feature": mean_feature,
        "max_feature": max_feature,
        "min_feature": min_feature,
    }


def build_modeling_spec(resources: DatasetResources, index: int) -> tuple[str, dict[str, Any]]:
    feature_one, feature_two, feature_three = select_distinct_features(
        resources,
        index=index + 2,
        count=3,
        include_auxiliary=False,
    )
    target, task_kind = select_model_target(resources, index)
    pattern = index % 5

    if task_kind == "classification":
        metrics_text = "AUROC, AUPRC, accuracy, and F1"
        prediction_note = f"predicted probabilities and labels for `{target}`"
    else:
        metrics_text = "MAE, RMSE, and R2"
        prediction_note = f"predicted numeric values for `{target}`"

    if pattern == 0:
        task_brief = (
            f"Build a deterministic {task_kind} pipeline to predict `{target}` at the "
            f"{resources.config.unit_label}-level. Use the mean, max, and last available values "
            f"of `{feature_one}`, `{feature_two}`, and `{feature_three}` together with `Age` and "
            f"`Sex` when available."
        )
    elif pattern == 1:
        task_brief = (
            f"Build a deterministic {task_kind} model to predict `{target}` using first-record "
            f"values, last-record values, and first-to-last deltas for `{feature_one}`, "
            f"`{feature_two}`, and `{feature_three}` for each `{resources.config.entity_key}`."
        )
    elif pattern == 2:
        task_brief = (
            f"Train two reproducible {task_kind} baselines, logistic/linear regression and random "
            f"forest, to predict `{target}` using aggregated `{feature_one}`, `{feature_two}`, "
            f"`{feature_three}`, `Age`, and `Sex`. Compare the two models."
        )
    elif pattern == 3:
        task_brief = (
            f"Construct a missingness-aware {task_kind} model for `{target}`. Use the mean values "
            f"of `{feature_one}`, `{feature_two}`, and `{feature_three}` plus binary indicators "
            f"showing whether each feature is ever missing for a given "
            f"`{resources.config.entity_key}`."
        )
    else:
        task_brief = (
            f"Build a deterministic {task_kind} model for `{target}` that uses three feature "
            f"blocks for each `{resources.config.entity_key}`: the earliest available values of "
            f"`{feature_one}` and `{feature_two}`, the latest available value of `{feature_three}`, "
            f"and the total record count per {resources.config.unit_label}."
        )

    return task_brief, {
        "kind": "generated_modeling",
        "feature_one": feature_one,
        "feature_two": feature_two,
        "feature_three": feature_three,
        "target": target,
        "task_kind": task_kind,
        "pattern": pattern,
    }


def build_visualization_spec(resources: DatasetResources, index: int) -> tuple[str, dict[str, Any]]:
    if index % 2 == 0:
        feature_one, feature_two = select_distinct_features(
            resources,
            index=index,
            count=2,
            include_auxiliary=False,
        )
        group_target = select_binary_target(resources, index)
        pattern = (index // 2) % 5
        if pattern == 0:
            task_brief = (
                f"Create a line plot showing the average temporal trend of `{feature_one}` over "
                f"`RecordTime`, grouped by `{group_target}`."
            )
        elif pattern == 1:
            task_brief = (
                f"Create box plots comparing the distribution of `{feature_one}` across "
                f"`{group_target}` groups."
            )
        elif pattern == 2:
            task_brief = (
                f"Create a scatter plot of `Age` versus `LOS`, with points colored by "
                f"`{group_target}` and point sizes scaled by `{feature_one}`."
            )
        elif pattern == 3:
            task_brief = (
                f"Create side-by-side histograms of `{feature_one}` for each `{group_target}` "
                "group."
            )
        else:
            task_brief = (
                f"Create a correlation heatmap for `{feature_one}`, `{feature_two}`, `Age`, and "
                "`LOS` after aggregating the data to one row per "
                f"`{resources.config.entity_key}`."
            )
        return task_brief, {
            "kind": "generated_visualization_data_direct",
            "feature_one": feature_one,
            "feature_two": feature_two,
            "group_target": group_target,
            "pattern": pattern,
        }

    feature_one, feature_two, feature_three = select_distinct_features(
        resources,
        index=index + 3,
        count=3,
        include_auxiliary=False,
    )
    target, task_kind = select_model_target(resources, index + 1)
    pattern = (index // 2) % 4

    if task_kind == "classification":
        if pattern == 0:
            task_brief = (
                f"Train a reproducible classifier for `{target}` using aggregated `{feature_one}`, "
                f"`{feature_two}`, `{feature_three}`, `Age`, and `Sex`. Visualize the ROC curve "
                "and the top 10 feature importances."
            )
        elif pattern == 1:
            task_brief = (
                f"Train a reproducible classifier for `{target}` and create a precision-recall "
                f"curve plus a calibration plot using aggregated `{feature_one}`, `{feature_two}`, "
                f"and `{feature_three}`."
            )
        elif pattern == 2:
            task_brief = (
                f"Train a reproducible classifier for `{target}` and visualize a confusion matrix "
                f"together with the distribution of prediction scores using aggregated "
                f"`{feature_one}`, `{feature_two}`, and `{feature_three}`."
            )
        else:
            task_brief = (
                f"Train two reproducible classifiers for `{target}` and visualize their ROC "
                f"curves on the same chart using aggregated `{feature_one}`, `{feature_two}`, and "
                f"`{feature_three}`."
            )
    else:
        if pattern in {0, 1}:
            task_brief = (
                f"Train a reproducible regression model for `{target}` using aggregated "
                f"`{feature_one}`, `{feature_two}`, `{feature_three}`, `Age`, and `Sex`. "
                "Visualize predicted versus actual values and the residual distribution."
            )
        elif pattern == 2:
            task_brief = (
                f"Train two reproducible regression baselines for `{target}` and compare them with "
                f"a bar chart of MAE and RMSE using aggregated `{feature_one}`, `{feature_two}`, "
                f"and `{feature_three}`."
            )
        else:
            task_brief = (
                f"Train a reproducible regression model for `{target}` and visualize the top "
                f"feature importances together with the residual trend over predicted values using "
                f"aggregated `{feature_one}`, `{feature_two}`, and `{feature_three}`."
            )

    return task_brief, {
        "kind": "generated_visualization_model_analysis",
        "feature_one": feature_one,
        "feature_two": feature_two,
        "feature_three": feature_three,
        "target": target,
        "task_kind": task_kind,
        "pattern": pattern,
    }


def create_current_task_specs(dataset_path: Path, source_reference_root: Path) -> list[TaskSpec]:
    rows = load_jsonl(dataset_path)
    specs: list[TaskSpec] = []
    for row in rows:
        if row["task_type"] == TASK_TYPE_REPORT_GENERATION:
            continue
        qid = int(row["qid"])
        if row["task_type"] == TASK_TYPE_PREDICTIVE_MODELING and qid not in REFRESHED_TASK_OVERRIDES:
            continue
        task_brief = row["task_brief"]
        builder_kind = None
        builder_config: dict[str, Any] = {}
        protocol_notes: list[str] = []
        if qid in REFRESHED_TASK_OVERRIDES:
            override = REFRESHED_TASK_OVERRIDES[qid]
            task_brief = override["task_brief"]
            builder_kind = override["builder_kind"]
            protocol_notes = override["protocol_notes"]
            origin = "refreshed"
        else:
            origin = "current"
        required_outputs = required_outputs_for_task(
            task_type=row["task_type"],
            builder_kind=builder_kind,
            builder_config=builder_config,
            source_reference_root=source_reference_root,
            source_qid=qid,
        )
        task_text = format_task_text(
            row["dataset"],
            DATASET_CONFIGS[row["dataset"]].task_data_path,
            task_brief,
            required_outputs,
            row["task_type"],
        )
        specs.append(
            TaskSpec(
                dataset=row["dataset"],
                task_type=row["task_type"],
                task=task_text,
                task_brief=task_brief,
                origin=origin,
                source_qid=qid,
                builder_kind=builder_kind,
                builder_config=builder_config,
                protocol_notes=protocol_notes,
                required_outputs=required_outputs,
                judge_prompt_type=JUDGE_PROMPT_TYPE[row["task_type"]],
            )
        )
    return specs


def create_generated_task_specs(resources_by_name: dict[str, DatasetResources], existing_specs: list[TaskSpec]) -> list[TaskSpec]:
    counts_needed = {
        "TJH": {
            TASK_TYPE_DATA_EXTRACTION: 19,
            TASK_TYPE_PREDICTIVE_MODELING: 18,
            TASK_TYPE_VISUALIZATION: 18,
        },
        "MIMIC-IV": {
            TASK_TYPE_DATA_EXTRACTION: 19,
            TASK_TYPE_PREDICTIVE_MODELING: 18,
            TASK_TYPE_VISUALIZATION: 18,
        },
    }
    counts_current: dict[str, dict[str, int]] = {
        dataset: {
            TASK_TYPE_DATA_EXTRACTION: 0,
            TASK_TYPE_PREDICTIVE_MODELING: 0,
            TASK_TYPE_VISUALIZATION: 0,
        }
        for dataset in counts_needed
    }
    existing_briefs = {spec.task_brief for spec in existing_specs}
    for spec in existing_specs:
        counts_current[spec.dataset][spec.task_type] += 1

    generated_specs: list[TaskSpec] = []
    for dataset_name, target_counts in counts_needed.items():
        resources = resources_by_name[dataset_name]
        for task_type, target_count in target_counts.items():
            needed = target_count - counts_current[dataset_name][task_type]
            index = 0
            while needed > 0:
                if task_type == TASK_TYPE_DATA_EXTRACTION:
                    builder = [
                        build_data_wrangling_spec,
                        build_data_querying_spec,
                        build_data_statistics_spec,
                        build_data_preprocessing_spec,
                    ][index % 4]
                    task_brief, builder_config = builder(resources, index)
                elif task_type == TASK_TYPE_PREDICTIVE_MODELING:
                    task_brief, builder_config = build_modeling_spec(resources, index)
                else:
                    task_brief, builder_config = build_visualization_spec(resources, index)

                index += 1
                if task_brief in existing_briefs:
                    continue

                existing_briefs.add(task_brief)
                generated_specs.append(
                    TaskSpec(
                        dataset=dataset_name,
                        task_type=task_type,
                        task=format_task_text(
                            dataset_name,
                            resources.config.task_data_path,
                            task_brief,
                            required_outputs_for_task(
                                task_type=task_type,
                                builder_kind=builder_config["kind"],
                                builder_config=builder_config,
                            ),
                            task_type,
                        ),
                        task_brief=task_brief,
                        origin="generated",
                        builder_kind=builder_config["kind"],
                        builder_config=builder_config,
                        protocol_notes=["Generated filler task to satisfy the 3-type MedAgentBoard target counts."],
                        required_outputs=required_outputs_for_task(
                            task_type=task_type,
                            builder_kind=builder_config["kind"],
                            builder_config=builder_config,
                        ),
                        judge_prompt_type=JUDGE_PROMPT_TYPE[task_type],
                    )
                )
                needed -= 1
    return generated_specs


def assign_global_qids(specs: list[TaskSpec]) -> list[TaskSpec]:
    ordered: list[TaskSpec] = []
    for dataset_name in ["TJH", "MIMIC-IV"]:
        for task_type in [TASK_TYPE_DATA_EXTRACTION, TASK_TYPE_PREDICTIVE_MODELING, TASK_TYPE_VISUALIZATION]:
            bucket = [spec for spec in specs if spec.dataset == dataset_name and spec.task_type == task_type]
            ordered.extend(bucket)
    for qid, spec in enumerate(ordered, start=1):
        spec.qid = qid
    return ordered


def assign_splits(specs: list[TaskSpec]) -> tuple[list[TaskSpec], list[TaskSpec]]:
    rng = random.Random(SEED)
    train_targets = {
        ("TJH", TASK_TYPE_DATA_EXTRACTION): 2,
        ("TJH", TASK_TYPE_PREDICTIVE_MODELING): 1,
        ("TJH", TASK_TYPE_VISUALIZATION): 2,
        ("MIMIC-IV", TASK_TYPE_DATA_EXTRACTION): 2,
        ("MIMIC-IV", TASK_TYPE_PREDICTIVE_MODELING): 1,
        ("MIMIC-IV", TASK_TYPE_VISUALIZATION): 2,
    }
    train_qids: set[int] = set()
    for key, train_count in train_targets.items():
        bucket_qids = [spec.qid for spec in specs if (spec.dataset, spec.task_type) == key]
        selected = rng.sample(bucket_qids, train_count)
        train_qids.update(selected)

    train_specs: list[TaskSpec] = []
    test_specs: list[TaskSpec] = []
    for spec in specs:
        if spec.qid in train_qids:
            spec.split = "train"
            train_specs.append(spec)
        else:
            spec.split = "test"
            test_specs.append(spec)
    return train_specs, test_specs


def assign_split_qids(train_specs: list[TaskSpec], test_specs: list[TaskSpec]) -> None:
    for split_qid, spec in enumerate(train_specs, start=1):
        spec.split_qid = split_qid
    for split_qid, spec in enumerate(test_specs, start=1):
        spec.split_qid = split_qid


def aggregate_entity_rows(
    resources: DatasetResources,
    feature_ops: list[tuple[str, str, str]],
    *,
    include_targets: bool = True,
    include_demographics: bool = True,
    include_split: bool = False,
    frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source_frame = frame if frame is not None else resources.frame
    sort_columns = [column for column in entity_sort_columns(resources) if column in source_frame.columns]
    sorted_frame = source_frame.sort_values(sort_columns)
    rows: list[dict[str, Any]] = []
    for entity_id, group in sorted_frame.groupby(resources.config.entity_key, sort=False):
        row: dict[str, Any] = {resources.config.entity_key: entity_id}
        if include_split:
            split_value = first_non_null(group[resources.config.split_key])
            row["split_key"] = split_value
            row["split"] = resources.split_key_to_split.get(str(split_value))
        for output_name, source_feature, op_name in feature_ops:
            series = group[source_feature]
            if op_name == "first":
                row[output_name] = first_non_null(series)
            elif op_name == "last":
                row[output_name] = last_non_null(series)
            elif op_name == "mean":
                row[output_name] = pd.to_numeric(series, errors="coerce").mean()
            elif op_name == "max":
                row[output_name] = pd.to_numeric(series, errors="coerce").max()
            elif op_name == "min":
                row[output_name] = pd.to_numeric(series, errors="coerce").min()
            elif op_name == "std":
                row[output_name] = pd.to_numeric(series, errors="coerce").std()
            elif op_name == "range":
                row[output_name] = range_or_nan(series)
            elif op_name == "count_non_null":
                row[output_name] = int(series.notna().sum())
            elif op_name == "row_count":
                row[output_name] = int(len(group))
            else:
                raise ValueError(f"unsupported op: {op_name}")
        if include_targets:
            for target in ["Outcome", "LOS", "Readmission"]:
                if target in group.columns:
                    row[target] = last_non_null(group[target])
        if include_demographics:
            for column in ["Age", "Sex"]:
                if column in group.columns and column not in row:
                    row[column] = first_non_null(group[column])
        rows.append(row)
    aggregated = pd.DataFrame(rows)
    if not aggregated.empty:
        aggregated = aggregated.sort_values(resources.config.entity_key).reset_index(drop=True)
    return aggregated


def split_train_test(resources: DatasetResources, entity_frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = entity_frame.dropna(subset=["split", target]).copy()
    train_mask = usable["split"].isin(resources.train_like_splits)
    test_mask = usable["split"].isin(resources.test_splits)
    train_frame = usable.loc[train_mask].reset_index(drop=True)
    test_frame = usable.loc[test_mask].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError(f"empty train/test split for {resources.config.display_name} and target {target}")
    return train_frame, test_frame


def numeric_feature_columns(frame: pd.DataFrame, excluded: set[str]) -> list[str]:
    feature_columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            feature_columns.append(column)
    return feature_columns


def classification_pipeline(model_name: str) -> Pipeline:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, random_state=SEED)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=SEED,
        )
    else:
        raise ValueError(f"unsupported classifier: {model_name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def regression_pipeline(model_name: str) -> Pipeline:
    if model_name == "linear_regression":
        estimator = LinearRegression()
    elif model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=SEED,
        )
    else:
        raise ValueError(f"unsupported regressor: {model_name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def compute_classification_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if len(pd.Series(y_true).unique()) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, probabilities))
        metrics["auprc"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None
    return metrics


def compute_regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    rmse = math.sqrt(mean_squared_error(y_true, predictions))
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, predictions)),
    }


def extract_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_).reshape(-1)
        importances = np.abs(coefficients)
    else:
        importances = np.zeros(len(feature_names))
    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(feature_importance: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    top = feature_importance.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.5 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def structured_frame_from_object(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        if all(not isinstance(value, (dict, list)) for value in payload.values()):
            return pd.DataFrame([{str(key): serialize_cell(value) for key, value in payload.items()}])
        return pd.DataFrame(
            [
                {
                    "field": str(key),
                    "value": serialize_cell(value),
                }
                for key, value in payload.items()
            ]
        )
    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            return pd.DataFrame(payload)
        return pd.DataFrame({"value": [serialize_cell(item) for item in payload]})
    return pd.DataFrame([{"value": serialize_cell(payload)}])


def structured_frame_from_text(text: str) -> pd.DataFrame:
    stripped = text.strip()
    if not stripped:
        return pd.DataFrame(columns=["value"])

    for parser in (json.loads, literal_eval):
        try:
            return structured_frame_from_object(parser(stripped))
        except Exception:
            pass

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    key_values: dict[str, Any] = {}
    for line in lines:
        match = re.match(r"^(.*?)[=:]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", line)
        if not match:
            key_values = {}
            break
        key_values[match.group(1).strip()] = float(match.group(2))
    if key_values:
        return pd.DataFrame([key_values])

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", stripped):
        return pd.DataFrame([{"value": float(stripped)}])

    return pd.DataFrame([{"value": stripped}])


def load_structured_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return structured_frame_from_object(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".txt":
        return structured_frame_from_text(path.read_text(encoding="utf-8", errors="replace"))
    return pd.DataFrame([{"value": path.read_text(encoding="utf-8", errors="replace")}])


def bundle_structured_frames(items: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for file_name, frame in items:
        normalized = frame.copy()
        normalized.columns = [str(column) for column in normalized.columns]
        if normalized.empty and not normalized.columns.tolist():
            rows.append({"source_file": file_name, "row_index": 0, "field": "value", "value": ""})
            continue
        for row_index, record in normalized.reset_index(drop=True).iterrows():
            for field, value in record.items():
                rows.append(
                    {
                        "source_file": file_name,
                        "row_index": int(row_index),
                        "field": str(field),
                        "value": serialize_cell(value),
                    }
                )
    return pd.DataFrame(rows)


def normalize_data_extraction_output_dir(output_dir: Path) -> None:
    source_files = [
        path
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in {"answer_manifest.json", "result.csv"}
    ]
    if not source_files and (output_dir / "result.csv").exists():
        return

    if not source_files:
        raise ValueError(f"data_extraction task is missing source files in {output_dir}")

    structured_items = [(path.name, load_structured_frame(path)) for path in source_files]
    if len(structured_items) == 1:
        result_frame = structured_items[0][1]
    else:
        result_frame = bundle_structured_frames(structured_items)
    result_frame.to_csv(output_dir / "result.csv", index=False)


def normalize_predictive_modeling_output_dir(output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        for candidate_name in ["evaluation_metrics.json", "results.json"]:
            candidate = output_dir / candidate_name
            if candidate.exists():
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                write_json(metrics_path, payload)
                break

    predictions_path = output_dir / "predictions.csv"
    if not predictions_path.exists():
        for candidate_name in ["test_predictions.csv", "final_predictions.csv"]:
            candidate = output_dir / candidate_name
            if candidate.exists():
                shutil.copy2(candidate, predictions_path)
                break

    answer_path = output_dir / "answer.json"
    if not answer_path.exists():
        answer_payload: dict[str, Any] = {}
        for candidate_name in ["feature_definition.txt", "feature_list.txt", "features.txt"]:
            candidate = output_dir / candidate_name
            if candidate.exists():
                answer_payload["notes"] = candidate.read_text(encoding="utf-8", errors="replace").strip()
                break
        if not answer_payload:
            answer_payload["notes"] = "No additional summary provided."
        write_json(answer_path, answer_payload)


def prune_output_dir(output_dir: Path, required_outputs: list[str]) -> None:
    keep = set(required_outputs) | {"answer_manifest.json"}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name not in keep:
            path.unlink()


def normalize_output_dir_for_task(task: TaskSpec, output_dir: Path) -> None:
    if task.task_type == TASK_TYPE_DATA_EXTRACTION:
        normalize_data_extraction_output_dir(output_dir)
    elif task.task_type == TASK_TYPE_PREDICTIVE_MODELING:
        normalize_predictive_modeling_output_dir(output_dir)

    prune_output_dir(output_dir, task.required_outputs)
    missing = [file_name for file_name in task.required_outputs if not (output_dir / file_name).exists()]
    if missing:
        raise FileNotFoundError(f"task qid={task.qid} missing normalized outputs: {missing}")


def save_answer_manifest(task: TaskSpec, output_dir: Path) -> None:
    if task.split is None or task.split_qid is None:
        raise ValueError("split-local qid is required before writing manifests")
    files = [
        f"reference_answers/{task.split}/{task.split_qid}/{path.relative_to(output_dir).as_posix()}"
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "answer_manifest.json"
    ]
    manifest = {
        "qid": task.split_qid,
        "dataset": task.dataset,
        "task_type": task.task_type,
        "judge_prompt_type": task.judge_prompt_type or JUDGE_PROMPT_TYPE[task.task_type],
        "score_scale": {
            "min": 0,
            "max": SCORE_SCALE_MAX,
            "pass_threshold": PASS_THRESHOLD,
        },
        "required_outputs": [
            {
                "file_name": file_name,
                "reference_path": f"reference_answers/{task.split}/{task.split_qid}/{file_name}",
                "media_type": guess_media_type(file_name),
            }
            for file_name in task.required_outputs
        ],
        "all_outputs": files,
    }
    write_json(output_dir / "answer_manifest.json", manifest)


def build_generated_data_wrangling(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    config = task.builder_config
    entity_frame = aggregate_entity_rows(
        resources,
        [
            (config["first_feature"], config["first_feature"], "first"),
            (config["last_feature"], config["last_feature"], "last"),
            (config["mean_feature"], config["mean_feature"], "mean"),
        ],
    )
    output_csv = output_dir / "entity_features.csv"
    entity_frame.to_csv(output_csv, index=False)
    write_json(
        output_dir / "answer.json",
        {
            "row_count": int(len(entity_frame)),
            "column_count": int(len(entity_frame.columns)),
        },
    )
    return [
        str(output_csv.relative_to(PROCESSED_ROOT)),
        str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
    ]


def build_generated_data_querying(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    config = task.builder_config
    grouped = resources.frame.groupby(resources.config.entity_key)
    selected_ids: list[Any] = []
    for entity_id, group in grouped:
        feature_values = pd.to_numeric(group[config["feature"]], errors="coerce")
        target_values = pd.to_numeric(group[config["target"]], errors="coerce")
        if feature_values.dropna().empty:
            continue
        if (feature_values > config["threshold"]).any() and (target_values == 1).any():
            selected_ids.append(entity_id)
    cohort = pd.DataFrame({resources.config.entity_key: sorted(selected_ids)})
    output_csv = output_dir / "cohort.csv"
    cohort.to_csv(output_csv, index=False)
    write_json(
        output_dir / "answer.json",
        {
            "count": int(len(cohort)),
            "threshold": config["threshold"],
        },
    )
    return [
        str(output_csv.relative_to(PROCESSED_ROOT)),
        str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
    ]


def build_generated_data_statistics(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    config = task.builder_config
    entity_frame = aggregate_entity_rows(
        resources,
        [
            (f"{config['mean_feature']}__mean", config["mean_feature"], "mean"),
            (f"{config['max_feature']}__max", config["max_feature"], "max"),
        ],
    )
    summary = (
        entity_frame.groupby(config["group_target"])[
            [f"{config['mean_feature']}__mean", f"{config['max_feature']}__max"]
        ]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    summary.columns = [
        "__".join(str(part) for part in column if part != "")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    output_csv = output_dir / "summary.csv"
    summary.to_csv(output_csv, index=False)
    write_json(
        output_dir / "answer.json",
        {
            "rows": int(len(summary)),
            "group_target": config["group_target"],
        },
    )
    return [
        str(output_csv.relative_to(PROCESSED_ROOT)),
        str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
    ]


def build_generated_data_preprocessing(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    config = task.builder_config
    feature_ops = [
        (f"{config['mean_feature']}__mean", config["mean_feature"], "mean"),
        (f"{config['max_feature']}__max", config["max_feature"], "max"),
        (f"{config['min_feature']}__min", config["min_feature"], "min"),
    ]
    entity_frame = aggregate_entity_rows(resources, feature_ops)
    derived_features = [name for name, _, _ in feature_ops]
    entity_frame["missing_value_count"] = entity_frame[derived_features].isna().sum(axis=1)
    output_csv = output_dir / "patient_features.csv"
    entity_frame.to_csv(output_csv, index=False)
    write_json(
        output_dir / "answer.json",
        {
            "missing_rate": {
                column: float(entity_frame[column].isna().mean()) for column in derived_features
            }
        },
    )
    return [
        str(output_csv.relative_to(PROCESSED_ROOT)),
        str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
    ]


def build_model_dataset(resources: DatasetResources, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    pattern = config["pattern"]
    feature_one = config["feature_one"]
    feature_two = config["feature_two"]
    feature_three = config["feature_three"]
    feature_ops: list[tuple[str, str, str]]
    if pattern == 0:
        feature_ops = [
            (f"{feature_one}__mean", feature_one, "mean"),
            (f"{feature_one}__max", feature_one, "max"),
            (f"{feature_one}__last", feature_one, "last"),
            (f"{feature_two}__mean", feature_two, "mean"),
            (f"{feature_two}__max", feature_two, "max"),
            (f"{feature_two}__last", feature_two, "last"),
            (f"{feature_three}__mean", feature_three, "mean"),
            (f"{feature_three}__max", feature_three, "max"),
            (f"{feature_three}__last", feature_three, "last"),
        ]
    elif pattern == 1:
        feature_ops = [
            (f"{feature_one}__first", feature_one, "first"),
            (f"{feature_one}__last", feature_one, "last"),
            (f"{feature_two}__first", feature_two, "first"),
            (f"{feature_two}__last", feature_two, "last"),
            (f"{feature_three}__first", feature_three, "first"),
            (f"{feature_three}__last", feature_three, "last"),
        ]
    elif pattern == 2:
        feature_ops = [
            (f"{feature_one}__mean", feature_one, "mean"),
            (f"{feature_two}__mean", feature_two, "mean"),
            (f"{feature_three}__mean", feature_three, "mean"),
            (f"{feature_one}__std", feature_one, "std"),
            (f"{feature_two}__std", feature_two, "std"),
            (f"{feature_three}__std", feature_three, "std"),
        ]
    elif pattern == 3:
        feature_ops = [
            (f"{feature_one}__mean", feature_one, "mean"),
            (f"{feature_two}__mean", feature_two, "mean"),
            (f"{feature_three}__mean", feature_three, "mean"),
            (f"{feature_one}__missing_count", feature_one, "count_non_null"),
            (f"{feature_two}__missing_count", feature_two, "count_non_null"),
            (f"{feature_three}__missing_count", feature_three, "count_non_null"),
            ("record_count", feature_one, "row_count"),
        ]
    else:
        feature_ops = [
            (f"{feature_one}__first", feature_one, "first"),
            (f"{feature_two}__first", feature_two, "first"),
            (f"{feature_three}__last", feature_three, "last"),
            ("record_count", feature_one, "row_count"),
        ]
    entity_frame = aggregate_entity_rows(resources, feature_ops, include_split=True)
    if pattern == 1:
        entity_frame[f"{feature_one}__delta"] = entity_frame[f"{feature_one}__last"] - entity_frame[f"{feature_one}__first"]
        entity_frame[f"{feature_two}__delta"] = entity_frame[f"{feature_two}__last"] - entity_frame[f"{feature_two}__first"]
        entity_frame[f"{feature_three}__delta"] = entity_frame[f"{feature_three}__last"] - entity_frame[f"{feature_three}__first"]
    if pattern == 3:
        for feature in [feature_one, feature_two, feature_three]:
            count_column = f"{feature}__missing_count"
            entity_frame[count_column] = entity_frame["record_count"] - entity_frame[count_column]
            entity_frame[f"{feature}__ever_missing"] = entity_frame[count_column] > 0
    feature_columns = numeric_feature_columns(
        entity_frame,
        excluded={
            resources.config.entity_key,
            "split",
            "split_key",
            "Outcome",
            "LOS",
            "Readmission",
        },
    )
    return entity_frame, feature_columns


def build_generated_modeling(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    config = task.builder_config
    entity_frame, feature_columns = build_model_dataset(resources, config)
    target = config["target"]
    train_frame, test_frame = split_train_test(resources, entity_frame, target)
    x_train = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    x_test = test_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    primary_outputs: list[str] = []

    if config["task_kind"] == "classification":
        if config["pattern"] == 2:
            metrics_payload: dict[str, Any] = {}
            best_name = None
            best_probability = None
            best_model = None
            best_score = -math.inf
            for model_name in ["logistic_regression", "random_forest"]:
                pipeline = classification_pipeline(model_name)
                pipeline.fit(x_train, train_frame[target].astype(int))
                probability = pipeline.predict_proba(x_test)[:, 1]
                metrics = compute_classification_metrics(test_frame[target].astype(int), probability)
                metrics_payload[model_name] = metrics
                score = metrics["auroc"] if metrics["auroc"] is not None else -math.inf
                if score > best_score:
                    best_score = score
                    best_name = model_name
                    best_probability = probability
                    best_model = pipeline
            write_json(output_dir / "metrics.json", metrics_payload)
            predictions = test_frame[[resources.config.entity_key, target]].copy()
            predictions["predicted_probability"] = best_probability
            predictions["predicted_label"] = (best_probability >= 0.5).astype(int)
            predictions.to_csv(output_dir / "predictions.csv", index=False)
            write_json(output_dir / "answer.json", {"winning_model": best_name, "feature_columns": feature_columns})
            dump(best_model, output_dir / "best_model.joblib")
            primary_outputs.extend(
                [
                    str((output_dir / "metrics.json").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "predictions.csv").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
                ]
            )
        else:
            pipeline = classification_pipeline("logistic_regression")
            pipeline.fit(x_train, train_frame[target].astype(int))
            probability = pipeline.predict_proba(x_test)[:, 1]
            metrics = compute_classification_metrics(test_frame[target].astype(int), probability)
            write_json(output_dir / "metrics.json", metrics)
            predictions = test_frame[[resources.config.entity_key, target]].copy()
            predictions["predicted_probability"] = probability
            predictions["predicted_label"] = (probability >= 0.5).astype(int)
            predictions.to_csv(output_dir / "predictions.csv", index=False)
            answer_payload: dict[str, Any] = {"feature_columns": feature_columns}
            if config["pattern"] == 3:
                missing_cols = [column for column in feature_columns if column.endswith("__ever_missing")]
                feature_importance = extract_feature_importance(pipeline, feature_columns)
                strongest_missing = feature_importance.loc[
                    feature_importance["feature"].isin(missing_cols)
                ].head(1)
                answer_payload["strongest_missingness_signal"] = (
                    strongest_missing.iloc[0]["feature"] if not strongest_missing.empty else None
                )
            write_json(output_dir / "answer.json", answer_payload)
            dump(pipeline, output_dir / "trained_model.joblib")
            primary_outputs.extend(
                [
                    str((output_dir / "metrics.json").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "predictions.csv").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
                ]
            )
    else:
        if config["pattern"] == 2:
            metrics_payload: dict[str, Any] = {}
            best_name = None
            best_predictions = None
            best_model = None
            best_rmse = math.inf
            for model_name in ["linear_regression", "random_forest"]:
                pipeline = regression_pipeline(model_name)
                pipeline.fit(x_train, train_frame[target])
                predictions = pipeline.predict(x_test)
                metrics = compute_regression_metrics(test_frame[target], predictions)
                metrics_payload[model_name] = metrics
                if metrics["rmse"] < best_rmse:
                    best_rmse = metrics["rmse"]
                    best_name = model_name
                    best_predictions = predictions
                    best_model = pipeline
            write_json(output_dir / "metrics.json", metrics_payload)
            predictions_frame = test_frame[[resources.config.entity_key, target]].copy()
            predictions_frame["predicted_value"] = best_predictions
            predictions_frame.to_csv(output_dir / "predictions.csv", index=False)
            write_json(output_dir / "answer.json", {"winning_model": best_name, "feature_columns": feature_columns})
            dump(best_model, output_dir / "best_model.joblib")
            primary_outputs.extend(
                [
                    str((output_dir / "metrics.json").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "predictions.csv").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
                ]
            )
        else:
            pipeline = regression_pipeline("linear_regression")
            pipeline.fit(x_train, train_frame[target])
            predictions = pipeline.predict(x_test)
            metrics = compute_regression_metrics(test_frame[target], predictions)
            write_json(output_dir / "metrics.json", metrics)
            predictions_frame = test_frame[[resources.config.entity_key, target]].copy()
            predictions_frame["predicted_value"] = predictions
            predictions_frame.to_csv(output_dir / "predictions.csv", index=False)
            feature_importance = extract_feature_importance(pipeline, feature_columns)
            strongest_feature = feature_importance.head(1)["feature"].iloc[0] if not feature_importance.empty else None
            write_json(output_dir / "answer.json", {"strongest_predictor": strongest_feature, "feature_columns": feature_columns})
            dump(pipeline, output_dir / "trained_model.joblib")
            primary_outputs.extend(
                [
                    str((output_dir / "metrics.json").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "predictions.csv").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
                ]
            )
    return primary_outputs


def build_visualization_data_direct(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    config = task.builder_config
    feature_one = config["feature_one"]
    group_target = config["group_target"]
    pattern = config["pattern"]
    primary_outputs: list[str] = []

    if pattern == 0:
        plot_frame = resources.frame[[resources.config.entity_key, "RecordTime", "__RecordTime_dt", group_target, feature_one]].copy()
        if "__AdmissionTime_dt" in resources.frame.columns:
            plot_frame["day_from_admission"] = (
                resources.frame["__RecordTime_dt"] - resources.frame["__AdmissionTime_dt"]
            ).dt.days
        else:
            plot_frame["day_from_admission"] = np.arange(len(plot_frame))
        plot_frame = plot_frame.dropna(subset=[group_target, feature_one, "day_from_admission"])
        summary = (
            plot_frame.groupby(["day_from_admission", group_target])[feature_one]
            .mean()
            .reset_index()
            .rename(columns={feature_one: "mean_value"})
        )
        summary.to_csv(output_dir / "plot_summary.csv", index=False)
        plt.figure(figsize=(8, 5))
        for label, group in summary.groupby(group_target):
            plt.plot(group["day_from_admission"], group["mean_value"], marker="o", label=str(label))
        plt.xlabel("Days from admission")
        plt.ylabel(feature_one)
        plt.legend(title=group_target)
        plt.tight_layout()
        plt.savefig(output_dir / "trend_plot.png", dpi=200)
        plt.close()
        max_diff = None
        if summary[group_target].nunique() >= 2:
            pivot = summary.pivot(index="day_from_admission", columns=group_target, values="mean_value")
            if pivot.shape[1] >= 2:
                max_diff = float((pivot.iloc[:, 0] - pivot.iloc[:, 1]).abs().max())
        write_json(output_dir / "answer.json", {"largest_between_group_difference": max_diff})
        primary_outputs.extend(
            [
                str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                str((output_dir / "trend_plot.png").relative_to(PROCESSED_ROOT)),
                str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
            ]
        )
    elif pattern == 1:
        entity_frame = aggregate_entity_rows(resources, [(feature_one, feature_one, "mean")])
        plot_summary = entity_frame[unique_columns([resources.config.entity_key, feature_one, group_target])].dropna()
        plot_summary.to_csv(output_dir / "plot_summary.csv", index=False)
        groups = [group[feature_one].values for _, group in plot_summary.groupby(group_target)]
        labels = [str(label) for label, _ in plot_summary.groupby(group_target)]
        plt.figure(figsize=(7, 5))
        if groups:
            plt.boxplot(groups, tick_labels=labels)
        plt.ylabel(feature_one)
        plt.tight_layout()
        plt.savefig(output_dir / "boxplot.png", dpi=200)
        plt.close()
        medians = plot_summary.groupby(group_target)[feature_one].median().to_dict()
        write_json(output_dir / "answer.json", {"group_medians": {str(key): float(value) for key, value in medians.items()}})
        primary_outputs.extend(
            [
                str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                str((output_dir / "boxplot.png").relative_to(PROCESSED_ROOT)),
                str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
            ]
        )
    elif pattern == 2:
        entity_frame = aggregate_entity_rows(resources, [(feature_one, feature_one, "mean")])
        plot_summary = entity_frame[
            unique_columns([resources.config.entity_key, "Age", "LOS", group_target, feature_one])
        ].dropna()
        plot_summary.to_csv(output_dir / "plot_summary.csv", index=False)
        plt.figure(figsize=(7, 5))
        scatter = plt.scatter(
            plot_summary["Age"],
            plot_summary["LOS"],
            c=plot_summary[group_target],
            s=np.clip(plot_summary[feature_one].fillna(0), a_min=5, a_max=None) * 5,
            alpha=0.7,
        )
        plt.xlabel("Age")
        plt.ylabel("LOS")
        plt.colorbar(scatter, label=group_target)
        plt.tight_layout()
        plt.savefig(output_dir / "age_los_scatter.png", dpi=200)
        plt.close()
        counts = plot_summary.groupby(group_target)[resources.config.entity_key].count().to_dict()
        medians = plot_summary.groupby(group_target)[feature_one].median().to_dict()
        write_json(
            output_dir / "answer.json",
            {
                "group_counts": {str(key): int(value) for key, value in counts.items()},
                "feature_medians": {str(key): float(value) for key, value in medians.items()},
            },
        )
        primary_outputs.extend(
            [
                str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                str((output_dir / "age_los_scatter.png").relative_to(PROCESSED_ROOT)),
                str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
            ]
        )
    elif pattern == 3:
        entity_frame = aggregate_entity_rows(resources, [(feature_one, feature_one, "mean")])
        plot_summary = entity_frame[[resources.config.entity_key, feature_one, group_target]].dropna()
        plot_summary.to_csv(output_dir / "plot_summary.csv", index=False)
        unique_groups = list(plot_summary[group_target].dropna().unique())
        plt.figure(figsize=(7, 5))
        for group_value in unique_groups:
            plt.hist(
                plot_summary.loc[plot_summary[group_target] == group_value, feature_one],
                alpha=0.6,
                bins=15,
                label=str(group_value),
            )
        plt.xlabel(feature_one)
        plt.ylabel("Frequency")
        plt.legend(title=group_target)
        plt.tight_layout()
        plt.savefig(output_dir / "histogram.png", dpi=200)
        plt.close()
        means = plot_summary.groupby(group_target)[feature_one].mean().to_dict()
        higher_group = None
        if means:
            higher_group = max(means.items(), key=lambda item: item[1])[0]
        write_json(output_dir / "answer.json", {"higher_mean_group": maybe_python_scalar(higher_group)})
        primary_outputs.extend(
            [
                str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                str((output_dir / "histogram.png").relative_to(PROCESSED_ROOT)),
                str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
            ]
        )
    else:
        feature_two = config["feature_two"]
        entity_frame = aggregate_entity_rows(
            resources,
            [
                (feature_one, feature_one, "mean"),
                (feature_two, feature_two, "mean"),
            ],
        )
        plot_summary = entity_frame[
            unique_columns([resources.config.entity_key, feature_one, feature_two, "Age", "LOS"])
        ].dropna()
        plot_summary.to_csv(output_dir / "plot_summary.csv", index=False)
        correlation = plot_summary[[feature_one, feature_two, "Age", "LOS"]].corr()
        plt.figure(figsize=(6, 5))
        plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
        plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
        plt.yticks(range(len(correlation.index)), correlation.index)
        plt.colorbar(label="Correlation")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
        plt.close()
        strongest = (
            correlation.where(~np.eye(len(correlation), dtype=bool)).abs().stack().sort_values(ascending=False).head(1)
        )
        strongest_pair = strongest.index[0] if not strongest.empty else None
        write_json(
            output_dir / "answer.json",
            {
                "strongest_absolute_correlation": {
                    "pair": list(strongest_pair) if strongest_pair else None,
                    "value": float(strongest.iloc[0]) if not strongest.empty else None,
                }
            },
        )
        primary_outputs.extend(
            [
                str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                str((output_dir / "correlation_heatmap.png").relative_to(PROCESSED_ROOT)),
                str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
            ]
        )
    return primary_outputs


def build_visualization_model_analysis(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    config = task.builder_config
    model_frame, feature_columns = build_model_dataset(resources, config)
    target = config["target"]
    train_frame, test_frame = split_train_test(resources, model_frame, target)
    x_train = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    x_test = test_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    primary_outputs: list[str] = []

    if config["task_kind"] == "classification":
        y_train = train_frame[target].astype(int)
        y_test = test_frame[target].astype(int)
        if config["pattern"] == 3:
            roc_payload: list[dict[str, Any]] = []
            best_model_name = None
            best_score = -math.inf
            for model_name in ["logistic_regression", "random_forest"]:
                pipeline = classification_pipeline(model_name)
                pipeline.fit(x_train, y_train)
                probabilities = pipeline.predict_proba(x_test)[:, 1]
                metrics = compute_classification_metrics(y_test, probabilities)
                if metrics["auroc"] is not None and metrics["auroc"] > best_score:
                    best_score = metrics["auroc"]
                    best_model_name = model_name
                if len(pd.Series(y_test).unique()) > 1:
                    fpr, tpr, _ = roc_curve(y_test, probabilities)
                    roc_payload.append(
                        pd.DataFrame({"fpr": fpr, "tpr": tpr, "model": model_name}).to_dict("records")
                    )
            plot_summary = pd.DataFrame([item for sublist in roc_payload for item in sublist])
            plot_summary.to_csv(output_dir / "plot_summary.csv", index=False)
            plt.figure(figsize=(7, 5))
            for model_name, group in plot_summary.groupby("model"):
                plt.plot(group["fpr"], group["tpr"], label=model_name)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "model_comparison.png", dpi=200)
            plt.close()
            write_json(output_dir / "answer.json", {"better_model": best_model_name})
            primary_outputs.extend(
                [
                    str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "model_comparison.png").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
                ]
            )
            return primary_outputs

        pipeline = classification_pipeline("random_forest" if config["pattern"] == 0 else "logistic_regression")
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = compute_classification_metrics(y_test, probabilities)
        write_json(output_dir / "metrics.json", metrics)

        if config["pattern"] == 0:
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            plot_summary = pd.DataFrame({"fpr": fpr, "tpr": tpr})
            plot_summary.to_csv(output_dir / "plot_summary.csv", index=False)
            plt.figure(figsize=(7, 5))
            plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.tight_layout()
            plt.savefig(output_dir / "roc_curve.png", dpi=200)
            plt.close()
            feature_importance = extract_feature_importance(pipeline, feature_columns)
            feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
            save_feature_importance_plot(feature_importance, output_dir / "feature_importance.png")
            write_json(output_dir / "answer.json", {"auroc": metrics["auroc"]})
            primary_outputs.extend(
                [
                    str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "roc_curve.png").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "feature_importance.png").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
                ]
            )
        elif config["pattern"] == 1:
            precision_recall_frame = pd.DataFrame(
                calibration_curve(y_test, probabilities, n_bins=5, strategy="uniform")
            ).T
            pr_curve_precision, pr_curve_recall = [], []
            sorted_pairs = sorted(zip(probabilities, y_test), reverse=True)
            true_positives = 0
            for idx, (probability, outcome) in enumerate(sorted_pairs, start=1):
                if outcome == 1:
                    true_positives += 1
                precision = true_positives / idx
                recall = true_positives / max(int(y_test.sum()), 1)
                pr_curve_precision.append(precision)
                pr_curve_recall.append(recall)
            plot_summary = pd.DataFrame(
                {
                    "precision": pr_curve_precision,
                    "recall": pr_curve_recall,
                }
            )
            plot_summary.to_csv(output_dir / "plot_summary.csv", index=False)
            plt.figure(figsize=(7, 5))
            plt.plot(plot_summary["recall"], plot_summary["precision"])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.tight_layout()
            plt.savefig(output_dir / "pr_curve.png", dpi=200)
            plt.close()
            frac_pos, mean_pred = calibration_curve(y_test, probabilities, n_bins=5, strategy="uniform")
            calibration_df = pd.DataFrame({"mean_predicted_probability": mean_pred, "fraction_positive": frac_pos})
            calibration_df.to_csv(output_dir / "calibration_curve.csv", index=False)
            plt.figure(figsize=(7, 5))
            plt.plot(mean_pred, frac_pos, marker="o")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Fraction positive")
            plt.tight_layout()
            plt.savefig(output_dir / "calibration.png", dpi=200)
            plt.close()
            grouped_probability = (
                pd.DataFrame({"outcome": y_test, "predicted_probability": probabilities})
                .groupby("outcome")["predicted_probability"]
                .mean()
                .to_dict()
            )
            write_json(
                output_dir / "answer.json",
                {"average_predicted_probability_by_outcome": {str(key): float(value) for key, value in grouped_probability.items()}},
            )
            primary_outputs.extend(
                [
                    str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "pr_curve.png").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "calibration.png").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
                ]
            )
        else:
            plot_summary = pd.DataFrame({"actual": y_test, "predicted_probability": probabilities, "predicted_label": predictions})
            plot_summary.to_csv(output_dir / "plot_summary.csv", index=False)
            confusion = pd.crosstab(pd.Series(y_test, name="actual"), pd.Series(predictions, name="predicted"))
            confusion.to_csv(output_dir / "confusion_matrix.csv")
            plt.figure(figsize=(5, 4))
            plt.imshow(confusion, cmap="Blues")
            plt.xticks(range(len(confusion.columns)), confusion.columns)
            plt.yticks(range(len(confusion.index)), confusion.index)
            for row_idx in range(confusion.shape[0]):
                for col_idx in range(confusion.shape[1]):
                    plt.text(col_idx, row_idx, str(confusion.iloc[row_idx, col_idx]), ha="center", va="center")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
            plt.close()
            plt.figure(figsize=(7, 5))
            plt.hist(probabilities[y_test == 0], bins=10, alpha=0.6, label="Outcome 0")
            plt.hist(probabilities[y_test == 1], bins=10, alpha=0.6, label="Outcome 1")
            plt.xlabel("Predicted probability")
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "score_distribution.png", dpi=200)
            plt.close()
            write_json(output_dir / "answer.json", {"accuracy": metrics["accuracy"]})
            primary_outputs.extend(
                [
                    str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "confusion_matrix.png").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "score_distribution.png").relative_to(PROCESSED_ROOT)),
                    str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
                ]
            )
        return primary_outputs

    y_train = train_frame[target]
    y_test = test_frame[target]
    if config["pattern"] == 2:
        results = []
        metric_rows = []
        for model_name in ["linear_regression", "random_forest"]:
            pipeline = regression_pipeline(model_name)
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            metrics = compute_regression_metrics(y_test, predictions)
            metric_rows.append({"model": model_name, **metrics})
        plot_summary = pd.DataFrame(metric_rows)
        plot_summary.to_csv(output_dir / "plot_summary.csv", index=False)
        plt.figure(figsize=(7, 5))
        x_positions = np.arange(len(plot_summary))
        plt.bar(x_positions - 0.15, plot_summary["mae"], width=0.3, label="MAE")
        plt.bar(x_positions + 0.15, plot_summary["rmse"], width=0.3, label="RMSE")
        plt.xticks(x_positions, plot_summary["model"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "regression_comparison.png", dpi=200)
        plt.close()
        best_model = plot_summary.sort_values("rmse").iloc[0]["model"]
        write_json(output_dir / "answer.json", {"better_model": best_model})
        primary_outputs.extend(
            [
                str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                str((output_dir / "regression_comparison.png").relative_to(PROCESSED_ROOT)),
                str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
            ]
        )
        return primary_outputs

    pipeline = regression_pipeline("random_forest" if config["pattern"] == 3 else "linear_regression")
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = compute_regression_metrics(y_test, predictions)
    write_json(output_dir / "metrics.json", metrics)
    plot_summary = pd.DataFrame({"actual": y_test, "predicted": predictions})
    plot_summary.to_csv(output_dir / "plot_summary.csv", index=False)

    if config["pattern"] in {0, 1}:
        plt.figure(figsize=(7, 5))
        plt.scatter(plot_summary["actual"], plot_summary["predicted"], alpha=0.7)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(output_dir / "predicted_vs_actual.png", dpi=200)
        plt.close()
        residuals = plot_summary["actual"] - plot_summary["predicted"]
        plt.figure(figsize=(7, 5))
        plt.hist(residuals, bins=15)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_dir / "residuals.png", dpi=200)
        plt.close()
        write_json(output_dir / "answer.json", {"rmse": metrics["rmse"]})
        primary_outputs.extend(
            [
                str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                str((output_dir / "predicted_vs_actual.png").relative_to(PROCESSED_ROOT)),
                str((output_dir / "residuals.png").relative_to(PROCESSED_ROOT)),
                str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
            ]
        )
    else:
        feature_importance = extract_feature_importance(pipeline, feature_columns)
        feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
        save_feature_importance_plot(feature_importance, output_dir / "feature_importance.png")
        residuals = plot_summary["actual"] - plot_summary["predicted"]
        plt.figure(figsize=(7, 5))
        plt.scatter(plot_summary["predicted"], residuals, alpha=0.7)
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.tight_layout()
        plt.savefig(output_dir / "residual_trend.png", dpi=200)
        plt.close()
        strongest_predictor = feature_importance.head(1)["feature"].iloc[0] if not feature_importance.empty else None
        write_json(output_dir / "answer.json", {"strongest_predictor": strongest_predictor})
        primary_outputs.extend(
            [
                str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
                str((output_dir / "feature_importance.png").relative_to(PROCESSED_ROOT)),
                str((output_dir / "residual_trend.png").relative_to(PROCESSED_ROOT)),
                str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
            ]
        )
    return primary_outputs


def build_refresh_q21(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    feature_set = [
        "Hypersensitive c-reactive protein",
        "White blood cell count",
        "hemoglobin",
        "Platelet count",
    ]
    time_points = [1, 3, 5, 7, 10, 14]
    metrics_rows = []
    for time_point in time_points:
        limited = resources.frame.copy()
        time_window = (
            limited["__RecordTime_dt"] - limited["__AdmissionTime_dt"]
        ).dt.days
        limited = limited.loc[time_window.fillna(0) <= time_point].copy()
        if limited.empty:
            continue
        entity_frame = aggregate_entity_rows(
            resources,
            [
                (f"{feature}__mean", feature, "mean") for feature in feature_set
            ]
            + [
                (f"{feature}__last", feature, "last") for feature in feature_set
            ]
            + [
                ("record_count", feature_set[0], "count_non_null")
            ],
            include_targets=True,
            include_demographics=True,
            include_split=True,
            frame=limited,
        )
        feature_columns = numeric_feature_columns(
            entity_frame,
            excluded={resources.config.entity_key, "split", "split_key", "Outcome", "LOS", "Readmission"},
        )
        train_frame, test_frame = split_train_test(resources, entity_frame, "Outcome")
        pipeline = classification_pipeline("logistic_regression")
        x_train = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        x_test = test_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        pipeline.fit(x_train, train_frame["Outcome"].astype(int))
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics = compute_classification_metrics(test_frame["Outcome"].astype(int), probabilities)
        metrics_rows.append(
            {
                "time_window_days": time_point,
                **metrics,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
            }
        )
        dump(pipeline, output_dir / f"model_T_{time_point}.joblib")
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.plot(metrics_frame["time_window_days"], metrics_frame["auroc"], marker="o")
    plt.xlabel("Time window (days)")
    plt.ylabel("AUROC")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_trajectory.png", dpi=200)
    plt.close()
    best_row = metrics_frame.assign(auroc_rank=metrics_frame["auroc"].fillna(-1.0)).sort_values("auroc_rank", ascending=False).iloc[0].to_dict()
    write_json(
        output_dir / "answer.json",
        {
            "best_time_window_days": int(best_row["time_window_days"]),
            "best_auroc": maybe_python_scalar(best_row["auroc"]),
        },
    )
    return [
        str((output_dir / "evaluation_metrics.csv").relative_to(PROCESSED_ROOT)),
        str((output_dir / "performance_trajectory.png").relative_to(PROCESSED_ROOT)),
        str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
    ]


def build_refresh_q54(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    record_id = "10002428_23473524"
    plot_frame = resources.frame.loc[resources.frame["RecordID"] == record_id, ["RecordID", "RecordTime", "Heart Rate"]].copy()
    plot_frame = plot_frame.sort_values("RecordTime")
    plot_frame.to_csv(output_dir / "plot_summary.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(plot_frame["RecordTime"], plot_frame["Heart Rate"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Heart Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "heart_rate_trend.png", dpi=200)
    plt.close()
    write_json(output_dir / "answer.json", {"record_id": record_id, "row_count": int(len(plot_frame))})
    return [
        str((output_dir / "plot_summary.csv").relative_to(PROCESSED_ROOT)),
        str((output_dir / "heart_rate_trend.png").relative_to(PROCESSED_ROOT)),
        str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
    ]


def build_refresh_q68(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    feature_set = [
        "Heart Rate",
        "Mean blood pressure",
        "Respiratory rate",
        "Oxygen saturation",
        "Temperature",
        "Glucose",
        "pH",
    ]
    windows = [6, 12, 24, 48]
    rows: list[dict[str, Any]] = []
    for window in windows + ["full_stay"]:
        limited = resources.frame.copy()
        if window != "full_stay":
            hours = (
                limited["__RecordTime_dt"] - limited["__RecordTime_dt"].groupby(limited["AdmissionID"]).transform("min")
            ).dt.total_seconds() / 3600.0
            limited = limited.loc[hours.fillna(0) <= float(window)].copy()
        entity_frame = aggregate_entity_rows(
            resources,
            [(f"{feature}__mean", feature, "mean") for feature in feature_set]
            + [(f"{feature}__last", feature, "last") for feature in feature_set]
            + [("record_count", feature_set[0], "count_non_null")],
            include_targets=True,
            include_demographics=True,
            include_split=True,
            frame=limited,
        )
        feature_columns = numeric_feature_columns(
            entity_frame,
            excluded={resources.config.entity_key, "split", "split_key", "Outcome", "LOS", "Readmission"},
        )
        train_frame, test_frame = split_train_test(resources, entity_frame, "Outcome")
        pipeline = classification_pipeline("logistic_regression")
        x_train = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        x_test = test_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        pipeline.fit(x_train, train_frame["Outcome"].astype(int))
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics = compute_classification_metrics(test_frame["Outcome"].astype(int), probabilities)
        rows.append(
            {
                "window": window,
                **metrics,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
            }
        )
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(output_dir / "window_metrics.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.plot(metrics_frame["window"].astype(str), metrics_frame["auroc"], marker="o")
    plt.xlabel("Window")
    plt.ylabel("AUROC")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_by_window.png", dpi=200)
    plt.close()
    best_row = metrics_frame.assign(auroc_rank=metrics_frame["auroc"].fillna(-1.0)).sort_values("auroc_rank", ascending=False).iloc[0]
    write_json(
        output_dir / "answer.json",
        {"best_window": str(best_row["window"]), "best_auroc": maybe_python_scalar(best_row["auroc"])},
    )
    return [
        str((output_dir / "window_metrics.csv").relative_to(PROCESSED_ROOT)),
        str((output_dir / "performance_by_window.png").relative_to(PROCESSED_ROOT)),
        str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
    ]


def build_refresh_q71(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    feature_set = [
        "Heart Rate",
        "Mean blood pressure",
        "Respiratory rate",
        "Oxygen saturation",
        "Temperature",
        "Glucose",
        "pH",
    ]
    hours = (
        resources.frame["__RecordTime_dt"] - resources.frame["__RecordTime_dt"].groupby(resources.frame["AdmissionID"]).transform("min")
    ).dt.total_seconds() / 3600.0
    limited = resources.frame.loc[hours.fillna(0) <= 24].copy()
    entity_frame = aggregate_entity_rows(
        resources,
        [(f"{feature}__mean", feature, "mean") for feature in feature_set],
        include_targets=True,
        include_demographics=True,
        frame=limited,
    )
    feature_columns = numeric_feature_columns(
        entity_frame,
        excluded={resources.config.entity_key, "Outcome", "LOS", "Readmission"},
    )
    imputer = SimpleImputer(strategy="median")
    scaled_input = StandardScaler().fit_transform(imputer.fit_transform(entity_frame[feature_columns]))
    model = KMeans(n_clusters=3, random_state=SEED, n_init=20)
    clusters = model.fit_predict(scaled_input)
    entity_frame["cluster"] = clusters
    entity_frame.to_csv(output_dir / "clustered_admissions.csv", index=False)
    cluster_sizes = entity_frame["cluster"].value_counts().sort_index().to_dict()
    metrics = {
        "n_clusters": 3,
        "cluster_sizes": {str(key): int(value) for key, value in cluster_sizes.items()},
        "silhouette_score": float(silhouette_score(scaled_input, clusters)),
    }
    write_json(output_dir / "clustering_metrics.json", metrics)
    cluster_feature_means = entity_frame.groupby("cluster")[feature_columns + ["LOS", "Outcome"]].mean().reset_index()
    cluster_feature_means.to_csv(output_dir / "cluster_feature_means.csv", index=False)
    plt.figure(figsize=(7, 5))
    los_groups = [group["LOS"].dropna().values for _, group in entity_frame.groupby("cluster")]
    plt.boxplot(los_groups, tick_labels=[str(label) for label in sorted(entity_frame["cluster"].unique())])
    plt.xlabel("Cluster")
    plt.ylabel("LOS")
    plt.tight_layout()
    plt.savefig(output_dir / "los_distribution_by_cluster.png", dpi=200)
    plt.close()
    write_json(output_dir / "answer.json", {"n_clusters": 3, "silhouette_score": metrics["silhouette_score"]})
    return [
        str((output_dir / "clustered_admissions.csv").relative_to(PROCESSED_ROOT)),
        str((output_dir / "clustering_metrics.json").relative_to(PROCESSED_ROOT)),
        str((output_dir / "los_distribution_by_cluster.png").relative_to(PROCESSED_ROOT)),
    ]


def build_refresh_q91(task: TaskSpec, resources: DatasetResources, output_dir: Path) -> list[str]:
    feature_set = [
        "Heart Rate",
        "Mean blood pressure",
        "Respiratory rate",
        "Oxygen saturation",
        "Glucose",
        "Temperature",
    ]
    admission_start = resources.frame.groupby("AdmissionID")["__RecordTime_dt"].transform("min")
    hours_from_start = (resources.frame["__RecordTime_dt"] - admission_start).dt.total_seconds() / 3600.0
    feature_window = resources.frame.loc[hours_from_start.fillna(0) <= 6].copy()
    entity_frame = aggregate_entity_rows(
        resources,
        [(f"{feature}__mean", feature, "mean") for feature in feature_set]
        + [(f"{feature}__last", feature, "last") for feature in feature_set],
        include_targets=True,
        include_demographics=True,
        include_split=True,
        frame=feature_window,
    )
    target_rows = []
    for admission_id, group in resources.frame.groupby("AdmissionID"):
        group = group.dropna(subset=["Glascow coma scale total"]).copy()
        if group.empty:
            continue
        start_time = group["__RecordTime_dt"].min()
        group["distance_to_24h"] = (
            ((group["__RecordTime_dt"] - start_time).dt.total_seconds() / 3600.0) - 24.0
        ).abs()
        group["hours_from_start"] = (group["__RecordTime_dt"] - start_time).dt.total_seconds() / 3600.0
        selected = group.sort_values(["distance_to_24h", "hours_from_start"], ascending=[True, False]).iloc[0]
        target_rows.append({"AdmissionID": admission_id, "target_gcs_24h": selected["Glascow coma scale total"]})
    targets = pd.DataFrame(target_rows)
    entity_frame = entity_frame.merge(targets, on="AdmissionID", how="inner")
    feature_columns = numeric_feature_columns(
        entity_frame,
        excluded={"AdmissionID", "split", "split_key", "Outcome", "LOS", "Readmission", "target_gcs_24h"},
    )
    train_frame, test_frame = split_train_test(resources, entity_frame, "target_gcs_24h")
    pipeline = regression_pipeline("random_forest")
    x_train = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    x_test = test_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    pipeline.fit(x_train, train_frame["target_gcs_24h"])
    predictions = pipeline.predict(x_test)
    metrics = compute_regression_metrics(test_frame["target_gcs_24h"], predictions)
    write_json(output_dir / "metrics.json", metrics)
    predictions_frame = test_frame[["AdmissionID", "target_gcs_24h"]].copy()
    predictions_frame["predicted_gcs_24h"] = predictions
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)
    dump(pipeline, output_dir / "gcs_predictor.joblib")
    write_json(
        output_dir / "answer.json",
        {
            "rmse": metrics["rmse"],
            "feature_columns": feature_columns,
            "target_protocol": (
                "Target is the non-null Glascow coma scale total measurement closest to 24 hours "
                "after admission; ties use the later record."
            ),
        },
    )
    return [
        str((output_dir / "metrics.json").relative_to(PROCESSED_ROOT)),
        str((output_dir / "predictions.csv").relative_to(PROCESSED_ROOT)),
        str((output_dir / "answer.json").relative_to(PROCESSED_ROOT)),
    ]


BUILDERS = {
    "generated_data_wrangling": build_generated_data_wrangling,
    "generated_data_querying": build_generated_data_querying,
    "generated_data_statistics": build_generated_data_statistics,
    "generated_data_preprocessing": build_generated_data_preprocessing,
    "generated_modeling": build_generated_modeling,
    "generated_visualization_data_direct": build_visualization_data_direct,
    "generated_visualization_model_analysis": build_visualization_model_analysis,
    "refresh_q21": build_refresh_q21,
    "refresh_q54": build_refresh_q54,
    "refresh_q68": build_refresh_q68,
    "refresh_q71": build_refresh_q71,
    "refresh_q91": build_refresh_q91,
}


def build_reference_answers(
    specs: list[TaskSpec],
    resources_by_name: dict[str, DatasetResources],
    staging_reference_root: Path,
    source_reference_root: Path,
) -> list[dict[str, Any]]:
    qid_remap: list[dict[str, Any]] = []
    for spec in specs:
        if spec.split is None or spec.split_qid is None:
            raise ValueError("split-local qid is required before building reference answers")
        output_dir = staging_reference_root / spec.split / str(spec.split_qid)
        output_dir.mkdir(parents=True, exist_ok=True)
        if spec.origin == "current" and spec.source_qid is not None:
            source_dir = source_reference_root / str(spec.source_qid)
            shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)
        else:
            builder = BUILDERS[spec.builder_kind]
            builder(spec, resources_by_name[spec.dataset], output_dir)
        normalize_output_dir_for_task(spec, output_dir)
        save_answer_manifest(spec, output_dir)
        qid_remap.append(
            {
                "global_qid": spec.qid,
                "split": spec.split,
                "split_qid": spec.split_qid,
                "source_qid": spec.source_qid,
                "origin": spec.origin,
                "dataset": spec.dataset,
                "task_type": spec.task_type,
            }
        )
    return qid_remap


def make_backup() -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_dir = BACKUP_ROOT / f"medagentboard_3type_rebuild_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in [CURRENT_DATASET_PATH, TRAIN_PATH, TEST_PATH, MANIFEST_PATH]:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    if REFERENCE_ROOT.exists():
        shutil.copytree(REFERENCE_ROOT, backup_dir / "reference_answers")
    return backup_dir


def replace_processed_artifacts(staging_dir: Path) -> None:
    for path in [CURRENT_DATASET_PATH, TRAIN_PATH, TEST_PATH, MANIFEST_PATH]:
        if path.exists():
            path.unlink()
    if REFERENCE_ROOT.exists():
        shutil.rmtree(REFERENCE_ROOT)

    shutil.move(str(staging_dir / "medagentboard.jsonl"), str(CURRENT_DATASET_PATH))
    shutil.move(str(staging_dir / "train.jsonl"), str(TRAIN_PATH))
    shutil.move(str(staging_dir / "test.jsonl"), str(TEST_PATH))
    shutil.move(str(staging_dir / "subset_manifest.json"), str(MANIFEST_PATH))
    shutil.move(str(staging_dir / "reference_answers"), str(REFERENCE_ROOT))


def build_subset_manifest(
    specs: list[TaskSpec],
    train_specs: list[TaskSpec],
    test_specs: list[TaskSpec],
    qid_remap: list[dict[str, Any]],
    backup_dir: Path | None,
) -> dict[str, Any]:
    def count_rows(rows: list[TaskSpec]) -> dict[str, Any]:
        task_type_counts: dict[str, int] = {
            TASK_TYPE_DATA_EXTRACTION: 0,
            TASK_TYPE_PREDICTIVE_MODELING: 0,
            TASK_TYPE_VISUALIZATION: 0,
        }
        dataset_counts: dict[str, dict[str, int]] = {
            "TJH": {key: 0 for key in task_type_counts},
            "MIMIC-IV": {key: 0 for key in task_type_counts},
        }
        for row in rows:
            task_type_counts[row.task_type] += 1
            dataset_counts[row.dataset][row.task_type] += 1
        return {"task_type_counts": task_type_counts, "dataset_counts": dataset_counts}

    summary_all = count_rows(specs)
    summary_train = count_rows(train_specs)
    summary_test = count_rows(test_specs)

    return {
        "dataset": "medagentboard",
        "seed": SEED,
        "schema_version": "medagentboard_3type_v2",
        "task_types": [TASK_TYPE_DATA_EXTRACTION, TASK_TYPE_PREDICTIVE_MODELING, TASK_TYPE_VISUALIZATION],
        "global_qid_policy": "medagentboard.jsonl keeps global qids; train.jsonl and test.jsonl are renumbered within each split.",
        "reference_answer_field": "reference_answer",
        "reference_answer_contract": "Train/test rows point to reference_answers/<split>/<qid>/answer_manifest.json.",
        "source_files": {
            "TJH": DATASET_CONFIGS["TJH"].task_data_path,
            "MIMIC-IV": DATASET_CONFIGS["MIMIC-IV"].task_data_path,
        },
        "split_policy": {
            "train_count": len(train_specs),
            "test_count": len(test_specs),
            "per_dataset_train_mix": {
                "data_extraction": 2,
                "predictive_modeling": 1,
                "visualization": 2,
            },
        },
        "combined": {
            "count": len(specs),
            **summary_all,
        },
        "train": {
            "count": len(train_specs),
            **summary_train,
            "qids": [spec.split_qid for spec in train_specs],
        },
        "test": {
            "count": len(test_specs),
            **summary_test,
            "qids": [spec.split_qid for spec in test_specs],
        },
        "qid_remap": qid_remap,
        "refreshed_source_qids": [21, 54, 68, 71, 91],
        "deleted_report_source_qids": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
        "backup_dir": str(backup_dir.relative_to(PROCESSED_ROOT)) if backup_dir is not None else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def validate_rebuild(specs: list[TaskSpec], train_specs: list[TaskSpec], test_specs: list[TaskSpec]) -> None:
    if len(specs) != 110:
        raise ValueError(f"expected 110 total tasks, found {len(specs)}")
    if len(train_specs) != 10 or len(test_specs) != 100:
        raise ValueError(f"unexpected split counts: train={len(train_specs)} test={len(test_specs)}")
    if [spec.split_qid for spec in train_specs] != list(range(1, 11)):
        raise ValueError("train split qids are not numbered 1..10")
    if [spec.split_qid for spec in test_specs] != list(range(1, 101)):
        raise ValueError("test split qids are not numbered 1..100")
    for dataset_name in ["TJH", "MIMIC-IV"]:
        dataset_specs = [spec for spec in specs if spec.dataset == dataset_name]
        if len(dataset_specs) != 55:
            raise ValueError(f"{dataset_name} expected 55 tasks, found {len(dataset_specs)}")
        counts = {
            TASK_TYPE_DATA_EXTRACTION: sum(spec.task_type == TASK_TYPE_DATA_EXTRACTION for spec in dataset_specs),
            TASK_TYPE_PREDICTIVE_MODELING: sum(spec.task_type == TASK_TYPE_PREDICTIVE_MODELING for spec in dataset_specs),
            TASK_TYPE_VISUALIZATION: sum(spec.task_type == TASK_TYPE_VISUALIZATION for spec in dataset_specs),
        }
        expected = {
            TASK_TYPE_DATA_EXTRACTION: 19,
            TASK_TYPE_PREDICTIVE_MODELING: 18,
            TASK_TYPE_VISUALIZATION: 18,
        }
        if counts != expected:
            raise ValueError(f"{dataset_name} unexpected type counts: {counts}")
    if any(spec.task_type == TASK_TYPE_REPORT_GENERATION for spec in specs):
        raise ValueError("report_generation tasks remain after rebuild")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the MedAgentBoard benchmark corpus.")
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip copying the existing processed outputs before rebuilding.",
    )
    args = parser.parse_args()

    if not CURRENT_DATASET_PATH.exists():
        raise FileNotFoundError(f"missing dataset file: {CURRENT_DATASET_PATH}")
    if not REFERENCE_ROOT.exists():
        raise FileNotFoundError(f"missing reference answer directory: {REFERENCE_ROOT}")

    resources_by_name = load_resources()
    source_dataset_path, source_reference_root = resolve_source_artifacts()
    current_specs = create_current_task_specs(source_dataset_path, source_reference_root)
    generated_specs = create_generated_task_specs(resources_by_name, current_specs)
    combined_specs = assign_global_qids(current_specs + generated_specs)
    train_specs, test_specs = assign_splits(combined_specs)
    assign_split_qids(train_specs, test_specs)
    validate_rebuild(combined_specs, train_specs, test_specs)

    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    STAGING_ROOT.mkdir(parents=True, exist_ok=False)
    staging_reference_root = STAGING_ROOT / "reference_answers"
    staging_reference_root.mkdir(parents=True, exist_ok=False)

    backup_dir = None if args.skip_backup else make_backup()
    qid_remap = build_reference_answers(
        combined_specs,
        resources_by_name,
        staging_reference_root,
        source_reference_root,
    )

    write_jsonl(STAGING_ROOT / "medagentboard.jsonl", [spec.to_row() for spec in combined_specs])
    write_jsonl(STAGING_ROOT / "train.jsonl", [spec.to_row(use_split_qid=True) for spec in train_specs])
    write_jsonl(STAGING_ROOT / "test.jsonl", [spec.to_row(use_split_qid=True) for spec in test_specs])
    write_json(
        STAGING_ROOT / "subset_manifest.json",
        build_subset_manifest(combined_specs, train_specs, test_specs, qid_remap, backup_dir),
    )

    replace_processed_artifacts(STAGING_ROOT)
    shutil.rmtree(STAGING_ROOT, ignore_errors=True)

    print("MedAgentBoard rebuild completed.")
    print(f"Backup: {backup_dir if backup_dir is not None else 'skipped'}")
    print(f"Combined tasks: {len(combined_specs)}")
    print(f"Train tasks: {len(train_specs)}")
    print(f"Test tasks: {len(test_specs)}")


if __name__ == "__main__":
    main()
