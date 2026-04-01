from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = BENCHMARK_ROOT.parents[1]
PROCESSED_ROOT = BENCHMARK_ROOT / "processed"
TJH_PARQUET_PATH = PROCESSED_ROOT / "tjh" / "tjh_formatted_ehr.parquet"
TJH_SPLIT_METADATA_PATH = PROCESSED_ROOT / "tjh" / "split_metadata.json"
DEFAULT_BACKUP_PROCESSED_ROOT = REPO_ROOT.parent / "HealthFlow_backup" / "data" / "medagentboard" / "processed"
DEFAULT_SOURCE_EVAL_PATH = DEFAULT_BACKUP_PROCESSED_ROOT / "eval.jsonl"
DEFAULT_SOURCE_MANIFEST_PATH = DEFAULT_BACKUP_PROCESSED_ROOT / "task_manifest.jsonl"
DEFAULT_OUTPUT_EVAL_PATH = PROCESSED_ROOT / "eval.jsonl"
DEFAULT_OUTPUT_MANIFEST_PATH = PROCESSED_ROOT / "task_manifest.jsonl"

TJH_DATASET_PATH_LITERAL = "data/medagentboard/processed/tjh/tjh_formatted_ehr.parquet"
TJH_SPLIT_PATH_LITERAL = "data/medagentboard/processed/tjh/split_metadata.json"
SORT_COLUMNS = ["PatientID", "RecordTime", "AdmissionTime", "DischargeTime"]
QUERY_QIDS = set(range(1, 11))
SUMMARY_QIDS = set(range(11, 21))
PREPROCESS_QIDS = set(range(21, 31))
PLOT_QIDS = set(range(31, 41))
MODELING_QIDS = set(range(41, 46))
REPORT_QIDS = set(range(46, 51))

RE_QUERY = re.compile(r"Compute the patient-level `(?P<agg>[^`]+)` value of `(?P<feature>[^`]+)`")
RE_SUMMARY = re.compile(r"patient-level mean of `(?P<feature>[^`]+)`")
RE_FEATURE_TABLE = re.compile(r"aliases `(?P<features>[^`]+)` using the first, last, mean, and measurement count per patient")
RE_PLOT = re.compile(r"daily mean `(?P<feature>[^`]+)` over days since admission")
RE_MODEL = re.compile(r"predict `(?P<target>[^`]+)` from the patient-level TJH features `(?P<features>[^`]+)`")

ALIAS_MAP = {
    "age": "Age",
    "albumin": "albumin",
    "calcium": "calcium",
    "creatinine": "creatinine",
    "d_dimer": "D-D dimer",
    "ferritin": "ferritin",
    "hemoglobin": "hemoglobin",
    "hs_crp": "Hypersensitive c-reactive protein",
    "interleukin_6": "Interleukin 6",
    "ldh": "Lactate dehydrogenase",
    "los_last": "LOS__last",
    "outcome": "Outcome",
    "platelet_count": "Platelet count",
    "record_count": "record_count",
    "serum_chloride": "Serum chloride",
    "serum_potassium": "Serum potassium",
    "serum_sodium": "serum sodium",
    "sex": "Sex",
    "split": "split",
    "stay_days": "stay_days",
    "urea": "Urea",
    "white_blood_cell_count": "White blood cell count",
}


@dataclass
class TjhTaskContext:
    frame: pd.DataFrame
    schema_columns: set[str]
    patient_base: pd.DataFrame
    split_key_counts: dict[str, int]
    feature_cache: dict[str, pd.DataFrame]

    @classmethod
    def load(cls) -> TjhTaskContext:
        schema_columns = set(pq.read_schema(TJH_PARQUET_PATH).names)
        frame = pd.read_parquet(TJH_PARQUET_PATH)
        for column in ["RecordTime", "AdmissionTime", "DischargeTime"]:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame = frame.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)

        split_metadata = json.loads(TJH_SPLIT_METADATA_PATH.read_text(encoding="utf-8"))
        split_map: dict[int, str] = {}
        for split_name, patient_ids in split_metadata["split_keys"].items():
            for patient_id in patient_ids:
                split_map[int(patient_id)] = split_name

        grouped = frame.groupby("PatientID", sort=True, group_keys=False)
        patient_base = pd.DataFrame(index=sorted(frame["PatientID"].dropna().astype(int).unique()))
        patient_base.index.name = "PatientID"
        patient_base["Outcome"] = grouped["Outcome"].apply(first_non_null)
        patient_base["Age"] = grouped["Age"].apply(first_non_null)
        patient_base["Sex"] = grouped["Sex"].apply(first_non_null)
        patient_base["LOS__last"] = grouped["LOS"].apply(last_non_null)
        first_admission = grouped["AdmissionTime"].apply(first_non_null)
        last_discharge = grouped["DischargeTime"].apply(last_non_null)
        patient_base["stay_days"] = (last_discharge - first_admission).dt.days.clip(lower=0)
        patient_base["record_count"] = grouped.size().astype(int)
        patient_base["split"] = patient_base.index.map(split_map)

        patient_base["Outcome"] = patient_base["Outcome"].astype("Int64")
        patient_base["Sex"] = patient_base["Sex"].astype("Int64")

        return cls(
            frame=frame,
            schema_columns=schema_columns,
            patient_base=patient_base,
            split_key_counts={str(key): int(value) for key, value in split_metadata["split_key_counts"].items()},
            feature_cache={},
        )

    def map_feature(self, name: str) -> str:
        token = name.strip()
        if "__" in token:
            base_name, aggregate = token.rsplit("__", 1)
            mapped_base = self.map_feature(base_name)
            return f"{mapped_base}__{aggregate}"

        mapped = ALIAS_MAP.get(token, token)
        if mapped in {"record_count", "split", "LOS__last", "stay_days"}:
            return mapped
        if mapped not in self.schema_columns:
            raise ValueError(f"unknown TJH feature alias: {token}")
        return mapped

    def feature_aggregates(self, feature_name: str) -> pd.DataFrame:
        exact_name = self.map_feature(feature_name)
        if exact_name in {"record_count", "split", "LOS__last", "stay_days"}:
            raise ValueError(f"{exact_name} is not a longitudinal feature column")
        if exact_name in self.feature_cache:
            return self.feature_cache[exact_name]

        grouped = self.frame.groupby("PatientID", sort=True)[exact_name]
        aggregate_frame = pd.DataFrame(index=self.patient_base.index)
        aggregate_frame[f"{exact_name}__count"] = grouped.count().astype(int)
        aggregate_frame[f"{exact_name}__first"] = grouped.apply(first_non_null)
        aggregate_frame[f"{exact_name}__last"] = grouped.apply(last_non_null)
        aggregate_frame[f"{exact_name}__mean"] = grouped.mean()
        aggregate_frame[f"{exact_name}__min"] = grouped.min()
        aggregate_frame[f"{exact_name}__max"] = grouped.max()
        self.feature_cache[exact_name] = aggregate_frame
        return aggregate_frame

    def patient_feature_table(self, feature_names: list[str]) -> pd.DataFrame:
        feature_columns: list[str] = []
        table = self.patient_base.copy()
        for feature_name in feature_names:
            exact_name = self.map_feature(feature_name)
            aggregate_frame = self.feature_aggregates(exact_name)
            for aggregate in ["count", "first", "last", "mean"]:
                column_name = f"{exact_name}__{aggregate}"
                table[column_name] = aggregate_frame[column_name]
                feature_columns.append(column_name)

        ordered_columns = ["split", "Outcome", "Age", "Sex", "LOS__last", "record_count", *feature_columns]
        table = table[ordered_columns].reset_index()
        return table

    def modeling_table(self, feature_columns: list[str], target_column: str) -> pd.DataFrame:
        table = self.patient_base.copy()
        for feature_column in feature_columns:
            if feature_column in table.columns:
                continue
            if "__" not in feature_column:
                raise ValueError(f"unsupported patient-level feature: {feature_column}")
            base_name, aggregate = feature_column.rsplit("__", 1)
            if aggregate not in {"mean", "last", "max", "min", "first", "count"}:
                raise ValueError(f"unsupported aggregate: {aggregate}")
            aggregate_frame = self.feature_aggregates(base_name)
            source_column = f"{base_name}__{aggregate}"
            table[source_column] = aggregate_frame[source_column]

        if target_column not in table.columns:
            raise ValueError(f"unsupported target column: {target_column}")
        return table.reset_index()


def first_non_null(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return math.nan
    return values.iloc[0]


def last_non_null(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return math.nan
    return values.iloc[-1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_feature_list(value: str) -> list[str]:
    return [feature.strip() for feature in value.split(",") if feature.strip()]


def quoted_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def task_preamble(include_split_metadata: bool = False) -> list[str]:
    lines = [
        f"Use the local formatted EHR table at `{TJH_DATASET_PATH_LITERAL}`. Treat `PatientID` as the patient key and `RecordTime` as the longitudinal timestamp. Use deterministic computations only and keep the output machine-checkable.",
    ]
    if include_split_metadata:
        lines.append(
            f"Use patient-level split assignments from `{TJH_SPLIT_PATH_LITERAL}`. The only valid split labels are `train`, `val`, and `test`."
        )
    return lines


def build_query_task(feature: str, aggregate: str, threshold: float, comparator: str) -> str:
    derived_column = f"{feature}__{aggregate}"
    lines = [
        *task_preamble(),
        "",
        "Task:",
        f"Build a patient cohort from the formatted TJH table. Compute the patient-level `{aggregate}` of `{feature}` using non-null values only, then select patients whose value is `{comparator}` {threshold}. Exclude patients with no non-null `{feature}` measurements.",
        f"Write `cohort.csv` with columns `PatientID`, `Outcome`, `{derived_column}`, sorted by `PatientID` ascending.",
        "Write `answer.json` with keys `threshold`, `comparator`, and `patient_count`.",
        "",
        "Required deliverables:",
        "- `cohort.csv`",
        "- `answer.json`",
        "Return a concise textual summary, but the files are the canonical graded outputs.",
    ]
    return "\n".join(lines)


def build_summary_task(feature: str) -> str:
    lines = [
        *task_preamble(),
        "",
        "Task:",
        f"Create an `Outcome`- and `Sex`-stratified summary table for the patient-level mean of `{feature}`.",
        f"First compute each patient's mean `{feature}` across non-null longitudinal rows. Exclude patients with no non-null `{feature}` measurements. Then group by `Outcome` and `Sex`.",
        "Write `summary.csv` with columns `Outcome`, `Sex`, `patient_count`, `mean_feature`, `median_feature`, `mean_Age`, sorted by `Outcome`, `Sex` ascending.",
        "Write `answer.json` with keys `rows` and `feature`.",
        "",
        "Required deliverables:",
        "- `summary.csv`",
        "- `answer.json`",
        "Return a concise textual summary, but the files are the canonical graded outputs.",
    ]
    return "\n".join(lines)


def build_preprocess_task(feature_names: list[str], output_columns: list[str]) -> str:
    lines = [
        *task_preamble(include_split_metadata=True),
        "",
        "Task:",
        f"Construct a patient-level feature table for {quoted_list(feature_names)}.",
        "Within each patient, sort rows by `RecordTime`, `AdmissionTime`, and `DischargeTime` ascending. Use the first non-null patient-level values of `Outcome`, `Age`, and `Sex`; use the last non-null `LOS` as `LOS__last`; set `record_count` to the total number of longitudinal rows; and populate `split` from the split metadata file.",
        "For each requested longitudinal feature, compute `__count`, `__first`, `__last`, and `__mean` from non-null values only. If a feature has no non-null values for a patient, set `__count` to `0` and leave the other aggregates empty.",
        f"Write `patient_features.csv` with these columns in this exact order: {quoted_list(output_columns)}, sorted by `PatientID` ascending.",
        "Write `answer.json` with keys `rows` and `columns`.",
        "",
        "Required deliverables:",
        "- `patient_features.csv`",
        "- `answer.json`",
        "Return a concise textual summary, but the files are the canonical graded outputs.",
    ]
    return "\n".join(lines)


def build_plot_task(feature: str) -> str:
    lines = [
        *task_preamble(),
        "",
        "Task:",
        f"Create an `Outcome`-stratified temporal line plot for daily mean `{feature}` over days since admission.",
        f"Define `day` as the integer number of days between `RecordTime` and `AdmissionTime`. Keep only rows with `day >= 0` and non-null `{feature}` values. For each `Outcome` and `day` pair, compute `daily_mean` as the mean of `{feature}` and `patient_count` as the number of unique patients contributing at least one non-null measurement.",
        "Write `plot_summary.csv` with columns `day`, `Outcome`, `daily_mean`, `patient_count`, sorted by `Outcome`, `day` ascending.",
        "Save the chart to `plot.png`.",
        "Write `answer.json` with keys `feature` and `rows`.",
        "",
        "Required deliverables:",
        "- `plot_summary.csv`",
        "- `plot.png`",
        "- `answer.json`",
        "`plot_summary.csv` and `answer.json` are the canonical graded outputs. Return a concise textual summary, but the files are the canonical graded outputs.",
    ]
    return "\n".join(lines)


def build_logistic_model_task(feature_columns: list[str]) -> str:
    lines = [
        *task_preamble(include_split_metadata=True),
        "",
        "Task:",
        f"Construct patient-level modeling features {quoted_list(feature_columns)} and fit a deterministic logistic regression model to predict `Outcome`.",
        "Use the same per-patient aggregation rules as the preprocessing tasks: `__first`, `__last`, `__mean`, `__min`, and `__max` use non-null longitudinal values only; `record_count` is the total number of longitudinal rows; `LOS__last` is the last non-null `LOS`; and `Age`, `Sex`, and `Outcome` come from the first non-null patient-level values after sorting by `RecordTime`, `AdmissionTime`, and `DischargeTime` ascending.",
        "Train on the union of `train` and `val`, evaluate on `test`, and drop patients missing any required feature column.",
        "If `stay_days` is requested, define it as the integer number of days between the patient's first non-null `AdmissionTime` and last non-null `DischargeTime`.",
        "Fit `sklearn.linear_model.LogisticRegression` with `solver='liblinear'`, `penalty='l2'`, `C=1.0`, `max_iter=1000`, and prediction threshold `0.5`.",
        "Write `metrics.json` with keys `task_type`, `train_rows`, `test_rows`, `positive_rate_train`, `positive_rate_test`, `roc_auc`, `average_precision`, `accuracy`, and `f1`.",
        "Write `coefficients.csv` with columns `feature`, `coefficient`, sorted by `feature` ascending.",
        "Write `predictions.csv` with columns `PatientID`, `target`, `predicted_probability`, `predicted_label`, sorted by `PatientID` ascending.",
        "Write `answer.json` containing the metric fields above plus `model` and ordered `feature_columns`.",
        "",
        "Required deliverables:",
        "- `metrics.json`",
        "- `coefficients.csv`",
        "- `predictions.csv`",
        "- `answer.json`",
        "Return a concise textual summary, but the files are the canonical graded outputs.",
    ]
    return "\n".join(lines)


def build_report_task(feature_columns: list[str]) -> str:
    lines = [
        *task_preamble(include_split_metadata=True),
        "",
        "Task:",
        f"Construct patient-level modeling features {quoted_list(feature_columns)} and fit a deterministic linear regression model to predict `LOS__last`.",
        "Use the same per-patient aggregation rules as the preprocessing tasks: `__first`, `__last`, `__mean`, `__min`, and `__max` use non-null longitudinal values only; `record_count` is the total number of longitudinal rows; `LOS__last` is the last non-null `LOS`; and `Age`, `Sex`, and `Outcome` come from the first non-null patient-level values after sorting by `RecordTime`, `AdmissionTime`, and `DischargeTime` ascending.",
        "Train on the union of `train` and `val`, evaluate on `test`, and drop patients missing any required feature column.",
        "If `stay_days` is requested, define it as the integer number of days between the patient's first non-null `AdmissionTime` and last non-null `DischargeTime`.",
        "Fit `sklearn.linear_model.LinearRegression` and summarize the result as a short report.",
        "Write `report.md` with the exact Markdown headings `## Task`, `## Data Split`, `## Feature Set`, `## Model Specification`, `## Test Metrics`, `## Coefficient Interpretation`, and `## Limitations`.",
        "Under `## Test Metrics`, report `train_rows`, `test_rows`, `r2`, `mae`, and `rmse`.",
        "Write `answer.json` with keys `model`, `task_type`, `train_rows`, `test_rows`, `r2`, `mae`, `rmse`, `feature_columns`, `target_column`, and `report_file`.",
        "",
        "Required deliverables:",
        "- `report.md`",
        "- `answer.json`",
        "Return a concise textual summary, but the files are the canonical graded outputs.",
    ]
    return "\n".join(lines)


def compute_query_answer(context: TjhTaskContext, feature: str, aggregate: str, threshold: float, comparator: str) -> dict[str, Any]:
    exact_feature = context.map_feature(feature)
    derived_column = f"{exact_feature}__{aggregate}"
    table = context.patient_base[["Outcome"]].join(context.feature_aggregates(exact_feature)[[derived_column]], how="left")
    cohort = table.dropna(subset=[derived_column])
    if comparator == ">=":
        cohort = cohort[cohort[derived_column] >= threshold]
    elif comparator == "<=":
        cohort = cohort[cohort[derived_column] <= threshold]
    else:
        raise ValueError(f"unsupported comparator: {comparator}")
    return {
        "threshold": threshold,
        "comparator": comparator,
        "patient_count": int(len(cohort)),
    }


def compute_summary_answer(context: TjhTaskContext, feature: str) -> dict[str, Any]:
    exact_feature = context.map_feature(feature)
    mean_column = f"{exact_feature}__mean"
    table = context.patient_base[["Outcome", "Sex", "Age"]].join(
        context.feature_aggregates(exact_feature)[[mean_column]], how="left"
    )
    table = table.dropna(subset=[mean_column])
    summary = (
        table.groupby(["Outcome", "Sex"], sort=True)
        .agg(
            patient_count=(mean_column, "size"),
            mean_feature=(mean_column, "mean"),
            median_feature=(mean_column, "median"),
            mean_Age=("Age", "mean"),
        )
        .reset_index()
    )
    return {
        "rows": int(len(summary)),
        "feature": exact_feature,
    }


def compute_preprocess_answer(context: TjhTaskContext, feature_names: list[str]) -> dict[str, Any]:
    exact_features = [context.map_feature(feature_name) for feature_name in feature_names]
    table = context.patient_feature_table(exact_features)
    return {
        "rows": int(len(table)),
        "columns": list(table.columns),
    }


def compute_plot_answer(context: TjhTaskContext, feature: str) -> dict[str, Any]:
    exact_feature = context.map_feature(feature)
    plot_frame = context.frame[["PatientID", "Outcome", "RecordTime", "AdmissionTime", exact_feature]].copy()
    plot_frame["day"] = (plot_frame["RecordTime"] - plot_frame["AdmissionTime"]).dt.days
    plot_frame = plot_frame[(plot_frame["day"] >= 0) & plot_frame[exact_feature].notna()]
    summary = (
        plot_frame.groupby(["Outcome", "day"], sort=True)
        .agg(
            daily_mean=(exact_feature, "mean"),
            patient_count=("PatientID", "nunique"),
        )
        .reset_index()
    )
    return {
        "feature": exact_feature,
        "rows": int(len(summary)),
    }


def compute_logistic_answer(context: TjhTaskContext, feature_columns: list[str]) -> dict[str, Any]:
    exact_feature_columns = [context.map_feature(feature_column) for feature_column in feature_columns]
    table = context.modeling_table(exact_feature_columns, target_column="Outcome")
    required_columns = ["PatientID", "split", "Outcome", *exact_feature_columns]
    table = table[required_columns].dropna(subset=["Outcome", *exact_feature_columns]).copy()
    table["Outcome"] = table["Outcome"].astype(int)

    train_frame = table[table["split"].isin({"train", "val"})].copy()
    test_frame = table[table["split"] == "test"].copy()

    model = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=1000,
    )
    model.fit(train_frame[exact_feature_columns], train_frame["Outcome"])

    predicted_probability = model.predict_proba(test_frame[exact_feature_columns])[:, 1]
    predicted_label = (predicted_probability >= 0.5).astype(int)
    test_target = test_frame["Outcome"].to_numpy()

    return {
        "model": "logistic_regression",
        "task_type": "logistic_regression",
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "positive_rate_train": round(float(train_frame["Outcome"].mean()), 6),
        "positive_rate_test": round(float(test_frame["Outcome"].mean()), 6),
        "roc_auc": round(float(roc_auc_score(test_target, predicted_probability)), 6),
        "average_precision": round(float(average_precision_score(test_target, predicted_probability)), 6),
        "accuracy": round(float(accuracy_score(test_target, predicted_label)), 6),
        "f1": round(float(f1_score(test_target, predicted_label)), 6),
        "feature_columns": exact_feature_columns,
    }


def compute_report_answer(context: TjhTaskContext, feature_columns: list[str]) -> dict[str, Any]:
    exact_feature_columns = [context.map_feature(feature_column) for feature_column in feature_columns]
    table = context.modeling_table(exact_feature_columns, target_column="LOS__last")
    required_columns = ["PatientID", "split", "LOS__last", *exact_feature_columns]
    table = table[required_columns].dropna(subset=["LOS__last", *exact_feature_columns]).copy()

    train_frame = table[table["split"].isin({"train", "val"})].copy()
    test_frame = table[table["split"] == "test"].copy()

    model = LinearRegression()
    model.fit(train_frame[exact_feature_columns], train_frame["LOS__last"])

    prediction = model.predict(test_frame[exact_feature_columns])
    target = test_frame["LOS__last"].to_numpy()
    rmse = math.sqrt(mean_squared_error(target, prediction))

    return {
        "model": "linear_regression",
        "task_type": "report",
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "r2": round(float(r2_score(target, prediction)), 6),
        "mae": round(float(mean_absolute_error(target, prediction)), 6),
        "rmse": round(float(rmse), 6),
        "feature_columns": exact_feature_columns,
        "target_column": "LOS__last",
        "report_file": "report.md",
    }


def rewrite_eval_row(row: dict[str, Any], context: TjhTaskContext) -> dict[str, Any]:
    qid = int(row["qid"])
    rewritten = dict(row)

    if qid in QUERY_QIDS:
        match = RE_QUERY.search(row["task"])
        if not match:
            raise ValueError(f"unable to parse query task {qid}")
        aggregate = match.group("agg")
        feature = context.map_feature(match.group("feature"))
        threshold = float(row["answer"]["threshold"])
        comparator = row["answer"]["comparator"]
        rewritten["task"] = build_query_task(feature, aggregate, threshold, comparator)
        rewritten["answer"] = compute_query_answer(context, feature, aggregate, threshold, comparator)
        return rewritten

    if qid in SUMMARY_QIDS:
        match = RE_SUMMARY.search(row["task"])
        if not match:
            raise ValueError(f"unable to parse summary task {qid}")
        feature = context.map_feature(match.group("feature"))
        rewritten["task"] = build_summary_task(feature)
        rewritten["answer"] = compute_summary_answer(context, feature)
        return rewritten

    if qid in PREPROCESS_QIDS:
        match = RE_FEATURE_TABLE.search(row["task"])
        if not match:
            raise ValueError(f"unable to parse preprocessing task {qid}")
        feature_names = [context.map_feature(feature_name) for feature_name in parse_feature_list(match.group("features"))]
        output_columns = compute_preprocess_answer(context, feature_names)["columns"]
        rewritten["task"] = build_preprocess_task(feature_names, output_columns)
        rewritten["answer"] = {
            "rows": int(context.patient_base.shape[0]),
            "columns": output_columns,
        }
        return rewritten

    if qid in PLOT_QIDS:
        match = RE_PLOT.search(row["task"])
        if not match:
            raise ValueError(f"unable to parse visualization task {qid}")
        feature = context.map_feature(match.group("feature"))
        rewritten["task"] = build_plot_task(feature)
        rewritten["answer"] = compute_plot_answer(context, feature)
        return rewritten

    if qid in MODELING_QIDS:
        match = RE_MODEL.search(row["task"])
        if not match:
            raise ValueError(f"unable to parse modeling task {qid}")
        feature_columns = [context.map_feature(feature_name) for feature_name in parse_feature_list(match.group("features"))]
        rewritten["task"] = build_logistic_model_task(feature_columns)
        rewritten["answer"] = compute_logistic_answer(context, feature_columns)
        return rewritten

    if qid in REPORT_QIDS:
        match = RE_MODEL.search(row["task"])
        if not match:
            raise ValueError(f"unable to parse report task {qid}")
        feature_columns = [context.map_feature(feature_name) for feature_name in parse_feature_list(match.group("features"))]
        rewritten["task"] = build_report_task(feature_columns)
        rewritten["answer"] = compute_report_answer(context, feature_columns)
        return rewritten

    return rewritten


def rewrite_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    qid = int(row["qid"])
    rewritten = dict(row)
    if qid in REPORT_QIDS:
        rewritten["required_files"] = ["report.md", "answer.json"]
    return rewritten


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite TJH benchmark tasks against the current formatted parquet dataset.")
    parser.add_argument("--source-eval", type=Path, default=DEFAULT_SOURCE_EVAL_PATH)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST_PATH)
    parser.add_argument("--output-eval", type=Path, default=DEFAULT_OUTPUT_EVAL_PATH)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = TjhTaskContext.load()

    eval_rows = read_jsonl(args.source_eval)
    manifest_rows = read_jsonl(args.source_manifest)

    rewritten_eval_rows = [rewrite_eval_row(row, context) if int(row["qid"]) <= 50 else row for row in eval_rows]
    rewritten_manifest_rows = [
        rewrite_manifest_row(row) if row.get("dataset") == "TJH" else row
        for row in manifest_rows
    ]

    write_jsonl(args.output_eval, rewritten_eval_rows)
    write_jsonl(args.output_manifest, rewritten_manifest_rows)

    summary = {
        "rewritten_tjh_tasks": 50,
        "source_eval": str(args.source_eval),
        "source_manifest": str(args.source_manifest),
        "output_eval": str(args.output_eval),
        "output_manifest": str(args.output_manifest),
        "split_key_counts": context.split_key_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
