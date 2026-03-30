from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
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
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TJH_FEATURES: list[str] = [
    "White blood cell count",
    "hemoglobin",
    "Platelet count",
    "creatinine",
    "albumin",
    "Serum potassium",
    "serum sodium",
    "Serum chloride",
    "calcium",
    "Urea",
    "Hypersensitive c-reactive protein",
    "Lactate dehydrogenase",
    "D-D dimer",
    "ferritin",
    "Interleukin 6",
]

TJH_ALIAS = {
    "White blood cell count": "white_blood_cell_count",
    "hemoglobin": "hemoglobin",
    "Platelet count": "platelet_count",
    "creatinine": "creatinine",
    "albumin": "albumin",
    "Serum potassium": "serum_potassium",
    "serum sodium": "serum_sodium",
    "Serum chloride": "serum_chloride",
    "calcium": "calcium",
    "Urea": "urea",
    "Hypersensitive c-reactive protein": "hs_crp",
    "Lactate dehydrogenase": "ldh",
    "D-D dimer": "d_dimer",
    "ferritin": "ferritin",
    "Interleukin 6": "interleukin_6",
}

MIMIC_LABS: dict[str, tuple[str, str]] = {
    "potassium": ("LAB//50971//mEq/L", "Potassium [Moles/volume] in Serum or Plasma"),
    "sodium": ("LAB//50983//mEq/L", "Sodium [Moles/volume] in Serum or Plasma"),
    "creatinine": ("LAB//50912//mg/dL", "Creatinine [Mass/volume] in Serum or Plasma"),
    "chloride": ("LAB//50902//mEq/L", "Chloride [Moles/volume] in Serum or Plasma"),
    "urea_nitrogen": ("LAB//51006//mg/dL", "Urea nitrogen [Mass/volume] in Serum or Plasma"),
    "hematocrit": ("LAB//51221//%", "Hematocrit [Volume Fraction] of Blood by Automated count"),
    "platelets": ("LAB//51265//K/uL", "Platelets [#/volume] in Blood by Automated count"),
    "hemoglobin": ("LAB//51222//g/dL", "Hemoglobin [Mass/volume] in Blood"),
    "leukocytes": ("LAB//51301//K/uL", "Leukocytes [#/volume] in Blood by Automated count"),
}

BENCHMARK_ROOT_NAME = "benchmark_ground_truth"
PROVENANCE_ROOT_NAME = "benchmark_provenance"


@dataclass
class GeneratedTask:
    task: dict[str, Any]
    ground_truth_dir: Path


@dataclass
class TjhViews:
    raw: pd.DataFrame
    patient_summary: pd.DataFrame
    temporal_daily: pd.DataFrame
    split_counts: dict[str, int]


@dataclass
class MimicViews:
    raw: pd.DataFrame
    numeric_events: pd.DataFrame
    subject_summary: pd.DataFrame
    temporal_daily: pd.DataFrame
    split_counts: dict[str, int]


def find_workspace_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "data" / "TJH.csv").exists():
            return parent
    raise FileNotFoundError("Could not locate workspace root containing data/TJH.csv")


WORKSPACE_ROOT = find_workspace_root()
HEALTHFLOW_ROOT = WORKSPACE_ROOT / "code" / "HealthFlow"
DATA_ROOT = WORKSPACE_ROOT / "data"
PROVENANCE_ROOT = DATA_ROOT / PROVENANCE_ROOT_NAME
GROUND_TRUTH_ROOT = DATA_ROOT / BENCHMARK_ROOT_NAME
MIMIC_ROOT = DATA_ROOT / "mimic_iv_demo_meds"


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def round_float(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [round_float(item, digits=digits) for item in value]
    if isinstance(value, dict):
        return {key: round_float(item, digits=digits) for key, item in value.items()}
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        return round(float(value), digits)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(round_float(payload), handle, indent=2, ensure_ascii=False)


def text_dump(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def csv_dump(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def stable_split(identifier: Any) -> str:
    digest = hashlib.sha256(str(identifier).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 20
    if bucket < 14:
        return "train"
    if bucket < 17:
        return "tuning"
    return "held_out"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataframe_signature(frame: pd.DataFrame) -> str:
    temp = frame.copy()
    for column in temp.columns:
        if pd.api.types.is_float_dtype(temp[column]):
            temp[column] = temp[column].round(6)
    csv_bytes = temp.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def feature_stat_column(alias: str, stat: str) -> str:
    return f"{alias}__{stat}"


def build_tjh_views() -> TjhViews:
    raw = pd.read_csv(DATA_ROOT / "TJH.csv")
    raw["PatientID"] = raw["PatientID"].astype(int)
    raw["RecordTime"] = pd.to_datetime(raw["RecordTime"])
    raw["AdmissionTime"] = pd.to_datetime(raw["AdmissionTime"])
    raw["DischargeTime"] = pd.to_datetime(raw["DischargeTime"])
    raw["Outcome"] = raw["Outcome"].astype(float)
    raw["LOS"] = raw["LOS"].astype(float)
    raw["Sex"] = raw["Sex"].astype(float)
    raw["Age"] = raw["Age"].astype(float)
    raw = raw.sort_values(["PatientID", "RecordTime"]).reset_index(drop=True)
    raw["days_from_admission"] = (raw["RecordTime"] - raw["AdmissionTime"]).dt.total_seconds() / 86400.0

    for feature in TJH_FEATURES:
        raw[feature] = pd.to_numeric(raw[feature], errors="coerce")

    patient_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    for patient_id, group in raw.groupby("PatientID", sort=True):
        group = group.sort_values("RecordTime")
        summary: dict[str, Any] = {
            "patient_id": int(patient_id),
            "split": stable_split(patient_id),
            "record_count": int(len(group)),
            "outcome": int(group["Outcome"].max()),
            "los_last": round(float(group["LOS"].iloc[-1]), 6),
            "age": round(float(group["Age"].iloc[0]), 6),
            "sex": int(group["Sex"].iloc[0]),
            "stay_days": round(float((group["DischargeTime"].iloc[0] - group["AdmissionTime"].iloc[0]).total_seconds() / 86400.0), 6),
            "admission_time": group["AdmissionTime"].iloc[0],
            "discharge_time": group["DischargeTime"].iloc[0],
        }
        for feature in TJH_FEATURES:
            alias = TJH_ALIAS[feature]
            values = group[feature].dropna()
            early3 = group.loc[group["days_from_admission"] <= 3.0, feature].dropna()
            early7 = group.loc[group["days_from_admission"] <= 7.0, feature].dropna()
            summary[feature_stat_column(alias, "count")] = int(values.size)
            for stat_name, stat_value in {
                "first": values.iloc[0] if not values.empty else None,
                "last": values.iloc[-1] if not values.empty else None,
                "mean": values.mean() if not values.empty else None,
                "max": values.max() if not values.empty else None,
                "min": values.min() if not values.empty else None,
                "early3_mean": early3.mean() if not early3.empty else None,
                "early7_mean": early7.mean() if not early7.empty else None,
            }.items():
                summary[feature_stat_column(alias, stat_name)] = round_float(stat_value)
        patient_rows.append(summary)

        daily = (
            group.assign(day=group["days_from_admission"].floordiv(1).astype(int))
            .loc[:, ["PatientID", "day", "Outcome", *TJH_FEATURES]]
            .groupby(["PatientID", "day", "Outcome"], as_index=False)
            .mean(numeric_only=True)
        )
        daily = daily.rename(columns={"Outcome": "outcome"})
        daily["split"] = stable_split(patient_id)
        daily["patient_id"] = daily["PatientID"].astype(int)
        temporal_rows.append(daily.drop(columns=["PatientID"]))

    patient_summary = pd.DataFrame(patient_rows).sort_values("patient_id").reset_index(drop=True)
    temporal_daily = pd.concat(temporal_rows, ignore_index=True).sort_values(["patient_id", "day"]).reset_index(drop=True)
    split_counts = patient_summary["split"].value_counts().sort_index().to_dict()
    return TjhViews(raw=raw, patient_summary=patient_summary, temporal_daily=temporal_daily, split_counts=split_counts)


def build_mimic_views() -> MimicViews:
    split_frames: list[pd.DataFrame] = []
    for split_name in ["train", "tuning", "held_out"]:
        split_frame = pd.read_parquet(MIMIC_ROOT / "data" / split_name / "0.parquet")
        split_frame["split"] = split_name
        split_frames.append(split_frame)
    raw = pd.concat(split_frames, ignore_index=True)
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    raw["numeric_value"] = pd.to_numeric(raw["numeric_value"], errors="coerce")
    raw = raw.sort_values(["subject_id", "time", "code"]).reset_index(drop=True)

    codes = pd.read_parquet(MIMIC_ROOT / "metadata" / "codes.parquet")
    code_to_description = dict(zip(codes["code"], codes["description"]))

    birth_times = raw.loc[raw["code"] == "MEDS_BIRTH", ["subject_id", "time"]].rename(columns={"time": "birth_time"})
    first_gender = (
        raw.loc[raw["code"].str.startswith("GENDER//", na=False), ["subject_id", "code"]]
        .drop_duplicates("subject_id")
        .assign(gender=lambda frame: frame["code"].str.replace("GENDER//", "", regex=False))
        .loc[:, ["subject_id", "gender"]]
    )
    first_admission = (
        raw.loc[raw["code"].str.startswith("HOSPITAL_ADMISSION//", na=False), ["subject_id", "time", "code"]]
        .sort_values(["subject_id", "time"])
        .groupby("subject_id", as_index=False)
        .first()
        .rename(columns={"time": "first_admission_time", "code": "first_admission_code"})
    )
    last_discharge = (
        raw.loc[raw["code"].str.startswith("HOSPITAL_DISCHARGE//", na=False), ["subject_id", "time", "code"]]
        .sort_values(["subject_id", "time"])
        .groupby("subject_id", as_index=False)
        .last()
        .rename(columns={"time": "last_discharge_time", "code": "last_discharge_code"})
    )

    subject_summary = pd.DataFrame({"subject_id": sorted(raw["subject_id"].unique())})
    subject_summary["split"] = subject_summary["subject_id"].map(
        pd.read_parquet(MIMIC_ROOT / "metadata" / "subject_splits.parquet").set_index("subject_id")["split"]
    )
    subject_summary = subject_summary.merge(birth_times, on="subject_id", how="left")
    subject_summary = subject_summary.merge(first_gender, on="subject_id", how="left")
    subject_summary = subject_summary.merge(first_admission, on="subject_id", how="left")
    subject_summary = subject_summary.merge(last_discharge, on="subject_id", how="left")
    subject_summary["age_at_first_admission"] = (
        (subject_summary["first_admission_time"] - subject_summary["birth_time"]).dt.total_seconds() / (365.25 * 86400.0)
    ).round(3)
    subject_summary["discharge_died"] = subject_summary["last_discharge_code"].eq("HOSPITAL_DISCHARGE//DIED").astype(int)
    subject_summary["discharge_home"] = subject_summary["last_discharge_code"].isin(
        ["HOSPITAL_DISCHARGE//HOME", "HOSPITAL_DISCHARGE//HOME HEALTH CARE"]
    ).astype(int)

    for prefix_name, prefix_value in {
        "hospital_admission_count": "HOSPITAL_ADMISSION//",
        "hospital_discharge_count": "HOSPITAL_DISCHARGE//",
        "icu_admission_count": "ICU_ADMISSION//",
        "icu_discharge_count": "ICU_DISCHARGE//",
        "transfer_count": "TRANSFER_TO//",
        "diagnosis_event_count": "DIAGNOSIS//",
        "procedure_event_count": "PROCEDURE//",
        "drg_event_count": "DRG//",
    }.items():
        counts = (
            raw.loc[raw["code"].str.startswith(prefix_value, na=False)]
            .groupby("subject_id")
            .size()
            .rename(prefix_name)
        )
        subject_summary[prefix_name] = subject_summary["subject_id"].map(counts).fillna(0).astype(int)

    total_events = raw.groupby("subject_id").size().rename("event_count")
    subject_summary["event_count"] = subject_summary["subject_id"].map(total_events).fillna(0).astype(int)

    numeric_frames: list[pd.DataFrame] = []
    for alias, (code, description) in MIMIC_LABS.items():
        subset = raw.loc[raw["code"] == code, ["subject_id", "time", "numeric_value", "split"]].dropna(subset=["numeric_value"])
        subset = subset.sort_values(["subject_id", "time"])
        subset["lab_alias"] = alias
        subset["lab_description"] = description
        numeric_frames.append(subset)

        stats = (
            subset.groupby("subject_id")["numeric_value"]
            .agg(["count", "first", "last", "mean", "max", "min"])
            .reset_index()
            .rename(
                columns={
                    "count": feature_stat_column(alias, "count"),
                    "first": feature_stat_column(alias, "first"),
                    "last": feature_stat_column(alias, "last"),
                    "mean": feature_stat_column(alias, "mean"),
                    "max": feature_stat_column(alias, "max"),
                    "min": feature_stat_column(alias, "min"),
                }
            )
        )
        stats[feature_stat_column(alias, "count")] = stats[feature_stat_column(alias, "count")].astype(int)
        subject_summary = subject_summary.merge(stats, on="subject_id", how="left")

    numeric_events = pd.concat(numeric_frames, ignore_index=True).sort_values(["subject_id", "time", "lab_alias"]).reset_index(drop=True)
    for alias in MIMIC_LABS:
        for suffix in ["count", "first", "last", "mean", "max", "min"]:
            column = feature_stat_column(alias, suffix)
            if column not in subject_summary:
                continue
            if suffix == "count":
                subject_summary[column] = subject_summary[column].fillna(0).astype(int)
            else:
                subject_summary[column] = subject_summary[column].map(round_float)

    temporal_daily = (
        numeric_events.assign(day=lambda frame: (frame["time"] - frame.groupby("subject_id")["time"].transform("min")).dt.total_seconds() / 86400.0)
        .assign(day=lambda frame: frame["day"].floordiv(1).astype(int))
        .groupby(["subject_id", "split", "lab_alias", "lab_description", "day"], as_index=False)["numeric_value"]
        .mean()
        .rename(columns={"numeric_value": "daily_mean"})
    )
    split_counts = subject_summary["split"].value_counts().sort_index().to_dict()
    subject_summary = subject_summary.sort_values("subject_id").reset_index(drop=True)
    return MimicViews(
        raw=raw,
        numeric_events=numeric_events,
        subject_summary=subject_summary,
        temporal_daily=temporal_daily,
        split_counts=split_counts,
    )


def title_from_markdown_dir(name: str) -> str:
    parts = name.split("_", 1)
    if len(parts) < 2:
        return name
    title = parts[1].split(".pdf", 1)[0]
    title = title.replace("_", " ").strip()
    return re.sub(r"\s+", " ", title)


def classify_extracted_task(category: str, text: str) -> tuple[str, list[str]]:
    lowered = text.lower()
    reasons: list[str] = []
    if any(token in lowered for token in ["mimic-iii", "mimic-iv", "eicu", "physionet", "clinical note", "discharge summary"]):
        reasons.append("inaccessible_data")
    if any(token in lowered for token in ["auc", "auroc", "auprc", "accuracy", "precision@", "f1", "macro recall", "microauc"]):
        reasons.append("paper_metric_only")
    if any(token in lowered for token in ["implement", "loss function", "attention mechanism", "transformer", "knowledge graph", "gan", "gru", "bert"]):
        reasons.append("non_executable")
    if any(token in lowered for token in ["table ", "figure ", "section "]):
        reasons.append("weak_comparability")

    if reasons:
        return "reject", reasons

    if any(token in lowered for token in ["cohort", "count the number", "calculate the average", "preprocess", "survival analysis"]):
        return "rewrite", []
    if category.lower() in {"data analysis", "cohort definition", "data preprocessing"}:
        return "rewrite", []
    return "reject", ["non_verifiable"]


def build_paper_audit_outputs() -> dict[str, Any]:
    markdown_dirs = {
        directory.name.split("_", 1)[0]: directory.name
        for directory in (HEALTHFLOW_ROOT / "scripts" / "extract_task" / "assets" / "markdowns").iterdir()
        if directory.is_dir()
    }
    selected_ids = (HEALTHFLOW_ROOT / "scripts" / "filter_paper" / "results" / "final_selected_ID.txt").read_text(encoding="utf-8").splitlines()
    filtered = pd.read_csv(HEALTHFLOW_ROOT / "scripts" / "filter_paper" / "results" / "filtered_results_combined.csv")

    extracted_rows: list[dict[str, Any]] = []
    for task_file in sorted((HEALTHFLOW_ROOT / "scripts" / "extract_task" / "tasks").glob("*_tasks.jsonl")):
        paper_id = task_file.name.split("_", 1)[0]
        with task_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                action, reasons = classify_extracted_task(row.get("category", ""), f"{row.get('task', '')} {row.get('answer', '')}")
                extracted_rows.append(
                    {
                        "paper_id": paper_id,
                        "paper_title": title_from_markdown_dir(markdown_dirs.get(paper_id, paper_id)),
                        "task_index": line_number,
                        "category": row.get("category", ""),
                        "task": row.get("task", ""),
                        "answer": row.get("answer", ""),
                        "recommended_action": action,
                        "reason_codes": "|".join(reasons),
                    }
                )
    extracted_df = pd.DataFrame(extracted_rows)
    csv_ids = set(filtered["ID"].astype(str))
    markdown_ids = set(markdown_dirs)
    selected_id_set = set(selected_ids)

    current_ehr = read_jsonl(DATA_ROOT / "ehrflowbench.jsonl")
    current_ehr_train = read_jsonl(DATA_ROOT / "ehrflowbench_train.jsonl")
    current_mab = read_jsonl(DATA_ROOT / "medagentboard.jsonl")

    current_ehr_flagged = [
        row
        for row in current_ehr + current_ehr_train
        if classify_extracted_task(row.get("category", ""), f"{row.get('task', '')} {row.get('answer', '')}")[0] != "rewrite"
    ]
    blank_medagentboard_qids = [row["qid"] for row in current_mab if not str(row.get("answer", "")).strip() and "qid" in row]
    inaccessible_medagentboard_qids = [
        row["qid"]
        for row in current_mab
        if "qid" in row
        and (
            "/home/projects/HealthFlow" in json.dumps(row, ensure_ascii=False)
            or "MIMIC-IV.parquet" in json.dumps(row, ensure_ascii=False)
        )
    ]
    reason_code_counts: Counter[str] = Counter()
    for reason_codes in extracted_df["reason_codes"].fillna(""):
        if not str(reason_codes).strip():
            continue
        for reason_code in str(reason_codes).split("|"):
            if reason_code:
                reason_code_counts[reason_code] += 1

    audit_summary = {
        "filtered_rows": int(len(filtered)),
        "markdown_directories": int(len(markdown_ids)),
        "selected_ids": int(len(selected_id_set)),
        "extracted_task_files": int(extracted_df["paper_id"].nunique()),
        "extracted_tasks": int(len(extracted_df)),
        "filter_only_rows_without_markdown": sorted(csv_ids - markdown_ids),
        "selected_without_markdown": sorted(selected_id_set - markdown_ids),
        "recommended_action_counts": extracted_df["recommended_action"].value_counts().to_dict(),
        "reason_code_counts": dict(reason_code_counts),
        "current_ehrflowbench_eval_tasks": len(current_ehr),
        "current_ehrflowbench_train_tasks": len(current_ehr_train),
        "current_ehrflowbench_flagged_unverifiable_or_inaccessible": len(current_ehr_flagged),
        "current_medagentboard_tasks": len(current_mab),
        "current_medagentboard_blank_answers": len(blank_medagentboard_qids),
        "current_medagentboard_inaccessible_paths": len(inaccessible_medagentboard_qids),
    }

    ensure_empty_dir(PROVENANCE_ROOT)
    csv_dump(PROVENANCE_ROOT / "extracted_task_audit.csv", extracted_df)
    json_dump(PROVENANCE_ROOT / "paper_provenance_manifest.json", audit_summary)
    json_dump(
        PROVENANCE_ROOT / "filter_markdown_reconciliation.json",
        {
            "csv_not_markdown": sorted(csv_ids - markdown_ids),
            "markdown_not_csv": sorted(markdown_ids - csv_ids),
            "selected_not_markdown": sorted(selected_id_set - markdown_ids),
        },
    )
    json_dump(
        PROVENANCE_ROOT / "current_benchmark_audit.json",
        {
            "ehrflowbench_flagged_qids": [row["qid"] for row in current_ehr_flagged if "qid" in row],
            "medagentboard_blank_answer_qids": blank_medagentboard_qids,
            "medagentboard_inaccessible_path_qids": inaccessible_medagentboard_qids,
        },
    )
    return audit_summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_line(path: Path, frame: pd.DataFrame, x_col: str, y_col: str, group_col: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for group_name, group in frame.groupby(group_col):
        ax.plot(group[x_col], group[y_col], marker="o", label=str(group_name))
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bar(path: Path, frame: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(frame[x_col].astype(str), frame[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hist(path: Path, values: pd.Series, title: str) -> tuple[list[float], list[int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts, edges = np.histogram(values.dropna(), bins=min(10, max(3, values.dropna().nunique())))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values.dropna(), bins=len(edges) - 1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return edges.tolist(), counts.astype(int).tolist()


def prompt_header(dataset: str) -> str:
    if dataset == "TJH":
        return (
            "Use the local longitudinal EHR table at `../../data/TJH.csv`. "
            "Treat `PatientID` as the patient key and `RecordTime` as the longitudinal timestamp. "
        )
    return (
        "Use the local open MEDS dataset rooted at `../../data/mimic_iv_demo_meds/`. "
        "Read the three split parquet files under `data/` together with `metadata/subject_splits.parquet` and `metadata/codes.parquet`. "
        "Treat `subject_id` as the patient key and `time` as the event timestamp. "
    )


def write_task_ground_truth(ground_truth_dir: Path, artifacts: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    file_specs: dict[str, Any] = {}
    required_files: list[str] = []
    for relative_name, payload in artifacts.items():
        required_files.append(relative_name)
        target = ground_truth_dir / relative_name
        if isinstance(payload, pd.DataFrame):
            csv_dump(target, payload)
            file_specs[relative_name] = {
                "compare": "csv",
                "rows": int(len(payload)),
                "columns": list(payload.columns),
                "sha256": dataframe_signature(payload),
                "float_tolerance": 1e-6,
            }
        elif isinstance(payload, str) and relative_name.endswith(".md"):
            text_dump(target, payload)
            file_specs[relative_name] = {"compare": "text", "sha256": sha256_file(target)}
        elif isinstance(payload, (dict, list)):
            json_dump(target, payload)
            file_specs[relative_name] = {"compare": "json", "sha256": sha256_file(target), "float_tolerance": 1e-6}
        elif isinstance(payload, Path):
            target.parent.mkdir(parents=True, exist_ok=True)
            if payload.resolve() != target.resolve():
                shutil.copy2(payload, target)
            file_specs[relative_name] = {"compare": "exists", "sha256": sha256_file(target)}
        else:
            raise TypeError(f"Unsupported artifact payload for {relative_name}: {type(payload)!r}")
    return required_files, {"files": file_specs}


def build_logistic_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    id_column: str,
    split_column: str = "split",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train = frame[frame[split_column].isin(["train", "tuning"])].copy()
    held_out = frame[frame[split_column] == "held_out"].copy()
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[feature_columns])
    x_test = imputer.transform(held_out[feature_columns])
    y_train = train[target_column].astype(int).to_numpy()
    y_test = held_out[target_column].astype(int).to_numpy()

    model = LogisticRegression(max_iter=500, random_state=0, solver="liblinear")
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "task_type": "logistic_regression",
        "train_rows": int(len(train)),
        "held_out_rows": int(len(held_out)),
        "positive_rate_train": round_float(y_train.mean()),
        "positive_rate_held_out": round_float(y_test.mean()),
        "roc_auc": round_float(roc_auc_score(y_test, probabilities)),
        "average_precision": round_float(average_precision_score(y_test, probabilities)),
        "accuracy": round_float(accuracy_score(y_test, predictions)),
        "f1": round_float(f1_score(y_test, predictions)),
    }
    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": np.round(model.coef_[0], 6),
        }
    ).sort_values("feature")
    predictions_df = pd.DataFrame(
        {
            id_column: held_out[id_column].astype(int),
            "target": y_test.astype(int),
            "predicted_probability": np.round(probabilities, 6),
            "predicted_label": predictions.astype(int),
        }
    ).sort_values(id_column)
    return metrics, coefficients, predictions_df


def build_linear_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    id_column: str,
    split_column: str = "split",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train = frame[frame[split_column].isin(["train", "tuning"])].copy()
    held_out = frame[frame[split_column] == "held_out"].copy()
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[feature_columns])
    x_test = imputer.transform(held_out[feature_columns])
    y_train = train[target_column].astype(float).to_numpy()
    y_test = held_out[target_column].astype(float).to_numpy()

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "task_type": "linear_regression",
        "train_rows": int(len(train)),
        "held_out_rows": int(len(held_out)),
        "r2": round_float(r2_score(y_test, predictions)),
        "mae": round_float(mean_absolute_error(y_test, predictions)),
        "rmse": round_float(math.sqrt(mean_squared_error(y_test, predictions))),
    }
    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": np.round(model.coef_, 6),
        }
    ).sort_values("feature")
    predictions_df = pd.DataFrame(
        {
            id_column: held_out[id_column].astype(int),
            "target": np.round(y_test, 6),
            "prediction": np.round(predictions, 6),
        }
    ).sort_values(id_column)
    return metrics, coefficients, predictions_df


def current_benchmark_prompt(dataset: str, task_brief: str, deliverables: Iterable[str]) -> str:
    deliverable_list = "".join(f"\n- `{item}`" for item in deliverables)
    return (
        f"{prompt_header(dataset)}"
        "Use deterministic computations only and keep the output machine-checkable.\n\n"
        f"Task:\n{task_brief}\n\n"
        "Required deliverables:"
        f"{deliverable_list}\n"
        "Return a concise textual summary, but the files are the canonical graded outputs."
    )


def make_task_record(
    *,
    benchmark: str,
    qid: int,
    split: str,
    dataset: str,
    category: str,
    task_family: str,
    task_type: str,
    task_brief: str,
    answer_payload: dict[str, Any],
    ground_truth_dir: Path,
    artifacts: dict[str, Any],
    derivation_label: str = "dataset_native",
    source_paper_id: str | None = None,
    source_paper_title: str | None = None,
    provenance_note: str | None = None,
) -> GeneratedTask:
    required_files, verification_spec = write_task_ground_truth(ground_truth_dir, artifacts)
    answer_payload = round_float(answer_payload)
    prompt = current_benchmark_prompt(dataset, task_brief, required_files)
    task = {
        "qid": qid,
        "split": split,
        "category": category,
        "task": prompt,
        "task_brief": task_brief,
        "answer": answer_payload,
        "dataset": dataset,
        "task_family": task_family,
        "task_type": task_type,
        "answer_type": "structured_artifact_bundle",
        "required_files": required_files,
        "verification_type": "artifact_bundle",
        "verification_spec": verification_spec,
        "ground_truth_ref": str(ground_truth_dir.relative_to(WORKSPACE_ROOT)),
        "derivation_label": derivation_label,
        "source_paper_id": source_paper_id,
        "source_paper_title": source_paper_title,
        "provenance_note": provenance_note or "Rebuilt from fully local, free-to-use data with deterministic answer generation.",
    }
    return GeneratedTask(task=task, ground_truth_dir=ground_truth_dir)


def build_tjh_medagentboard(views: TjhViews) -> list[GeneratedTask]:
    tasks: list[GeneratedTask] = []
    summary = views.patient_summary.copy()
    gt_root = GROUND_TRUTH_ROOT / "medagentboard"

    high_feature_plan = [
        ("hs_crp", "max", 0.75),
        ("ldh", "max", 0.75),
        ("d_dimer", "max", 0.75),
        ("ferritin", "max", 0.75),
        ("interleukin_6", "max", 0.75),
        ("albumin", "min", 0.25),
        ("hemoglobin", "min", 0.25),
        ("platelet_count", "min", 0.25),
        ("creatinine", "mean", 0.75),
        ("white_blood_cell_count", "mean", 0.75),
    ]
    for index, (alias, stat, quantile) in enumerate(high_feature_plan, start=1):
        column = feature_stat_column(alias, stat)
        series = summary[column].dropna()
        threshold = round(float(series.quantile(quantile)), 6)
        comparator = ">=" if quantile >= 0.5 else "<="
        if comparator == ">=":
            cohort = summary.loc[summary[column] >= threshold, ["patient_id", "outcome", column]].sort_values("patient_id")
        else:
            cohort = summary.loc[summary[column] <= threshold, ["patient_id", "outcome", column]].sort_values("patient_id")
        task_brief = (
            f"Build a patient cohort using the TJH longitudinal table. Compute the patient-level `{stat}` value of `{alias}` and select patients "
            f"whose value is `{comparator} {threshold}`. Write `cohort.csv` with `patient_id`, `outcome`, and the derived `{column}` column, "
            "and write `answer.json` with the threshold, comparator, and patient count."
        )
        answer = {
            "threshold": threshold,
            "comparator": comparator,
            "patient_count": int(len(cohort)),
            "patient_ids": cohort["patient_id"].astype(int).tolist(),
        }
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=index,
                split="eval",
                dataset="TJH",
                category="Data Querying",
                task_family="data_querying",
                task_type="data_querying",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(index),
                artifacts={"cohort.csv": cohort, "answer.json": answer},
            )
        )

    stats_features = ["white_blood_cell_count", "hemoglobin", "platelet_count", "creatinine", "albumin", "serum_potassium", "serum_sodium", "serum_chloride", "calcium", "urea"]
    for offset, alias in enumerate(stats_features, start=11):
        mean_col = feature_stat_column(alias, "mean")
        group = (
            summary.loc[:, ["outcome", "sex", "age", mean_col]]
            .dropna()
            .groupby(["outcome", "sex"], as_index=False)
            .agg(
                patient_count=("age", "size"),
                mean_feature=(mean_col, "mean"),
                median_feature=(mean_col, "median"),
                mean_age=("age", "mean"),
            )
            .sort_values(["outcome", "sex"])
        )
        answer = {"rows": int(len(group)), "feature": alias, "sha256": dataframe_signature(group)}
        task_brief = (
            f"Create an outcome- and sex-stratified summary table for the patient-level mean of `{alias}`. "
            "Write `summary.csv` with columns `outcome`, `sex`, `patient_count`, `mean_feature`, `median_feature`, and `mean_age`, "
            "then write `answer.json` with the feature alias, row count, and summary hash."
        )
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=offset,
                split="eval",
                dataset="TJH",
                category="Data Statistics",
                task_family="data_statistics",
                task_type="data_statistics",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(offset),
                artifacts={"summary.csv": group, "answer.json": answer},
            )
        )

    preprocess_groups = [
        ["white_blood_cell_count", "hemoglobin", "platelet_count"],
        ["creatinine", "albumin", "urea"],
        ["serum_sodium", "serum_potassium", "serum_chloride"],
        ["calcium", "hs_crp", "ldh"],
        ["d_dimer", "ferritin", "interleukin_6"],
        ["white_blood_cell_count", "creatinine", "albumin"],
        ["hemoglobin", "platelet_count", "urea"],
        ["serum_sodium", "calcium", "hs_crp"],
        ["serum_potassium", "creatinine", "d_dimer"],
        ["serum_chloride", "albumin", "ldh"],
    ]
    for offset, aliases in enumerate(preprocess_groups, start=21):
        columns = ["patient_id", "split", "outcome", "age", "sex", "los_last", "record_count"]
        for alias in aliases:
            columns.extend(
                [
                    feature_stat_column(alias, "count"),
                    feature_stat_column(alias, "first"),
                    feature_stat_column(alias, "last"),
                    feature_stat_column(alias, "mean"),
                ]
            )
        processed = summary.loc[:, columns].sort_values("patient_id").reset_index(drop=True)
        answer = {"rows": int(len(processed)), "columns": columns, "sha256": dataframe_signature(processed)}
        task_brief = (
            f"Construct a patient-level feature table for the aliases `{', '.join(aliases)}` using the first, last, mean, and measurement count per patient. "
            "Keep the static columns `patient_id`, `split`, `outcome`, `age`, `sex`, `los_last`, and `record_count`. "
            "Write the result to `patient_features.csv` and write `answer.json` with the row count, ordered columns, and table hash."
        )
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=offset,
                split="eval",
                dataset="TJH",
                category="Data Preprocessing",
                task_family="data_preprocessing",
                task_type="data_preprocessing",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(offset),
                artifacts={"patient_features.csv": processed, "answer.json": answer},
            )
        )

    plot_features = ["white_blood_cell_count", "hemoglobin", "platelet_count", "creatinine", "albumin", "serum_sodium", "serum_potassium", "hs_crp", "ldh", "d_dimer"]
    for offset, alias in enumerate(plot_features, start=31):
        feature_name = next(name for name, mapped_alias in TJH_ALIAS.items() if mapped_alias == alias)
        daily = (
            views.temporal_daily.loc[:, ["day", "outcome", feature_name]]
            .rename(columns={feature_name: "daily_mean"})
            .dropna()
            .groupby(["day", "outcome"], as_index=False)
            .agg(daily_mean=("daily_mean", "mean"), patient_count=("daily_mean", "size"))
            .sort_values(["outcome", "day"])
        )
        plot_path = gt_root / str(offset) / "plot.png"
        plot_line(plot_path, daily, "day", "daily_mean", "outcome", f"TJH temporal trend: {alias}")
        summary_payload = {
            "feature": alias,
            "rows": int(len(daily)),
            "sha256": dataframe_signature(daily),
        }
        task_brief = (
            f"Create an outcome-stratified temporal line plot for daily mean `{alias}` over days since admission. "
            "Write the machine-checkable aggregation to `plot_summary.csv`, save the chart to `plot.png`, and write `answer.json` with the feature alias, row count, and hash."
        )
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=offset,
                split="eval",
                dataset="TJH",
                category="Visualization",
                task_family="visualization",
                task_type="visualization",
                task_brief=task_brief,
                answer_payload=summary_payload,
                ground_truth_dir=gt_root / str(offset),
                artifacts={"plot_summary.csv": daily, "plot.png": plot_path, "answer.json": summary_payload},
            )
        )

    classification_feature_sets = [
        ["white_blood_cell_count__mean", "hemoglobin__mean", "platelet_count__mean", "creatinine__mean", "albumin__mean"],
        ["serum_sodium__mean", "serum_potassium__mean", "serum_chloride__mean", "calcium__mean", "urea__mean"],
        ["white_blood_cell_count__last", "creatinine__last", "albumin__last", "hs_crp__max", "ldh__max"],
        ["platelet_count__last", "hemoglobin__last", "d_dimer__max", "ferritin__max", "interleukin_6__max"],
        ["record_count", "stay_days", "age", "white_blood_cell_count__mean", "creatinine__mean"],
    ]
    regression_feature_sets = [
        ["white_blood_cell_count__mean", "hemoglobin__mean", "platelet_count__mean", "creatinine__mean", "albumin__mean"],
        ["serum_sodium__mean", "serum_potassium__mean", "serum_chloride__mean", "calcium__mean", "urea__mean"],
        ["record_count", "stay_days", "age", "white_blood_cell_count__max", "hs_crp__max"],
        ["hemoglobin__last", "platelet_count__last", "creatinine__last", "albumin__last", "d_dimer__max"],
        ["age", "sex", "white_blood_cell_count__mean", "serum_sodium__mean", "creatinine__mean"],
    ]
    for index, features in enumerate(classification_feature_sets, start=41):
        metrics, coefficients, predictions = build_logistic_model(summary, features, "outcome", "patient_id")
        answer = {"model": "logistic_regression", **metrics, "feature_columns": features}
        task_brief = (
            f"Fit a deterministic logistic regression model to predict `outcome` from the patient-level TJH features `{', '.join(features)}`. "
            "Use the pre-defined split convention embedded in the benchmark metadata, train on `train+tuning`, evaluate on `held_out`, and write `metrics.json`, `coefficients.csv`, `predictions.csv`, and `answer.json`."
        )
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=index,
                split="eval",
                dataset="TJH",
                category="Modeling",
                task_family="modeling_or_report",
                task_type="modeling",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(index),
                artifacts={
                    "metrics.json": metrics,
                    "coefficients.csv": coefficients,
                    "predictions.csv": predictions,
                    "answer.json": answer,
                },
            )
        )
    for index, features in enumerate(regression_feature_sets, start=46):
        metrics, coefficients, predictions = build_linear_model(summary, features, "los_last", "patient_id")
        answer = {"model": "linear_regression", **metrics, "feature_columns": features}
        task_brief = (
            f"Fit a deterministic linear regression model to predict `los_last` from the patient-level TJH features `{', '.join(features)}`. "
            "Use the same fixed split convention, then write `metrics.json`, `coefficients.csv`, `predictions.csv`, and `answer.json`."
        )
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=index,
                split="eval",
                dataset="TJH",
                category="Report",
                task_family="modeling_or_report",
                task_type="report",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(index),
                artifacts={
                    "metrics.json": metrics,
                    "coefficients.csv": coefficients,
                    "predictions.csv": predictions,
                    "answer.json": answer,
                },
            )
        )
    return tasks


def build_mimic_medagentboard(views: MimicViews) -> list[GeneratedTask]:
    tasks: list[GeneratedTask] = []
    summary = views.subject_summary.copy()
    gt_root = GROUND_TRUTH_ROOT / "medagentboard"

    query_aliases = list(MIMIC_LABS.keys())[:9] + ["event_count"]
    for idx, alias in enumerate(query_aliases, start=51):
        if alias == "event_count":
            column = "event_count"
            threshold = int(summary[column].quantile(0.75))
            cohort = summary.loc[summary[column] >= threshold, ["subject_id", "split", column]].sort_values("subject_id")
            task_brief = (
                f"Build a MEDS subject cohort containing every subject with `event_count >= {threshold}`. "
                "Write `cohort.csv` with `subject_id`, `split`, and `event_count`, and `answer.json` with the threshold and cohort size."
            )
        else:
            column = feature_stat_column(alias, "mean")
            threshold = round(float(summary[column].dropna().quantile(0.75)), 6)
            cohort = summary.loc[summary[column] >= threshold, ["subject_id", "split", column]].sort_values("subject_id")
            description = MIMIC_LABS[alias][1]
            task_brief = (
                f"Compute the subject-level mean of `{alias}` (`{description}`) from the MEDS event stream and select subjects with `{column} >= {threshold}`. "
                "Write `cohort.csv` and `answer.json` with the threshold and subject count."
            )
        answer = {"threshold": threshold, "subject_count": int(len(cohort)), "subject_ids": cohort["subject_id"].astype(int).tolist()}
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=idx,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Data Querying",
                task_family="data_querying",
                task_type="data_querying",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(idx),
                artifacts={"cohort.csv": cohort, "answer.json": answer},
            )
        )

    stats_aliases = list(MIMIC_LABS.keys()) + ["event_count"]
    for idx, alias in enumerate(stats_aliases, start=61):
        mean_col = feature_stat_column(alias, "mean") if alias != "event_count" else "event_count"
        group = (
            summary.loc[:, ["split", "discharge_died", "discharge_home", mean_col]]
            .dropna()
            .groupby(["split", "discharge_died", "discharge_home"], as_index=False)
            .agg(
                subject_count=(mean_col, "size"),
                mean_feature=(mean_col, "mean"),
                median_feature=(mean_col, "median"),
            )
            .sort_values(["split", "discharge_died", "discharge_home"])
        )
        answer = {"feature": alias, "rows": int(len(group)), "sha256": dataframe_signature(group)}
        description = MIMIC_LABS[alias][1] if alias != "event_count" else "Total MEDS event count per subject"
        task_brief = (
            f"Create a split- and discharge-status summary table for the subject-level mean of `{alias}` (`{description}`). "
            "Write `summary.csv` and `answer.json`."
        )
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=idx,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Data Statistics",
                task_family="data_statistics",
                task_type="data_statistics",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(idx),
                artifacts={"summary.csv": group, "answer.json": answer},
            )
        )

    preprocess_groups = [
        ["potassium", "sodium", "chloride"],
        ["creatinine", "urea_nitrogen", "hematocrit"],
        ["platelets", "hemoglobin", "leukocytes"],
        ["potassium", "creatinine", "hemoglobin"],
        ["sodium", "chloride", "platelets"],
        ["urea_nitrogen", "hematocrit", "leukocytes"],
        ["creatinine", "platelets", "hemoglobin"],
        ["potassium", "sodium", "urea_nitrogen"],
        ["chloride", "hematocrit", "leukocytes"],
        ["sodium", "creatinine", "platelets"],
    ]
    for idx, aliases in enumerate(preprocess_groups, start=71):
        columns = [
            "subject_id",
            "split",
            "gender",
            "age_at_first_admission",
            "discharge_died",
            "discharge_home",
            "event_count",
            "hospital_admission_count",
            "icu_admission_count",
        ]
        for alias in aliases:
            columns.extend(
                [
                    feature_stat_column(alias, "count"),
                    feature_stat_column(alias, "first"),
                    feature_stat_column(alias, "last"),
                    feature_stat_column(alias, "mean"),
                ]
            )
        processed = summary.loc[:, columns].sort_values("subject_id").reset_index(drop=True)
        answer = {"rows": int(len(processed)), "columns": columns, "sha256": dataframe_signature(processed)}
        task_brief = (
            f"Build a subject-level MEDS feature table for `{', '.join(aliases)}` together with the fixed demographic and admission-count columns. "
            "Write `subject_features.csv` and `answer.json`."
        )
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=idx,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Data Preprocessing",
                task_family="data_preprocessing",
                task_type="data_preprocessing",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(idx),
                artifacts={"subject_features.csv": processed, "answer.json": answer},
            )
        )

    plot_aliases = list(MIMIC_LABS.keys()) + ["event_count"]
    for idx, alias in enumerate(plot_aliases, start=81):
        plot_path = gt_root / str(idx) / "plot.png"
        if alias == "event_count":
            edges, counts = plot_hist(plot_path, summary["event_count"], "MIMIC distribution: event_count")
            daily = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})
            description = "Total MEDS event count per subject"
            task_brief = (
                "Create a histogram of `event_count` across MEDS subjects. "
                "Write `plot_summary.csv`, `plot.png`, and `answer.json`."
            )
        else:
            daily = (
                views.temporal_daily.loc[views.temporal_daily["lab_alias"] == alias, ["split", "day", "daily_mean"]]
                .groupby(["split", "day"], as_index=False)
                .agg(daily_mean=("daily_mean", "mean"), subject_count=("daily_mean", "size"))
                .sort_values(["split", "day"])
            )
            plot_line(plot_path, daily, "day", "daily_mean", "split", f"MIMIC temporal trend: {alias}")
            description = MIMIC_LABS[alias][1]
            task_brief = (
                f"Create a split-stratified temporal line plot for daily mean `{alias}` (`{description}`) from the MEDS event stream. "
                "Write `plot_summary.csv`, `plot.png`, and `answer.json`."
            )
        answer = {"feature": alias, "rows": int(len(daily)), "sha256": dataframe_signature(daily)}
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=idx,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Visualization",
                task_family="visualization",
                task_type="visualization",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(idx),
                artifacts={"plot_summary.csv": daily, "plot.png": plot_path, "answer.json": answer},
            )
        )

    mimic_classification_sets = [
        ["potassium__mean", "sodium__mean", "creatinine__mean", "chloride__mean", "urea_nitrogen__mean"],
        ["hematocrit__mean", "platelets__mean", "hemoglobin__mean", "leukocytes__mean", "event_count"],
        ["potassium__last", "sodium__last", "creatinine__last", "chloride__last", "urea_nitrogen__last"],
        ["hematocrit__last", "platelets__last", "hemoglobin__last", "leukocytes__last", "icu_admission_count"],
        ["age_at_first_admission", "event_count", "hospital_admission_count", "icu_admission_count", "diagnosis_event_count"],
    ]
    mimic_regression_sets = [
        ["potassium__mean", "sodium__mean", "creatinine__mean", "chloride__mean", "urea_nitrogen__mean"],
        ["hematocrit__mean", "platelets__mean", "hemoglobin__mean", "leukocytes__mean", "event_count"],
        ["potassium__last", "sodium__last", "creatinine__last", "chloride__last", "urea_nitrogen__last"],
        ["age_at_first_admission", "event_count", "hospital_admission_count", "icu_admission_count", "diagnosis_event_count"],
        ["discharge_died", "discharge_home", "event_count", "transfer_count", "procedure_event_count"],
    ]
    for idx, features in enumerate(mimic_classification_sets, start=91):
        metrics, coefficients, predictions = build_logistic_model(summary, features, "discharge_died", "subject_id")
        answer = {"model": "logistic_regression", "target": "discharge_died", **metrics, "feature_columns": features}
        task_brief = (
            f"Fit a deterministic logistic regression model to predict `discharge_died` from `{', '.join(features)}`. "
            "Use the published MEDS `train`, `tuning`, and `held_out` splits exactly, and write `metrics.json`, `coefficients.csv`, `predictions.csv`, and `answer.json`."
        )
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=idx,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Modeling",
                task_family="modeling_or_report",
                task_type="modeling",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(idx),
                artifacts={
                    "metrics.json": metrics,
                    "coefficients.csv": coefficients,
                    "predictions.csv": predictions,
                    "answer.json": answer,
                },
            )
        )
    for idx, features in enumerate(mimic_regression_sets, start=96):
        metrics, coefficients, predictions = build_linear_model(summary, features, "event_count", "subject_id")
        answer = {"model": "linear_regression", "target": "event_count", **metrics, "feature_columns": features}
        task_brief = (
            f"Fit a deterministic linear regression model to predict `event_count` from `{', '.join(features)}`. "
            "Use the MEDS split file as-is and write `metrics.json`, `coefficients.csv`, `predictions.csv`, and `answer.json`."
        )
        tasks.append(
            make_task_record(
                benchmark="medagentboard",
                qid=idx,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Report",
                task_family="modeling_or_report",
                task_type="report",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(idx),
                artifacts={
                    "metrics.json": metrics,
                    "coefficients.csv": coefficients,
                    "predictions.csv": predictions,
                    "answer.json": answer,
                },
            )
        )
    return tasks


def build_tjh_ehrflowbench(views: TjhViews) -> tuple[list[GeneratedTask], list[GeneratedTask]]:
    eval_tasks: list[GeneratedTask] = []
    train_tasks: list[GeneratedTask] = []
    summary = views.patient_summary.copy()
    gt_root = GROUND_TRUTH_ROOT / "ehrflowbench"
    qid = 1
    train_qid = 101

    cohort_plan = [
        ("white_blood_cell_count", "max", 0.75),
        ("hemoglobin", "min", 0.25),
        ("platelet_count", "min", 0.25),
        ("creatinine", "mean", 0.75),
        ("albumin", "mean", 0.25),
        ("serum_potassium", "mean", 0.75),
        ("serum_sodium", "mean", 0.25),
        ("serum_chloride", "mean", 0.25),
        ("hs_crp", "max", 0.75),
        ("ldh", "max", 0.75),
    ]
    for alias, stat, quantile in cohort_plan:
        column = feature_stat_column(alias, stat)
        threshold = round(float(summary[column].dropna().quantile(quantile)), 6)
        comparator = ">=" if quantile >= 0.5 else "<="
        cohort = summary.loc[summary[column] >= threshold if comparator == ">=" else summary[column] <= threshold, ["patient_id", "split", "outcome", column]].sort_values("patient_id")
        answer = {"threshold": threshold, "comparator": comparator, "patient_count": int(len(cohort)), "sha256": dataframe_signature(cohort)}
        task_brief = (
            f"Translate the longitudinal TJH table into a patient cohort defined by the patient-level `{stat}` of `{alias}` being `{comparator} {threshold}`. "
            "Write `cohort.csv` and `answer.json`."
        )
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="TJH",
                category="Cohort Query",
                task_family="cohort_query",
                task_type="cohort_query",
                task_brief=task_brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"cohort.csv": cohort, "answer.json": answer},
            )
        )
        qid += 1

    train_cohort_alias = "d_dimer"
    train_cohort_column = feature_stat_column(train_cohort_alias, "max")
    train_threshold = round(float(summary[train_cohort_column].dropna().quantile(0.75)), 6)
    train_cohort = summary.loc[summary[train_cohort_column] >= train_threshold, ["patient_id", "split", "outcome", train_cohort_column]].sort_values("patient_id")
    train_tasks.append(
        make_task_record(
            benchmark="ehrflowbench",
            qid=train_qid,
            split="train",
            dataset="TJH",
            category="Cohort Query",
            task_family="cohort_query",
            task_type="cohort_query",
            task_brief=(
                f"Build a training-example patient cohort where the patient-level max `d_dimer` is `>= {train_threshold}`. "
                "Write `cohort.csv` and `answer.json`."
            ),
            answer_payload={"threshold": train_threshold, "patient_count": int(len(train_cohort)), "sha256": dataframe_signature(train_cohort)},
            ground_truth_dir=gt_root / str(train_qid),
            artifacts={"cohort.csv": train_cohort, "answer.json": {"threshold": train_threshold, "patient_count": int(len(train_cohort)), "sha256": dataframe_signature(train_cohort)}},
        )
    )
    train_qid += 1

    temporal_aliases = ["white_blood_cell_count", "hemoglobin", "platelet_count", "creatinine", "albumin", "serum_sodium", "serum_potassium", "hs_crp", "ldh", "d_dimer"]
    for alias in temporal_aliases:
        feature_name = next(name for name, mapped_alias in TJH_ALIAS.items() if mapped_alias == alias)
        summary_frame = (
            views.temporal_daily.loc[:, ["day", "split", "outcome", feature_name]]
            .rename(columns={feature_name: "daily_mean"})
            .dropna()
            .groupby(["split", "outcome", "day"], as_index=False)
            .agg(daily_mean=("daily_mean", "mean"), patient_count=("daily_mean", "size"))
            .sort_values(["split", "outcome", "day"])
        )
        answer = {"feature": alias, "rows": int(len(summary_frame)), "sha256": dataframe_signature(summary_frame)}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="TJH",
                category="Temporal Statistics",
                task_family="temporal_statistics",
                task_type="temporal_statistics",
                task_brief=(
                    f"Summarize the daily temporal trajectory of `{alias}` across TJH patients, stratified by split and outcome. "
                    "Write `summary.csv` and `answer.json`."
                ),
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"summary.csv": summary_frame, "answer.json": answer},
            )
        )
        qid += 1

    train_temporal_alias = "serum_chloride"
    train_feature_name = next(name for name, mapped_alias in TJH_ALIAS.items() if mapped_alias == train_temporal_alias)
    train_temporal = (
        views.temporal_daily.loc[:, ["day", "split", "outcome", train_feature_name]]
        .rename(columns={train_feature_name: "daily_mean"})
        .dropna()
        .groupby(["split", "outcome", "day"], as_index=False)
        .agg(daily_mean=("daily_mean", "mean"), patient_count=("daily_mean", "size"))
    )
    train_tasks.append(
        make_task_record(
            benchmark="ehrflowbench",
            qid=train_qid,
            split="train",
            dataset="TJH",
            category="Temporal Statistics",
            task_family="temporal_statistics",
            task_type="temporal_statistics",
            task_brief=f"Summarize the daily temporal trajectory of `{train_temporal_alias}` across TJH patients. Write `summary.csv` and `answer.json`.",
            answer_payload={"feature": train_temporal_alias, "rows": int(len(train_temporal)), "sha256": dataframe_signature(train_temporal)},
            ground_truth_dir=gt_root / str(train_qid),
            artifacts={"summary.csv": train_temporal, "answer.json": {"feature": train_temporal_alias, "rows": int(len(train_temporal)), "sha256": dataframe_signature(train_temporal)}},
        )
    )
    train_qid += 1

    preprocess_groups = [
        ["white_blood_cell_count", "hemoglobin", "platelet_count"],
        ["creatinine", "albumin", "serum_sodium"],
        ["serum_potassium", "serum_chloride", "calcium"],
        ["urea", "hs_crp", "ldh"],
        ["d_dimer", "ferritin", "interleukin_6"],
        ["white_blood_cell_count", "creatinine", "albumin"],
        ["hemoglobin", "platelet_count", "serum_sodium"],
        ["serum_potassium", "urea", "hs_crp"],
        ["serum_chloride", "calcium", "ldh"],
        ["white_blood_cell_count", "d_dimer", "albumin"],
    ]
    for aliases in preprocess_groups:
        columns = ["patient_id", "split", "outcome", "age", "sex", "los_last", "record_count"]
        for alias in aliases:
            columns.extend(
                [
                    feature_stat_column(alias, "count"),
                    feature_stat_column(alias, "first"),
                    feature_stat_column(alias, "last"),
                    feature_stat_column(alias, "mean"),
                    feature_stat_column(alias, "early3_mean"),
                ]
            )
        frame = summary.loc[:, columns].sort_values("patient_id").reset_index(drop=True)
        answer = {"rows": int(len(frame)), "columns": columns, "sha256": dataframe_signature(frame)}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="TJH",
                category="Preprocessing Feature Engineering",
                task_family="preprocessing_feature_engineering",
                task_type="preprocessing_feature_engineering",
                task_brief=(
                    f"Build a TJH patient-level feature matrix for `{', '.join(aliases)}` using first, last, mean, early-3-day mean, and count statistics. "
                    "Write `patient_features.csv` and `answer.json`."
                ),
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"patient_features.csv": frame, "answer.json": answer},
            )
        )
        qid += 1

    train_preprocess_aliases = ["creatinine", "albumin", "hs_crp"]
    train_columns = ["patient_id", "split", "outcome", "age", "sex", "los_last", "record_count"]
    for alias in train_preprocess_aliases:
        train_columns.extend(
            [
                feature_stat_column(alias, "count"),
                feature_stat_column(alias, "first"),
                feature_stat_column(alias, "last"),
                feature_stat_column(alias, "mean"),
                feature_stat_column(alias, "early3_mean"),
            ]
        )
    train_frame = summary.loc[:, train_columns].sort_values("patient_id").reset_index(drop=True)
    train_tasks.append(
        make_task_record(
            benchmark="ehrflowbench",
            qid=train_qid,
            split="train",
            dataset="TJH",
            category="Preprocessing Feature Engineering",
            task_family="preprocessing_feature_engineering",
            task_type="preprocessing_feature_engineering",
            task_brief="Build a TJH patient-level feature matrix for `creatinine`, `albumin`, and `hs_crp`. Write `patient_features.csv` and `answer.json`.",
            answer_payload={"rows": int(len(train_frame)), "columns": train_columns, "sha256": dataframe_signature(train_frame)},
            ground_truth_dir=gt_root / str(train_qid),
            artifacts={"patient_features.csv": train_frame, "answer.json": {"rows": int(len(train_frame)), "columns": train_columns, "sha256": dataframe_signature(train_frame)}},
        )
    )
    train_qid += 1

    artifact_aliases = ["white_blood_cell_count", "hemoglobin", "platelet_count", "creatinine", "albumin", "serum_sodium", "serum_potassium", "hs_crp", "ldh", "d_dimer"]
    for alias in artifact_aliases:
        values = summary[feature_stat_column(alias, "mean")]
        hist_path = gt_root / str(qid) / "plot.png"
        edges, counts = plot_hist(hist_path, values, f"TJH distribution: {alias}")
        plot_summary = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})
        answer = {"feature": alias, "rows": int(len(plot_summary)), "sha256": dataframe_signature(plot_summary)}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="TJH",
                category="Artifact Generation",
                task_family="artifact_generation",
                task_type="artifact_generation",
                task_brief=(
                    f"Create a histogram of the patient-level mean `{alias}` values in TJH. "
                    "Write `plot_summary.csv`, save `plot.png`, and write `answer.json`."
                ),
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"plot_summary.csv": plot_summary, "plot.png": hist_path, "answer.json": answer},
            )
        )
        qid += 1

    train_hist_alias = "urea"
    train_hist_path = gt_root / str(train_qid) / "plot.png"
    train_edges, train_counts = plot_hist(train_hist_path, summary[feature_stat_column(train_hist_alias, "mean")], f"TJH distribution: {train_hist_alias}")
    train_hist_frame = pd.DataFrame({"bin_left": train_edges[:-1], "bin_right": train_edges[1:], "count": train_counts})
    train_tasks.append(
        make_task_record(
            benchmark="ehrflowbench",
            qid=train_qid,
            split="train",
            dataset="TJH",
            category="Artifact Generation",
            task_family="artifact_generation",
            task_type="artifact_generation",
            task_brief=f"Create a histogram of the patient-level mean `{train_hist_alias}` values in TJH. Write `plot_summary.csv`, `plot.png`, and `answer.json`.",
            answer_payload={"feature": train_hist_alias, "rows": int(len(train_hist_frame)), "sha256": dataframe_signature(train_hist_frame)},
            ground_truth_dir=gt_root / str(train_qid),
            artifacts={"plot_summary.csv": train_hist_frame, "plot.png": train_hist_path, "answer.json": {"feature": train_hist_alias, "rows": int(len(train_hist_frame)), "sha256": dataframe_signature(train_hist_frame)}},
        )
    )
    train_qid += 1

    logistic_sets = [
        ["white_blood_cell_count__mean", "hemoglobin__mean", "platelet_count__mean", "creatinine__mean", "albumin__mean"],
        ["serum_sodium__mean", "serum_potassium__mean", "serum_chloride__mean", "calcium__mean", "urea__mean"],
        ["white_blood_cell_count__last", "creatinine__last", "albumin__last", "hs_crp__max", "ldh__max"],
        ["platelet_count__last", "hemoglobin__last", "d_dimer__max", "ferritin__max", "interleukin_6__max"],
        ["record_count", "stay_days", "age", "white_blood_cell_count__mean", "creatinine__mean"],
    ]
    linear_sets = [
        ["white_blood_cell_count__mean", "hemoglobin__mean", "platelet_count__mean", "creatinine__mean", "albumin__mean"],
        ["serum_sodium__mean", "serum_potassium__mean", "serum_chloride__mean", "calcium__mean", "urea__mean"],
        ["record_count", "stay_days", "age", "white_blood_cell_count__max", "hs_crp__max"],
        ["hemoglobin__last", "platelet_count__last", "creatinine__last", "albumin__last", "d_dimer__max"],
        ["age", "sex", "white_blood_cell_count__mean", "serum_sodium__mean", "creatinine__mean"],
    ]
    for features in logistic_sets:
        metrics, coefficients, predictions = build_logistic_model(summary, features, "outcome", "patient_id")
        answer = {"model": "logistic_regression", "target": "outcome", **metrics, "feature_columns": features}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="TJH",
                category="Fixed Spec Modeling",
                task_family="fixed_spec_modeling",
                task_type="fixed_spec_modeling",
                task_brief=(
                    f"Train a deterministic logistic regression model to predict `outcome` from `{', '.join(features)}` using the benchmark split convention. "
                    "Write `metrics.json`, `coefficients.csv`, `predictions.csv`, and `answer.json`."
                ),
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"metrics.json": metrics, "coefficients.csv": coefficients, "predictions.csv": predictions, "answer.json": answer},
            )
        )
        qid += 1
    for features in linear_sets:
        metrics, coefficients, predictions = build_linear_model(summary, features, "los_last", "patient_id")
        answer = {"model": "linear_regression", "target": "los_last", **metrics, "feature_columns": features}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="TJH",
                category="Fixed Spec Modeling",
                task_family="fixed_spec_modeling",
                task_type="fixed_spec_modeling",
                task_brief=(
                    f"Train a deterministic linear regression model to predict `los_last` from `{', '.join(features)}` using the benchmark split convention. "
                    "Write `metrics.json`, `coefficients.csv`, `predictions.csv`, and `answer.json`."
                ),
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"metrics.json": metrics, "coefficients.csv": coefficients, "predictions.csv": predictions, "answer.json": answer},
            )
        )
        qid += 1

    train_features = ["white_blood_cell_count__mean", "hemoglobin__mean", "platelet_count__mean", "creatinine__mean", "albumin__mean"]
    train_metrics, train_coeffs, train_preds = build_logistic_model(summary, train_features, "outcome", "patient_id")
    train_answer = {"model": "logistic_regression", "target": "outcome", **train_metrics, "feature_columns": train_features}
    train_tasks.append(
        make_task_record(
            benchmark="ehrflowbench",
            qid=train_qid,
            split="train",
            dataset="TJH",
            category="Fixed Spec Modeling",
            task_family="fixed_spec_modeling",
            task_type="fixed_spec_modeling",
            task_brief="Train a deterministic logistic regression model to predict `outcome` from the baseline five TJH features. Write `metrics.json`, `coefficients.csv`, `predictions.csv`, and `answer.json`.",
            answer_payload=train_answer,
            ground_truth_dir=gt_root / str(train_qid),
            artifacts={"metrics.json": train_metrics, "coefficients.csv": train_coeffs, "predictions.csv": train_preds, "answer.json": train_answer},
        )
    )
    return eval_tasks, train_tasks


def build_mimic_ehrflowbench(views: MimicViews) -> tuple[list[GeneratedTask], list[GeneratedTask]]:
    eval_tasks: list[GeneratedTask] = []
    train_tasks: list[GeneratedTask] = []
    summary = views.subject_summary.copy()
    gt_root = GROUND_TRUTH_ROOT / "ehrflowbench"
    qid = 51
    train_qid = 106

    cohort_aliases = list(MIMIC_LABS.keys())[:9] + ["event_count"]
    for alias in cohort_aliases:
        if alias == "event_count":
            column = "event_count"
            threshold = int(summary[column].quantile(0.75))
            cohort = summary.loc[summary[column] >= threshold, ["subject_id", "split", column]].sort_values("subject_id")
            brief = f"Build a subject cohort with `event_count >= {threshold}`. Write `cohort.csv` and `answer.json`."
        else:
            column = feature_stat_column(alias, "mean")
            threshold = round(float(summary[column].dropna().quantile(0.75)), 6)
            cohort = summary.loc[summary[column] >= threshold, ["subject_id", "split", "discharge_died", column]].sort_values("subject_id")
            brief = f"Build a subject cohort where `{column} >= {threshold}` for the MEDS lab alias `{alias}`. Write `cohort.csv` and `answer.json`."
        answer = {"threshold": threshold, "subject_count": int(len(cohort)), "sha256": dataframe_signature(cohort)}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Cohort Query",
                task_family="cohort_query",
                task_type="cohort_query",
                task_brief=brief,
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"cohort.csv": cohort, "answer.json": answer},
            )
        )
        qid += 1

    train_cohort = summary.loc[summary["discharge_died"] == 1, ["subject_id", "split", "discharge_died"]].sort_values("subject_id")
    train_tasks.append(
        make_task_record(
            benchmark="ehrflowbench",
            qid=train_qid,
            split="train",
            dataset="MIMIC-IV-demo-MEDS",
            category="Cohort Query",
            task_family="cohort_query",
            task_type="cohort_query",
            task_brief="Build the discharge-death cohort from the MEDS demo. Write `cohort.csv` and `answer.json`.",
            answer_payload={"subject_count": int(len(train_cohort)), "sha256": dataframe_signature(train_cohort)},
            ground_truth_dir=gt_root / str(train_qid),
            artifacts={"cohort.csv": train_cohort, "answer.json": {"subject_count": int(len(train_cohort)), "sha256": dataframe_signature(train_cohort)}},
        )
    )
    train_qid += 1

    temporal_aliases = list(MIMIC_LABS.keys()) + ["all_numeric_events"]
    for alias in temporal_aliases:
        if alias == "all_numeric_events":
            summary_frame = (
                views.numeric_events.assign(day=lambda frame: (frame["time"] - frame.groupby("subject_id")["time"].transform("min")).dt.total_seconds() / 86400.0)
                .assign(day=lambda frame: frame["day"].floordiv(1).astype(int))
                .groupby(["split", "day"], as_index=False)
                .agg(daily_mean=("numeric_value", "mean"), subject_count=("numeric_value", "size"))
                .sort_values(["split", "day"])
            )
        else:
            summary_frame = (
                views.temporal_daily.loc[views.temporal_daily["lab_alias"] == alias, ["split", "day", "daily_mean"]]
                .groupby(["split", "day"], as_index=False)
                .agg(daily_mean=("daily_mean", "mean"), subject_count=("daily_mean", "size"))
                .sort_values(["split", "day"])
            )
        answer = {"feature": alias, "rows": int(len(summary_frame)), "sha256": dataframe_signature(summary_frame)}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Temporal Statistics",
                task_family="temporal_statistics",
                task_type="temporal_statistics",
                task_brief=f"Summarize the split-stratified temporal trajectory of `{alias}`. Write `summary.csv` and `answer.json`.",
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"summary.csv": summary_frame, "answer.json": answer},
            )
        )
        qid += 1

    train_temporal_alias = "leukocytes"
    train_temporal = (
        views.temporal_daily.loc[views.temporal_daily["lab_alias"] == train_temporal_alias, ["split", "day", "daily_mean"]]
        .groupby(["split", "day"], as_index=False)
        .agg(daily_mean=("daily_mean", "mean"), subject_count=("daily_mean", "size"))
    )
    train_tasks.append(
        make_task_record(
            benchmark="ehrflowbench",
            qid=train_qid,
            split="train",
            dataset="MIMIC-IV-demo-MEDS",
            category="Temporal Statistics",
            task_family="temporal_statistics",
            task_type="temporal_statistics",
            task_brief="Summarize the split-stratified temporal trajectory of `leukocytes`. Write `summary.csv` and `answer.json`.",
            answer_payload={"feature": train_temporal_alias, "rows": int(len(train_temporal)), "sha256": dataframe_signature(train_temporal)},
            ground_truth_dir=gt_root / str(train_qid),
            artifacts={"summary.csv": train_temporal, "answer.json": {"feature": train_temporal_alias, "rows": int(len(train_temporal)), "sha256": dataframe_signature(train_temporal)}},
        )
    )
    train_qid += 1

    preprocess_groups = [
        ["potassium", "sodium", "chloride"],
        ["creatinine", "urea_nitrogen", "hematocrit"],
        ["platelets", "hemoglobin", "leukocytes"],
        ["potassium", "creatinine", "hemoglobin"],
        ["sodium", "chloride", "platelets"],
        ["urea_nitrogen", "hematocrit", "leukocytes"],
        ["creatinine", "platelets", "hemoglobin"],
        ["potassium", "sodium", "urea_nitrogen"],
        ["chloride", "hematocrit", "leukocytes"],
        ["sodium", "creatinine", "platelets"],
    ]
    for aliases in preprocess_groups:
        columns = [
            "subject_id",
            "split",
            "gender",
            "age_at_first_admission",
            "discharge_died",
            "discharge_home",
            "event_count",
            "hospital_admission_count",
            "icu_admission_count",
        ]
        for alias in aliases:
            columns.extend(
                [
                    feature_stat_column(alias, "count"),
                    feature_stat_column(alias, "first"),
                    feature_stat_column(alias, "last"),
                    feature_stat_column(alias, "mean"),
                ]
            )
        frame = summary.loc[:, columns].sort_values("subject_id").reset_index(drop=True)
        answer = {"rows": int(len(frame)), "columns": columns, "sha256": dataframe_signature(frame)}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Preprocessing Feature Engineering",
                task_family="preprocessing_feature_engineering",
                task_type="preprocessing_feature_engineering",
                task_brief=f"Build a subject-level MEDS feature matrix for `{', '.join(aliases)}`. Write `subject_features.csv` and `answer.json`.",
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"subject_features.csv": frame, "answer.json": answer},
            )
        )
        qid += 1

    train_group = ["creatinine", "hemoglobin", "leukocytes"]
    train_columns = [
        "subject_id",
        "split",
        "gender",
        "age_at_first_admission",
        "discharge_died",
        "discharge_home",
        "event_count",
        "hospital_admission_count",
        "icu_admission_count",
    ]
    for alias in train_group:
        train_columns.extend(
            [
                feature_stat_column(alias, "count"),
                feature_stat_column(alias, "first"),
                feature_stat_column(alias, "last"),
                feature_stat_column(alias, "mean"),
            ]
        )
    train_frame = summary.loc[:, train_columns].sort_values("subject_id").reset_index(drop=True)
    train_tasks.append(
        make_task_record(
            benchmark="ehrflowbench",
            qid=train_qid,
            split="train",
            dataset="MIMIC-IV-demo-MEDS",
            category="Preprocessing Feature Engineering",
            task_family="preprocessing_feature_engineering",
            task_type="preprocessing_feature_engineering",
            task_brief="Build a subject-level MEDS feature matrix for `creatinine`, `hemoglobin`, and `leukocytes`. Write `subject_features.csv` and `answer.json`.",
            answer_payload={"rows": int(len(train_frame)), "columns": train_columns, "sha256": dataframe_signature(train_frame)},
            ground_truth_dir=gt_root / str(train_qid),
            artifacts={"subject_features.csv": train_frame, "answer.json": {"rows": int(len(train_frame)), "columns": train_columns, "sha256": dataframe_signature(train_frame)}},
        )
    )
    train_qid += 1

    artifact_aliases = list(MIMIC_LABS.keys()) + ["event_count"]
    for alias in artifact_aliases:
        hist_path = gt_root / str(qid) / "plot.png"
        values = summary["event_count"] if alias == "event_count" else summary[feature_stat_column(alias, "mean")]
        edges, counts = plot_hist(hist_path, values, f"MIMIC distribution: {alias}")
        plot_summary = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})
        answer = {"feature": alias, "rows": int(len(plot_summary)), "sha256": dataframe_signature(plot_summary)}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Artifact Generation",
                task_family="artifact_generation",
                task_type="artifact_generation",
                task_brief=f"Create a histogram of subject-level mean `{alias}` values in the MEDS demo. Write `plot_summary.csv`, `plot.png`, and `answer.json`.",
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"plot_summary.csv": plot_summary, "plot.png": hist_path, "answer.json": answer},
            )
        )
        qid += 1

    train_hist_alias = "potassium"
    train_hist_path = gt_root / str(train_qid) / "plot.png"
    train_edges, train_counts = plot_hist(train_hist_path, summary[feature_stat_column(train_hist_alias, "mean")], f"MIMIC distribution: {train_hist_alias}")
    train_hist_frame = pd.DataFrame({"bin_left": train_edges[:-1], "bin_right": train_edges[1:], "count": train_counts})
    train_tasks.append(
        make_task_record(
            benchmark="ehrflowbench",
            qid=train_qid,
            split="train",
            dataset="MIMIC-IV-demo-MEDS",
            category="Artifact Generation",
            task_family="artifact_generation",
            task_type="artifact_generation",
            task_brief="Create a histogram of subject-level mean `potassium` values in the MEDS demo. Write `plot_summary.csv`, `plot.png`, and `answer.json`.",
            answer_payload={"feature": train_hist_alias, "rows": int(len(train_hist_frame)), "sha256": dataframe_signature(train_hist_frame)},
            ground_truth_dir=gt_root / str(train_qid),
            artifacts={"plot_summary.csv": train_hist_frame, "plot.png": train_hist_path, "answer.json": {"feature": train_hist_alias, "rows": int(len(train_hist_frame)), "sha256": dataframe_signature(train_hist_frame)}},
        )
    )
    train_qid += 1

    logistic_sets = [
        ["potassium__mean", "sodium__mean", "creatinine__mean", "chloride__mean", "urea_nitrogen__mean"],
        ["hematocrit__mean", "platelets__mean", "hemoglobin__mean", "leukocytes__mean", "event_count"],
        ["potassium__last", "sodium__last", "creatinine__last", "chloride__last", "urea_nitrogen__last"],
        ["hematocrit__last", "platelets__last", "hemoglobin__last", "leukocytes__last", "icu_admission_count"],
        ["age_at_first_admission", "event_count", "hospital_admission_count", "icu_admission_count", "diagnosis_event_count"],
    ]
    linear_sets = [
        ["potassium__mean", "sodium__mean", "creatinine__mean", "chloride__mean", "urea_nitrogen__mean"],
        ["hematocrit__mean", "platelets__mean", "hemoglobin__mean", "leukocytes__mean", "event_count"],
        ["potassium__last", "sodium__last", "creatinine__last", "chloride__last", "urea_nitrogen__last"],
        ["age_at_first_admission", "event_count", "hospital_admission_count", "icu_admission_count", "diagnosis_event_count"],
        ["discharge_died", "discharge_home", "event_count", "transfer_count", "procedure_event_count"],
    ]
    for features in logistic_sets:
        metrics, coefficients, predictions = build_logistic_model(summary, features, "discharge_died", "subject_id")
        answer = {"model": "logistic_regression", "target": "discharge_died", **metrics, "feature_columns": features}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Fixed Spec Modeling",
                task_family="fixed_spec_modeling",
                task_type="fixed_spec_modeling",
                task_brief=f"Train a deterministic logistic regression model to predict `discharge_died` from `{', '.join(features)}` using the MEDS split file exactly. Write `metrics.json`, `coefficients.csv`, `predictions.csv`, and `answer.json`.",
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"metrics.json": metrics, "coefficients.csv": coefficients, "predictions.csv": predictions, "answer.json": answer},
            )
        )
        qid += 1
    for features in linear_sets:
        metrics, coefficients, predictions = build_linear_model(summary, features, "event_count", "subject_id")
        answer = {"model": "linear_regression", "target": "event_count", **metrics, "feature_columns": features}
        eval_tasks.append(
            make_task_record(
                benchmark="ehrflowbench",
                qid=qid,
                split="eval",
                dataset="MIMIC-IV-demo-MEDS",
                category="Fixed Spec Modeling",
                task_family="fixed_spec_modeling",
                task_type="fixed_spec_modeling",
                task_brief=f"Train a deterministic linear regression model to predict `event_count` from `{', '.join(features)}` using the MEDS split file exactly. Write `metrics.json`, `coefficients.csv`, `predictions.csv`, and `answer.json`.",
                answer_payload=answer,
                ground_truth_dir=gt_root / str(qid),
                artifacts={"metrics.json": metrics, "coefficients.csv": coefficients, "predictions.csv": predictions, "answer.json": answer},
            )
        )
        qid += 1

    train_features = ["potassium__mean", "sodium__mean", "creatinine__mean", "chloride__mean", "urea_nitrogen__mean"]
    train_metrics, train_coeffs, train_preds = build_logistic_model(summary, train_features, "discharge_died", "subject_id")
    train_answer = {"model": "logistic_regression", "target": "discharge_died", **train_metrics, "feature_columns": train_features}
    train_tasks.append(
        make_task_record(
            benchmark="ehrflowbench",
            qid=train_qid,
            split="train",
            dataset="MIMIC-IV-demo-MEDS",
            category="Fixed Spec Modeling",
            task_family="fixed_spec_modeling",
            task_type="fixed_spec_modeling",
            task_brief="Train a deterministic logistic regression model to predict `discharge_died` from the baseline five MEDS lab features. Write `metrics.json`, `coefficients.csv`, `predictions.csv`, and `answer.json`.",
            answer_payload=train_answer,
            ground_truth_dir=gt_root / str(train_qid),
            artifacts={"metrics.json": train_metrics, "coefficients.csv": train_coeffs, "predictions.csv": train_preds, "answer.json": train_answer},
        )
    )
    return eval_tasks, train_tasks


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(round_float(row), ensure_ascii=False) + "\n")


def write_docs(
    provenance_summary: dict[str, Any],
    tjh_views: TjhViews,
    mimic_views: MimicViews,
    medagentboard_rows: list[dict[str, Any]],
    ehr_eval_rows: list[dict[str, Any]],
    ehr_train_rows: list[dict[str, Any]],
) -> None:
    medagentboard_dataset_counts = pd.Series([row["dataset"] for row in medagentboard_rows]).value_counts().to_dict()
    ehr_dataset_counts = pd.Series([row["dataset"] for row in ehr_eval_rows]).value_counts().to_dict()
    reproducibility_md = f"""# Benchmark Reproducibility

This benchmark rebuild is generated by `scripts/evaluation/rebuild_benchmarks.py`.

## Datasets

- `TJH.csv`: {len(tjh_views.raw):,} rows, {tjh_views.patient_summary['patient_id'].nunique()} patients, split counts {tjh_views.split_counts}
- `mimic_iv_demo_meds`: {len(mimic_views.raw):,} events, {mimic_views.subject_summary['subject_id'].nunique()} subjects, split counts {mimic_views.split_counts}

The open MEDS snapshot is checked into `data/mimic_iv_demo_meds/` together with the upstream checksum files and license.

## Canonical files

- `data/medagentboard.jsonl`: {len(medagentboard_rows)} tasks with dataset counts {medagentboard_dataset_counts}
- `data/ehrflowbench.jsonl`: {len(ehr_eval_rows)} eval tasks with dataset counts {ehr_dataset_counts}
- `data/ehrflowbench_train.jsonl`: {len(ehr_train_rows)} train tasks

All canonical benchmark rows include:

- deterministic output contracts
- machine-readable `verification_spec`
- `ground_truth_ref` folders under `data/{BENCHMARK_ROOT_NAME}/`
- dataset provenance metadata

## Paper audit

The original paper pipeline is preserved as provenance only:

- filtered rows: {provenance_summary['filtered_rows']}
- markdown directories: {provenance_summary['markdown_directories']}
- selected ids: {provenance_summary['selected_ids']}
- extracted tasks: {provenance_summary['extracted_tasks']}
- extracted task decisions: {provenance_summary['recommended_action_counts']}
- extracted task reason counts: {provenance_summary['reason_code_counts']}
- filter-only rows without markdown: {provenance_summary['filter_only_rows_without_markdown']}

The extracted-task audit lives in `data/{PROVENANCE_ROOT_NAME}/extracted_task_audit.csv`.

## Scoring

Canonical scoring is deterministic. The evaluator compares required output files against the committed ground-truth artifacts with exact or tolerance-bounded comparisons. LLM-judge scoring is no longer required for EHRFlowBench or MedAgentBoard.
"""
    text_dump(HEALTHFLOW_ROOT / "BENCHMARK_REPRODUCIBILITY.md", reproducibility_md)

    rebuttal_md = f"""# Benchmark Audit

## Current benchmark failures addressed

- The canonical `medagentboard.jsonl` now has {provenance_summary['current_medagentboard_blank_answers']} blank answers and {provenance_summary['current_medagentboard_inaccessible_paths']} inaccessible path references. Legacy blank-answer items and credential-bound path references were rebuilt or dropped before inclusion.
- The previous EHRFlowBench files inherited paper-extracted prompts directly; {provenance_summary['current_ehrflowbench_flagged_unverifiable_or_inaccessible']} of the 110 current eval+train rows were flagged as non-verifiable, inaccessible, or paper-metric-only during the rebuild audit.
- The paper audit reviewed {provenance_summary['extracted_tasks']} extracted tasks from {provenance_summary['selected_ids']} selected papers. {provenance_summary['recommended_action_counts']} and reason counts {provenance_summary['reason_code_counts']} are recorded in the provenance manifest.
- The filtering corpus contains {provenance_summary['filtered_rows']} rows but only {provenance_summary['markdown_directories']} local markdown directories. IDs {provenance_summary['filter_only_rows_without_markdown']} are now explicitly quarantined in the provenance manifest.

## Rebuild policy

- Canonical tasks only depend on free-to-use local data in this repo.
- All canonical tasks require structured outputs and deterministic verification.
- The paper markdown corpus is retained as an audited provenance input, not as canonical benchmark ground truth.

## Generated artifacts

- `data/{PROVENANCE_ROOT_NAME}/paper_provenance_manifest.json`
- `data/{PROVENANCE_ROOT_NAME}/filter_markdown_reconciliation.json`
- `data/{PROVENANCE_ROOT_NAME}/extracted_task_audit.csv`
- `data/{BENCHMARK_ROOT_NAME}/`

## Dataset balance

- MedAgentBoard: {medagentboard_dataset_counts}
- EHRFlowBench eval: {ehr_dataset_counts}
- TJH split counts: {tjh_views.split_counts}
- MIMIC demo MEDS split counts: {mimic_views.split_counts}
"""
    text_dump(WORKSPACE_ROOT / "rebuttal_202603" / "benchmark_audit.md", rebuttal_md)


def validate_rows(medagentboard_rows: list[dict[str, Any]], ehr_eval_rows: list[dict[str, Any]], ehr_train_rows: list[dict[str, Any]]) -> None:
    assert len(medagentboard_rows) == 100, len(medagentboard_rows)
    assert len(ehr_eval_rows) == 100, len(ehr_eval_rows)
    assert len(ehr_train_rows) == 10, len(ehr_train_rows)
    for rows in [medagentboard_rows, ehr_eval_rows, ehr_train_rows]:
        qids = [row["qid"] for row in rows]
        assert len(qids) == len(set(qids))
        for row in rows:
            assert row["answer"]
            assert row["required_files"]
            assert row["ground_truth_ref"]
            assert "/home/projects/HealthFlow" not in json.dumps(row, ensure_ascii=False)


def main() -> None:
    tjh_views = build_tjh_views()
    mimic_views = build_mimic_views()
    provenance_summary = build_paper_audit_outputs()

    ensure_empty_dir(GROUND_TRUTH_ROOT / "medagentboard")
    ensure_empty_dir(GROUND_TRUTH_ROOT / "ehrflowbench")

    medagentboard_tasks = build_tjh_medagentboard(tjh_views) + build_mimic_medagentboard(mimic_views)
    ehr_eval_tjh, ehr_train_tjh = build_tjh_ehrflowbench(tjh_views)
    ehr_eval_mimic, ehr_train_mimic = build_mimic_ehrflowbench(mimic_views)
    ehr_eval_tasks = ehr_eval_tjh + ehr_eval_mimic
    ehr_train_tasks = ehr_train_tjh + ehr_train_mimic

    medagentboard_rows = [item.task for item in medagentboard_tasks]
    ehr_eval_rows = [item.task for item in ehr_eval_tasks]
    ehr_train_rows = [item.task for item in ehr_train_tasks]
    validate_rows(medagentboard_rows, ehr_eval_rows, ehr_train_rows)

    write_jsonl(DATA_ROOT / "medagentboard.jsonl", medagentboard_rows)
    write_jsonl(DATA_ROOT / "ehrflowbench.jsonl", ehr_eval_rows)
    write_jsonl(DATA_ROOT / "ehrflowbench_train.jsonl", ehr_train_rows)

    write_docs(provenance_summary, tjh_views, mimic_views, medagentboard_rows, ehr_eval_rows, ehr_train_rows)
    print(
        json.dumps(
            {
                "medagentboard_tasks": len(medagentboard_rows),
                "ehrflowbench_eval_tasks": len(ehr_eval_rows),
                "ehrflowbench_train_tasks": len(ehr_train_rows),
                "tjh_patients": int(tjh_views.patient_summary["patient_id"].nunique()),
                "mimic_subjects": int(mimic_views.subject_summary["subject_id"].nunique()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
