from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from ehr_common import (
    build_stratified_split_metadata,
    extract_gzip_file,
    write_json,
    write_parquet_with_metadata,
)


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = BENCHMARK_ROOT / "raw" / "mimic_iv_demo"
PROCESSED_ROOT = BENCHMARK_ROOT / "processed" / "mimic_iv_demo"
EXTRACTED_ROOT = PROCESSED_ROOT / "_extracted"
OUTPUT_PATH = PROCESSED_ROOT / "mimic_iv_demo_formatted_ehr.parquet"
SPLIT_METADATA_PATH = PROCESSED_ROOT / "split_metadata.json"
METADATA_BUNDLE_PATH = BENCHMARK_ROOT / "metadata" / "mimic_iv_demo_ehr_metadata.json"
MAX_WINDOW = 7
SEED = 42

EXTRACT_TARGETS = {
    RAW_ROOT / "hosp" / "patients.csv.gz": EXTRACTED_ROOT / "hosp" / "patients.csv",
    RAW_ROOT / "hosp" / "admissions.csv.gz": EXTRACTED_ROOT / "hosp" / "admissions.csv",
    RAW_ROOT / "icu" / "icustays.csv.gz": EXTRACTED_ROOT / "icu" / "icustays.csv",
    RAW_ROOT / "icu" / "chartevents.csv.gz": EXTRACTED_ROOT / "icu" / "chartevents.csv",
}

PATIENT_USECOLS = ["subject_id", "gender", "anchor_age", "dod"]
ADMISSION_USECOLS = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"]
ICUSTAY_USECOLS = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"]
CHARTEVENT_USECOLS = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "value"]


def load_metadata_bundle() -> dict[str, Any]:
    return json.loads(METADATA_BUNDLE_PATH.read_text(encoding="utf-8"))


def prepare_extracted_sources() -> dict[str, Path]:
    return {
        source_path.name.replace(".csv.gz", ".csv"): extract_gzip_file(source_path, target_path)
        for source_path, target_path in EXTRACT_TARGETS.items()
    }


def format_capillary_refill_rate(values: pd.Series) -> pd.Series:
    normalized = values.fillna("").astype(str).str.strip()
    result = pd.Series(np.nan, index=values.index, dtype="object")
    result.loc[normalized.str.contains("normal", case=False)] = "0.0"
    result.loc[normalized.str.contains("abnormal", case=False)] = "1.0"
    return result


def format_gcs(values: pd.Series) -> pd.Series:
    normalized = values.fillna("").astype(str).str.strip()
    normalized = normalized.replace("", np.nan)
    return normalized


def format_temperature(values: pd.Series, mimic_labels: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    fahrenheit_mask = mimic_labels.fillna("").astype(str).str.contains("f", case=False)
    numeric.loc[fahrenheit_mask] = (numeric.loc[fahrenheit_mask] - 32.0) * 5.0 / 9.0
    return numeric.round(2)


def format_weight(values: pd.Series, mimic_labels: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    pounds_mask = mimic_labels.fillna("").astype(str).str.contains("lbs", case=False)
    numeric.loc[pounds_mask] = numeric.loc[pounds_mask] * 0.453592
    return numeric.round(2)


def format_height(values: pd.Series, mimic_labels: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    non_cm_mask = ~mimic_labels.fillna("").astype(str).str.contains("cm", case=False)
    numeric.loc[non_cm_mask] = numeric.loc[non_cm_mask] * 2.54
    return numeric.round(2)


def process_patients(extracted_paths: dict[str, Path]) -> pd.DataFrame:
    patients = pd.read_csv(extracted_paths["patients.csv"], usecols=PATIENT_USECOLS)
    admissions = pd.read_csv(extracted_paths["admissions.csv"], usecols=ADMISSION_USECOLS)
    icustays = pd.read_csv(extracted_paths["icustays.csv"], usecols=ICUSTAY_USECOLS)

    for frame, columns in [
        (patients, ["dod"]),
        (admissions, ["admittime", "dischtime", "deathtime"]),
        (icustays, ["intime", "outtime"]),
    ]:
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    stays = icustays.merge(admissions, how="inner", on=["subject_id", "hadm_id"])
    stays = stays.merge(patients, how="inner", on=["subject_id"])

    stays = stays.sort_values(by=["subject_id", "intime"]).reset_index(drop=True)
    stays["next_intime"] = stays.groupby("subject_id")["intime"].shift(-1)
    days_to_next = (stays["next_intime"] - stays["outtime"]).dt.total_seconds() / (24 * 3600)
    days_to_death = (stays["dod"] - stays["outtime"]).dt.total_seconds() / (24 * 3600)
    readmit_cond = (days_to_next > 0) & (days_to_next <= 30)
    death_cond = stays["next_intime"].isna() & (days_to_death > 0) & (days_to_death <= 30)
    stays["Readmission"] = (readmit_cond | death_cond).astype(int)

    mortality_inhospital = stays["dod"].notna() & (stays["admittime"] <= stays["dod"]) & (stays["dischtime"] >= stays["dod"])
    mortality_inhospital |= (
        stays["deathtime"].notna()
        & (stays["admittime"] <= stays["deathtime"])
        & (stays["dischtime"] >= stays["deathtime"])
    )

    mortality_inunit = stays["dod"].notna() & (stays["intime"] <= stays["dod"]) & (stays["outtime"] >= stays["dod"])
    mortality_inunit |= (
        stays["deathtime"].notna() & (stays["intime"] <= stays["deathtime"]) & (stays["outtime"] >= stays["deathtime"])
    )

    processed = pd.DataFrame(
        {
            "PatientID": stays["subject_id"].astype("int64"),
            "AdmissionID": stays["hadm_id"].astype("int64"),
            "StayID": stays["stay_id"].astype("int64"),
            "Outcome": (mortality_inhospital | mortality_inunit).astype(int),
            "LOS": 24 * stays["los"].astype(float),
            "Readmission": stays["Readmission"].astype(int),
            "Age": stays["anchor_age"].astype("int64"),
            "Sex": stays["gender"].map({"M": 1, "F": 0}).astype("Int64"),
        }
    )
    return processed


def process_events(
    extracted_paths: dict[str, Path],
    config: dict[str, Any],
    item2var_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    item2var = pd.DataFrame(item2var_rows)
    item2var["ItemID"] = pd.to_numeric(item2var["ItemID"], errors="raise").astype("int64")
    chartevents = pd.read_csv(extracted_paths["chartevents.csv"], usecols=CHARTEVENT_USECOLS, low_memory=False)
    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"], errors="coerce")

    events = chartevents.merge(item2var, how="inner", left_on="itemid", right_on="ItemID")
    events = events.drop_duplicates(
        subset=["subject_id", "charttime", "hadm_id", "stay_id", "Variable"],
        keep="last",
    ).reset_index(drop=True)

    categorical_features = {
        feature
        for feature, is_categorical in config["is_categorical_channel"].items()
        if is_categorical
    }
    numeric_value = pd.to_numeric(events["valuenum"], errors="coerce")
    fallback_numeric_value = pd.to_numeric(events["value"], errors="coerce")
    events["FormattedValue"] = numeric_value.where(numeric_value.notna(), fallback_numeric_value).astype("object")

    variable = events["Variable"]
    events.loc[variable == "Capillary refill rate", "FormattedValue"] = format_capillary_refill_rate(
        events.loc[variable == "Capillary refill rate", "value"]
    )
    for gcs_feature in config["gcs_features"]:
        events.loc[variable == gcs_feature, "FormattedValue"] = format_gcs(events.loc[variable == gcs_feature, "value"])
    events.loc[variable == "Temperature", "FormattedValue"] = format_temperature(
        events.loc[variable == "Temperature", "value"],
        events.loc[variable == "Temperature", "MimicLabel"],
    )
    events.loc[variable == "Weight", "FormattedValue"] = format_weight(
        events.loc[variable == "Weight", "value"],
        events.loc[variable == "Weight", "MimicLabel"],
    )
    events.loc[variable == "Height", "FormattedValue"] = format_height(
        events.loc[variable == "Height", "value"],
        events.loc[variable == "Height", "MimicLabel"],
    )

    numeric_feature_mask = ~events["Variable"].isin(categorical_features)
    events.loc[numeric_feature_mask, "FormattedValue"] = pd.to_numeric(
        events.loc[numeric_feature_mask, "FormattedValue"],
        errors="coerce",
    )

    empty_categorical_mask = events["Variable"].isin(categorical_features) & (
        events["FormattedValue"].isna() | (events["FormattedValue"].astype(str).str.strip() == "")
    )
    events = events.loc[~empty_categorical_mask].copy()
    events = events.loc[
        events["Variable"].isin(categorical_features) | events["FormattedValue"].notna()
    ].copy()

    value_events = (
        events.rename(
            columns={
                "subject_id": "PatientID",
                "hadm_id": "AdmissionID",
                "stay_id": "StayID",
                "charttime": "RecordTime",
            }
        )[["PatientID", "AdmissionID", "StayID", "RecordTime", "Variable", "FormattedValue"]]
        .pivot_table(
            index=["PatientID", "AdmissionID", "StayID", "RecordTime"],
            columns="Variable",
            values="FormattedValue",
            aggfunc="last",
        )
        .reset_index()
    )

    for feature in config["labtest_features"]:
        if feature not in value_events.columns:
            value_events[feature] = pd.NA

    gcs_numeric_df = pd.concat(
        [
            pd.to_numeric(
                value_events[gcs_feature].map(lambda value: config["mapping_values"].get(gcs_feature, {}).get(str(value), value)),
                errors="coerce",
            )
            for gcs_feature in config["gcs_features"]
        ],
        axis=1,
    )
    value_events["Glascow coma scale total"] = gcs_numeric_df.sum(axis=1).astype("Int64")

    for feature in config["labtest_features"]:
        if feature in value_events.columns and feature not in categorical_features:
            value_events[feature] = pd.to_numeric(value_events[feature], errors="coerce")

    selected_columns = ["PatientID", "AdmissionID", "StayID", "RecordTime"] + config["labtest_features"]
    return value_events[selected_columns]


def aggregate_ehr(
    patients: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = patients.merge(events, how="inner", on=["PatientID", "AdmissionID", "StayID"])
    frame = frame[["PatientID", "AdmissionID", "RecordTime"] + config["label_features"] + config["demographic_features"] + config["labtest_features"]]
    frame["RecordTime"] = pd.to_datetime(frame["RecordTime"], errors="coerce").dt.date
    frame = frame.sort_values(by=["PatientID", "AdmissionID", "RecordTime"]).reset_index(drop=True)

    unique_dates = frame.drop_duplicates(subset=["PatientID", "AdmissionID", "RecordTime"]).copy()
    unique_dates["date_rank"] = unique_dates.groupby(["PatientID", "AdmissionID"])["RecordTime"].rank(
        method="dense",
        ascending=False,
    )
    split_dates = unique_dates.loc[
        unique_dates["date_rank"] == MAX_WINDOW + 1,
        ["PatientID", "AdmissionID", "RecordTime"],
    ].rename(columns={"RecordTime": "SplitTime"})

    frame = frame.merge(split_dates, how="left", on=["PatientID", "AdmissionID"])
    frame["RecordTime"] = np.where(
        frame["RecordTime"] <= frame["SplitTime"].fillna(pd.Timestamp.min.date()),
        frame["SplitTime"],
        frame["RecordTime"],
    )
    frame = frame.drop(columns=["SplitTime"]).groupby(["PatientID", "AdmissionID", "RecordTime"]).last().reset_index()

    for feature in config["labtest_features"]:
        if feature not in frame.columns:
            frame[feature] = pd.NA

    gcs_numeric_df = pd.concat(
        [
            pd.to_numeric(
                frame[gcs_feature].map(lambda value: config["mapping_values"].get(gcs_feature, {}).get(str(value), value)),
                errors="coerce",
            )
            for gcs_feature in config["gcs_features"]
        ],
        axis=1,
    )
    frame["Glascow coma scale total"] = gcs_numeric_df.sum(axis=1).astype("Int64")

    categorical_features = [
        feature for feature, is_categorical in config["is_categorical_channel"].items() if is_categorical
    ]
    numeric_features = [
        feature for feature in config["labtest_features"] if not config["is_categorical_channel"].get(feature, False)
    ]
    for feature in numeric_features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    frame["RecordID"] = frame["PatientID"].astype(str) + "_" + frame["AdmissionID"].astype(str)
    ordered_columns = ["RecordID", "PatientID", "AdmissionID", "RecordTime"] + config["label_features"] + config["demographic_features"] + numeric_features
    one_hot_columns: list[str] = []
    for feature in categorical_features:
        values = config["possible_values"].get(feature, [])
        feature_series = frame[feature].astype(str)
        for value in values:
            column_name = f"{feature}->{value}"
            frame[column_name] = (feature_series == str(value)).astype(int)
            one_hot_columns.append(column_name)

    frame = frame.drop(columns=categorical_features)
    frame["RecordTime"] = frame["RecordTime"].astype(str)
    frame["PatientID"] = frame["PatientID"].astype("int64")
    frame["AdmissionID"] = frame["AdmissionID"].astype("int64")
    frame["Outcome"] = frame["Outcome"].astype(int)
    frame["Readmission"] = frame["Readmission"].astype(int)
    frame["Age"] = frame["Age"].astype("int64")
    frame["Sex"] = frame["Sex"].astype("Int64")

    return frame[ordered_columns + one_hot_columns]


def main() -> None:
    metadata_bundle = load_metadata_bundle()
    config = metadata_bundle["config"]
    extracted_paths = prepare_extracted_sources()
    patients = process_patients(extracted_paths)
    events = process_events(extracted_paths, config, metadata_bundle["item2var"])
    frame = aggregate_ehr(patients, events, config)

    split_metadata = build_stratified_split_metadata(
        frame,
        dataset="mimic_iv_demo",
        key_column="RecordID",
        label_column="Outcome",
        seed=SEED,
    )
    write_json(SPLIT_METADATA_PATH, split_metadata)

    metadata = {
        "healthflow.source_dataset": "mimic_iv_demo",
        "healthflow.source_dir": str(RAW_ROOT.relative_to(BENCHMARK_ROOT)),
        "healthflow.key_column": "RecordID",
        "healthflow.metadata_bundle_path": str(METADATA_BUNDLE_PATH.relative_to(BENCHMARK_ROOT)),
        "healthflow.split_metadata_path": str(SPLIT_METADATA_PATH.relative_to(BENCHMARK_ROOT)),
        "healthflow.metadata_bundle.json": metadata_bundle,
        "healthflow.processing.json": {
            "seed": SEED,
            "max_window": MAX_WINDOW,
            "one_hot_encode_categorical": True,
            "split_strategy": {
                "method": "stratified",
                "ratios": {
                    "train": 0.7,
                    "val": 0.2,
                    "test": 0.1,
                },
            },
            "categorical_features": [
                feature for feature, is_categorical in config["is_categorical_channel"].items() if is_categorical
            ],
            "numeric_features": [
                feature for feature in config["labtest_features"] if not config["is_categorical_channel"].get(feature, False)
            ],
            "extracted_sources": {
                str(source.relative_to(BENCHMARK_ROOT)): str(target.relative_to(BENCHMARK_ROOT))
                for source, target in EXTRACT_TARGETS.items()
            },
        },
    }
    write_parquet_with_metadata(frame, OUTPUT_PATH, metadata=metadata)

    summary = {
        "rows": len(frame),
        "patients": int(frame["PatientID"].nunique()),
        "admissions": int(frame["AdmissionID"].nunique()),
        "columns": len(frame.columns),
        "metadata_bundle": str(METADATA_BUNDLE_PATH),
        "split_metadata_file": str(SPLIT_METADATA_PATH),
        "split_key_counts": split_metadata["split_key_counts"],
        "output_file": str(OUTPUT_PATH),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
