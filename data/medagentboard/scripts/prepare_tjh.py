from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from ehr_common import build_stratified_split_metadata, write_json, write_parquet_with_metadata


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = BENCHMARK_ROOT / "raw" / "tjh" / "time_series_375_prerpocess_en.xlsx"
PROCESSED_ROOT = BENCHMARK_ROOT / "processed" / "tjh"
OUTPUT_PATH = BENCHMARK_ROOT / "processed" / "tjh" / "tjh_formatted_ehr.parquet"
SPLIT_METADATA_PATH = PROCESSED_ROOT / "split_metadata.json"
SOURCE_DATASET = "tjh"
SEED = 42
DROP_COLUMNS = ["2019-nCoV nucleic acid detection"]

BASIC_RECORDS = ["PatientID", "RecordTime", "AdmissionTime", "DischargeTime"]
TARGET_FEATURES = ["Outcome", "LOS"]
DEMOGRAPHIC_FEATURES = ["Sex", "Age"]
LABTEST_FEATURES = [
    "Hypersensitive cardiac troponinI",
    "hemoglobin",
    "Serum chloride",
    "Prothrombin time",
    "procalcitonin",
    "eosinophils(%)",
    "Interleukin 2 receptor",
    "Alkaline phosphatase",
    "albumin",
    "basophil(%)",
    "Interleukin 10",
    "Total bilirubin",
    "Platelet count",
    "monocytes(%)",
    "antithrombin",
    "Interleukin 8",
    "indirect bilirubin",
    "Red blood cell distribution width ",
    "neutrophils(%)",
    "total protein",
    "Quantification of Treponema pallidum antibodies",
    "Prothrombin activity",
    "HBsAg",
    "mean corpuscular volume",
    "hematocrit",
    "White blood cell count",
    "Tumor necrosis factorα",
    "mean corpuscular hemoglobin concentration",
    "fibrinogen",
    "Interleukin 1β",
    "Urea",
    "lymphocyte count",
    "PH value",
    "Red blood cell count",
    "Eosinophil count",
    "Corrected calcium",
    "Serum potassium",
    "glucose",
    "neutrophils count",
    "Direct bilirubin",
    "Mean platelet volume",
    "ferritin",
    "RBC distribution width SD",
    "Thrombin time",
    "(%)lymphocyte",
    "HCV antibody quantification",
    "D-D dimer",
    "Total cholesterol",
    "aspartate aminotransferase",
    "Uric acid",
    "HCO3-",
    "calcium",
    "Amino-terminal brain natriuretic peptide precursor(NT-proBNP)",
    "Lactate dehydrogenase",
    "platelet large cell ratio ",
    "Interleukin 6",
    "Fibrin degradation products",
    "monocytes count",
    "PLT distribution width",
    "globulin",
    "γ-glutamyl transpeptidase",
    "International standard ratio",
    "basophil count(#)",
    "mean corpuscular hemoglobin ",
    "Activation of partial thromboplastin time",
    "High sensitivity C-reactive protein",
    "HIV antibody quantification",
    "serum sodium",
    "thrombocytocrit",
    "ESR",
    "glutamic-pyruvic transaminase",
    "eGFR",
    "creatinine",
]

RENAME_MAP = {
    "PATIENT_ID": "PatientID",
    "outcome": "Outcome",
    "gender": "Sex",
    "age": "Age",
    "RE_DATE": "RecordTime",
    "Admission time": "AdmissionTime",
    "Discharge time": "DischargeTime",
}


def format_datetime_column(frame: pd.DataFrame, column: str) -> None:
    frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")


def main() -> None:
    frame = pd.read_excel(RAW_PATH)
    frame = frame.rename(columns=RENAME_MAP)
    frame["PatientID"] = frame["PatientID"].ffill()
    frame["Sex"] = frame["Sex"].replace(2, 0)

    for column in TARGET_FEATURES + DEMOGRAPHIC_FEATURES + LABTEST_FEATURES:
        if column not in frame.columns:
            frame[column] = np.nan

    for column in ["RecordTime", "AdmissionTime", "DischargeTime"]:
        format_datetime_column(frame, column)

    frame = frame.dropna(subset=["PatientID", "RecordTime", "DischargeTime"], how="any")
    frame["LOS"] = (
        pd.to_datetime(frame["DischargeTime"], errors="coerce") - pd.to_datetime(frame["RecordTime"], errors="coerce")
    ).dt.days
    frame["LOS"] = frame["LOS"].clip(lower=0)

    drop_columns = [column for column in DROP_COLUMNS if column in frame.columns]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)

    frame[DEMOGRAPHIC_FEATURES + LABTEST_FEATURES] = frame[DEMOGRAPHIC_FEATURES + LABTEST_FEATURES].mask(
        frame[DEMOGRAPHIC_FEATURES + LABTEST_FEATURES] < 0
    )

    frame = frame.groupby(BASIC_RECORDS, dropna=True, as_index=False).mean(numeric_only=True)
    ordered_columns = BASIC_RECORDS + TARGET_FEATURES + DEMOGRAPHIC_FEATURES + LABTEST_FEATURES
    frame = frame[ordered_columns]
    frame["PatientID"] = pd.to_numeric(frame["PatientID"], errors="raise").astype("int64")

    split_metadata = build_stratified_split_metadata(
        frame,
        dataset=SOURCE_DATASET,
        key_column="PatientID",
        label_column="Outcome",
        seed=SEED,
    )
    write_json(SPLIT_METADATA_PATH, split_metadata)

    metadata = {
        "healthflow.source_dataset": SOURCE_DATASET,
        "healthflow.source_file": str(RAW_PATH.relative_to(BENCHMARK_ROOT)),
        "healthflow.key_column": "PatientID",
        "healthflow.split_metadata_path": str(SPLIT_METADATA_PATH.relative_to(BENCHMARK_ROOT)),
        "healthflow.processing.json": {
            "seed": SEED,
            "rename_map": RENAME_MAP,
            "dropped_columns": DROP_COLUMNS,
            "basic_records": BASIC_RECORDS,
            "target_features": TARGET_FEATURES,
            "demographic_features": DEMOGRAPHIC_FEATURES,
            "labtest_features": LABTEST_FEATURES,
            "split_strategy": {
                "method": "stratified",
                "ratios": {
                    "train": 0.7,
                    "val": 0.2,
                    "test": 0.1,
                },
            },
        },
    }
    write_parquet_with_metadata(frame, OUTPUT_PATH, metadata=metadata)

    summary = {
        "rows": len(frame),
        "patients": int(frame["PatientID"].nunique()),
        "columns": len(frame.columns),
        "split_metadata_file": str(SPLIT_METADATA_PATH),
        "split_key_counts": split_metadata["split_key_counts"],
        "output_file": str(OUTPUT_PATH),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
