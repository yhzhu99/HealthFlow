from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_gzip_file(source_path: Path, target_path: Path) -> Path:
    ensure_directory(target_path.parent)
    if target_path.exists():
        return target_path

    with gzip.open(source_path, "rb") as source_handle:
        with target_path.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
    return target_path


def normalize_metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_key_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def sort_key_value(value: Any) -> tuple[int, Any]:
    value = normalize_key_value(value)
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, float) and value.is_integer():
        return (0, int(value))
    return (1, str(value))


def build_stratified_split_metadata(
    frame: pd.DataFrame,
    *,
    dataset: str,
    key_column: str,
    label_column: str,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> dict[str, Any]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")

    key_frame = frame.groupby(key_column, as_index=False)[label_column].first()
    labels = key_frame[label_column]
    label_counts = labels.value_counts(dropna=False)
    if len(label_counts) < 2 or int(label_counts.min()) < 2:
        raise ValueError(f"{dataset}: stratified split requires at least two labels with count >= 2")

    train_val_keys, test_keys = train_test_split(
        key_frame[key_column],
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )

    train_val_frame = key_frame[key_frame[key_column].isin(train_val_keys)].reset_index(drop=True)
    val_share = val_ratio / (train_ratio + val_ratio)
    train_keys, val_keys = train_test_split(
        train_val_frame[key_column],
        test_size=val_share,
        random_state=seed,
        stratify=train_val_frame[label_column],
    )

    split_keys = {
        "train": sorted((normalize_key_value(value) for value in train_keys), key=sort_key_value),
        "val": sorted((normalize_key_value(value) for value in val_keys), key=sort_key_value),
        "test": sorted((normalize_key_value(value) for value in test_keys), key=sort_key_value),
    }
    split_key_sets = {
        split: set(keys)
        for split, keys in split_keys.items()
    }

    return {
        "dataset": dataset,
        "seed": seed,
        "key_column": key_column,
        "label_column": label_column,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "key_count": len(key_frame),
        "row_count": len(frame),
        "label_distribution": {
            str(normalize_key_value(label)): int(count)
            for label, count in label_counts.items()
        },
        "split_key_counts": {
            split: len(keys)
            for split, keys in split_keys.items()
        },
        "split_row_counts": {
            split: int(frame[frame[key_column].isin(key_set)].shape[0])
            for split, key_set in split_key_sets.items()
        },
        "split_keys": split_keys,
    }


def write_parquet_with_metadata(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    ensure_directory(output_path.parent)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if metadata:
        existing_metadata = dict(table.schema.metadata or {})
        existing_metadata.update(
            {
                key.encode("utf-8"): normalize_metadata_value(value).encode("utf-8")
                for key, value in metadata.items()
            }
        )
        table = table.replace_schema_metadata(existing_metadata)
    pq.write_table(table, output_path)
