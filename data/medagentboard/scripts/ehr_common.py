from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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

