from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .models import DataProfile, SchemaSummary
from .risk import IDENTIFIER_COLUMNS, TARGET_COLUMNS, TIME_COLUMNS
from .tasking import classify_task_family


def _hash_parts(parts: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:12]


def _match_columns(columns: list[str], vocabulary: set[str]) -> list[str]:
    return sorted(column for column in columns if column.lower() in vocabulary)


def _profile_csv(path: Path, max_preview_rows: int) -> SchemaSummary:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    header = rows[0] if rows else []
    preview_rows = [", ".join(row[: min(5, len(row))]) for row in rows[1 : 1 + max_preview_rows]]
    return SchemaSummary(
        file_name=path.name,
        file_type="csv",
        columns=header,
        preview_rows=preview_rows,
        row_count=max(len(rows) - 1, 0),
        patient_id_columns=_match_columns(header, IDENTIFIER_COLUMNS),
        target_columns=_match_columns(header, TARGET_COLUMNS),
        time_columns=_match_columns(header, TIME_COLUMNS),
    )


def _profile_json(path: Path, max_preview_rows: int) -> SchemaSummary:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(data, list) and data:
        first = data[0]
        columns = list(first.keys()) if isinstance(first, dict) else []
        preview_rows = [json.dumps(item, ensure_ascii=False)[:120] for item in data[:max_preview_rows]]
        row_count = len(data)
    elif isinstance(data, dict):
        columns = list(data.keys())
        preview_rows = [json.dumps(data, ensure_ascii=False)[:120]]
        row_count = 1
    else:
        columns = []
        preview_rows = [str(data)[:120]]
        row_count = 1
    return SchemaSummary(
        file_name=path.name,
        file_type="json",
        columns=columns,
        preview_rows=preview_rows,
        row_count=row_count,
        patient_id_columns=_match_columns(columns, IDENTIFIER_COLUMNS),
        target_columns=_match_columns(columns, TARGET_COLUMNS),
        time_columns=_match_columns(columns, TIME_COLUMNS),
    )


def _profile_text(path: Path, max_preview_rows: int) -> SchemaSummary:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return SchemaSummary(
        file_name=path.name,
        file_type=path.suffix.lstrip(".") or "text",
        columns=[],
        preview_rows=lines[:max_preview_rows],
        row_count=len(lines),
    )


def _detect_artifact_hints(workspace_dir: Path) -> list[str]:
    hints: list[str] = []
    analyze_dir = workspace_dir / "analyze"
    if (workspace_dir / "manifest.json").exists():
        hints.append("oneehr_manifest")
    if (workspace_dir / "preprocess" / "split.json").exists():
        hints.append("oneehr_split")
    if (workspace_dir / "test" / "metrics.json").exists():
        hints.append("oneehr_metrics")
    if analyze_dir.exists() and any(analyze_dir.glob("*.json")):
        hints.append("oneehr_analysis")
    return hints


def profile_workspace_data(workspace_dir: Path, user_request: str, max_preview_rows: int = 3) -> DataProfile:
    task_family = classify_task_family(user_request)
    schemas: list[SchemaSummary] = []
    modalities = set()
    signature_parts = [task_family]

    total_rows = 0
    patient_id_columns = set()
    target_columns = set()
    time_columns = set()

    for path in sorted(workspace_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            schema = _profile_csv(path, max_preview_rows)
            modalities.add("structured_tabular")
        elif suffix == ".json":
            schema = _profile_json(path, max_preview_rows)
            modalities.add("structured_json")
        elif suffix in {".txt", ".md"}:
            schema = _profile_text(path, max_preview_rows)
            modalities.add("clinical_text")
        else:
            continue

        schemas.append(schema)
        total_rows += schema.row_count
        patient_id_columns.update(schema.patient_id_columns)
        target_columns.update(schema.target_columns)
        time_columns.update(schema.time_columns)
        signature_parts.append(schema.file_name)
        signature_parts.extend(schema.columns[:10])

    notes = []
    if not schemas:
        notes.append("No profileable structured inputs were uploaded for this task.")
    if patient_id_columns:
        notes.append("Patient-level identifiers were detected and should drive split logic.")
    if target_columns:
        notes.append("Target-like columns were detected and require leakage checks.")
    if time_columns:
        notes.append("Time-like columns were detected and can support temporal validation.")

    return DataProfile(
        task_family=task_family,
        dataset_signature=_hash_parts(signature_parts),
        modalities=sorted(modalities),
        schemas=schemas,
        row_count=total_rows,
        patient_id_columns=sorted(patient_id_columns),
        target_columns=sorted(target_columns),
        time_columns=sorted(time_columns),
        artifact_hints=_detect_artifact_hints(workspace_dir),
        notes=notes,
    )
