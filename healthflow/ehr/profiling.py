from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from .models import DataProfile, SchemaSummary
from .risk import GROUP_ID_COLUMNS, PATIENT_ID_COLUMNS, TARGET_COLUMNS, TIME_COLUMNS
from .tasking import classify_task_family, detect_domain_focus


def _hash_parts(parts: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:12]


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _match_columns(columns: list[str], vocabulary: set[str]) -> list[str]:
    normalized_vocabulary = {_normalize_token(item) for item in vocabulary}
    return sorted(column for column in columns if _normalize_token(column) in normalized_vocabulary)


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
        group_id_columns=_match_columns(header, GROUP_ID_COLUMNS),
        patient_id_columns=_match_columns(header, PATIENT_ID_COLUMNS),
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
        group_id_columns=_match_columns(columns, GROUP_ID_COLUMNS),
        patient_id_columns=_match_columns(columns, PATIENT_ID_COLUMNS),
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


def profile_workspace_data(workspace_dir: Path, user_request: str, max_preview_rows: int = 3) -> DataProfile:
    task_family = classify_task_family(user_request)
    schemas: list[SchemaSummary] = []
    modalities = set()
    signature_parts = [task_family]

    total_rows = 0
    group_id_columns = set()
    patient_id_columns = set()
    target_columns = set()
    time_columns = set()
    file_names: list[str] = []
    schema_columns: list[str] = []

    for path in sorted(workspace_dir.iterdir()):
        if not path.is_file():
            continue
        file_names.append(path.name)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            schema = _profile_csv(path, max_preview_rows)
            modalities.add("structured_tabular")
        elif suffix == ".json":
            schema = _profile_json(path, max_preview_rows)
            modalities.add("structured_json")
        elif suffix in {".txt", ".md"}:
            schema = _profile_text(path, max_preview_rows)
            modalities.add("text")
        else:
            continue

        schemas.append(schema)
        total_rows += schema.row_count
        group_id_columns.update(schema.group_id_columns)
        patient_id_columns.update(schema.patient_id_columns)
        target_columns.update(schema.target_columns)
        time_columns.update(schema.time_columns)
        signature_parts.append(schema.file_name)
        signature_parts.extend(schema.columns[:10])
        schema_columns.extend(schema.columns)

    domain_focus, domain_signals = detect_domain_focus(
        user_request,
        file_names=file_names,
        columns=schema_columns,
    )
    if domain_focus == "ehr" and "text" in modalities:
        modalities.remove("text")
        modalities.add("clinical_text")

    notes = []
    if not schemas:
        notes.append("No profileable structured inputs were uploaded for this task.")
    if group_id_columns:
        notes.append("Group/entity identifiers were detected and may require entity-aware validation or splitting.")
    if patient_id_columns:
        notes.append("Patient-level identifiers were detected and should drive split logic.")
    if target_columns:
        notes.append("Target-like columns were detected and require leakage checks.")
    if time_columns:
        notes.append("Time-like columns were detected and can support temporal validation.")
    if domain_focus == "ehr":
        notes.append("EHR domain signals were detected, so healthcare-specific safeguards should be applied selectively.")

    return DataProfile(
        task_family=task_family,
        dataset_signature=_hash_parts(signature_parts),
        domain_focus=domain_focus,
        domain_signals=domain_signals,
        modalities=sorted(modalities),
        schemas=schemas,
        row_count=total_rows,
        group_id_columns=sorted(group_id_columns),
        patient_id_columns=sorted(patient_id_columns),
        target_columns=sorted(target_columns),
        time_columns=sorted(time_columns),
        notes=notes,
    )
