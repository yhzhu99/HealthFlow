from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .models import DataProfile, SchemaSummary
from .risk import GROUP_ID_COLUMNS, PATIENT_ID_COLUMNS, TARGET_COLUMNS, TIME_COLUMNS
from .tasking import classify_task_family, detect_domain_focus


_REQUEST_PATH_RE = re.compile(
    r"(?P<path>(?:~?/|\.{1,2}/)?[^\s'\"`]+?\.(?:csv|tsv|json|jsonl|txt|md|xlsx|xls|parquet|feather|arrow))",
    re.IGNORECASE,
)
_TRAILING_PATH_PUNCTUATION = ".,;:!?)]}>"


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


def _build_schema_summary(
    *,
    path: Path,
    file_type: str,
    columns: list[str],
    preview_rows: list[str],
    row_count: int,
) -> SchemaSummary:
    return SchemaSummary(
        file_name=path.name,
        file_type=file_type,
        columns=columns,
        preview_rows=preview_rows,
        row_count=row_count,
        group_id_columns=_match_columns(columns, GROUP_ID_COLUMNS),
        patient_id_columns=_match_columns(columns, PATIENT_ID_COLUMNS),
        target_columns=_match_columns(columns, TARGET_COLUMNS),
        time_columns=_match_columns(columns, TIME_COLUMNS),
    )


def _format_preview_rows(rows: list[dict[str, object]], columns: list[str], max_preview_rows: int) -> list[str]:
    preview_columns = columns[:5]
    formatted_rows: list[str] = []
    for row in rows[:max_preview_rows]:
        values = []
        for column in preview_columns:
            value = row.get(column, "")
            values.append("" if value is None else str(value))
        formatted_rows.append(", ".join(values))
    return formatted_rows


def _profile_delimited(path: Path, max_preview_rows: int, *, delimiter: str, file_type: str) -> SchemaSummary:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        rows = list(reader)
    header = rows[0] if rows else []
    preview_rows = [", ".join(row[: min(5, len(row))]) for row in rows[1 : 1 + max_preview_rows]]
    return _build_schema_summary(
        path=path,
        file_type=file_type,
        columns=header,
        preview_rows=preview_rows,
        row_count=max(len(rows) - 1, 0),
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
    return _build_schema_summary(
        path=path,
        file_type="json",
        columns=columns,
        preview_rows=preview_rows,
        row_count=row_count,
    )


def _profile_text(path: Path, max_preview_rows: int) -> SchemaSummary:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return _build_schema_summary(
        path=path,
        file_type=path.suffix.lstrip(".") or "text",
        columns=[],
        preview_rows=lines[:max_preview_rows],
        row_count=len(lines),
    )


def _profile_jsonl(path: Path, max_preview_rows: int) -> SchemaSummary:
    rows: list[dict[str, object]] = []
    columns: list[str] = []
    row_count = 0

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row_count += 1
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                if not columns:
                    columns = [str(item) for item in payload.keys()]
                if len(rows) < max_preview_rows:
                    rows.append({str(key): value for key, value in payload.items()})

    return _build_schema_summary(
        path=path,
        file_type="jsonl",
        columns=columns,
        preview_rows=_format_preview_rows(rows, columns, max_preview_rows),
        row_count=row_count,
    )


def _profile_excel(path: Path, max_preview_rows: int) -> SchemaSummary:
    preview_frame = pd.read_excel(path, nrows=max_preview_rows)
    columns = [str(column) for column in preview_frame.columns]
    row_count = len(pd.read_excel(path, usecols=[0]))
    preview_rows = _format_preview_rows(preview_frame.to_dict(orient="records"), columns, max_preview_rows)
    return _build_schema_summary(
        path=path,
        file_type=path.suffix.lstrip("."),
        columns=columns,
        preview_rows=preview_rows,
        row_count=row_count,
    )


def _profile_parquet(path: Path, max_preview_rows: int) -> SchemaSummary:
    parquet_file = pq.ParquetFile(path)
    columns = [str(column) for column in parquet_file.schema.names]
    preview_table = parquet_file.read().slice(0, max_preview_rows)
    preview_rows = _format_preview_rows(preview_table.to_pylist(), columns, max_preview_rows)
    return _build_schema_summary(
        path=path,
        file_type="parquet",
        columns=columns,
        preview_rows=preview_rows,
        row_count=parquet_file.metadata.num_rows,
    )


def _profile_feather(path: Path, max_preview_rows: int) -> SchemaSummary:
    table = feather.read_table(path, memory_map=True)
    columns = [str(column) for column in table.column_names]
    preview_rows = _format_preview_rows(table.slice(0, max_preview_rows).to_pylist(), columns, max_preview_rows)
    return _build_schema_summary(
        path=path,
        file_type="feather",
        columns=columns,
        preview_rows=preview_rows,
        row_count=table.num_rows,
    )


def _profile_arrow(path: Path, max_preview_rows: int) -> SchemaSummary:
    with pa.memory_map(path, "rb") as source:
        try:
            reader = ipc.open_file(source)
        except pa.ArrowInvalid:
            reader = ipc.open_stream(source)
        table = reader.read_all()
    columns = [str(column) for column in table.column_names]
    preview_rows = _format_preview_rows(table.slice(0, max_preview_rows).to_pylist(), columns, max_preview_rows)
    return _build_schema_summary(
        path=path,
        file_type="arrow",
        columns=columns,
        preview_rows=preview_rows,
        row_count=table.num_rows,
    )


def _profile_supported_file(path: Path, max_preview_rows: int) -> SchemaSummary | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _profile_delimited(path, max_preview_rows, delimiter=",", file_type="csv")
    if suffix == ".tsv":
        return _profile_delimited(path, max_preview_rows, delimiter="\t", file_type="tsv")
    if suffix == ".json":
        return _profile_json(path, max_preview_rows)
    if suffix == ".jsonl":
        return _profile_jsonl(path, max_preview_rows)
    if suffix in {".txt", ".md"}:
        return _profile_text(path, max_preview_rows)
    if suffix in {".xlsx", ".xls"}:
        return _profile_excel(path, max_preview_rows)
    if suffix == ".parquet":
        return _profile_parquet(path, max_preview_rows)
    if suffix == ".feather":
        return _profile_feather(path, max_preview_rows)
    if suffix == ".arrow":
        return _profile_arrow(path, max_preview_rows)
    return None


def _request_referenced_paths(user_request: str) -> list[Path]:
    paths: list[Path] = []
    for match in _REQUEST_PATH_RE.finditer(user_request):
        raw_path = match.group("path").strip().strip("'\"`").rstrip(_TRAILING_PATH_PUNCTUATION)
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.is_file():
            paths.append(candidate)
    return list(dict.fromkeys(paths))


def _profile_paths(workspace_dir: Path, user_request: str) -> list[Path]:
    candidate_paths: list[Path] = []
    seen_paths: set[str] = set()

    for path in sorted(workspace_dir.iterdir()):
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidate_paths.append(path)

    for path in _request_referenced_paths(user_request):
        resolved = str(path.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidate_paths.append(path)

    return candidate_paths


def profile_workspace_data(workspace_dir: Path, user_request: str, max_preview_rows: int = 3) -> DataProfile:
    task_family = classify_task_family(user_request)
    schemas: list[SchemaSummary] = []
    modalities = set()
    signature_parts: list[str] = []

    total_rows = 0
    group_id_columns = set()
    patient_id_columns = set()
    target_columns = set()
    time_columns = set()
    file_names: list[str] = []
    schema_columns: list[str] = []

    for path in _profile_paths(workspace_dir, user_request):
        file_names.append(path.name)
        schema = _profile_supported_file(path, max_preview_rows)
        if schema is None:
            continue

        if schema.file_type in {"csv", "tsv", "xlsx", "xls", "parquet", "feather", "arrow"}:
            modalities.add("structured_tabular")
        elif schema.file_type in {"json", "jsonl"}:
            modalities.add("structured_json")
        elif schema.file_type in {"txt", "md", "text"}:
            modalities.add("text")

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
        notes.append("Group/entity identifier columns were detected in the profiled inputs.")
    if patient_id_columns:
        notes.append("Patient identifier columns were detected in the profiled inputs.")
    if target_columns:
        notes.append("Target-like columns were detected in the profiled inputs.")
    if time_columns:
        notes.append("Time-like columns were detected in the profiled inputs.")
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
