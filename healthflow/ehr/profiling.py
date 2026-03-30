from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .models import DataProfile, SchemaSummary
from .tasking import classify_task_family


def _hash_parts(parts: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:12]


def _profile_csv(path: Path, max_preview_rows: int) -> SchemaSummary:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    header = rows[0] if rows else []
    preview_rows = [", ".join(row[: min(5, len(row))]) for row in rows[1 : 1 + max_preview_rows]]
    return SchemaSummary(file_name=path.name, file_type="csv", columns=header, preview_rows=preview_rows)


def _profile_json(path: Path, max_preview_rows: int) -> SchemaSummary:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(data, list) and data:
        first = data[0]
        columns = list(first.keys()) if isinstance(first, dict) else []
        preview_rows = [json.dumps(item, ensure_ascii=False)[:120] for item in data[:max_preview_rows]]
    elif isinstance(data, dict):
        columns = list(data.keys())
        preview_rows = [json.dumps(data, ensure_ascii=False)[:120]]
    else:
        columns = []
        preview_rows = [str(data)[:120]]
    return SchemaSummary(file_name=path.name, file_type="json", columns=columns, preview_rows=preview_rows)


def _profile_text(path: Path, max_preview_rows: int) -> SchemaSummary:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return SchemaSummary(
        file_name=path.name,
        file_type=path.suffix.lstrip(".") or "text",
        columns=[],
        preview_rows=lines[:max_preview_rows],
    )


def profile_workspace_data(workspace_dir: Path, user_request: str, max_preview_rows: int = 3) -> DataProfile:
    task_family = classify_task_family(user_request)
    schemas: list[SchemaSummary] = []
    modalities = set()
    signature_parts = [task_family]

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
        signature_parts.append(schema.file_name)
        signature_parts.extend(schema.columns[:10])

    notes = []
    if not schemas:
        notes.append("No profileable structured inputs were uploaded for this task.")

    return DataProfile(
        task_family=task_family,
        dataset_signature=_hash_parts(signature_parts),
        modalities=sorted(modalities),
        schemas=schemas,
        notes=notes,
    )
