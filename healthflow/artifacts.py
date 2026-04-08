from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .session import TaskTurnRecord

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
_REPORT_EXTENSIONS = {".md", ".txt", ".pdf", ".rst", ".doc", ".docx"}
_CODE_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".r",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
}
_DATA_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".feather", ".xlsx", ".xls", ".arrow"}
_TEXT_EXTENSIONS = _REPORT_EXTENSIONS | _CODE_EXTENSIONS | _DATA_EXTENSIONS | {".yaml", ".yml", ".toml"}
_EXCLUDED_RUNTIME_FILES = {"report.md"}
_COMMENT_PREFIXES = ("#", "//", "--", ";", "/*", "*")
_WHITESPACE_RE = re.compile(r"\s+")
_PREVIEW_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".ipynb": "json",
    ".json": "json",
    ".jsonl": "json",
    ".md": "markdown",
    ".rst": "markdown",
    ".html": "html",
    ".css": "css",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "yaml",
    ".txt": "markdown",
    ".log": "markdown",
    ".csv": "markdown",
    ".tsv": "markdown",
}


def collect_report_deliverables(sandbox_dir: Path, report_path: Path) -> list[dict[str, Any]]:
    deliverables: list[dict[str, Any]] = []
    if not sandbox_dir.exists():
        return deliverables

    for path in sorted(sandbox_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(sandbox_dir).as_posix()
        if _is_runtime_path(relative_path):
            continue
        report_relative_path = os.path.relpath(path, report_path.parent).replace("\\", "/")
        deliverables.append(
            {
                "path": report_relative_path,
                "sandbox_relative_path": relative_path,
                "source_path": path,
                "category": artifact_category(path),
                "descriptor": artifact_descriptor(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return deliverables


def collect_task_artifacts(task_root: Path, history: Sequence["TaskTurnRecord"]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    uploaded_sandbox_paths: set[str] = set()

    for record in history:
        for upload in record.uploaded_files:
            sandbox_relative_path = str(upload.get("sandbox_path") or "").strip()
            if sandbox_relative_path:
                uploaded_sandbox_paths.add(sandbox_relative_path)

            upload_relative_path = str(upload.get("upload_path") or "").strip()
            preferred_relative_path = upload_relative_path or sandbox_relative_path
            if not preferred_relative_path:
                continue
            source_path = task_root / preferred_relative_path
            if not source_path.exists():
                fallback_relative_path = sandbox_relative_path or upload_relative_path
                if not fallback_relative_path:
                    continue
                source_path = task_root / fallback_relative_path
                preferred_relative_path = fallback_relative_path
                if not source_path.exists():
                    continue

            artifacts.append(
                _artifact_record(
                    task_root=task_root,
                    source_path=source_path,
                    origin="uploaded",
                    display_name=str(upload.get("original_name") or source_path.name),
                )
            )
            seen_paths.add(preferred_relative_path)

    uploads_dir = task_root / "uploads"
    if uploads_dir.exists():
        for path in sorted(uploads_dir.rglob("*")):
            if not path.is_file():
                continue
            task_relative_path = path.relative_to(task_root).as_posix()
            if task_relative_path in seen_paths:
                continue
            artifacts.append(
                _artifact_record(
                    task_root=task_root,
                    source_path=path,
                    origin="uploaded",
                    display_name=path.name,
                )
            )
            seen_paths.add(task_relative_path)

    report_path = task_root / "runtime" / "report.md"
    if report_path.exists():
        report_relative_path = report_path.relative_to(task_root).as_posix()
        artifacts.append(
            _artifact_record(
                task_root=task_root,
                source_path=report_path,
                origin="report",
                display_name="report.md",
            )
        )
        seen_paths.add(report_relative_path)

    sandbox_dir = task_root / "sandbox"
    if sandbox_dir.exists():
        for path in sorted(sandbox_dir.rglob("*")):
            if not path.is_file():
                continue
            sandbox_relative_path = path.relative_to(sandbox_dir).as_posix()
            task_relative_path = path.relative_to(task_root).as_posix()
            if _is_runtime_path(sandbox_relative_path):
                continue
            if sandbox_relative_path in uploaded_sandbox_paths or task_relative_path in seen_paths:
                continue
            artifacts.append(
                _artifact_record(
                    task_root=task_root,
                    source_path=path,
                    origin="generated",
                    display_name=path.name,
                )
            )
            seen_paths.add(task_relative_path)

    artifacts.sort(
        key=lambda item: (
            item.get("updated_at_epoch", 0.0),
            item.get("origin_priority", 0),
            item.get("task_relative_path", ""),
        ),
        reverse=True,
    )
    for item in artifacts:
        item.pop("origin_priority", None)
        item.pop("updated_at_epoch", None)
    return artifacts


def artifact_category(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "images"
    if suffix in _CODE_EXTENSIONS:
        return "code/notebooks"
    if suffix in _DATA_EXTENSIONS:
        return "tables/data"
    if suffix in _REPORT_EXTENSIONS:
        return "reports/docs"
    return "other outputs"


def artifact_descriptor(path: Path) -> str:
    if path.suffix.lower() == ".md":
        heading = _first_markdown_heading(path)
        if heading:
            return heading
    if path.suffix.lower() == ".json":
        json_value = _json_descriptor(path)
        if json_value:
            return json_value
    if path.suffix.lower() in {".csv", ".tsv"}:
        csv_value = _csv_descriptor(path)
        if csv_value:
            return csv_value
    if path.suffix.lower() == ".ipynb":
        notebook_value = _notebook_descriptor(path)
        if notebook_value:
            return notebook_value
    if path.suffix.lower() == ".py":
        python_value = _python_descriptor(path)
        if python_value:
            return python_value
    text_value = _text_descriptor(path)
    if text_value:
        return text_value
    return _generic_descriptor(path)


def artifact_preview_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "markdown"
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix == ".json":
        preview = read_structured_preview(path)
        if preview is not None:
            return "table"
    if suffix in _TEXT_EXTENSIONS or suffix == ".log":
        return "code"
    return "download"


def artifact_preview_language(path: Path) -> str | None:
    return _PREVIEW_LANGUAGE_BY_SUFFIX.get(path.suffix.lower())


def read_structured_preview(
    path: Path,
    *,
    max_rows: int = 20,
    max_cols: int = 8,
) -> dict[str, Any] | None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _json_structured_preview(path, max_rows=max_rows, max_cols=max_cols)
    if suffix in {".csv", ".tsv"}:
        return _tabular_preview(path, max_rows=max_rows, max_cols=max_cols)
    return None


def _artifact_record(
    *,
    task_root: Path,
    source_path: Path,
    origin: str,
    display_name: str,
) -> dict[str, Any]:
    stat = source_path.stat()
    origin_priority = {"report": 3, "generated": 2, "uploaded": 1}.get(origin, 0)
    updated_at = stat.st_mtime
    task_relative_path = source_path.relative_to(task_root).as_posix()
    return {
        "task_relative_path": task_relative_path,
        "source_path": str(source_path),
        "display_name": Path(display_name).name or source_path.name,
        "descriptor": artifact_descriptor(source_path),
        "category": artifact_category(source_path),
        "origin": origin,
        "size_bytes": stat.st_size,
        "updated_at_epoch": updated_at,
        "updated_at": _format_timestamp(updated_at),
        "preview_kind": artifact_preview_kind(source_path),
        "preview_language": artifact_preview_language(source_path),
        "origin_priority": origin_priority,
    }


def _json_structured_preview(path: Path, *, max_rows: int, max_cols: int) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    flattened = _flatten_json_scalars(payload)
    if flattened and len(flattened) <= max_rows * 2:
        rows = [[key, _format_scalar(value)] for key, value in flattened[:max_rows]]
        return {
            "headers": ["Field", "Value"],
            "rows": rows,
            "truncated": len(flattened) > max_rows,
        }

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        headers = [str(key) for key in list(payload[0].keys())[:max_cols]]
        rows: list[list[str]] = []
        for item in payload[:max_rows]:
            row = [_format_scalar(item.get(header)) for header in headers]
            rows.append(row)
        return {
            "headers": headers,
            "rows": rows,
            "truncated": len(payload) > max_rows or len(payload[0]) > max_cols,
        }

    return None


def _tabular_preview(path: Path, *, max_rows: int, max_cols: int) -> dict[str, Any] | None:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, [])
            if not header:
                return None
            headers = [str(item) for item in header[:max_cols]]
            rows: list[list[str]] = []
            row_count = 0
            for row in reader:
                row_count += 1
                if len(rows) < max_rows:
                    rows.append([str(item) for item in row[:max_cols]])
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration):
        return None

    return {
        "headers": headers,
        "rows": rows,
        "truncated": row_count > max_rows or len(header) > max_cols,
    }


def _flatten_json_scalars(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_json_scalars(nested_value, nested_prefix))
        return rows
    if isinstance(value, list):
        if not value:
            return [(prefix, "[]")] if prefix else []
        if all(_is_scalar(item) for item in value) and len(value) <= 8:
            return [(prefix, "; ".join(_format_scalar(item) for item in value))]
        return []
    if _is_scalar(value):
        return [(prefix, value)] if prefix else []
    return []


def _first_markdown_heading(path: Path) -> str | None:
    for line in _iter_text_lines(path, max_lines=40):
        stripped = line.strip()
        if stripped.startswith("#"):
            return _normalize_descriptor(stripped.lstrip("#").strip())
    return None


def _notebook_descriptor(path: Path) -> str | None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for cell in notebook.get("cells", [])[:10]:
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue
        if cell.get("cell_type") == "markdown":
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    return _normalize_descriptor(stripped.lstrip("#").strip())
                if stripped:
                    return _normalize_descriptor(stripped)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(_COMMENT_PREFIXES):
                return _normalize_descriptor(_strip_comment_prefix(stripped))
            if stripped:
                return _normalize_descriptor(stripped)
    return None


def _python_descriptor(path: Path) -> str | None:
    lines = list(_iter_text_lines(path, max_lines=40))
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    docstring = _leading_docstring(lines)
    if docstring:
        return docstring
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            return _normalize_descriptor(_strip_comment_prefix(stripped))
        if stripped and not stripped.startswith(("import ", "from ", "def ", "class ")):
            return _normalize_descriptor(stripped)
    return None


def _leading_docstring(lines: list[str]) -> str | None:
    content = "\n".join(lines)
    match = re.search(r'^[\s\r\n]*("""|\'\'\')(.+?)(\1)', content, flags=re.DOTALL)
    if not match:
        return None
    first_line = next((line.strip() for line in match.group(2).splitlines() if line.strip()), "")
    return _normalize_descriptor(first_line) if first_line else None


def _text_descriptor(path: Path) -> str | None:
    if path.suffix.lower() not in _TEXT_EXTENSIONS:
        return None
    for line in _iter_text_lines(path, max_lines=30):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("==="):
            continue
        if stripped.lower().startswith(("summary statistics:", "generated artifacts:", "analysis date:")):
            continue
        if path.suffix.lower() == ".json" and stripped in {"{", "}", "[", "]"}:
            continue
        if stripped.startswith(_COMMENT_PREFIXES):
            return _normalize_descriptor(_strip_comment_prefix(stripped))
        if path.suffix.lower() == ".md" and stripped.startswith("#"):
            return _normalize_descriptor(stripped.lstrip("#").strip())
        return _normalize_descriptor(stripped)
    return None


def _json_descriptor(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and payload:
        keys = ", ".join(str(key) for key in list(payload.keys())[:4])
        return _normalize_descriptor(f"JSON fields: {keys}")
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        keys = ", ".join(str(key) for key in list(payload[0].keys())[:4])
        return _normalize_descriptor(f"JSON table fields: {keys}")
    return None


def _csv_descriptor(path: Path) -> str | None:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, [])
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration):
        return None
    if not header:
        return None
    return _normalize_descriptor(f"Columns: {', '.join(header[:6])}")


def _iter_text_lines(path: Path, max_lines: int) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return text.splitlines()[:max_lines]


def _strip_comment_prefix(text: str) -> str:
    for prefix in _COMMENT_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :].strip(" */")
    return text.strip()


def _normalize_descriptor(text: str) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", text).strip("`*_ ")
    return _truncate_text(cleaned, 140)


def _generic_descriptor(path: Path) -> str:
    category = artifact_category(path)
    fallback = {
        "reports/docs": "Document artifact",
        "images": "Image artifact",
        "code/notebooks": "Code or notebook artifact",
        "tables/data": "Data artifact",
        "other outputs": "Output artifact",
    }
    return fallback[category]


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_scalar(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_timestamp(epoch_seconds: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _is_runtime_path(relative_path: str) -> bool:
    path = Path(relative_path)
    if any(part.startswith(".") for part in path.parts):
        return True
    if path.name in _EXCLUDED_RUNTIME_FILES:
        return True
    if path.parts and path.parts[0] == ".healthflow_pi_agent":
        return True
    return False


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))
