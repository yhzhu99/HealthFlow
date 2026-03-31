from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def find_healthflow_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    search_roots = [current] + list(current.parents)
    for parent in search_roots:
        if (parent / "pyproject.toml").exists() and (parent / "run_benchmark.py").exists():
            return parent
    raise FileNotFoundError("Could not locate HealthFlow root")


HEALTHFLOW_ROOT = find_healthflow_root()


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def benchmark_root(name: str) -> Path:
    return HEALTHFLOW_ROOT / "data" / name


def benchmark_raw_root(name: str) -> Path:
    return benchmark_root(name) / "raw"


def benchmark_processed_root(name: str) -> Path:
    return benchmark_root(name) / "processed"


def benchmark_runtime_root(name: str) -> Path:
    return benchmark_processed_root(name) / "runtime"


def benchmark_expected_root(name: str) -> Path:
    return benchmark_processed_root(name) / "expected"


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def text_dump(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def csv_dump(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

