from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from healthflow.benchmarks.common import (
    benchmark_processed_root,
    benchmark_raw_root,
    json_dump,
    load_jsonl,
    write_jsonl,
)


SimpleRowFilter = Callable[[dict[str, object]], bool]


@dataclass(frozen=True)
class SplitBuildResult:
    split: str
    row_count: int
    qids: list[str]


def canonicalize_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    canonical_rows: list[dict[str, object]] = []
    for row in rows:
        canonical_rows.append(
            {
                "qid": row["qid"],
                "task": row["task"],
                "answer": row["answer"],
            }
        )
    return canonical_rows


def sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def sort_key(row: dict[str, object]) -> tuple[int, str]:
        qid_text = str(row["qid"])
        return (0, f"{int(qid_text):09d}") if qid_text.isdigit() else (1, qid_text)

    return sorted(rows, key=sort_key)


def build_simple_processed_split(
    *,
    benchmark_name: str,
    source_filename: str,
    split_name: str,
    row_filter: SimpleRowFilter | None = None,
) -> SplitBuildResult:
    source_path = benchmark_raw_root(benchmark_name) / source_filename
    rows = load_jsonl(source_path)
    if row_filter is not None:
        rows = [row for row in rows if row_filter(row)]
    canonical_rows = sort_rows(canonicalize_rows(rows))
    write_jsonl(benchmark_processed_root(benchmark_name) / f"{split_name}.jsonl", canonical_rows)
    return SplitBuildResult(
        split=split_name,
        row_count=len(canonical_rows),
        qids=[str(row["qid"]) for row in canonical_rows],
    )


def write_subset_manifest(benchmark_name: str, manifest: dict[str, object]) -> None:
    json_dump(benchmark_processed_root(benchmark_name) / "subset_manifest.json", manifest)
