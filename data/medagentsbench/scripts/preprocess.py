from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = BENCHMARK_ROOT / "raw"
PROCESSED_ROOT = BENCHMARK_ROOT / "processed"
OUTPUT_PATH = PROCESSED_ROOT / "eval.jsonl"
MANIFEST_PATH = PROCESSED_ROOT / "subset_manifest.json"
SEED = 42
SAMPLE_SIZE_PER_DATASET = 10
SUBDATASETS = [
    "AfrimedQA",
    "MMLU",
    "MMLU-Pro",
    "MedBullets",
    "MedExQA",
    "MedMCQA",
    "MedQA",
    "MedXpertQA-R",
    "MedXpertQA-U",
    "PubMedQA",
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):09d}") if value.isdigit() else (1, value)


def normalize_options(raw_options: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    if not isinstance(raw_options, dict):
        raise ValueError("options must be a dict")

    normalized: dict[str, str] = {}
    removed_empty_labels: list[str] = []
    for label, text in raw_options.items():
        normalized_label = str(label).strip().upper()
        normalized_text = "" if text is None else str(text).strip()
        if not normalized_text:
            removed_empty_labels.append(normalized_label)
            continue
        normalized[normalized_label] = normalized_text

    if len(normalized) < 2:
        raise ValueError("expected at least two non-empty options")

    return normalized, removed_empty_labels


def main() -> None:
    all_selected_rows: list[dict[str, Any]] = []
    manifest_subdatasets: list[dict[str, Any]] = []

    for dataset_name in SUBDATASETS:
        source_path = RAW_ROOT / dataset_name / "test_hard-00000-of-00001.parquet"
        frame = pd.read_parquet(source_path)

        candidates: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            source_id = str(row["realidx"]).strip()
            question = str(row["question"]).strip()
            ground_truth = str(row["answer_idx"]).strip().upper()
            options, removed_empty_labels = normalize_options(row["options"])

            if not question:
                raise ValueError(f"{dataset_name}:{source_id}: empty question")
            if ground_truth not in options:
                raise ValueError(f"{dataset_name}:{source_id}: invalid ground truth {ground_truth}")

            candidates.append(
                {
                    "source_dataset": dataset_name,
                    "source_id": source_id,
                    "question": question,
                    "question_type": "multi_choice",
                    "options": options,
                    "ground_truth": ground_truth,
                    "removed_empty_labels": removed_empty_labels,
                }
            )

        candidates.sort(key=lambda row: sort_key(row["source_id"]))
        if len(candidates) < SAMPLE_SIZE_PER_DATASET:
            raise ValueError(
                f"{dataset_name}: not enough candidates: expected {SAMPLE_SIZE_PER_DATASET}, found {len(candidates)}"
            )

        sampled = random.Random(SEED).sample(candidates, SAMPLE_SIZE_PER_DATASET)
        sampled.sort(key=lambda row: sort_key(row["source_id"]))
        all_selected_rows.extend(sampled)
        manifest_subdatasets.append(
            {
                "name": dataset_name,
                "source_file": str(source_path.relative_to(BENCHMARK_ROOT)),
                "seed": SEED,
                "candidate_count": len(candidates),
                "selected_count": len(sampled),
                "selected_rows": [
                    {
                        "source_id": row["source_id"],
                        "removed_empty_labels": row["removed_empty_labels"],
                    }
                    for row in sampled
                ],
            }
        )

    processed_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for qid, row in enumerate(all_selected_rows, start=1):
        processed_rows.append(
            {
                "qid": qid,
                "question": row["question"],
                "question_type": row["question_type"],
                "options": row["options"],
                "ground_truth": row["ground_truth"],
            }
        )
        manifest_rows.append(
            {
                "qid": qid,
                "source_dataset": row["source_dataset"],
                "source_id": row["source_id"],
                "removed_empty_labels": row["removed_empty_labels"],
            }
        )

    manifest = {
        "dataset": "medagentsbench",
        "seed": SEED,
        "sample_size_per_dataset": SAMPLE_SIZE_PER_DATASET,
        "selected_count": len(processed_rows),
        "subdatasets": manifest_subdatasets,
        "selected_rows": manifest_rows,
    }

    write_jsonl(OUTPUT_PATH, processed_rows)
    write_json(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
