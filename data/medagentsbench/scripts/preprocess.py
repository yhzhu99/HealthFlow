from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = BENCHMARK_ROOT / "raw"
PROCESSED_ROOT = BENCHMARK_ROOT / "processed"
TRAIN_OUTPUT_PATH = PROCESSED_ROOT / "train.jsonl"
TEST_OUTPUT_PATH = PROCESSED_ROOT / "test.jsonl"
MANIFEST_PATH = PROCESSED_ROOT / "subset_manifest.json"
SEED = 42
TRAIN_SIZE_PER_DATASET = 1
TEST_SIZE_PER_DATASET = 10
SAMPLE_SIZE_PER_DATASET = TRAIN_SIZE_PER_DATASET + TEST_SIZE_PER_DATASET
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


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = rows[:TRAIN_SIZE_PER_DATASET]
    test_rows = rows[TRAIN_SIZE_PER_DATASET:]
    if len(train_rows) != TRAIN_SIZE_PER_DATASET or len(test_rows) != TEST_SIZE_PER_DATASET:
        raise ValueError(
            f"unexpected split sizes: train={len(train_rows)} test={len(test_rows)} "
            f"expected train={TRAIN_SIZE_PER_DATASET} test={TEST_SIZE_PER_DATASET}"
        )
    train_rows.sort(key=lambda row: sort_key(row["source_id"]))
    test_rows.sort(key=lambda row: sort_key(row["source_id"]))
    return train_rows, test_rows


def build_split_output(
    rows: list[dict[str, Any]],
    *,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    processed_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for qid, row in enumerate(rows, start=1):
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
                "split": split,
                "qid": qid,
                "source_dataset": row["source_dataset"],
                "source_id": row["source_id"],
                "removed_empty_labels": row["removed_empty_labels"],
            }
        )
    return processed_rows, manifest_rows


def main() -> None:
    all_train_rows: list[dict[str, Any]] = []
    all_test_rows: list[dict[str, Any]] = []
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
        train_rows, test_rows = split_rows(sampled)
        all_train_rows.extend(train_rows)
        all_test_rows.extend(test_rows)
        manifest_subdatasets.append(
            {
                "name": dataset_name,
                "source_file": str(source_path.relative_to(BENCHMARK_ROOT)),
                "seed": SEED,
                "candidate_count": len(candidates),
                "selected_count": len(train_rows) + len(test_rows),
                "split_counts": {
                    "train": len(train_rows),
                    "test": len(test_rows),
                },
                "selected_rows": [
                    {
                        "source_id": row["source_id"],
                        "removed_empty_labels": row["removed_empty_labels"],
                        "split": "train",
                    }
                    for row in train_rows
                ]
                + [
                    {
                        "source_id": row["source_id"],
                        "removed_empty_labels": row["removed_empty_labels"],
                        "split": "test",
                    }
                    for row in test_rows
                ],
            }
        )

    train_processed_rows, train_manifest_rows = build_split_output(all_train_rows, split="train")
    test_processed_rows, test_manifest_rows = build_split_output(all_test_rows, split="test")
    manifest_rows = train_manifest_rows + test_manifest_rows

    manifest = {
        "dataset": "medagentsbench",
        "seed": SEED,
        "sample_size_per_dataset": SAMPLE_SIZE_PER_DATASET,
        "split_counts": {
            "train": len(train_processed_rows),
            "test": len(test_processed_rows),
        },
        "selected_count": len(train_processed_rows) + len(test_processed_rows),
        "subdatasets": manifest_subdatasets,
        "selected_rows": manifest_rows,
    }

    write_jsonl(TRAIN_OUTPUT_PATH, train_processed_rows)
    write_jsonl(TEST_OUTPUT_PATH, test_processed_rows)
    write_json(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
