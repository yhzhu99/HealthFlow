from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = BENCHMARK_ROOT / "raw" / "curebench_valset_phase1.jsonl"
PROCESSED_ROOT = BENCHMARK_ROOT / "processed"
TRAIN_OUTPUT_PATH = PROCESSED_ROOT / "train.jsonl"
TEST_OUTPUT_PATH = PROCESSED_ROOT / "test.jsonl"
MANIFEST_PATH = PROCESSED_ROOT / "subset_manifest.json"
SEED = 42
TRAIN_SIZE = 10
TEST_SIZE = 100
SAMPLE_SIZE = TRAIN_SIZE + TEST_SIZE
ALLOWED_QUESTION_TYPES = {"multi_choice", "open_ended_multi_choice"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def normalize_options(raw_options: dict[str, Any]) -> dict[str, str]:
    if not isinstance(raw_options, dict):
        raise ValueError("options must be a dict")
    normalized: dict[str, str] = {}
    for label, text in raw_options.items():
        normalized_label = str(label).strip().upper()
        normalized_text = str(text).strip()
        if not normalized_text:
            raise ValueError(f"empty option text for {normalized_label}")
        normalized[normalized_label] = normalized_text
    if len(normalized) < 2:
        raise ValueError("expected at least two options")
    return normalized


def build_candidate(row: dict[str, Any]) -> dict[str, Any]:
    source_id = str(row["id"]).strip()
    source_question_type = str(row["question_type"]).strip()
    question = str(row["question"]).strip()
    ground_truth = str(row["correct_answer"]).strip().upper()
    options = normalize_options(row["options"])

    if not question:
        raise ValueError(f"{source_id}: empty question")
    if ground_truth not in options:
        raise ValueError(f"{source_id}: invalid ground truth {ground_truth}")

    return {
        "source_id": source_id,
        "source_question_type": source_question_type,
        "question": question,
        "question_type": "multi_choice",
        "options": options,
        "ground_truth": ground_truth,
    }


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = rows[:TRAIN_SIZE]
    test_rows = rows[TRAIN_SIZE:]
    if len(train_rows) != TRAIN_SIZE or len(test_rows) != TEST_SIZE:
        raise ValueError(
            f"unexpected split sizes: train={len(train_rows)} test={len(test_rows)} "
            f"expected train={TRAIN_SIZE} test={TEST_SIZE}"
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
                "source_id": row["source_id"],
                "source_question_type": row["source_question_type"],
            }
        )
    return processed_rows, manifest_rows


def main() -> None:
    source_rows = load_jsonl(RAW_PATH)
    question_type_counts = Counter(str(row.get("question_type", "")).strip() for row in source_rows)

    candidates = [
        build_candidate(row)
        for row in source_rows
        if str(row.get("question_type", "")).strip() in ALLOWED_QUESTION_TYPES
    ]
    candidates.sort(key=lambda row: sort_key(row["source_id"]))
    if len(candidates) < SAMPLE_SIZE:
        raise ValueError(f"not enough candidates: expected {SAMPLE_SIZE}, found {len(candidates)}")

    sampled = random.Random(SEED).sample(candidates, SAMPLE_SIZE)
    train_rows, test_rows = split_rows(sampled)

    train_processed_rows, train_manifest_rows = build_split_output(train_rows, split="train")
    test_processed_rows, test_manifest_rows = build_split_output(test_rows, split="test")
    manifest_rows = train_manifest_rows + test_manifest_rows

    manifest = {
        "dataset": "curebench",
        "source_file": str(RAW_PATH.relative_to(BENCHMARK_ROOT)),
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "split_counts": {
            "train": len(train_processed_rows),
            "test": len(test_processed_rows),
        },
        "filters": {
            "allowed_question_types": sorted(ALLOWED_QUESTION_TYPES),
        },
        "source_question_type_counts": dict(question_type_counts),
        "candidate_count": len(candidates),
        "selected_count": len(train_processed_rows) + len(test_processed_rows),
        "selected_rows": manifest_rows,
    }

    write_jsonl(TRAIN_OUTPUT_PATH, train_processed_rows)
    write_jsonl(TEST_OUTPUT_PATH, test_processed_rows)
    write_json(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
