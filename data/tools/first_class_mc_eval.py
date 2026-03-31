from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sort_qids(values: list[str]) -> list[str]:
    return sorted(values, key=lambda value: (0, f"{int(value):09d}") if value.isdigit() else (1, value))


def normalize_choice(value: Any, *, field_name: str, qid: str) -> str:
    normalized = str(value).strip().upper()
    if len(normalized) != 1 or not normalized.isalpha():
        raise ValueError(f"qid {qid}: {field_name} must be a single letter, found {value!r}")
    return normalized


def load_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"benchmark file is empty: {path}")

    seen_qids: set[str] = set()
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        if not qid:
            raise ValueError("benchmark row is missing qid")
        if qid in seen_qids:
            raise ValueError(f"duplicate benchmark qid: {qid}")
        seen_qids.add(qid)
        if "ground_truth" not in row:
            raise ValueError(f"benchmark row {qid} is missing ground_truth")
        normalize_choice(row["ground_truth"], field_name="ground_truth", qid=qid)
    return rows


def load_prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"predictions file is empty: {path}")

    predictions: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        if not qid:
            raise ValueError("prediction row is missing qid")
        if qid in predictions:
            raise ValueError(f"duplicate prediction qid: {qid}")
        if "predicted_answer" not in row:
            raise ValueError(f"prediction row {qid} is missing predicted_answer")
        predictions[qid] = row
    return predictions


def default_output_path(predictions_file: Path) -> Path:
    if predictions_file.suffix:
        return predictions_file.with_suffix(".eval.json")
    return predictions_file.parent / f"{predictions_file.name}.eval.json"


def evaluate_predictions(benchmark_file: Path, predictions_file: Path) -> dict[str, Any]:
    benchmark_rows = load_benchmark_rows(benchmark_file)
    prediction_rows = load_prediction_rows(predictions_file)

    benchmark_qids = [str(row["qid"]) for row in benchmark_rows]
    benchmark_qid_set = set(benchmark_qids)
    prediction_qid_set = set(prediction_rows)

    missing_qids = sort_qids(list(benchmark_qid_set - prediction_qid_set))
    unexpected_qids = sort_qids(list(prediction_qid_set - benchmark_qid_set))
    if missing_qids or unexpected_qids:
        raise ValueError(
            "prediction qids do not align with benchmark qids: "
            f"missing={missing_qids}, unexpected={unexpected_qids}"
        )

    results: list[dict[str, Any]] = []
    correct_questions = 0
    for benchmark_row in benchmark_rows:
        qid = str(benchmark_row["qid"])
        ground_truth = normalize_choice(benchmark_row["ground_truth"], field_name="ground_truth", qid=qid)
        predicted_answer = normalize_choice(
            prediction_rows[qid]["predicted_answer"],
            field_name="predicted_answer",
            qid=qid,
        )
        is_correct = predicted_answer == ground_truth
        correct_questions += int(is_correct)
        results.append(
            {
                "qid": benchmark_row["qid"],
                "predicted_answer": predicted_answer,
                "ground_truth": ground_truth,
                "correct": is_correct,
            }
        )

    total_questions = len(results)
    summary = {
        "benchmark_file": str(benchmark_file),
        "predictions_file": str(predictions_file),
        "total_questions": total_questions,
        "correct_questions": correct_questions,
        "accuracy": correct_questions / total_questions if total_questions else 0.0,
        "results": results,
    }
    return summary


def run_cli(default_benchmark_file: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate first-class multiple-choice benchmark predictions.")
    parser.add_argument(
        "--predictions-file",
        type=Path,
        required=True,
        help="Path to the jsonl file that contains qid and predicted_answer.",
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=Path(default_benchmark_file) if default_benchmark_file else None,
        help="Optional override for the reference processed eval jsonl.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output json path. Defaults to <predictions-file>.eval.json.",
    )
    args = parser.parse_args()

    if args.benchmark_file is None:
        raise ValueError("--benchmark-file is required when no default benchmark file is configured")

    output_file = args.output_file or default_output_path(args.predictions_file)
    payload = evaluate_predictions(args.benchmark_file, args.predictions_file)
    write_json(output_file, payload)

    print(
        json.dumps(
            {
                "total_questions": payload["total_questions"],
                "correct_questions": payload["correct_questions"],
                "accuracy": payload["accuracy"],
                "output_file": str(output_file),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run_cli()
