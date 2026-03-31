from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from healthflow.benchmarks.first_class_mc_eval import default_output_path, evaluate_predictions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class FirstClassMCEvalTests(unittest.TestCase):
    def test_evaluate_predictions_returns_accuracy_and_per_qid_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "benchmark.jsonl"
            predictions_path = tmp_path / "predictions.jsonl"

            write_jsonl(
                benchmark_path,
                [
                    {
                        "qid": 1,
                        "question": "Q1",
                        "question_type": "multi_choice",
                        "options": {"A": "x", "B": "y"},
                        "ground_truth": "A",
                    },
                    {
                        "qid": 2,
                        "question": "Q2",
                        "question_type": "multi_choice",
                        "options": {"A": "x", "B": "y"},
                        "ground_truth": "B",
                    },
                ],
            )
            write_jsonl(
                predictions_path,
                [
                    {"qid": 1, "predicted_answer": "a"},
                    {"qid": 2, "predicted_answer": "A"},
                ],
            )

            payload = evaluate_predictions(benchmark_path, predictions_path)
            self.assertEqual(payload["total_questions"], 2)
            self.assertEqual(payload["correct_questions"], 1)
            self.assertEqual(payload["accuracy"], 0.5)
            self.assertEqual(
                payload["results"],
                [
                    {"qid": 1, "predicted_answer": "A", "ground_truth": "A", "correct": True},
                    {"qid": 2, "predicted_answer": "A", "ground_truth": "B", "correct": False},
                ],
            )

    def test_evaluate_predictions_requires_exact_qid_alignment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "benchmark.jsonl"
            predictions_path = tmp_path / "predictions.jsonl"

            write_jsonl(
                benchmark_path,
                [
                    {
                        "qid": 1,
                        "question": "Q1",
                        "question_type": "multi_choice",
                        "options": {"A": "x", "B": "y"},
                        "ground_truth": "A",
                    }
                ],
            )
            write_jsonl(
                predictions_path,
                [
                    {"qid": 2, "predicted_answer": "A"},
                ],
            )

            with self.assertRaisesRegex(ValueError, "prediction qids do not align"):
                evaluate_predictions(benchmark_path, predictions_path)

    def test_default_output_path_replaces_jsonl_suffix(self):
        output_path = default_output_path(Path("predictions.jsonl"))
        self.assertEqual(output_path, Path("predictions.eval.json"))


if __name__ == "__main__":
    unittest.main()
