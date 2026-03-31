from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from healthflow.benchmarks.deterministic_eval import compare_csv, compare_json, evaluate_single_task, load_benchmark_rows

HEALTHFLOW_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class BenchmarkRebuildTests(unittest.TestCase):
    def assert_legacy_row(self, row: dict) -> None:
        self.assertEqual(set(row), {"qid", "task", "answer"})
        self.assertTrue(str(row["task"]).strip())
        self.assertNotEqual(row["answer"], "")
        row_text = json.dumps(row, ensure_ascii=False)
        self.assertNotIn("/home/projects/HealthFlow", row_text)
        self.assertNotIn('"sha256"', row_text)

    def assert_first_class_row(self, row: dict) -> None:
        self.assertEqual(set(row), {"qid", "question", "question_type", "options", "ground_truth"})
        self.assertTrue(isinstance(row["qid"], int))
        self.assertTrue(str(row["question"]).strip())
        self.assertEqual(row["question_type"], "multi_choice")
        self.assertTrue(isinstance(row["options"], dict))
        self.assertGreaterEqual(len(row["options"]), 2)
        for label, option_text in row["options"].items():
            self.assertTrue(str(label).isalpha())
            self.assertEqual(str(label), str(label).upper())
            self.assertTrue(str(option_text).strip())
        self.assertEqual(len(str(row["ground_truth"])), 1)
        self.assertIn(row["ground_truth"], row["options"])
        row_text = json.dumps(row, ensure_ascii=False)
        self.assertNotIn("/home/projects/HealthFlow", row_text)
        self.assertNotIn('"sha256"', row_text)

    def test_canonical_processed_counts_and_schema(self):
        legacy_cases = {
            ("medagentboard", "eval"): 100,
            ("ehrflowbench", "eval"): 100,
            ("ehrflowbench", "train"): 10,
        }
        first_class_cases = {
            ("curebench", "eval"): 100,
            ("hle", "eval"): 100,
            ("medagentsbench", "eval"): 100,
        }

        for (benchmark, split), expected_count in legacy_cases.items():
            rows = load_jsonl(HEALTHFLOW_ROOT / "data" / benchmark / "processed" / f"{split}.jsonl")
            self.assertEqual(len(rows), expected_count)
            self.assertEqual(len(rows), len({str(row["qid"]) for row in rows}))
            for row in rows:
                self.assert_legacy_row(row)

        for (benchmark, split), expected_count in first_class_cases.items():
            rows = load_jsonl(HEALTHFLOW_ROOT / "data" / benchmark / "processed" / f"{split}.jsonl")
            self.assertEqual(len(rows), expected_count)
            self.assertEqual(len(rows), len({str(row["qid"]) for row in rows}))
            self.assertEqual([row["qid"] for row in rows], list(range(1, expected_count + 1)))
            for row in rows:
                self.assert_first_class_row(row)

    def test_subset_manifests_exist(self):
        for benchmark in ["curebench", "hle", "medagentsbench", "medagentboard", "ehrflowbench"]:
            subset_manifest = HEALTHFLOW_ROOT / "data" / benchmark / "processed" / "subset_manifest.json"
            self.assertTrue(subset_manifest.exists(), subset_manifest)

    def test_first_class_subset_manifests_capture_sampling_details(self):
        curebench_manifest = json.loads(
            (HEALTHFLOW_ROOT / "data" / "curebench" / "processed" / "subset_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(curebench_manifest["dataset"], "curebench")
        self.assertEqual(curebench_manifest["seed"], 42)
        self.assertEqual(curebench_manifest["candidate_count"], 413)
        self.assertEqual(curebench_manifest["selected_count"], 100)
        self.assertEqual(len(curebench_manifest["selected_rows"]), 100)

        hle_manifest = json.loads(
            (HEALTHFLOW_ROOT / "data" / "hle" / "processed" / "subset_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(hle_manifest["dataset"], "hle")
        self.assertEqual(hle_manifest["seed"], 42)
        self.assertEqual(hle_manifest["filters"]["category"], "Biology/Medicine")
        self.assertEqual(hle_manifest["filters"]["answer_type"], "multipleChoice")
        self.assertEqual(hle_manifest["image_filtered_count"], 19)
        self.assertEqual(hle_manifest["parse_rejection_count"], 0)
        self.assertEqual(hle_manifest["candidate_count"], 147)
        self.assertEqual(hle_manifest["selected_count"], 100)
        self.assertEqual(len(hle_manifest["selected_rows"]), 100)

        medagents_manifest = json.loads(
            (HEALTHFLOW_ROOT / "data" / "medagentsbench" / "processed" / "subset_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(medagents_manifest["dataset"], "medagentsbench")
        self.assertEqual(medagents_manifest["seed"], 42)
        self.assertEqual(medagents_manifest["sample_size_per_dataset"], 10)
        self.assertEqual(medagents_manifest["selected_count"], 100)
        self.assertEqual(len(medagents_manifest["selected_rows"]), 100)
        self.assertEqual([item["name"] for item in medagents_manifest["subdatasets"]], [
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
        ])
        self.assertTrue(all(item["selected_count"] == 10 for item in medagents_manifest["subdatasets"]))

    def test_expected_artifacts_exist_for_file_verified_benchmarks(self):
        cases = [
            ("medagentboard", "eval"),
            ("ehrflowbench", "eval"),
            ("ehrflowbench", "train"),
        ]
        for benchmark, split in cases:
            rows = load_jsonl(HEALTHFLOW_ROOT / "data" / benchmark / "processed" / f"{split}.jsonl")
            expected_root = HEALTHFLOW_ROOT / "data" / benchmark / "processed" / "expected"
            for row in rows:
                expected_dir = expected_root / str(row["qid"])
                self.assertTrue(expected_dir.exists(), expected_dir)
                self.assertTrue(any(path.is_file() for path in expected_dir.rglob("*")), expected_dir)

    def test_ehrflowbench_paper_map_covers_all_tasks(self):
        eval_rows = load_jsonl(HEALTHFLOW_ROOT / "data" / "ehrflowbench" / "processed" / "eval.jsonl")
        train_rows = load_jsonl(HEALTHFLOW_ROOT / "data" / "ehrflowbench" / "processed" / "train.jsonl")
        all_qids = {str(row["qid"]) for row in eval_rows + train_rows}

        paper_map_path = HEALTHFLOW_ROOT / "data" / "ehrflowbench" / "processed" / "paper_map.csv"
        rejected_path = HEALTHFLOW_ROOT / "data" / "ehrflowbench" / "processed" / "rejected_tasks.csv"
        self.assertTrue(paper_map_path.exists())
        self.assertTrue(rejected_path.exists())

        with paper_map_path.open("r", encoding="utf-8") as handle:
            paper_rows = list(csv.DictReader(handle))

        self.assertEqual(len(paper_rows), 110)
        self.assertEqual({row["qid"] for row in paper_rows}, all_qids)
        for row in paper_rows:
            self.assertTrue(row["source_paper_id"])
            self.assertTrue(row["source_paper_title"])
            self.assertFalse(str(row["source_paper_title"]).isdigit())
            self.assertEqual(row["source_task_eligibility"], "proxy_candidate")
            self.assertEqual(row["linkage_mode"], "task_proxy")
            self.assertEqual(row["approval_status"], "seeded_for_human_review")

    def test_deterministic_comparators_and_file_eval(self):
        self.assertEqual(compare_json({"a": 1.0, "b": [2, 3]}, {"a": 1.0000001, "b": [2, 3]}, 1e-5)[0], True)
        self.assertEqual(compare_json({"a": 1.0}, {"a": 1.2}, 1e-6)[0], False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expected = tmp_path / "expected.csv"
            actual = tmp_path / "actual.csv"
            pd.DataFrame({"id": [2, 1], "value": [1.5, 2.5]}).to_csv(expected, index=False)
            pd.DataFrame({"id": [1, 2], "value": [2.5, 1.5]}).to_csv(actual, index=False)
            matched, _ = compare_csv(expected, actual, 1e-6)
            self.assertTrue(matched)

        benchmark_file = HEALTHFLOW_ROOT / "data" / "medagentboard" / "processed" / "eval.jsonl"
        expected_root = HEALTHFLOW_ROOT / "data" / "medagentboard" / "processed" / "expected"
        benchmark_rows = load_benchmark_rows(benchmark_file)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            actual_qid_dir = tmp_path / "1"
            shutil.copytree(expected_root / "1", actual_qid_dir)
            evaluation = evaluate_single_task(actual_qid_dir, {"qid": "1"}, benchmark_rows, benchmark_file, None)
            self.assertTrue(evaluation["passed"])
            self.assertEqual(evaluation["comparison_mode"], "files")


if __name__ == "__main__":
    unittest.main()
