from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "evaluation"
if str(EVAL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPTS_DIR))

from deterministic_benchmark_eval import compare_csv, compare_json  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class BenchmarkRebuildTests(unittest.TestCase):
    def test_canonical_benchmark_counts_and_ground_truth(self):
        medagentboard = load_jsonl(PAPER_ROOT / "data" / "medagentboard.jsonl")
        ehr_eval = load_jsonl(PAPER_ROOT / "data" / "ehrflowbench.jsonl")
        ehr_train = load_jsonl(PAPER_ROOT / "data" / "ehrflowbench_train.jsonl")

        self.assertEqual(len(medagentboard), 100)
        self.assertEqual(len(ehr_eval), 100)
        self.assertEqual(len(ehr_train), 10)

        self.assertEqual(pd.Series([row["dataset"] for row in medagentboard]).value_counts().to_dict(), {"TJH": 50, "MIMIC-IV-demo-MEDS": 50})
        self.assertEqual(pd.Series([row["dataset"] for row in ehr_eval]).value_counts().to_dict(), {"TJH": 50, "MIMIC-IV-demo-MEDS": 50})
        self.assertEqual(pd.Series([row["dataset"] for row in ehr_train]).value_counts().to_dict(), {"TJH": 5, "MIMIC-IV-demo-MEDS": 5})

        for rows in [medagentboard, ehr_eval, ehr_train]:
            self.assertEqual(len(rows), len({row["qid"] for row in rows}))
            for row in rows:
                self.assertTrue(row["answer"])
                self.assertTrue(row["required_files"])
                self.assertTrue(row["verification_spec"]["files"])
                self.assertNotIn("/home/projects/HealthFlow", json.dumps(row, ensure_ascii=False))
                ground_truth_dir = PAPER_ROOT / row["ground_truth_ref"]
                self.assertTrue(ground_truth_dir.exists(), ground_truth_dir)
                for relative_name in row["required_files"]:
                    self.assertTrue((ground_truth_dir / relative_name).exists(), ground_truth_dir / relative_name)

    def test_deterministic_comparators(self):
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


if __name__ == "__main__":
    unittest.main()
