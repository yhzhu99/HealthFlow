import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from run_benchmark import aggregate_cost_totals, load_dataset


class BenchmarkRunnerTests(unittest.TestCase):
    def test_load_dataset_accepts_reference_answer_only(self):
        with TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.jsonl"
            dataset_path.write_text(
                '{"qid": 1, "task": "task one", "reference_answer": "reference_answers/1/answer_manifest.json"}\n',
                encoding="utf-8",
            )

            rows = load_dataset(dataset_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["answer"], "reference_answers/1/answer_manifest.json")
        self.assertEqual(rows[0]["reference_answer"], "reference_answers/1/answer_manifest.json")

    def test_aggregate_cost_totals_combines_stage_and_executor_costs(self):
        results = [
            {
                "cost_summary": {
                    "llm_estimated_cost_usd": 0.01,
                    "executor_estimated_cost_usd": 0.2,
                    "total_estimated_cost_usd": 0.21,
                },
                "cost_analysis": {
                    "run_total": {
                        "planning": {"estimated_cost_usd": 0.004},
                        "execution": {"estimated_cost_usd": 0.2},
                        "evaluation": {"estimated_cost_usd": 0.006},
                        "reflection": {},
                        "total_estimated_cost_usd": 0.21,
                    }
                },
            },
            {
                "cost_summary": {
                    "llm_estimated_cost_usd": 0.015,
                    "executor_estimated_cost_usd": 0.05,
                    "total_estimated_cost_usd": 0.065,
                },
                "cost_analysis": {
                    "run_total": {
                        "planning": {"estimated_cost_usd": 0.005},
                        "execution": {"estimated_cost_usd": 0.05},
                        "evaluation": {"estimated_cost_usd": 0.007},
                        "reflection": {"estimated_cost_usd": 0.003},
                        "total_estimated_cost_usd": 0.065,
                    }
                },
            },
        ]

        totals = aggregate_cost_totals(results)

        self.assertEqual(totals["total_llm_estimated_cost_usd"], 0.025)
        self.assertEqual(totals["total_executor_estimated_cost_usd"], 0.25)
        self.assertEqual(totals["total_estimated_cost_usd"], 0.275)
        self.assertEqual(
            totals["stage_cost_totals_usd"],
            {
                "planning": 0.009,
                "execution": 0.25,
                "evaluation": 0.013,
                "reflection": 0.003,
                "total": 0.275,
            },
        )


if __name__ == "__main__":
    unittest.main()
