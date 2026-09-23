import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from data.ehrflowbench.scripts.prepare_tasks.select_balanced_subset import run_selection


def build_source_task(*, dataset_key: str, paper_id: int, task_idx: int) -> dict:
    if dataset_key == "tjh":
        required_inputs = [
            "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
            "data/ehrflowbench/processed/tjh/split_metadata.json",
        ]
        task_text = f"Use only TJH and write report #{task_idx}."
    else:
        required_inputs = [
            "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet",
            "data/ehrflowbench/processed/mimic_iv_demo/split_metadata.json",
            "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_value_reference.md",
        ]
        task_text = f"Use only MIMIC-IV-demo and write report #{task_idx}."
    return {
        "task_brief": f"task {paper_id}-{task_idx}",
        "task_type": "report_generation",
        "focus_areas": ["prediction", "temporal modeling"],
        "task": task_text,
        "required_inputs": required_inputs,
        "deliverables": ["report.md", "metrics.json", "tables/result.csv", "figures/overview.png"],
        "report_requirements": [
            "State the objective.",
            "Describe the data.",
            "Explain the method.",
            "Report quantitative results.",
            "Provide figure and/or table evidence.",
            "State the final conclusion.",
        ],
        "paper_id": paper_id,
        "paper_title": f"Paper {paper_id}",
        "source_task_idx": task_idx,
    }


class EHRFlowBenchExportSubsetTests(TestCase):
    def test_run_selection_exports_raw_tasks_without_prompt_wrapping(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "processed"
            output_root.mkdir(parents=True, exist_ok=True)
            reference_root = output_root / "reference_answers"
            input_path = root / "final_220_tasks.json"

            tasks = []
            for paper_id in (1, 2, 3):
                tasks.append(build_source_task(dataset_key="tjh", paper_id=paper_id, task_idx=1))
                tasks.append(build_source_task(dataset_key="mimic_iv_demo", paper_id=paper_id, task_idx=2))
            input_path.write_text(
                json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            summary = run_selection(
                input_path=input_path,
                train_path=output_root / "train.jsonl",
                test_path=output_root / "test.jsonl",
                combined_path=output_root / "ehrflowbench.jsonl",
                subset_manifest_path=output_root / "subset_manifest.json",
                distribution_report_path=output_root / "subset_distribution.md",
                reference_root=reference_root,
                seed=42,
                select_count_per_dataset=2,
                train_count_per_dataset=1,
            )

            self.assertEqual(summary["selected_task_count"], 4)
            self.assertEqual(summary["train_task_count"], 2)
            self.assertEqual(summary["test_task_count"], 2)
            self.assertEqual(summary["selected_dataset_counts"], {"TJH": 2, "MIMIC-IV-demo": 2})

            train_rows = [
                json.loads(line)
                for line in (output_root / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            test_rows = [
                json.loads(line)
                for line in (output_root / "test.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            combined_rows = [
                json.loads(line)
                for line in (output_root / "ehrflowbench.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual([row["qid"] for row in train_rows], [1, 2])
            self.assertEqual([row["qid"] for row in test_rows], [1, 2])
            self.assertEqual([row["qid"] for row in combined_rows], [1, 2, 3, 4])
            self.assertEqual({row["dataset"] for row in combined_rows}, {"TJH", "MIMIC-IV-demo"})
            self.assertIn("Use only", combined_rows[0]["task"])
            self.assertNotIn("As an expert AI agent", combined_rows[0]["task"])

            manifest = json.loads(
                (reference_root / "train" / "1" / "answer_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                sorted(manifest.keys()),
                ["dataset", "primary_category", "qid", "required_inputs", "required_outputs", "task_type"],
            )
            self.assertEqual(
                [item["file_name"] for item in manifest["required_outputs"]],
                ["report.md", "metrics.json", "tables/result.csv", "figures/overview.png"],
            )
            self.assertEqual(
                [item["reference_path"] for item in manifest["required_outputs"]],
                [
                    "reference_answers/train/1/report.md",
                    "reference_answers/train/1/metrics.json",
                    "reference_answers/train/1/tables/result.csv",
                    "reference_answers/train/1/figures/overview.png",
                ],
            )
            self.assertFalse((reference_root / "train" / "1" / "report.md").exists())

            subset_manifest = json.loads((output_root / "subset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(subset_manifest["selection_seed"], 42)
            self.assertEqual(subset_manifest["selected_task_count"], 4)
            self.assertEqual(subset_manifest["dataset_counts"]["selected"], {"TJH": 2, "MIMIC-IV-demo": 2})
            self.assertEqual(subset_manifest["dataset_counts"]["train"], {"TJH": 1, "MIMIC-IV-demo": 1})
            self.assertEqual(subset_manifest["dataset_counts"]["test"], {"TJH": 1, "MIMIC-IV-demo": 1})
            self.assertEqual(len(subset_manifest["selected_tasks"]), 4)
            self.assertTrue((output_root / "subset_distribution.md").exists())
