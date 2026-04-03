import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from data.ehrflowbench.scripts.curate_generated_tasks import run_curation


def build_generated_task(*, dataset_key: str, task_idx: int) -> dict:
    if dataset_key == "tjh":
        required_inputs = [
            "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
            "data/ehrflowbench/processed/tjh/split_metadata.json",
        ]
        dataset_name = "TJH"
    else:
        required_inputs = [
            "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet",
            "data/ehrflowbench/processed/mimic_iv_demo/split_metadata.json",
            "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_value_reference.md",
        ]
        dataset_name = "MIMIC-IV-demo"
    return {
        "task_brief": f"{dataset_name} task {task_idx}",
        "task_type": "report_generation",
        "focus_areas": ["prediction", "temporal modeling"],
        "task": f"Use only the {dataset_name} dataset and produce a deterministic report.",
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
    }


class EHRFlowBenchExportSubsetTests(TestCase):
    def test_run_curation_writes_medagentboard_style_subset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generated_root = root / "generated_tasks"
            generated_root.mkdir(parents=True, exist_ok=True)
            paper_titles_path = root / "paper_titles.csv"
            output_root = root / "processed"

            paper_titles_path.write_text(
                "\n".join(
                    [
                        "paper_id,paper_title",
                        "1,Temporal prediction paper",
                        "2,Graph paper",
                        "3,Interpretability paper",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            for paper_id in (1, 2, 3):
                payload = {
                    "tasks": [
                        build_generated_task(dataset_key="tjh", task_idx=1),
                        build_generated_task(dataset_key="mimic_iv_demo", task_idx=2),
                    ]
                }
                (generated_root / f"{paper_id}_tasks.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                (generated_root / f"{paper_id}_response.json").write_text(
                    json.dumps({"paper_id": paper_id}) + "\n",
                    encoding="utf-8",
                )

            summary = run_curation(
                seed=42,
                sample_count_per_dataset=2,
                train_count_per_dataset=1,
                output_root=output_root,
                generated_tasks_root=generated_root,
                paper_titles_path=paper_titles_path,
            )

            self.assertEqual(summary["selected_count"], 4)
            self.assertEqual(summary["train_count"], 2)
            self.assertEqual(summary["test_count"], 2)

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
            self.assertEqual({row["dataset"] for row in train_rows}, {"TJH", "MIMIC-IV-demo"})
            self.assertEqual({row["dataset"] for row in test_rows}, {"TJH", "MIMIC-IV-demo"})

            manifest_train = json.loads(
                (output_root / "reference_answers" / "train" / "1" / "answer_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_train["task_type"], "report_generation")
            self.assertEqual(manifest_train["judge_prompt_type"], "report_generation")
            self.assertEqual(
                [item["file_name"] for item in manifest_train["required_outputs"]],
                ["report.md", "metrics.json", "tables/result.csv", "figures/overview.png"],
            )
            self.assertFalse((output_root / "reference_answers" / "train" / "1" / "report.md").exists())

            subset_manifest = json.loads((output_root / "subset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(subset_manifest["combined"]["count"], 4)
            self.assertEqual(subset_manifest["train"]["count"], 2)
            self.assertEqual(subset_manifest["test"]["count"], 2)
            self.assertEqual(
                subset_manifest["combined"]["dataset_counts"],
                {"TJH": 2, "MIMIC-IV-demo": 2},
            )
