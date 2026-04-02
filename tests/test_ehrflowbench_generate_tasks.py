from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from data.ehrflowbench.scripts import generate_tasks
from data.ehrflowbench.scripts.generate_tasks import DatasetPromptConfig
from data.ehrflowbench.scripts.generate_tasks import LLMGeneratedTask
from data.ehrflowbench.scripts.generate_tasks import LLMGeneratedTaskBundle


FAKE_DATASET_CONFIGS = (
    DatasetPromptConfig(
        key="tjh",
        display_name="TJH",
        parquet_path=Path("/tmp/tjh.parquet"),
        split_metadata_path=Path("/tmp/tjh_split.json"),
        required_inputs=(
            "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
            "data/ehrflowbench/processed/tjh/split_metadata.json",
        ),
    ),
    DatasetPromptConfig(
        key="mimic_iv_demo",
        display_name="MIMIC-IV-demo",
        parquet_path=Path("/tmp/mimic.parquet"),
        split_metadata_path=Path("/tmp/mimic_split.json"),
        required_inputs=(
            "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet",
            "data/ehrflowbench/processed/mimic_iv_demo/split_metadata.json",
            "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_value_reference.md",
        ),
        value_reference_path=Path("/tmp/mimic_value_reference.md"),
    ),
)


class EHRFlowBenchGenerateTasksTests(TestCase):
    def test_adapt_prompt_injects_metadata_blocks(self) -> None:
        prompt_body = "\n".join(
            [
                "## Injected Local EHR Metadata",
                "",
                generate_tasks.PROMPT_DATASET_METADATA_PLACEHOLDER,
                "",
                "## Injected Task Allocation",
                "",
                generate_tasks.PROMPT_TASK_ASSIGNMENT_PLACEHOLDER,
            ]
        )

        with patch.object(generate_tasks, "DATASET_PROMPT_CONFIGS", FAKE_DATASET_CONFIGS):
            with patch.object(generate_tasks, "build_dataset_metadata_block", return_value="DATASET BLOCK"):
                with patch.object(generate_tasks, "build_task_assignment_block", return_value="ASSIGNMENT BLOCK"):
                    adapted = generate_tasks.adapt_prompt(prompt_body, task_count=2)

        self.assertIn("DATASET BLOCK", adapted)
        self.assertIn("ASSIGNMENT BLOCK", adapted)
        self.assertIn("Generate exactly 2 tasks.", adapted)
        self.assertIn("Do not mention `task_type`, `required_inputs`, or `report_requirements`", adapted)

    def test_enrich_generated_bundle_appends_fixed_fields(self) -> None:
        bundle = LLMGeneratedTaskBundle(
            tasks=[
                LLMGeneratedTask(
                    task_brief="Analyze temporal severity progression in TJH.",
                    focus_areas=["temporal modeling", "outcome analysis"],
                    task="Build a deterministic longitudinal analysis on the assigned dataset only.",
                    deliverables=["metrics.json", "figures/trajectory.png"],
                ),
                LLMGeneratedTask(
                    task_brief="Profile ICU physiologic phenotypes in MIMIC-IV-demo.",
                    focus_areas=["phenotyping", "descriptive analytics"],
                    task="Cluster patients in the assigned dataset only and report stable subgroup statistics.",
                    deliverables=["report.md", "tables/cluster_summary.csv"],
                ),
            ]
        )

        with patch.object(generate_tasks, "DATASET_PROMPT_CONFIGS", FAKE_DATASET_CONFIGS):
            enriched = generate_tasks.enrich_generated_bundle(bundle)

        self.assertEqual(enriched.tasks[0].task_type, "report_generation")
        self.assertEqual(enriched.tasks[0].required_inputs, list(FAKE_DATASET_CONFIGS[0].required_inputs))
        self.assertEqual(enriched.tasks[1].required_inputs, list(FAKE_DATASET_CONFIGS[1].required_inputs))
        self.assertEqual(len(enriched.tasks[0].report_requirements), 6)
        self.assertEqual(enriched.tasks[0].deliverables[0], "report.md")

    def test_enrich_generated_bundle_rejects_cross_dataset_mentions(self) -> None:
        bundle = LLMGeneratedTaskBundle(
            tasks=[
                LLMGeneratedTask(
                    task_brief="Analyze TJH and MIMIC together.",
                    focus_areas=["prediction", "comparison"],
                    task="Run the same workflow on TJH and MIMIC-IV-demo and compare them.",
                    deliverables=["report.md", "metrics.json"],
                ),
                LLMGeneratedTask(
                    task_brief="Profile ICU trajectories.",
                    focus_areas=["temporal analysis", "phenotyping"],
                    task="Use only the assigned dataset.",
                    deliverables=["report.md", "tables/summary.csv"],
                ),
            ]
        )

        with patch.object(generate_tasks, "DATASET_PROMPT_CONFIGS", FAKE_DATASET_CONFIGS):
            with self.assertRaisesRegex(ValueError, "single-task single-dataset contract violated"):
                generate_tasks.enrich_generated_bundle(bundle)
