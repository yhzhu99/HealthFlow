import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace

from data.ehrflowbench.scripts import generate_tasks
from data.ehrflowbench.scripts.generate_tasks import DatasetPromptConfig
from data.ehrflowbench.scripts.generate_tasks import LLMGeneratedTask
from data.ehrflowbench.scripts.generate_tasks import LLMGeneratedTaskBundle
from data.ehrflowbench.scripts.generate_tasks import PaperPaths


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
        self.assertNotIn(generate_tasks.PROMPT_DATASET_METADATA_PLACEHOLDER, adapted)
        self.assertNotIn(generate_tasks.PROMPT_TASK_ASSIGNMENT_PLACEHOLDER, adapted)

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

    def test_write_prompt_output_writes_prompt_file(self) -> None:
        paper_paths = PaperPaths(
            paper_id=7,
            paper_dir=Path("/tmp/7_paper"),
            pdf_path=Path("/tmp/7_paper/paper.pdf"),
            markdown_path=Path("/tmp/7_paper/full.md"),
        )

        with TemporaryDirectory() as tmpdir:
            prompt_path = generate_tasks.write_prompt_output(
                Path(tmpdir),
                paper_paths,
                "Prompt body for local inspection.",
                overwrite=False,
            )

            self.assertEqual(prompt_path.name, "7_prompt.md")
            content = prompt_path.read_text(encoding="utf-8")
            self.assertIn("# Prompt Input", content)
            self.assertIn("- pdf_path: `/tmp/7_paper/paper.pdf`", content)
            self.assertIn("## Prompt Text", content)
            self.assertIn("Prompt body for local inspection.", content)

    def test_main_output_prompt_only_skips_api_calls(self) -> None:
        args = SimpleNamespace(
            paper_id=7,
            paper_dir=None,
            model_key="unused",
            task_count=2,
            output_prompt_only=True,
            max_output_tokens=123,
            output_dir=Path("/tmp/generated_tasks"),
            overwrite=False,
        )
        paper_paths = PaperPaths(
            paper_id=7,
            paper_dir=Path("/tmp/7_paper"),
            pdf_path=Path("/tmp/7_paper/paper.pdf"),
            markdown_path=None,
        )

        with patch.object(generate_tasks, "ARGS", args, create=True):
            with patch.object(generate_tasks, "resolve_paper_paths", return_value=paper_paths):
                with patch.object(generate_tasks, "extract_prompt_body", return_value="prompt template"):
                    with patch.object(generate_tasks, "adapt_prompt", return_value="final prompt"):
                        with patch.object(
                            generate_tasks,
                            "write_prompt_output",
                            return_value=Path("/tmp/generated_tasks/7_prompt.md"),
                        ) as write_prompt_output:
                            with patch.object(generate_tasks, "load_llm_config", side_effect=AssertionError("unexpected")):
                                with patch.object(generate_tasks, "build_client", side_effect=AssertionError("unexpected")):
                                    with patch.object(
                                        generate_tasks,
                                        "call_generation_api",
                                        side_effect=AssertionError("unexpected"),
                                    ):
                                        with patch("builtins.print") as mock_print:
                                            generate_tasks.main()

        write_prompt_output.assert_called_once_with(
            Path("/tmp/generated_tasks"),
            paper_paths,
            "final prompt",
            False,
        )
        payload = json.loads(mock_print.call_args.args[0])
        self.assertEqual(payload["paper_id"], 7)
        self.assertEqual(payload["prompt_path"], "/tmp/generated_tasks/7_prompt.md")
        self.assertTrue(payload["output_prompt_only"])
