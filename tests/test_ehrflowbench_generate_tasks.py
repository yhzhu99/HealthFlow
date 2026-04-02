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
        dataset_config = FAKE_DATASET_CONFIGS[0]

        with patch.object(generate_tasks, "build_dataset_metadata_block", return_value="DATASET BLOCK") as build_dataset_metadata_block:
            with patch.object(generate_tasks, "build_task_assignment_block", return_value="ASSIGNMENT BLOCK") as build_task_assignment_block:
                adapted = generate_tasks.adapt_prompt(prompt_body, task_count=2, dataset_config=dataset_config)

        self.assertIn("DATASET BLOCK", adapted)
        self.assertIn("ASSIGNMENT BLOCK", adapted)
        self.assertNotIn(generate_tasks.PROMPT_DATASET_METADATA_PLACEHOLDER, adapted)
        self.assertNotIn(generate_tasks.PROMPT_TASK_ASSIGNMENT_PLACEHOLDER, adapted)
        build_dataset_metadata_block.assert_called_once_with((dataset_config,))
        build_task_assignment_block.assert_called_once_with(dataset_config)

    def test_extract_generated_task_appends_fixed_fields(self) -> None:
        bundle = LLMGeneratedTaskBundle(
            task=LLMGeneratedTask(
                task_brief="Analyze temporal severity progression in TJH.",
                focus_areas=["temporal modeling", "outcome analysis"],
                task="Build a deterministic longitudinal analysis on the assigned dataset only.",
                deliverables=["metrics.json", "figures/trajectory.png"],
            )
        )

        enriched = generate_tasks.extract_generated_task(bundle, FAKE_DATASET_CONFIGS[0])

        self.assertEqual(enriched.task_type, "report_generation")
        self.assertEqual(enriched.required_inputs, list(FAKE_DATASET_CONFIGS[0].required_inputs))
        self.assertEqual(len(enriched.report_requirements), 6)
        self.assertEqual(enriched.deliverables[0], "report.md")

    def test_extract_generated_task_rejects_cross_dataset_mentions(self) -> None:
        bundle = LLMGeneratedTaskBundle(
            task=LLMGeneratedTask(
                task_brief="Analyze TJH and MIMIC together.",
                focus_areas=["prediction", "comparison"],
                task="Run the same workflow on TJH and MIMIC-IV-demo and compare them.",
                deliverables=["report.md", "metrics.json"],
            )
        )

        with self.assertRaisesRegex(ValueError, "single-task single-dataset contract violated"):
            generate_tasks.extract_generated_task(bundle, FAKE_DATASET_CONFIGS[0])

    def test_call_generation_api_uses_bundle_text_format(self) -> None:
        parsed_bundle = LLMGeneratedTaskBundle(
            task=LLMGeneratedTask(
                task_brief="Analyze temporal severity progression in TJH.",
                focus_areas=["temporal modeling", "outcome analysis"],
                task="Build a deterministic longitudinal analysis on the assigned dataset only.",
                deliverables=["metrics.json", "figures/trajectory.png"],
            )
        )
        response = SimpleNamespace(
            id="resp-1",
            output_parsed=parsed_bundle,
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            output_text='{"task":{}}',
        )
        client = SimpleNamespace(responses=SimpleNamespace(parse=lambda **kwargs: response))
        llm_config = generate_tasks.LLMConfig(
            api_key_env="OPENAI_API_KEY",
            base_url="https://example.com/v1",
            model_name="test-model",
            reasoning_effort="medium",
            input_cost_per_million_tokens=1.0,
            output_cost_per_million_tokens=2.0,
        )
        paper_paths = PaperPaths(
            paper_id=7,
            paper_dir=Path("/tmp/7_paper"),
            pdf_path=Path("/tmp/7_paper/paper.pdf"),
            markdown_path=None,
        )

        with patch.object(generate_tasks, "build_generation_request_input", return_value=[{"role": "user"}]):
            with patch.object(client.responses, "parse", return_value=response) as parse:
                task, metadata = generate_tasks.call_generation_api(
                    client=client,
                    llm_config=llm_config,
                    prompt_text="prompt",
                    paper_paths=paper_paths,
                    dataset_config=FAKE_DATASET_CONFIGS[0],
                    max_output_tokens=123,
                )

        self.assertEqual(task.task_brief, "Analyze temporal severity progression in TJH.")
        self.assertEqual(metadata["response_id"], "resp-1")
        self.assertEqual(parse.call_args.kwargs["text_format"], LLMGeneratedTaskBundle)

    def test_write_prompt_outputs_writes_one_file_per_dataset(self) -> None:
        paper_paths = PaperPaths(
            paper_id=7,
            paper_dir=Path("/tmp/7_paper"),
            pdf_path=Path("/tmp/7_paper/paper.pdf"),
            markdown_path=Path("/tmp/7_paper/full.md"),
        )
        prompt_payloads = [
            (FAKE_DATASET_CONFIGS[0], "Prompt body for TJH."),
            (FAKE_DATASET_CONFIGS[1], "Prompt body for MIMIC."),
        ]

        with TemporaryDirectory() as tmpdir:
            prompt_paths = generate_tasks.write_prompt_outputs(
                Path(tmpdir),
                paper_paths,
                prompt_payloads,
                overwrite=False,
            )

            self.assertEqual([path.name for path in prompt_paths], ["7_tjh_prompt.md", "7_mimic_iv_demo_prompt.md"])
            first_content = prompt_paths[0].read_text(encoding="utf-8")
            second_content = prompt_paths[1].read_text(encoding="utf-8")
            self.assertIn("- dataset_key: `tjh`", first_content)
            self.assertIn("Prompt body for TJH.", first_content)
            self.assertIn("- dataset_key: `mimic_iv_demo`", second_content)
            self.assertIn("Prompt body for MIMIC.", second_content)

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
            with patch.object(generate_tasks, "DATASET_PROMPT_CONFIGS", FAKE_DATASET_CONFIGS):
                with patch.object(generate_tasks, "resolve_paper_paths", return_value=paper_paths):
                    with patch.object(generate_tasks, "extract_prompt_body", return_value="prompt template"):
                        with patch.object(generate_tasks, "adapt_prompt", side_effect=["final prompt 1", "final prompt 2"]) as adapt_prompt:
                            with patch.object(
                                generate_tasks,
                                "write_prompt_outputs",
                                return_value=[
                                    Path("/tmp/generated_tasks/7_tjh_prompt.md"),
                                    Path("/tmp/generated_tasks/7_mimic_iv_demo_prompt.md"),
                                ],
                            ) as write_prompt_outputs:
                                with patch.object(generate_tasks, "load_llm_config", side_effect=AssertionError("unexpected")):
                                    with patch.object(generate_tasks, "build_client", side_effect=AssertionError("unexpected")):
                                        with patch.object(
                                            generate_tasks,
                                            "call_generation_api",
                                            side_effect=AssertionError("unexpected"),
                                        ):
                                            with patch("builtins.print") as mock_print:
                                                generate_tasks.main()

        self.assertEqual(adapt_prompt.call_count, 2)
        write_prompt_outputs.assert_called_once_with(
            Path("/tmp/generated_tasks"),
            paper_paths,
            [
                (FAKE_DATASET_CONFIGS[0], "final prompt 1"),
                (FAKE_DATASET_CONFIGS[1], "final prompt 2"),
            ],
            False,
        )
        payload = json.loads(mock_print.call_args.args[0])
        self.assertEqual(payload["paper_id"], 7)
        self.assertEqual(
            payload["prompt_paths"],
            [
                "/tmp/generated_tasks/7_tjh_prompt.md",
                "/tmp/generated_tasks/7_mimic_iv_demo_prompt.md",
            ],
        )
        self.assertTrue(payload["output_prompt_only"])

    def test_main_skips_existing_outputs_without_overwrite(self) -> None:
        args = SimpleNamespace(
            paper_id=7,
            paper_dir=None,
            model_key="unused",
            task_count=2,
            output_prompt_only=False,
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
                with patch.object(
                    generate_tasks,
                    "existing_output_paths",
                    return_value=[Path("/tmp/generated_tasks/7_tasks.json")],
                ) as existing_output_paths:
                    with patch.object(generate_tasks, "extract_prompt_body", side_effect=AssertionError("unexpected")):
                        with patch.object(generate_tasks, "load_llm_config", side_effect=AssertionError("unexpected")):
                            with patch.object(generate_tasks, "build_client", side_effect=AssertionError("unexpected")):
                                with patch.object(generate_tasks, "call_generation_api", side_effect=AssertionError("unexpected")):
                                    with patch("builtins.print") as mock_print:
                                        generate_tasks.main()

        existing_output_paths.assert_called_once_with(Path("/tmp/generated_tasks"), paper_paths, False)
        payload = json.loads(mock_print.call_args.args[0])
        self.assertTrue(payload["skipped"])
        self.assertEqual(payload["existing_paths"], ["/tmp/generated_tasks/7_tasks.json"])
        self.assertFalse(payload["output_prompt_only"])

    def test_main_skips_existing_prompt_outputs_without_overwrite(self) -> None:
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
                with patch.object(
                    generate_tasks,
                    "existing_output_paths",
                    return_value=[Path("/tmp/generated_tasks/7_tjh_prompt.md")],
                ) as existing_output_paths:
                    with patch.object(generate_tasks, "extract_prompt_body", side_effect=AssertionError("unexpected")):
                        with patch.object(generate_tasks, "write_prompt_outputs", side_effect=AssertionError("unexpected")):
                            with patch("builtins.print") as mock_print:
                                generate_tasks.main()

        existing_output_paths.assert_called_once_with(Path("/tmp/generated_tasks"), paper_paths, True)
        payload = json.loads(mock_print.call_args.args[0])
        self.assertTrue(payload["skipped"])
        self.assertEqual(payload["existing_paths"], ["/tmp/generated_tasks/7_tjh_prompt.md"])
        self.assertTrue(payload["output_prompt_only"])

    def test_main_calls_api_twice_and_combines_tasks(self) -> None:
        args = SimpleNamespace(
            paper_id=7,
            paper_dir=None,
            model_key="unused",
            task_count=2,
            output_prompt_only=False,
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
        generated_task_1 = generate_tasks.GeneratedTask(
            task_brief="Task one.",
            task_type="report_generation",
            focus_areas=["a", "b"],
            task="Do task one.",
            required_inputs=["input1"],
            deliverables=["report.md", "metrics.json"],
            report_requirements=list(generate_tasks.FIXED_REPORT_REQUIREMENTS),
        )
        generated_task_2 = generate_tasks.GeneratedTask(
            task_brief="Task two.",
            task_type="report_generation",
            focus_areas=["c", "d"],
            task="Do task two.",
            required_inputs=["input2"],
            deliverables=["report.md", "tables/out.csv"],
            report_requirements=list(generate_tasks.FIXED_REPORT_REQUIREMENTS),
        )

        with patch.object(generate_tasks, "ARGS", args, create=True):
            with patch.object(generate_tasks, "DATASET_PROMPT_CONFIGS", FAKE_DATASET_CONFIGS):
                with patch.object(generate_tasks, "resolve_paper_paths", return_value=paper_paths):
                    with patch.object(generate_tasks, "extract_prompt_body", return_value="prompt template"):
                        with patch.object(generate_tasks, "adapt_prompt", side_effect=["prompt 1", "prompt 2"]) as adapt_prompt:
                            with patch.object(generate_tasks, "load_llm_config", return_value="llm-config") as load_llm_config:
                                with patch.object(generate_tasks, "build_client", return_value="client") as build_client:
                                    with patch.object(
                                        generate_tasks,
                                        "call_generation_api",
                                        side_effect=[
                                            (generated_task_1, {"response_id": "resp-1", "usage": {"input_tokens": 10}, "estimated_cost": {"total_cost_usd": 1.0}}),
                                            (generated_task_2, {"response_id": "resp-2", "usage": {"input_tokens": 20}, "estimated_cost": {"total_cost_usd": 2.0}}),
                                        ],
                                    ) as call_generation_api:
                                        with patch.object(
                                            generate_tasks,
                                            "combine_generation_metadata",
                                            return_value={
                                                "response_ids": ["resp-1", "resp-2"],
                                                "estimated_cost": {"total_cost_usd": 3.0},
                                            },
                                        ) as combine_generation_metadata:
                                            with patch.object(
                                                generate_tasks,
                                                "write_outputs",
                                                return_value=(
                                                    Path("/tmp/generated_tasks/7_tasks.json"),
                                                    Path("/tmp/generated_tasks/7_response.json"),
                                                ),
                                            ) as write_outputs:
                                                with patch("builtins.print") as mock_print:
                                                    generate_tasks.main()

        load_llm_config.assert_called_once()
        build_client.assert_called_once_with("llm-config")
        self.assertEqual(adapt_prompt.call_count, 2)
        self.assertEqual(call_generation_api.call_count, 2)
        self.assertEqual(call_generation_api.call_args_list[0].kwargs["dataset_config"], FAKE_DATASET_CONFIGS[0])
        self.assertEqual(call_generation_api.call_args_list[0].kwargs["prompt_text"], "prompt 1")
        self.assertEqual(call_generation_api.call_args_list[1].kwargs["dataset_config"], FAKE_DATASET_CONFIGS[1])
        self.assertEqual(call_generation_api.call_args_list[1].kwargs["prompt_text"], "prompt 2")
        combine_generation_metadata.assert_called_once_with(
            "llm-config",
            paper_paths,
            [
                {"response_id": "resp-1", "usage": {"input_tokens": 10}, "estimated_cost": {"total_cost_usd": 1.0}},
                {"response_id": "resp-2", "usage": {"input_tokens": 20}, "estimated_cost": {"total_cost_usd": 2.0}},
            ],
        )
        written_bundle = write_outputs.call_args.args[2]
        self.assertEqual(len(written_bundle.tasks), 2)
        self.assertEqual(written_bundle.tasks[0], generated_task_1)
        self.assertEqual(written_bundle.tasks[1], generated_task_2)
        payload = json.loads(mock_print.call_args.args[0])
        self.assertEqual(payload["response_ids"], ["resp-1", "resp-2"])
