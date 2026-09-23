import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.tools.ehrflowbench_report_judge import DEFAULT_PASS_THRESHOLD
from data.tools.ehrflowbench_report_judge import evaluate_predictions
from data.tools.ehrflowbench_report_judge import extract_json_object
from data.tools.ehrflowbench_report_judge import find_report_reference_path
from data.tools.ehrflowbench_report_judge import find_submission_report
from data.tools.ehrflowbench_report_judge import load_manifest_outputs
from data.tools.ehrflowbench_report_judge import normalize_judge_payload
from data.tools.ehrflowbench_report_judge import render_markdown_to_pdf
from data.tools.ehrflowbench_report_judge import resolve_judge_config


def _write_manifest(root: Path, qid: int, reference_path: str) -> None:
    manifest_path = root / "reference_answers" / "test" / str(qid) / "answer_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "qid": qid,
                "dataset": "TJH",
                "task_type": "report_generation",
                "required_inputs": [],
                "required_outputs": [
                    {
                        "file_name": "report.md",
                        "reference_path": reference_path,
                        "media_type": "text",
                    },
                    {
                        "file_name": "metrics.json",
                        "reference_path": f"reference_answers/test/{qid}/metrics.json",
                        "media_type": "json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


class EhrFlowBenchReportJudgeTests(unittest.TestCase):
    def test_resolve_judge_config_uses_env_api_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm."openai/gpt-5.4"]
api_key_env = "TEST_EHRFLOWBENCH_JUDGE_KEY"
base_url = "https://example.com/v1"
model_name = "judge-model"
""".strip(),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TEST_EHRFLOWBENCH_JUDGE_KEY": "secret-key"}, clear=False):
                config = resolve_judge_config(config_path, None)

        self.assertEqual(config.llm_key, "openai/gpt-5.4")
        self.assertEqual(config.api_key, "secret-key")
        self.assertEqual(config.base_url, "https://example.com/v1")
        self.assertEqual(config.model_name, "judge-model")

    def test_load_manifest_outputs_and_report_reference_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_manifest(root, 7, "reference_answers/test/7/report.pdf")

            manifest, required_outputs = load_manifest_outputs(
                root,
                {"reference_answer": "reference_answers/test/7/answer_manifest.json"},
            )
            report_reference = find_report_reference_path(required_outputs, qid=7)

        self.assertEqual(manifest["task_type"], "report_generation")
        self.assertEqual([item.file_name for item in required_outputs], ["report.md", "metrics.json"])
        self.assertEqual(report_reference, "reference_answers/test/7/report.pdf")

    def test_find_report_reference_path_rejects_manifest_without_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_manifest(root, 8, "reference_answers/test/8/report.pdf")
            _, required_outputs = load_manifest_outputs(
                root,
                {"reference_answer": "reference_answers/test/8/answer_manifest.json"},
            )
            without_report = [item for item in required_outputs if Path(item.file_name).stem != "report"]

        with self.assertRaises(FileNotFoundError):
            find_report_reference_path(without_report, qid=8)

    def test_find_submission_report_prefers_pdf_over_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "3"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "report.md").write_text("# hi", encoding="utf-8")
            self.assertEqual(find_submission_report(task_dir).name, "report.md")

            (task_dir / "report.pdf").write_bytes(b"%PDF-1.4")
            self.assertEqual(find_submission_report(task_dir).name, "report.pdf")

    def test_render_markdown_to_pdf_writes_a_pdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = Path(tmpdir) / "report.md"
            markdown_path.write_text(
                "# Title\n\nSome **bold** text.\n\n- one\n- two\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n",
                encoding="utf-8",
            )
            pdf_path = render_markdown_to_pdf(markdown_path, Path(tmpdir) / "report.pdf")
            payload = pdf_path.read_bytes()

        self.assertTrue(payload.startswith(b"%PDF"))
        self.assertGreater(len(payload), 500)

    def test_extract_json_object_skips_surrounding_text(self):
        payload = extract_json_object(
            "Result:\n```json\n{\"overall_score\": {\"score\": 4}}\n```"
        )
        self.assertEqual(payload["overall_score"]["score"], 4)

    def test_normalize_judge_payload_clamps_dimensions_and_derives_correct(self):
        normalized = normalize_judge_payload(
            {
                "method_soundness": {"score": 9},
                "presentation_quality": 0,
                "artifact_generation": 3,
                "overall_score": {"score": 4},
                "summary": "solid",
                "file_level_notes": "not-a-list",
            },
            qid=12,
            task_type="report_generation",
            dataset="TJH",
            pass_threshold=DEFAULT_PASS_THRESHOLD,
            judge_model="openai/gpt-5.4",
        )

        self.assertEqual(normalized["method_soundness"], 5)
        self.assertEqual(normalized["presentation_quality"], 1)
        self.assertEqual(normalized["artifact_generation"], 3)
        self.assertEqual(normalized["score"], 4.0)
        self.assertTrue(normalized["correct"])
        self.assertEqual(normalized["status"], "scored")
        self.assertEqual(normalized["file_level_notes"], [])

    def test_normalize_judge_payload_honors_explicit_passed(self):
        normalized = normalize_judge_payload(
            {"overall_score": {"score": 1}, "passed": True},
            qid=13,
            task_type="report_generation",
            dataset="TJH",
            pass_threshold=DEFAULT_PASS_THRESHOLD,
            judge_model="openai/gpt-5.4",
        )

        self.assertTrue(normalized["passed"])
        self.assertTrue(normalized["correct"])

    def test_evaluate_predictions_marks_missing_reports_as_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_root = root / "processed"
            benchmark_root.mkdir(parents=True, exist_ok=True)
            benchmark_file = benchmark_root / "test.jsonl"
            benchmark_file.write_text(
                json.dumps(
                    {
                        "qid": 1,
                        "task": "Analyze the EHR.",
                        "task_brief": "Analyze the EHR.",
                        "dataset": "TJH",
                        "task_type": "report_generation",
                        "reference_answer": "reference_answers/test/1/answer_manifest.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write_manifest(benchmark_root, 1, "reference_answers/test/1/report.pdf")
            config_path = root / "config.toml"
            config_path.write_text(
                '[llm."openai/gpt-5.4"]\napi_key = "test-key"\nbase_url = "https://example.com/v1"\n',
                encoding="utf-8",
            )

            payload = evaluate_predictions(
                benchmark_file,
                root / "submission",
                config_path=config_path,
                judge_llm=None,
                prompt_root=root / "prompts",
                pass_threshold=DEFAULT_PASS_THRESHOLD,
            )

        self.assertEqual(payload["total_questions"], 1)
        self.assertEqual(payload["scored_questions"], 0)
        self.assertEqual(payload["failed_questions"], 1)
        self.assertEqual(payload["passed_questions"], 0)
        self.assertIsNone(payload["average_score"])
        self.assertEqual(payload["score_scale"]["max"], 5)
        self.assertEqual(payload["results"][0]["status"], "failed")
        self.assertFalse(payload["results"][0]["correct"])


if __name__ == "__main__":
    unittest.main()
