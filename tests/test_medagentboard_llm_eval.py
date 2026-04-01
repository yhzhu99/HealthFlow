import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.tools.medagentboard_llm_eval import SubmissionArtifact
from data.tools.medagentboard_llm_eval import build_text_context
from data.tools.medagentboard_llm_eval import extract_json_object
from data.tools.medagentboard_llm_eval import load_manifest_outputs
from data.tools.medagentboard_llm_eval import normalize_judge_payload
from data.tools.medagentboard_llm_eval import resolve_judge_config


class MedAgentBoardLlmEvalTests(unittest.TestCase):
    def test_resolve_judge_config_uses_env_api_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm."openai/gpt-5.4"]
api_key_env = "TEST_MEDAGENTBOARD_JUDGE_KEY"
base_url = "https://example.com/v1"
model_name = "judge-model"
""".strip(),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TEST_MEDAGENTBOARD_JUDGE_KEY": "secret-key"}, clear=False):
                config = resolve_judge_config(config_path, None)

        self.assertEqual(config.llm_key, "openai/gpt-5.4")
        self.assertEqual(config.api_key, "secret-key")
        self.assertEqual(config.base_url, "https://example.com/v1")
        self.assertEqual(config.model_name, "judge-model")

    def test_load_manifest_outputs_supports_v2_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = Path(tmpdir)
            manifest_path = benchmark_root / "reference_answers" / "1" / "answer_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "contract_version": "medagentboard_llm_v2",
                        "required_outputs": [
                            {
                                "file_name": "metrics.json",
                                "reference_path": "reference_answers/1/metrics.json",
                                "media_type": "json",
                            },
                            {
                                "file_name": "predictions.csv",
                                "reference_path": "reference_answers/1/predictions.csv",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest, required_outputs = load_manifest_outputs(
                benchmark_root,
                {"reference_answer": "reference_answers/1/answer_manifest.json"},
            )

        self.assertEqual(manifest["contract_version"], "medagentboard_llm_v2")
        self.assertEqual([item.file_name for item in required_outputs], ["metrics.json", "predictions.csv"])
        self.assertEqual([item.media_type for item in required_outputs], ["json", "csv"])

    def test_load_manifest_outputs_supports_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_root = Path(tmpdir)
            manifest_path = benchmark_root / "reference_answers" / "2" / "answer_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "primary_outputs": [
                            "reference_answers/2/result.csv",
                            "reference_answers/2/notes.txt",
                        ]
                    }
                ),
                encoding="utf-8",
            )

            _, required_outputs = load_manifest_outputs(
                benchmark_root,
                {"reference_answer": "reference_answers/2/answer_manifest.json"},
            )

        self.assertEqual([item.file_name for item in required_outputs], ["result.csv", "notes.txt"])
        self.assertEqual([item.media_type for item in required_outputs], ["csv", "text"])

    def test_build_text_context_includes_submission_and_reference_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_path = root / "reference_answers" / "7" / "result.csv"
            submission_path = root / "submission" / "7" / "result.csv"
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            submission_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_text("id,value\n1,10\n2,20\n", encoding="utf-8")
            submission_path.write_text("id,value\n1,10\n2,19\n", encoding="utf-8")

            context = build_text_context(
                {
                    "qid": 7,
                    "dataset": "TJH",
                    "task_type": "data_extraction",
                    "task_brief": "Summarize a cohort.",
                    "task": "Read the EHR data and produce result.csv.",
                },
                {
                    "contract_version": "medagentboard_llm_v2",
                    "origin": "generated",
                    "protocol_notes": ["Use the processed parquet only."],
                },
                [
                    SubmissionArtifact(
                        file_name="result.csv",
                        submission_path=submission_path,
                        reference_path=reference_path,
                        media_type="csv",
                    )
                ],
                pass_threshold=7.0,
            )

        payload = json.loads(context)
        self.assertEqual(payload["qid"], 7)
        self.assertEqual(payload["manifest"]["contract_version"], "medagentboard_llm_v2")
        self.assertEqual(payload["score_scale"]["pass_threshold"], 7.0)
        self.assertEqual(len(payload["required_outputs"]), 1)
        self.assertTrue(payload["required_outputs"][0]["submission_exists"])
        self.assertEqual(payload["required_outputs"][0]["submission_summary"]["row_count"], 2)
        self.assertEqual(payload["required_outputs"][0]["reference_summary"]["column_count"], 2)

    def test_extract_json_object_skips_surrounding_text(self):
        payload = extract_json_object(
            "Judge result:\n```json\n{\"score\": 8, \"passed\": true, \"summary\": \"ok\"}\n```"
        )
        self.assertEqual(payload["score"], 8)
        self.assertTrue(payload["passed"])

    def test_normalize_judge_payload_clamps_and_derives_defaults(self):
        normalized = normalize_judge_payload(
            {
                "score": 12,
                "passed": "yes",
                "summary": "accepted",
                "reason": None,
                "file_level_notes": "not-a-list",
            },
            qid=11,
            task_type="visualization",
            dataset="MIMIC-IV",
            pass_threshold=7.0,
            judge_model="openai/gpt-5.4",
        )

        self.assertEqual(normalized["score"], 10.0)
        self.assertTrue(normalized["passed"])
        self.assertEqual(normalized["summary"], "accepted")
        self.assertEqual(normalized["reason"], "None")
        self.assertEqual(normalized["file_level_notes"], [])
        self.assertEqual(normalized["judge_model"], "openai/gpt-5.4")


if __name__ == "__main__":
    unittest.main()
