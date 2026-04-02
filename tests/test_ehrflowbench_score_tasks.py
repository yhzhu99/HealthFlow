import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import httpx
from data.ehrflowbench.scripts import generate_tasks
from data.ehrflowbench.scripts import score_tasks
from openai import APITimeoutError


def make_candidate(task_idx: int = 1) -> score_tasks.TaskScoreCandidate:
    return score_tasks.TaskScoreCandidate(
        paper_id=100,
        paper_title="Paper 100",
        task_idx=task_idx,
        task_brief=f"Task {task_idx}",
        task_type="report_generation",
        focus_areas=("prediction", "robustness"),
        task="Use only the TJH dataset and produce report.md, metrics.json, and figures/overview.png.",
        required_inputs=(
            "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
            "data/ehrflowbench/processed/tjh/split_metadata.json",
        ),
        deliverables=("report.md", "metrics.json", "figures/overview.png"),
        report_requirements=(
            "State the objective.",
            "Describe the data.",
            "Explain the method.",
            "Report quantitative results.",
            "Provide figure and/or table evidence.",
            "State the final conclusion.",
        ),
        tasks_path=Path("/tmp/100_tasks.json"),
    )


class EHRFlowBenchScoreTasksTests(TestCase):
    def test_build_score_prompt_injects_candidate_and_context_json(self) -> None:
        prompt_body = "\n".join(
            [
                "Candidate",
                score_tasks.PROMPT_TASK_CANDIDATE_PLACEHOLDER,
                "Context",
                score_tasks.PROMPT_DERIVED_CONTEXT_PLACEHOLDER,
            ]
        )
        candidate = make_candidate()
        context = {
            "primary_bucket": "prediction",
            "method_family": "prediction",
            "target_family": "outcome_prediction",
            "input_dataset": "tjh",
            "has_valid_single_dataset_inputs": True,
            "has_split_metadata_inputs": True,
            "has_mixed_dataset_inputs": False,
            "has_numeric_artifact": True,
            "has_figure_or_table_artifact": True,
            "bucket_size": 4,
            "bucket_method_count": 1,
            "bucket_target_count": 2,
            "novelty_rarity_threshold": 1,
        }

        prompt = score_tasks.build_score_prompt(prompt_body, candidate, context)

        self.assertIn('"paper_id": 100', prompt)
        self.assertIn('"primary_bucket": "prediction"', prompt)
        self.assertNotIn(score_tasks.PROMPT_TASK_CANDIDATE_PLACEHOLDER, prompt)
        self.assertNotIn(score_tasks.PROMPT_DERIVED_CONTEXT_PLACEHOLDER, prompt)

    def test_parse_score_payload_from_response_validates_ranges(self) -> None:
        response = SimpleNamespace(
            output_text=json.dumps(
                {
                    "hard_reject": False,
                    "hard_reject_reasons": [],
                    "scores": {
                        "feasibility": 3,
                        "specificity": 2,
                        "evaluability": 2,
                        "practicality": 2,
                        "novelty": 1,
                    },
                    "flags": ["local_only"],
                    "rationale": "Good task.",
                }
            )
        )

        payload = score_tasks.parse_score_payload_from_response(response)

        self.assertFalse(payload.hard_reject)
        self.assertEqual(payload.scores.novelty, 1)

    def test_task_record_from_payload_computes_final_score_locally(self) -> None:
        candidate = make_candidate()
        context = {
            "primary_bucket": "prediction",
            "method_family": "prediction",
            "target_family": "outcome_prediction",
            "input_dataset": "tjh",
            "bucket_size": 4,
            "bucket_method_count": 1,
            "bucket_target_count": 2,
            "novelty_rarity_threshold": 1,
        }
        payload = score_tasks.LLMTaskScorePayload.model_validate(
            {
                "hard_reject": False,
                "hard_reject_reasons": [],
                "scores": {
                    "feasibility": 2,
                    "specificity": 1,
                    "evaluability": 2,
                    "practicality": 1,
                    "novelty": 1,
                },
                "flags": ["local_only"],
                "rationale": "Looks good.",
            }
        )

        record = score_tasks.task_record_from_payload(
            candidate=candidate,
            derived_context=context,
            payload=payload,
            raw_response_path=Path("/tmp/raw.json"),
        )

        self.assertEqual(record["final_score"], 7)
        self.assertEqual(record["rank_status"], "eligible")

    def test_score_single_candidate_writes_raw_before_parse_failure(self) -> None:
        candidate = make_candidate()
        context = {
            "primary_bucket": "prediction",
            "method_family": "prediction",
            "target_family": "outcome_prediction",
            "input_dataset": "tjh",
            "has_valid_single_dataset_inputs": True,
            "has_split_metadata_inputs": True,
            "has_mixed_dataset_inputs": False,
            "has_numeric_artifact": True,
            "has_figure_or_table_artifact": True,
            "bucket_size": 4,
            "bucket_method_count": 1,
            "bucket_target_count": 2,
            "novelty_rarity_threshold": 1,
        }
        response = SimpleNamespace(
            id="resp-1",
            status="completed",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            output_text="not-json",
            output=[],
            error=None,
            incomplete_details=None,
        )
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
        llm_config = generate_tasks.LLMConfig(
            api_key_env="OPENAI_API_KEY",
            base_url="https://example.com/v1",
            model_name="test-model",
            reasoning_effort="medium",
            input_cost_per_million_tokens=1.0,
            output_cost_per_million_tokens=2.0,
        )

        with TemporaryDirectory() as tmpdir:
            record, index_row = score_tasks.score_single_candidate(
                client=client,
                llm_config=llm_config,
                prompt_body="Prompt\n" + score_tasks.PROMPT_TASK_CANDIDATE_PLACEHOLDER + "\n" + score_tasks.PROMPT_DERIVED_CONTEXT_PLACEHOLDER,
                candidate=candidate,
                derived_context=context,
                output_dir=Path(tmpdir),
                max_output_tokens=256,
            )

            raw_path = Path(tmpdir) / score_tasks.RAW_RESPONSE_DIRNAME / "100_task_1.json"
            self.assertTrue(raw_path.exists())
            saved = json.loads(raw_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["response_id"], "resp-1")
        self.assertEqual(record["parse_status"], "parse_failed")
        self.assertEqual(index_row["parse_status"], "parse_failed")

    def test_call_scoring_api_retries_after_timeout(self) -> None:
        candidate = make_candidate()
        context = {
            "primary_bucket": "prediction",
            "method_family": "prediction",
            "target_family": "outcome_prediction",
            "input_dataset": "tjh",
            "has_valid_single_dataset_inputs": True,
            "has_split_metadata_inputs": True,
            "has_mixed_dataset_inputs": False,
            "has_numeric_artifact": True,
            "has_figure_or_table_artifact": True,
            "bucket_size": 4,
            "bucket_method_count": 1,
            "bucket_target_count": 2,
            "novelty_rarity_threshold": 1,
        }
        response = SimpleNamespace(
            id="resp-1",
            status="completed",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            output_text='{"hard_reject":false,"hard_reject_reasons":[],"scores":{"feasibility":3,"specificity":2,"evaluability":2,"practicality":2,"novelty":1},"flags":[],"rationale":"ok"}',
            output=[],
            error=None,
            incomplete_details=None,
        )
        request = httpx.Request("POST", "https://example.com/v1/responses")
        calls = {"count": 0}

        def create(**kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise APITimeoutError(request)
            return response

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        llm_config = generate_tasks.LLMConfig(
            api_key_env="OPENAI_API_KEY",
            base_url="https://example.com/v1",
            model_name="test-model",
            reasoning_effort="medium",
            input_cost_per_million_tokens=1.0,
            output_cost_per_million_tokens=2.0,
        )

        with patch.object(score_tasks.time, "sleep") as sleep:
            response_obj, metadata = score_tasks.call_scoring_api(
                client=client,
                llm_config=llm_config,
                candidate=candidate,
                prompt_text="prompt",
                derived_context=context,
                max_output_tokens=123,
            )

        self.assertEqual(response_obj.id, "resp-1")
        self.assertEqual(metadata["request"]["attempts"], 2)
        sleep.assert_called_once_with(generate_tasks.retry_sleep_seconds(1))

    def test_aggregate_scores_writes_sorted_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            score_tasks.write_json(
                output_dir / "100_scores.json",
                {
                    "paper_id": 100,
                    "paper_title": "Paper 100",
                    "rows": [
                        {
                            "paper_id": 100,
                            "paper_title": "Paper 100",
                            "task_idx": 1,
                            "task_brief": "Eligible task",
                            "input_dataset": "tjh",
                            "primary_bucket": "prediction",
                            "method_family": "prediction",
                            "target_family": "outcome_prediction",
                            "hard_reject": False,
                            "hard_reject_reasons": [],
                            "scores": {"feasibility": 3, "specificity": 2, "evaluability": 2, "practicality": 2, "novelty": 1},
                            "final_score": 10,
                            "rank_status": "eligible",
                            "raw_response_path": "raw_responses/100_task_1.json",
                            "parse_status": "parsed",
                        },
                        {
                            "paper_id": 100,
                            "paper_title": "Paper 100",
                            "task_idx": 2,
                            "task_brief": "Rejected task",
                            "input_dataset": "tjh",
                            "primary_bucket": "prediction",
                            "method_family": "prediction",
                            "target_family": "outcome_prediction",
                            "hard_reject": True,
                            "hard_reject_reasons": ["mentions_other_dataset"],
                            "scores": {"feasibility": 2, "specificity": 1, "evaluability": 2, "practicality": 1, "novelty": 0},
                            "final_score": 6,
                            "rank_status": "rejected",
                            "raw_response_path": "raw_responses/100_task_2.json",
                            "parse_status": "parsed",
                        },
                    ],
                },
            )
            score_tasks.write_json(
                output_dir / "101_scores.json",
                {
                    "paper_id": 101,
                    "paper_title": "Paper 101",
                    "rows": [
                        {
                            "paper_id": 101,
                            "paper_title": "Paper 101",
                            "task_idx": 1,
                            "task_brief": "Parse failed task",
                            "input_dataset": None,
                            "primary_bucket": "other",
                            "method_family": "prediction",
                            "target_family": "generic",
                            "hard_reject": None,
                            "hard_reject_reasons": [],
                            "scores": None,
                            "final_score": None,
                            "rank_status": "parse_failed",
                            "raw_response_path": "raw_responses/101_task_1.json",
                            "parse_status": "parse_failed",
                        }
                    ],
                },
            )

            summary = score_tasks.aggregate_scores(output_dir)
            ranking_rows = [json.loads(line) for line in (output_dir / "task_ranking.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary["task_count"], 3)
        self.assertEqual(ranking_rows[0]["task_brief"], "Eligible task")
        self.assertEqual(ranking_rows[1]["task_brief"], "Rejected task")
        self.assertEqual(ranking_rows[2]["task_brief"], "Parse failed task")
