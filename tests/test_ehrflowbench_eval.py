import json
from unittest.mock import patch

import pytest

from data.ehrflowbench.scripts.evaluate import bootstrap, evaluate_task, judge, select_attempt


def test_attempt_selection_uses_first_success_or_third_attempt():
    attempts = [{"attempt": 1, "evaluation": {"status": "success", "score": 0.81}},
                {"attempt": 2, "evaluation": {"status": "success", "score": 0.95}}]
    assert select_attempt(attempts)["attempt"] == 1
    assert select_attempt([{"attempt": 1, "evaluation": {"status": "failed", "score": 0.1}}]) is None
    assert select_attempt([{"attempt": 3, "evaluation": {"status": "failed", "score": 0.1}}])["attempt"] == 3


def test_bootstrap_preserves_native_scale_and_failure_denominator():
    summary = bootstrap([0, 5])
    assert summary["n"] == 2
    assert summary["sample_mean"] == 2.5
    assert summary == bootstrap([0, 5])


def test_missing_submission_scores_zero_but_missing_reference_is_an_error(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "report.md").write_text("Reference report")
    (reference / "manifest.json").write_text(json.dumps({"required_outputs": [
        {"file_name": "report.md", "reference_path": "report.md"}]}))
    row = {"qid": 1, "dataset": "TJH", "reference_answer": "manifest.json", "task": "write a report"}
    result = evaluate_task(row, reference, tmp_path / "submissions", tmp_path / "work", "fake")
    assert result["overall_score"] == 0
    (reference / "report.md").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_task(row, reference, tmp_path / "submissions", tmp_path / "work", "fake")


def test_judge_keeps_overall_independent_of_dimension_average():
    verdict = {"method_soundness": 2, "presentation_quality": 5, "artifact_generation": 4,
               "overall_score": 2, "reason": "Major methodological error"}
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}), patch("httpx.post") as post:
        post.return_value.json.return_value = {"candidates": [{"finishReason": "STOP", "content": {
            "parts": [{"text": json.dumps(verdict)}]}}]}
        parsed, _ = judge([{"text": "task"}], "fake")
    assert parsed == verdict
