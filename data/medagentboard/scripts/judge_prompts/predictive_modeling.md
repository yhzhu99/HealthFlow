You are grading a MedAgentBoard `predictive_modeling` task on a 0-10 scale.

Primary objective:
- Decide whether the submission correctly follows the task contract and produces outputs that are materially consistent with the reference solution.

Rubric:
- Give `9-10` when `metrics.json`, `predictions.csv`, and `answer.json` all align with the task objective and closely match the reference outputs.
- Give `7-8` when the submission is largely correct but has minor differences in reporting, naming, or non-critical metric drift.
- Give `4-6` when the modeling setup seems relevant but the metrics, predictions, or final answer are only partially consistent with the task.
- Give `1-3` when the submission misunderstands the target, split, task type, or output contract.
- Give `0` when required outputs are missing or the submission is clearly not a valid solution.

Important judging rules:
- Evaluate the outputs against the stated task, not against arbitrary modeling preferences.
- Treat small numeric differences as acceptable when they do not materially change correctness.
- Use the provided summaries of large prediction files to judge consistency; do not assume exact row-by-row identity is required unless the evidence clearly indicates a mismatch.
- For model-comparison tasks, check whether the reported winning model is supported by the reported metrics.
- If required files are missing, reflect that strongly in the score.

Return exactly one JSON object with this schema:
{
  "score": 0,
  "passed": false,
  "summary": "one short paragraph",
  "reason": "main reason for the score",
  "file_level_notes": [
    {
      "file": "metrics.json",
      "status": "correct | partially_correct | incorrect | missing",
      "note": "brief explanation"
    }
  ]
}
