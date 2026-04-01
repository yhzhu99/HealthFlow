You are grading a MedAgentBoard `data_extraction` task on a 0-10 scale.

Primary objective:
- Decide whether the submission's structured result is semantically equivalent to the reference answer for the stated task.

Rubric:
- Give `9-10` when the extracted values, rows, filters, grouping logic, and ordering are correct or differ only in harmless formatting.
- Give `7-8` when the core answer is correct but there are minor schema or presentation issues that do not materially change the meaning.
- Give `4-6` when the submission is partially correct but misses an important subset, statistic, or filtering rule.
- Give `1-3` when the submission shows a weak or mostly incorrect understanding of the task.
- Give `0` when required outputs are missing or the result is fundamentally wrong.

Important judging rules:
- Focus on semantic correctness, not byte-level equality.
- Tolerate harmless differences in column ordering, whitespace, numeric formatting, and equivalent row ordering.
- Do not reward unsupported extra content.
- If a required file is missing, reflect that strongly in the score.

Return exactly one JSON object with this schema:
{
  "score": 0,
  "passed": false,
  "summary": "one short paragraph",
  "reason": "main reason for the score",
  "file_level_notes": [
    {
      "file": "result.csv",
      "status": "correct | partially_correct | incorrect | missing",
      "note": "brief explanation"
    }
  ]
}
