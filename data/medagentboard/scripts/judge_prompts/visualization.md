You are grading a MedAgentBoard `visualization` task on a 0-10 scale.

Primary objective:
- Compare the submission image set against the reference image set and decide whether the submission correctly satisfies the requested visualization task.

Rubric:
- Give `9-10` when the submission reproduces the correct chart type, variables, visual structure, and main scientific pattern shown by the reference images.
- Give `7-8` when the submission is largely correct but has minor presentation differences such as styling, labeling, or layout changes that do not materially alter the interpretation.
- Give `4-6` when the submission captures part of the requested visualization but misses an important panel, axis meaning, grouping, or trend.
- Give `1-3` when the generated figure is only weakly related to the task or reference.
- Give `0` when required images are missing or clearly unrelated.

Important judging rules:
- Do not require pixel-perfect matching.
- Judge semantic fidelity: chart type, plotted variables, grouping, trends, highlighted comparisons, and whether all required images are present.
- Ignore harmless differences in color palette, font, exact dimensions, or minor stylistic choices.
- If the task requires multiple images, judge the whole set and penalize missing or incorrect panels.

Return exactly one JSON object with this schema:
{
  "score": 0,
  "passed": false,
  "summary": "one short paragraph",
  "reason": "main reason for the score",
  "file_level_notes": [
    {
      "file": "figure.png",
      "status": "correct | partially_correct | incorrect | missing",
      "note": "brief explanation"
    }
  ]
}
