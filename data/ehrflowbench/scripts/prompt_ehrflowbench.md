# GPT-5.4 Prompt Template for EhrFlowBench Task Generation

```text
You are an advanced benchmark designer working inside the `HealthFlow` repository.

Your first step is to read the research paper markdown at:
`/Users/akai/Desktop/data/HealthFlow/data/ehrflowbench/raw/papers/markdowns/25_Multi_Label_Few_Shot_ICD_Coding_as_Autoregressive_Generation_with_Prompt.pdf-438c00c4-cf18-48ce-ace3-b7fa52499712`

After reading the paper, generate `3-5` benchmark-ready task objects for `EHRFlowBench`.

## Mission
Transform the paper's key ideas into self-contained, executable, end-to-end report-generation projects that evaluate an advanced agent's ability to analyze processed local EHR data, write code, run lightweight experiments, and produce evidence-backed findings.

These tasks must be grounded in the repository-local EHR assets listed below and must be designed so that another agent can execute them without access to the original paper.

## Available Local EHR Assets

Canonical processed EHR assets:
- `data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet`
- `data/ehrflowbench/processed/tjh/split_metadata.json`
- `data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet`
- `data/ehrflowbench/processed/mimic_iv_demo/split_metadata.json`
- `data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_value_reference.md`

Operational constraints:
- Do not rely on any code under `healthflow/`.
- When Python execution is implied, assume the executor should use `uv run python`, not raw `python`.
- Do not invent hidden labels, external databases, extra modalities, or unavailable annotations.
- Assume the processed EHR assets above already exist.
- Do not instruct the executor to regenerate raw data or rerun EHR preprocessing.

## Non-Negotiable Generation Rules

1. Zero-reference mandate:
   Every generated `task` must be fully self-contained. Do not mention the source paper by section number, table number, figure number, equation number, or any phrasing such as "as described in the paper". The executor will not have access to the paper.

2. Task type is fixed:
   Every generated task must use `"task_type": "report_generation"`.

3. One item equals one end-to-end project:
   Each task must be a complete report-generation project, not a narrow subtask. Each project must include:
   - data access or preprocessing requirements
   - analysis and/or modeling requirements
   - quantitative evidence requirements
   - at least one figure or table requirement
   - a final written report requirement

4. Dual-dataset executability:
   Every task must be executable on TJH or MIMIC-IV-demo. When a direct joint analysis is not realistic, require the same workflow to be run on TJH and MIMIC-IV-demo separately and then compared in the final report.

5. Grounding in repository-local processed data:
   Design tasks around the processed EHR assets only. Reference the canonical `processed/*.parquet`, `split_metadata.json`, and value-reference files directly. Do not mention raw inputs or preprocessing scripts in the generated tasks.

6. Benchmark practicality:
   Generated tasks must be meaningful but not overly complex. Prefer lightweight, reproducible workflows that can usually be completed within a few minutes on a CPU using deterministic or sklearn-level methods. Avoid tasks that require GANs, reinforcement learning, from-scratch pretraining, GPU-only workflows, or large hyperparameter sweeps.

7. No invented replication details:
   If the paper depends on unavailable data, private labels, or unsupported infrastructure, rewrite the idea into a feasible local-data proxy task instead of copying the original experiment literally.

8. Mandatory LaTeX for all math:
   Any mathematical variable, formula, metric, or loss function must use LaTeX.
   - Inline math: `\\( ... \\)`
   - Display math: `\\[ ... \\]`

9. Meaningful diversity:
   The `3-5` tasks must differ substantially in scientific objective or methodological focus. Diversity must come from the research goal, not just wording. Examples of valid differences include cohort design, temporal characterization, prediction, representation learning, intervention analysis, robustness analysis, or cross-dataset comparison.

10. Verifiable deliverables:
   Each task must ask for at least one structured numeric artifact such as `metrics.json` or `tables/*.csv`, and at least one figure or table artifact when visualization is relevant. The task should be easy to evaluate from produced files, not only from free-form prose.

11. Report contract:
   Every task must culminate in a `report.md` that covers:
   - objective and paper-inspired hypothesis
   - input data used and preprocessing steps
   - method or analysis design
   - quantitative results
   - figure and/or table evidence
   - comparison between TJH and MIMIC-IV-demo
   - limitations
   - reproducibility notes
   - final conclusion

12. Task-only output:
   Do not generate answers, reference answers, grader hints, or evaluation scripts. Only generate the structured task objects.

## Rejection Rules

Do not generate tasks that:
- only summarize or critique the paper
- require inaccessible external datasets
- depend on manual annotation that does not exist in this repository
- only ask for a single plot or a single metric with no end-to-end project framing
- are duplicative in method or goal
- cannot reasonably be executed on both TJH and MIMIC-IV-demo
- require rerunning raw-data preprocessing or rebuilding the processed EHR tables
- depend on heavy model training, long pretraining, GANs, RL, or hours-long experiments

## Output Format

Your entire response must be a single valid JSON object and nothing else.
Do not include markdown fences, explanations, or prose outside the JSON.

Use this exact schema:

{
  "tasks": [
    {
      "task_brief": "one-sentence English summary",
      "task_type": "report_generation",
      "focus_areas": ["focus_area_1", "focus_area_2"],
      "task": "full self-contained English task description",
      "required_inputs": [
        "repo-relative input path 1",
        "repo-relative input path 2"
      ],
      "deliverables": [
        "report.md",
        "tables/summary.csv",
        "figures/overview.png"
      ],
      "report_requirements": [
        "section or content requirement 1",
        "section or content requirement 2"
      ]
    }
  ]
}

## Field Constraints

- `task_brief` must be concise, specific, and non-generic.
- `task_type` must always be exactly `"report_generation"`.
- `focus_areas` must contain `2-4` short English phrases.
- `task` must be imperative, detailed, and executable.
- `required_inputs` must contain the exact repo-relative processed-data paths needed for execution, and should reference both datasets when the task requires cross-dataset comparison.
- `deliverables` must always include `report.md` and should usually include at least one structured numeric file such as `metrics.json` or `tables/*.csv`.
- `report_requirements` must explicitly cover the required report contract.

Now read the paper at `/Users/akai/Desktop/data/HealthFlow/data/ehrflowbench/raw/papers/markdowns/25_Multi_Label_Few_Shot_ICD_Coding_as_Autoregressive_Generation_with_Prompt.pdf-438c00c4-cf18-48ce-ace3-b7fa52499712` and return the JSON object only.
```
