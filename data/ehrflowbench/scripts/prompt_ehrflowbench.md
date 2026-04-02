# GPT-5.4 Prompt Template for EhrFlowBench Task Generation

## Mission
Transform the uploaded paper's key ideas into exactly 2 self-contained, executable, end-to-end report-generation projects that evaluate an advanced agent's ability to analyze processed local EHR data, write code, run lightweight experiments, and produce evidence-backed findings.

These tasks must be grounded in the local EHR metadata provided inside this prompt and must be designed so that another agent can execute them without access to the original paper.

## API Execution Context

- You are running through a remote API call.
- You cannot open repository-local files, parquet tables, JSON manifests, or markdown files by path.
- The local pipeline will inject dataset metadata, data-format summaries, and task-allocation constraints into this prompt before it is sent to you.
- Treat the injected metadata as the only source of truth about the available local EHR assets.
- Do not assume any extra columns, labels, splits, modalities, annotations, or code mappings beyond what is explicitly included in this prompt.

## Injected Local EHR Metadata

{{DATASET_METADATA_BLOCK}}

## Injected Task Allocation

{{TASK_ASSIGNMENT_BLOCK}}

## Non-Negotiable Generation Rules

1. Zero-reference mandate:
   Every generated `task` must be fully self-contained. Do not mention the source paper by section number, table number, figure number, equation number, or any phrasing such as "as described in the paper". The executor will not have access to the paper.

2. One item equals one end-to-end project:
   Each task must be a complete report-generation project, not a narrow subtask. Each project must include:
   - data access or preprocessing requirements
   - analysis and/or modeling requirements
   - quantitative evidence requirements
   - at least one figure or table requirement
   - a final written report requirement

3. Single-dataset constraint:
   Each task must use exactly one assigned dataset. Do not design a single task that jointly uses TJH and MIMIC-IV-demo.

4. Grounding in provided metadata only:
   Design tasks only around the processed EHR metadata injected above. Use only the listed parquet schema, split metadata summary, and MIMIC value-reference content when relevant. Do not mention raw inputs or preprocessing scripts in the generated tasks.

5. Fixed fields are added locally:
   Do not output `task_type`, `required_inputs`, or `report_requirements`. The local Python pipeline will append these fixed fields after parsing your response.

6. Benchmark practicality:
   Generated tasks must be meaningful but not overly complex. Prefer lightweight, reproducible workflows that can usually be completed within a few minutes on a CPU using deterministic or sklearn-level methods. Avoid tasks that require GANs, reinforcement learning, from-scratch pretraining, GPU-only workflows, or large hyperparameter sweeps.

7. No invented replication details:
   If the paper depends on unavailable data, private labels, or unsupported infrastructure, rewrite the idea into a feasible local-data proxy task instead of copying the original experiment literally.

8. Mandatory LaTeX for all math:
   Any mathematical variable, formula, metric, or loss function must use LaTeX.
   - Inline math: `\\( ... \\)`
   - Display math: `\\[ ... \\]`

9. Meaningful diversity:
   The 2 tasks must differ substantially in scientific objective or methodological focus. Diversity must come from the research goal, not just wording. Examples of valid differences include cohort design, temporal characterization, prediction, representation learning, intervention analysis, robustness analysis, or descriptive clinical pattern analysis.

10. Verifiable deliverables:
    Each task must ask for at least one structured numeric artifact such as `metrics.json` or `tables/*.csv`, and at least one figure or table artifact when visualization is relevant. The task should be easy to evaluate from produced files, not only from free-form prose.

11. Report contract awareness:
    Every task must culminate in a `report.md`. The local pipeline will append a fixed 6-item report requirement list, so you do not need to output that list, but your task description must clearly imply a final report covering objective, data, method, results, evidence, and conclusion.

12. Task-only output:
    Do not generate answers, reference answers, grader hints, or evaluation scripts. Only generate the structured task objects.

## Rejection Rules

Do not generate tasks that:
- only summarize or critique the paper
- require inaccessible external datasets
- depend on manual annotation that does not exist in this repository
- only ask for a single plot or a single metric with no end-to-end project framing
- are duplicative in method or goal
- mix TJH and MIMIC-IV-demo inside one task
- require rerunning raw-data preprocessing or rebuilding the processed EHR tables
- depend on heavy model training, long pretraining, GANs, RL, or hours-long experiments

## Output Format

Your entire response must be a single valid JSON object and nothing else.
Do not include markdown fences, explanations, or prose outside the JSON.

The `tasks` array order must follow the injected task-allocation block exactly.

Use this exact schema:

{
  "tasks": [
    {
      "task_brief": "one-sentence English summary",
      "focus_areas": ["focus_area_1", "focus_area_2"],
      "task": "full self-contained English task description",
      "deliverables": [
        "report.md",
        "tables/summary.csv",
        "figures/overview.png"
      ]
    }
  ]
}

## Field Constraints

- `task_brief` must be concise, specific, and non-generic.
- `focus_areas` must contain `2-4` short English phrases.
- `task` must be imperative, detailed, and executable using only the assigned dataset metadata.
- `deliverables` must always include `report.md` and should usually include at least one structured numeric file such as `metrics.json` or `tables/*.csv`.
