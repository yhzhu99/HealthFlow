# Prompt for EhrFlowBench Task Generation

## Mission

Transform the uploaded paper's key ideas into exactly 1 self-contained, executable, end-to-end report-generation project that evaluates an advanced agent's ability to analyze processed local EHR data, write code, run lightweight experiments, and produce evidence-backed findings.

This task must be grounded in the local EHR metadata provided inside this prompt and must be designed so that another agent can execute it without access to the original paper.

## Injected Local EHR Metadata

{{TASK_ASSIGNMENT_BLOCK}}

{{DATASET_METADATA_BLOCK}}

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

3. Grounding in provided metadata only:
   Design tasks only around the processed EHR metadata injected above. Use only the listed parquet schema, split metadata summary, and MIMIC value-reference content when relevant. Do not assume any extra columns, labels, splits, modalities, annotations, or code mappings beyond what is explicitly included in this prompt.

4. Benchmark practicality:
   Generated tasks must be meaningful but not overly complex. Prefer lightweight, reproducible workflows that can usually be completed within 10 minutes on a CPU using deterministic or sklearn-level methods. Avoid tasks that require GANs, reinforcement learning, from-scratch pretraining, GPU-only workflows, or large hyperparameter sweeps.

5. No invented replication details:
   If the paper depends on unavailable data, private labels, or unsupported infrastructure, rewrite the idea into a feasible local-data proxy task instead of copying the original experiment literally.

6. Mandatory LaTeX for all math:
   Any mathematical variable, formula, metric, or loss function must use LaTeX.
   - Inline math: `\\( ... \\)`
   - Display math: `\\[ ... \\]`

7. Single-dataset exclusivity:
   Generate exactly 1 task, and it must use only the dataset identified in the injected task-allocation block. Do not compare against, combine with, or mention any other dataset.

8.  Verifiable deliverables:
    Each task must ask for at least one structured numeric artifact such as `metrics.json` or `tables/*.csv`, and at least one figure or table artifact when visualization is relevant. The task should be easy to evaluate from produced files, not only from free-form prose.

9.  Report contract awareness:
    Every task must culminate in a `report.md`, covering objective, data, method, results, evidence, and conclusion in that order.

10. Task-only output:
    Do not generate answers, reference answers, grader hints, or evaluation scripts. Only generate the structured task objects.

## Rejection Rules

Do not generate tasks that:
- only summarize or critique the paper
- require inaccessible external datasets
- depend on manual annotation that does not exist in this repository
- only ask for a single plot or a single metric with no end-to-end project framing
- depend on heavy model training, long pretraining, LLM inference, calling external APIs, GANs, RL, or hours-long experiments

## Output Format

Your entire response must be a single valid JSON object and nothing else.
Do not include markdown fences, explanations, or prose outside the JSON.

The `focus_areas` array must contain `2-4` short English phrases.

Use this exact schema:

{
  "task": {
      "task_brief": "one-sentence English summary",
      "focus_areas": ["focus_area_1", "focus_area_2"],
      "task": "full self-contained English task description",
      "deliverables": [
         "report.md",
         "tables/summary.csv",
         "figures/overview.png"
      ]
   }
}
