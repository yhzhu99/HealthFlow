# Prompt for EHRFlowBench Task Scoring

## Mission

Review exactly one candidate EHRFlowBench task and return a structured quality assessment.

You are **not** executing the task. You are auditing whether the task is suitable for inclusion in a benchmark of end-to-end, report-generation projects on processed EHR data. Your assessment must be based only on the information provided in this prompt.

## Task Background

EHRFlowBench candidate tasks are intended to be:

- self-contained report-generation projects
- scoped to exactly one local dataset and executable using the local processed EHR assets
- lightweight and reproducible on CPU
- verifiable from machine-readable artifacts and a final report

## Hard Rejection Rules

If any of the following is true, set `hard_reject=true` and return the matching canonical reason codes in `hard_reject_reasons`.

Use only the canonical reason codes listed below. Do not invent new codes.

### Canonical Hard Reject Codes

- `missing_report_deliverable`: Deliverables do not include `report.md`.
- `missing_numeric_artifact`: Deliverables do not include at least one structured numeric artifact.
- `missing_figure_or_table_artifact`: Deliverables do not include at least one figure/table artifact.
- `paper_summary_or_critique_task`: The task is only a paper summary, paper critique, or literature review rather than an executable project.
- `depends_on_external_data`: The task depends on external datasets, manual annotation, or human annotation.
- `depends_on_manual_annotation`: The task depends on manual annotation.
- `depends_on_external_api_or_llm`: The task depends on external APIs, OpenAI, GPT, LLM inference, or other external services.
- `heavy_execution_requirement`: The task requires obviously heavy workflows such as GAN training, reinforcement learning, from-scratch pretraining, GPU-only execution, or hours-long experiments.

## Scoring Rules

Return integer scores only.

### 1. Feasibility: 0-5

- If the required inputs are exactly one valid local dataset bundle
- If the task explicitly constrains itself to repository-local data / the assigned dataset only

### 2. Specificity: 0-5

- If the task names a concrete lightweight method, model family, or analysis procedure
- If the task names concrete metrics, explicit split usage, or similarly verifiable evaluation details

### 3. Evaluability: 0-5

- If the task asks for both `report.md` and at least one numeric artifact
- If the task asks for at least one figure/table artifact and the report requirements are sufficiently detailed

### 4. Practicality: 0-5

- If the task explicitly emphasizes lightweight / deterministic / CPU-friendly / sklearn-level execution
- If the workflow avoids heavy execution and the deliverable list remains reasonably compact

### 5. Novelty: 0-5

- If the task is novel and not a straightforward replication of a known method or dataset

## Input Task Candidate

```json
{{TASK_CANDIDATE_JSON}}
```

## Output Contract

Your entire response must be exactly one valid JSON object and nothing else.

Use this exact schema:

```json
{
  "hard_reject": false,
  "hard_reject_reasons": [],
  "scores": {
    "feasibility": 0,
    "specificity": 0,
    "evaluability": 0,
    "practicality": 0,
    "novelty": 0
  }
}
```
