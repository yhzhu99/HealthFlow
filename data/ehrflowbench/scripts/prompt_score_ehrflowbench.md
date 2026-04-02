# Prompt for EHRFlowBench Task Scoring

## Mission

Review exactly one candidate EHRFlowBench task and return a structured quality assessment.

You are **not** executing the task. You are auditing whether the task is suitable for inclusion in a benchmark of repository-local, end-to-end, report-generation projects on processed EHR data.

Your assessment must be based only on the information provided in this prompt. Do not assume hidden files, hidden labels, hidden annotations, or access to the original paper.

## Task Background

EHRFlowBench candidate tasks are intended to be:

- self-contained report-generation projects
- executable using only repository-local processed EHR assets
- scoped to exactly one local dataset
- lightweight and reproducible on CPU
- verifiable from machine-readable artifacts and a final report

Each task should end in `report.md` and should request enough structured outputs that a later reviewer can verify the work from files, not only prose.

## Accepted Local Dataset Input Sets

Only these single-dataset input bundles are valid:

### TJH

- `data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet`
- `data/ehrflowbench/processed/tjh/split_metadata.json`

### MIMIC-IV-demo

- `data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet`
- `data/ehrflowbench/processed/mimic_iv_demo/split_metadata.json`
- optional: `data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_value_reference.md`

Any task that mixes both datasets, depends on unspecified extra inputs, or uses an invalid input set must be rejected.

## Hard Rejection Rules

If any of the following is true, set `hard_reject=true` and return the matching canonical reason codes in `hard_reject_reasons`.

Use only the canonical reason codes listed below. Do not invent new codes.

### Canonical Hard Reject Codes

- `invalid_task_type`
- `missing_report_deliverable`
- `missing_numeric_artifact`
- `missing_figure_or_table_artifact`
- `mixed_dataset_inputs`
- `invalid_single_dataset_inputs`
- `unsupported_text_dependency`
- `mentions_other_dataset`
- `source_paper_reference`
- `legacy_target_fallback_instruction`
- `legacy_cross_dataset_target_instruction`
- `paper_summary_or_critique_task`
- `depends_on_healthflow_runtime`
- `depends_on_raw_rebuild`
- `depends_on_external_data`
- `depends_on_manual_annotation`
- `depends_on_external_api_or_llm`
- `heavy_execution_requirement`

### Meaning of the Hard Reject Rules

Reject if:

1. `task_type` is not `report_generation`.
2. Deliverables do not include `report.md`.
3. Deliverables do not include at least one structured numeric artifact.
   Numeric artifacts are files with suffix `.json`, `.csv`, `.parquet`, `.tsv`, or `.xlsx`.
4. Deliverables do not include at least one figure/table artifact.
   A figure/table artifact is any deliverable that:
   - starts with `tables/` or `figures/`, or
   - has suffix `.png`, `.jpg`, `.jpeg`, `.svg`, or `.pdf`, or
   - has suffix `.csv`, `.tsv`, or `.xlsx` and the filename stem contains `table`
5. The task depends on both datasets or otherwise uses mixed dataset inputs.
6. The required inputs are not exactly one valid local dataset bundle.
7. The task requires clinical notes, event text, unstructured text, bag-of-words, or TF-IDF style text processing.
8. The task positively mentions or compares against the other dataset instead of staying within the assigned dataset.
9. The task positively refers to the source paper, for example:
   - “as described in the paper”
   - “reproduce Table 2”
   - “follow Section 3”
   - “use Equation 4”
   Ignore purely negative instructions such as “do not refer to the paper”.
10. The task still follows old cross-dataset fallback wording such as:
   - “if a candidate target is unavailable”
   - “fall back to another directly available target”
   - “choose one binary target that is present in both TJH and MIMIC-IV-demo”
11. The task is only a paper summary, paper critique, or literature review rather than an executable project.
12. The task depends on code under `healthflow/`.
13. The task depends on rebuilding raw preprocessing, regenerating processed EHR data, or using raw inputs.
14. The task depends on external datasets, manual annotation, or human annotation.
15. The task depends on external APIs, OpenAI, GPT, LLM inference, or other external services.
16. The task requires obviously heavy workflows such as GAN training, reinforcement learning, from-scratch pretraining, GPU-only execution, or hours-long experiments.

## Scoring Rules

Return integer scores only.

### 1. Feasibility: 0-3

Add:

- `+1` if the required inputs are exactly one valid local dataset bundle
- `+1` if split metadata is included
- `+1` if the task explicitly constrains itself to repository-local data / the assigned dataset only

### 2. Specificity: 0-2

Add:

- `+1` if the task names a concrete lightweight method, model family, or analysis procedure
- `+1` if the task names concrete metrics, explicit split usage, or similarly verifiable evaluation details

### 3. Evaluability: 0-2

Add:

- `+1` if the task asks for both `report.md` and at least one numeric artifact
- `+1` if the task asks for at least one figure/table artifact and the report requirements are sufficiently detailed

### 4. Practicality: 0-2

Add:

- `+1` if the task explicitly emphasizes lightweight / deterministic / CPU-friendly / sklearn-level execution
- `+1` if the workflow avoids heavy execution and the deliverable list remains reasonably compact

### 5. Novelty: 0-1

Use the derived context object in this prompt.

Set novelty to `1` if either:

- `primary_bucket` is `causal` or `fairness`, or
- `bucket_method_count <= novelty_rarity_threshold`, or
- `bucket_target_count <= novelty_rarity_threshold`

Otherwise set novelty to `0`.

## Flags

`flags` is optional but helpful. Use short machine-readable strings only when clearly supported by the task, for example:

- `local_only`
- `lightweight`
- `input_dataset:tjh`
- `input_dataset:mimic_iv_demo`
- `graph_method`
- `unsupervised_method`

Do not include long explanations in `flags`.

## Input Task Candidate

### Raw Candidate JSON

```json
{{TASK_CANDIDATE_JSON}}
```

### Derived Context JSON

```json
{{DERIVED_CONTEXT_JSON}}
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
  },
  "flags": [],
  "rationale": "short audit summary"
}
```

## Final Reminders

- Do not output markdown fences.
- Do not output prose outside the JSON object.
- Use only the canonical hard reject codes listed above.
- Scores must be integers in range.
- If a task is hard rejected, still provide the score object.
