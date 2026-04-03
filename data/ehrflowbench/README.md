# EHRFlowBench

EHRFlowBench rebuilds paper-inspired EHR projects into repository-local `report_generation` tasks for TJH and MIMIC-IV-demo.
The benchmark artifacts under `data/ehrflowbench/processed/` are local rebuild outputs and are not committed to git.

The current canonical processed subset uses a MedAgentBoard-like storage contract:

- `processed/ehrflowbench.jsonl`
- `processed/train.jsonl`
- `processed/test.jsonl`
- `processed/reference_answers/<split>/<qid>/answer_manifest.json`
- `processed/subset_manifest.json`

At the moment EHRFlowBench does not ship real reference answers yet. The `reference_answers/` tree is manifest-only and freezes task contracts plus future output paths.

## Raw

These inputs are prepared locally and are not committed to git.

- `raw/papers/paper_titles.csv`
- `raw/papers/markdowns/`
- `raw/tjh/`
- `raw/mimic_iv_demo/`

Dataset sources described by the benchmark papers:

- TJH: https://github.com/HAIRLAB/Pre_Surv_COVID_19
- MIMIC-IV demo: https://physionet.org/content/mimic-iv-demo/2.2/

## Scripts

- `uv run python data/ehrflowbench/scripts/generate_tasks.py --paper-id 1`
- `uv run python data/ehrflowbench/scripts/curate_generated_tasks.py`

`generate_tasks.py` is the public entrypoint for paper-to-task generation. It writes intermediate `*_tasks.json` and `*_response.json` bundles under `processed/papers/generated_tasks/`.

`curate_generated_tasks.py` is the public entrypoint for subset export. It:

- reads parseable `*_tasks.json` bundles from `processed/papers/generated_tasks/`
- classifies each generated task to exactly one dataset using `required_inputs`
- samples a fixed 110-task subset with `seed=42`
- enforces a balanced dataset mix of `55` TJH tasks and `55` MIMIC-IV-demo tasks
- splits that subset into `10` train tasks and `100` test tasks
- writes the canonical JSONL files, manifest-only `reference_answers/`, and `subset_manifest.json`

Default export policy:

- `sample_count_per_dataset = 55`
- `train_count_per_dataset = 5`
- `test_count_per_dataset = 50`
- `seed = 42`

## Task Generation Contract

The paper-to-task stage emits intermediate task objects, not final benchmark rows.
Each generated object is expected to:

- use English only
- be serialized under a single JSON object with root key `tasks`
- keep `task_type` fixed to `report_generation`
- represent one end-to-end local-data project with a required final `report.md`
- use exactly one dataset, either TJH or MIMIC-IV-demo
- stay self-contained and avoid paper-internal references such as section, figure, table, or equation numbers

The canonical structured fields for intermediate generated tasks are:

- `task_brief`
- `task_type`
- `focus_areas`
- `task`
- `required_inputs`
- `deliverables`
- `report_requirements`

## Processed

These outputs are produced locally and are not committed to git.

- `processed/ehrflowbench.jsonl`
  - Combined 110-task subset with global `qid=1..110`
- `processed/train.jsonl`
  - Training subset with split-local `qid=1..10`
- `processed/test.jsonl`
  - Test subset with split-local `qid=1..100`
- `processed/subset_manifest.json`
  - Sampling metadata, dataset balance, split policy, and global-to-split qid remapping
- `processed/reference_answers/train/<qid>/answer_manifest.json`
- `processed/reference_answers/test/<qid>/answer_manifest.json`
  - Manifest-only task contracts for split-local task ids

The default subset composition is:

- `110` tasks total
- `55` TJH tasks
- `55` MIMIC-IV-demo tasks
- `10` train tasks
- `100` test tasks

### JSONL Row Contract

Each row in `processed/ehrflowbench.jsonl`, `processed/train.jsonl`, and `processed/test.jsonl` contains:

- `qid`
- `task`
- `task_brief`
- `dataset`
- `task_type`
- `reference_answer`
- `paper_id`
- `paper_title`
- `source_task_idx`
- `focus_areas`
- `primary_bucket`

`reference_answer` always points to `reference_answers/<split>/<qid>/answer_manifest.json`.

### Reference Manifest Contract

Each `answer_manifest.json` records:

- `contract_version`
- `qid`
- `dataset`
- `task_type`
- `judge_prompt_type`
- `score_scale`
- `required_inputs`
- `required_outputs`
- `all_outputs`
- `paper_id`
- `paper_title`
- `source_task_idx`
- `focus_areas`
- `primary_bucket`
- `report_requirements`

`required_outputs` is derived directly from the generated task `deliverables`.
Because reference answers are not available yet, the manifest freezes only future output paths and metadata; it does not create the referenced files themselves.

## Local Execution Notes

`data/ehrflowbench/scripts/local_codex_exec/run_codex_ehrflowbench.py` consumes:

- `processed/train.jsonl`
- `processed/test.jsonl`
- each row's `reference_answer` manifest

The local runner relies on:

- `required_inputs`
- `required_outputs`

from each `answer_manifest.json`.

## Current Limitation

EHRFlowBench currently has task manifests but no canonical reference answers.
That means:

- the subset can be sampled and executed locally
- required output filenames are frozen
- automatic answer judging for `report_generation` is not finalized yet
