# EHRFlowBench

EHRFlowBench turns paper-inspired EHR projects into repository-local `report_generation` tasks for TJH and MIMIC-IV-demo.
All files under `data/ehrflowbench/processed/` are local rebuild artifacts and are not committed to git.

## Scripts

- `uv run python data/ehrflowbench/scripts/prepare_tasks/generate_tasks.py --paper-id 1`
- `uv run python data/ehrflowbench/scripts/prepare_tasks/curate_generated_tasks.py`

`prepare_tasks/generate_tasks.py` writes intermediate `*_tasks.json` bundles under `processed/papers/generated_tasks/`.

`prepare_tasks/curate_generated_tasks.py` only does subset extraction:

- reads `processed/papers/generated_tasks/*_tasks.json`
- infers the dataset from `required_inputs`
- samples `55` TJH tasks and `55` MIMIC-IV-demo tasks with `seed=42`
- splits them into `10` train tasks and `100` test tasks
- writes the processed JSONL files and manifest-only `reference_answers/`

## Intermediate Task Contract

Each generated task bundle uses a JSON object with root key `tasks`.
Each task is expected to contain:

- `task_brief`
- `task_type`
- `focus_areas`
- `task`
- `required_inputs`
- `deliverables`
- `report_requirements`

`task_type` is fixed to `report_generation`.

## Processed Outputs

The extraction step writes:

- `processed/ehrflowbench.jsonl`
- `processed/train.jsonl`
- `processed/test.jsonl`
- `processed/subset_manifest.json`
- `processed/reference_answers/train/<qid>/answer_manifest.json`
- `processed/reference_answers/test/<qid>/answer_manifest.json`

Default subset composition:

- `110` tasks total
- `55` TJH tasks
- `55` MIMIC-IV-demo tasks
- `10` train tasks
- `100` test tasks

### JSONL Row Fields

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

`task` is the original generated task text. It is not wrapped into another prompt during extraction.

### Reference Manifest Fields

Each `answer_manifest.json` contains:

- `contract_version`
- `qid`
- `dataset`
- `task_type`
- `required_inputs`
- `required_outputs`
- `all_outputs`
- `paper_id`
- `paper_title`
- `source_task_idx`

`required_outputs` is derived directly from the generated task `deliverables`.

## Current Limitation

EHRFlowBench does not have real reference answers yet.
The `reference_answers/` tree currently stores manifests only and does not create placeholder output files.
