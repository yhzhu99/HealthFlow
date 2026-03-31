# EHRFlowBench

This benchmark is rebuilt from extracted paper tasks into reproducible local-data proxy tasks.

The canonical grading source of truth is:

- `processed/eval.jsonl` / `processed/train.jsonl`
- `processed/expected/<qid>/`
- the task manifests and subset manifest

The original papers provide provenance and task motivation only.

## Raw

- `raw/papers/selected_ids.txt`
- `raw/papers/extracted_tasks/`
- `raw/papers/paper_titles.csv`
- `raw/papers/markdowns/` (optional local audit mirror; not required for canonical rebuilds)
- `raw/tjh/`
- `raw/mimic_iv_demo_meds/`
- `raw/source_manifest.json`

Dataset sources described in the paper:

- TJH: https://github.com/HAIRLAB/Pre_Surv_COVID_19
- MIMIC-IV demo MEDS: https://physionet.org/content/mimic-iv-demo-meds/0.0.1/
- Extracted-paper corpus: `data/ehrflowbench/scripts/upstream/filter_paper/results/`, `data/ehrflowbench/scripts/upstream/extract_task/tasks/`, and an optional local markdown mirror under `data/ehrflowbench/scripts/upstream/extract_task/assets/markdowns/`

## Scripts

- `python data/ehrflowbench/scripts/prepare_raw.py`
- `python data/ehrflowbench/scripts/prepare_raw.py --include-markdowns`
- `python data/ehrflowbench/scripts/rebuild.py`
- `python data/ehrflowbench/scripts/evaluate.py --dataset-path <run_dir> --output-dir <eval_dir>`

`prepare_raw.py` always refreshes selected paper IDs and extracted task files. It refreshes `raw/papers/paper_titles.csv` and can materialize `raw/papers/markdowns/` only when the optional upstream markdown mirror is available locally.

## Processed

- `processed/eval.jsonl`
- `processed/train.jsonl`
- `processed/paper_map.csv`
- `processed/rejected_tasks.csv`
- `processed/task_manifest_eval.jsonl`
- `processed/task_manifest_train.jsonl`
- `processed/subset_manifest.json`
- `processed/runtime/`
- `processed/expected/`

`processed/paper_map.csv` records one provenance row per canonical task, including:

- source paper identifiers
- source task linkage mode (`task_proxy`)
- source task eligibility (`proxy_candidate`)
- proxy constraint flags
- current review status (`seeded_for_human_review`)
