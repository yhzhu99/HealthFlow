# EHRFlowBench

This benchmark is rebuilt from extracted paper tasks into reproducible local-data proxy tasks.
Dataset-specific preparation and curation workflows belong under `data/ehrflowbench/` and are intentionally decoupled from the `healthflow/` runtime package. `raw/`, `processed/`, and `scripts/upstream/extract_task/assets/` are local-only and gitignored.

The canonical grading source of truth is:

- `processed/eval.jsonl` / `processed/train.jsonl`
- `processed/expected/<qid>/`
- the task manifests and subset manifest

The original papers provide provenance and task motivation only.

## Raw

These inputs are prepared locally and are not committed to git.

- `raw/papers/selected_ids.txt`
- `raw/papers/extracted_tasks/`
- `raw/papers/paper_titles.csv`
- `raw/papers/markdowns/` (optional local audit mirror; not required for canonical rebuilds)
- `raw/tjh/`
- `raw/mimic_iv_demo/`
- `raw/source_manifest.json`

Dataset sources described in the paper:

- TJH: https://github.com/HAIRLAB/Pre_Surv_COVID_19
- MIMIC-IV demo: https://physionet.org/content/mimic-iv-demo/2.2/
- Extracted-paper corpus: `data/ehrflowbench/scripts/upstream/filter_paper/results/`, `data/ehrflowbench/scripts/upstream/extract_task/tasks/`, and an optional local markdown mirror under `data/ehrflowbench/scripts/upstream/extract_task/assets/markdowns/` (local cache only, not committed)

## Scripts

- `python data/ehrflowbench/scripts/prepare_raw.py`
- `python data/ehrflowbench/scripts/prepare_raw.py --include-markdowns`
- `uv run python data/ehrflowbench/scripts/prepare_ehr/prepare_tjh.py`
- `uv run python data/ehrflowbench/scripts/prepare_ehr/prepare_mimic_iv_demo.py`
- `uv run python data/ehrflowbench/scripts/generate_tasks_via_api.py --paper-id 1`
- `uv run python data/ehrflowbench/scripts/curate_generated_tasks.py`
- `uv run python data/ehrflowbench/scripts/curate_generated_tasks.py --allow-incomplete`

`prepare_raw.py` refreshes selected paper IDs and extracted task files. It also refreshes `raw/papers/paper_titles.csv` and can materialize `raw/papers/markdowns/` when the optional upstream markdown mirror is available locally. Other benchmark build/evaluation flows should stay under `data/` and evolve independently from `healthflow/`.

The reusable manual prompt for advanced tool-enabled models such as GPT-5.4 lives at `data/ehrflowbench/scripts/upstream/extract_task/prompt_gpt54_ehrflowbench.md`. It is intended for the paper-to-task stage and emits structured task objects only, not reference answers.

`curate_generated_tasks.py` is the local curation entrypoint that:

- waits for parseable `*_tasks.json` + `*_response.json` bundles under `processed/papers/generated_tasks/`
- scores candidate tasks, keeps at most one task per paper, applies bucket quotas and near-duplicate filtering
- writes the canonical `train.jsonl` / `eval.jsonl`, provenance CSVs, and `processed/expected/<qid>/task_manifest.json`

By default the curation step refuses to freeze an incomplete paper pool. Use `--allow-incomplete` only for provisional local rebuilds while generation is still running.

## Task Generation Contract

The paper-to-task stage emits intermediate task objects, not final benchmark rows. Each generated object is expected to:

- use English only
- be serialized under a single JSON object with root key `tasks`
- keep `task_type` fixed to `report_generation`
- represent one end-to-end project that includes data preparation, analysis or modeling, quantitative evidence, and a final `report.md`
- be executable on both TJH and MIMIC-IV-demo, using the local raw assets and, when needed, the canonical prepared tables under `processed/tjh/` and `processed/mimic_iv_demo/`
- remain fully self-contained and avoid paper-internal references such as section, figure, table, or equation numbers

The canonical structured fields for these intermediate objects are:

- `task_brief`
- `task_type`
- `focus_areas`
- `task`
- `required_inputs`
- `deliverables`
- `report_requirements`

Reference answers are generated later, after candidate tasks are accepted into the benchmark curation flow.

## Processed

These outputs are produced locally by dataset-specific workflow scripts and are not committed to git.

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

`processed/train.jsonl` and `processed/eval.jsonl` use the standard HealthFlow task schema with `qid`, `task`, and `reference_answer`, plus benchmark metadata such as `task_brief`, `paper_id`, and `primary_bucket`.

`reference_answer` points to `processed/expected/<qid>/task_manifest.json`. These manifests freeze the task contract, deliverable list, provenance, and review metadata before reference answers are generated.
