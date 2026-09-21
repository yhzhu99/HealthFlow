# EHRFlowBench

EHRFlowBench is the paper's benchmark of end-to-end EHR analysis tasks grounded in TJH and the MIMIC-IV Public Demo Release.
All files under `data/ehrflowbench/processed/` are local data artifacts and are not committed to git.

## Paper-Aligned Construction

The benchmark follows the construction process reported in the paper:

1. Screen 51,280 papers from major AI and data mining venues.
2. Identify 162 EHR-relevant candidates and manually retain 118 seed papers.
3. Generate one TJH task and one MIMIC-IV task from each seed paper, producing 236 dataset-grounded candidate tasks.
4. Curate the candidates into the fixed 100-task benchmark: 50 TJH tasks and 50 MIMIC-IV tasks.

The released `processed/test.jsonl` is the authoritative benchmark used by the paper. Treat it as a fixed evaluation set; do not resample or regenerate its membership from the candidate pool.

The 100 tasks cover the 10 categories reported in the paper:

- Temporal prediction: 19 tasks
- Graph and retrieval: 18 tasks
- Representation and features: 17 tasks
- Phenotyping and clustering: 12 tasks
- Robustness and missingness: 10 tasks
- Multi-task learning and transfer: 8 tasks
- Synthetic data and privacy: 7 tasks
- Causal and counterfactual analysis: 4 tasks
- Forecasting: 3 tasks
- Natural-language querying and reporting: 2 tasks

## Candidate Task Generation

Generate the two dataset-grounded candidate tasks for one seed paper with:

```bash
uv run python data/ehrflowbench/scripts/prepare_tasks/generate_tasks.py --paper-id 1
```

`generate_tasks.py` writes intermediate `*_tasks.json` bundles under `processed/papers/generated_tasks/`. Candidate generation does not determine membership in the fixed paper benchmark.

Each generated task bundle uses a JSON object with root key `tasks`. Each task contains:

- `task_brief`
- `task_type`
- `focus_areas`
- `task`
- `required_inputs`
- `deliverables`
- `report_requirements`

`task_type` is fixed to `report_generation`. Each task is a self-contained project grounded in exactly one of the two released EHR datasets.

## Released Benchmark

The dataset release provides `processed/test.jsonl` with 100 rows and sequential `qid` values. Its core task metadata include:

- `qid`
- `task`
- `task_brief`
- `dataset`
- `task_type`
- `paper_id`
- `paper_title`
- `source_task_idx`

The task text is the finalized, self-contained evaluation prompt and is not wrapped in another prompt before benchmark execution.

## Evaluation

`python data/ehrflowbench/scripts/evaluate.py --submission-root <submissions> --output-file <scores.json>` evaluates `<submissions>/<qid>/` against the reference artifacts. Install Pandoc and XeLaTeX and set `GEMINI_API_KEY`; the default judge is Gemini 3.1 Flash-Lite. Use `--dry-run` to check files and render reports without API calls.

As described in the published paper, reports are rendered to PDF and scored on a native 1–5 scale. The evaluator returns method soundness, presentation quality, artifact generation and a separate overall score, matching the fine-grained results table. Missing required artifacts receive 0. The reported 4.01 is on this scale. Results include 100-resample bootstrap means and standard deviations, plus the dimension breakdown for each dataset. The bootstrap seed defaults to 42 and can be set with `--bootstrap-seed`.

For HealthFlow exports containing `runtime/run/trajectory.json`, the evaluator selects the first successful attempt with score at least 0.8, otherwise attempt 3. Single-attempt or already selected submissions can place files directly under `<submissions>/<qid>/`. Multiple snapshots can instead be listed in `attempts.json` as an `attempts` array, each entry containing `attempt`, `evaluation` (`status`, `score`) and a relative `artifact_dir`.

The script supplies an implementation of the published evaluation description; historical per-task judge responses have not been recovered or rerun here.
