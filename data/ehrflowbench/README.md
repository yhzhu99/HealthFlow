# EHRFlowBench

EHRFlowBench is the paper's benchmark of end-to-end EHR analysis tasks grounded in TJH and the MIMIC-IV Public Demo Release.
All files under `data/ehrflowbench/processed/` are local data artifacts and are not committed to git.
The released corpus and its reference answers are distributed through GitHub Releases instead.

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

Prepare the benchmark-local EHR tables with:

- `uv run python data/ehrflowbench/scripts/prepare_ehr/prepare_tjh.py`
- `uv run python data/ehrflowbench/scripts/prepare_ehr/prepare_mimic_iv_demo.py`

Generate the two dataset-grounded candidate tasks for one seed paper with:

```bash
uv run python data/ehrflowbench/scripts/prepare_tasks/generate_tasks.py --paper-id 1
```

Batch-generate a paper range with:

```bash
uv run bash data/ehrflowbench/scripts/prepare_tasks/batch_generate_tasks.sh 1 10
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
- `reference_answer`
- `paper_id`
- `paper_title`
- `source_task_idx`

`reference_answer` names the answer manifest of the task, resolved relative to the benchmark jsonl file.
The task text is the finalized, self-contained evaluation prompt and is not wrapped in another prompt before benchmark execution.

The bundle also carries `processed/ehrflowbench.jsonl`, `processed/train.jsonl`, `processed/subset_manifest.json`, and the
`processed/reference_answers/` tree that `reference_answer` points into. Each answer manifest declares `qid`, `dataset`,
`task_type`, `primary_category`, `required_inputs` and `required_outputs`; every required output carries `file_name`,
`reference_path` and a `media_type`.

## Evaluation

EHRFlowBench is scored with a reference-based LLM judge. Unlike MedAgentBoard, which
compares per-artifact reference/submission summaries, the judge attaches the released
reference report and the submitted report as PDFs and scores four rubric dimensions on a
1-5 scale (`method_soundness`, `presentation_quality`, `artifact_generation`,
`overall_score`). A markdown-only report is rendered to PDF before judging.

```bash
uv run --with reportlab python data/ehrflowbench/scripts/evaluate.py \
  --submission-root benchmark_results/ehrflowbench/opencode/deepseek-chat
```

`--with reportlab` provides the renderer on demand, so the shared environment stays unchanged.

The submission root is laid out as `<root>/<qid>/`, which matches the directory written by
`run_benchmark.py`. Each task directory is expected to contain `report.pdf` or `report.md`.

Optional overrides:

```bash
uv run --with reportlab python data/ehrflowbench/scripts/evaluate.py \
  --submission-root benchmark_results/ehrflowbench/opencode/deepseek-chat \
  --judge-llm openai/gpt-5.4 \
  --pass-threshold 3 \
  --qid 1 --qid 2
```

Results are written to `<submission-root>/ehrflowbench.eval.json` and include
`total_questions`, `scored_questions`, `failed_questions`, `passed_questions`,
`average_score`, and per-dimension averages. Tasks whose reference report or submitted
report is missing are recorded with `status: "failed"` and score `0`, and are excluded from
the averages. `passed` defaults to `overall_score >= pass_threshold` with
`3` as the rubric-anchored default threshold.

Shared judge configuration lives in `config.toml` under `[llm."openai/gpt-5.4"]`; the
rubric lives in `data/ehrflowbench/scripts/judge_prompts/report_generation.md`.

### Reference Answers

The reference answers are not tracked in git. Download the released bundle from GitHub Releases and unpack it
anywhere. Reference paths are resolved relative to the benchmark jsonl file, so pointing `--benchmark-file` at your
own copy is enough:

```bash
uv run --with reportlab python data/ehrflowbench/scripts/evaluate.py \
  --submission-root benchmark_results/ehrflowbench/opencode/deepseek-chat \
  --benchmark-file /path/to/release/processed/test.jsonl
```

This matches the MedAgentBoard evaluator: the directory containing the benchmark jsonl is the benchmark root, the
`reference_answer` field of each row names the manifest, and the manifest's `reference_path` names the reference
report.
