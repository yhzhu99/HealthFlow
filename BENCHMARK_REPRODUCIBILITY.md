# Benchmark Reproducibility

Canonical benchmark data now lives under `code/HealthFlow/data/` and is rebuilt by `scripts/evaluation/rebuild_benchmarks.py`.

## Canonical Layout

- `data/ehrflowbench/{raw,scripts,processed}`
- `data/medagentboard/{raw,scripts,processed}`
- `data/curebench/{raw,scripts,processed}`
- `data/hle/{raw,scripts,processed}`
- `data/medagentsbench/{raw,scripts,processed}`

## Final JSONL schema

Every canonical task row now contains only:

- `qid`
- `task`
- `answer`

## Processed outputs

- `data/ehrflowbench/processed/eval.jsonl`: 100 rows
- `data/ehrflowbench/processed/train.jsonl`: 10 rows
- `data/medagentboard/processed/eval.jsonl`: 100 rows
- `data/curebench/processed/eval.jsonl`: 100 rows
- `data/curebench/processed/train.jsonl`: 10 rows
- `data/hle/processed/eval.jsonl`: 45 rows
- `data/medagentsbench/processed/eval.jsonl`: 100 rows

## Paper audit

EHRFlowBench is rebuilt from extracted paper tasks using benchmark-local raw paper inputs. The raw paper mirror is refreshed by `data/ehrflowbench/scripts/prepare_raw.py`, which syncs:

- selected IDs from `scripts/filter_paper/results/final_selected_ID.txt`
- extracted tasks from `scripts/extract_task/tasks/`
- markdowns from `scripts/extract_task/assets/markdowns/`

Paper audit summary:

- selected papers: 118
- markdown directories: 160
- extracted task files: 118
- extracted tasks: 590
- task audit decisions: {'reject': 569, 'rewrite': 21}
- task audit reason counts: {'inaccessible_data': 131, 'non_executable': 464, 'paper_metric_only': 317, 'weak_comparability': 88, 'non_verifiable': 20}

The machine-readable paper linkage is written to `data/ehrflowbench/processed/paper_map.csv`.

## Scoring

Deterministic expected outputs now live in `processed/expected/<qid>/`. The evaluator compares generated JSON/CSV/text outputs against those processed-side artifacts and falls back to direct answer comparison for text-only benchmarks.
