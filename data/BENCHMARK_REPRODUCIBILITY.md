# Benchmark Reproducibility

Canonical benchmark data lives entirely under `data/`. There is no benchmark-level root `scripts/` directory: each dataset owns its own `data/<dataset>/scripts/` entrypoints, and shared reusable rebuild/evaluation code lives in `healthflow/benchmarks/`.

## Canonical Layout

- `data/rebuild_all.py`
- `data/ehrflowbench/{raw,scripts,processed}`
- `data/medagentboard/{raw,scripts,processed}`
- `data/curebench/{raw,scripts,processed}`
- `data/hle/{raw,scripts,processed}`
- `data/medagentsbench/{raw,scripts,processed}`

Each `raw/` folder also contains a `source_manifest.json` that records the upstream dataset URLs, access notes, and the local files required for deterministic rebuilds.

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
- `data/hle/processed/eval.jsonl`: 30 rows
- `data/medagentsbench/processed/eval.jsonl`: 100 rows

## Rebuild entrypoints

- Rebuild all committed benchmarks: `python data/rebuild_all.py`
- Rebuild a single benchmark: `python data/<dataset>/scripts/rebuild.py`
- Run deterministic evaluation: `python data/<dataset>/scripts/evaluate.py --dataset-path <run_dir> --output-dir <eval_dir>`
- Refresh the EHRFlowBench paper raw mirror before rebuilding: `python data/ehrflowbench/scripts/prepare_raw.py`
- Materialize the optional large paper markdown mirror too: `python data/ehrflowbench/scripts/prepare_raw.py --include-markdowns`

## Paper audit

EHRFlowBench is rebuilt from extracted paper tasks using benchmark-local raw paper inputs. `data/ehrflowbench/scripts/rebuild.py` refreshes `raw/papers/selected_ids.txt` and `raw/papers/extracted_tasks/` from the dataset-local upstream curation tree before generating the canonical benchmark. If the optional upstream markdown mirror is available locally, `prepare_raw.py` also refreshes `raw/papers/paper_titles.csv`; otherwise the committed canonical `paper_titles.csv` is reused.

The upstream curation assets now live under:

- `data/ehrflowbench/scripts/upstream/filter_paper/`
- `data/ehrflowbench/scripts/upstream/extract_task/`
- `data/ehrflowbench/scripts/upstream/title_extract/`
- `data/ehrflowbench/scripts/upstream/utils/`

Paper linkage stays machine-verifiable through `data/ehrflowbench/processed/paper_map.csv`, and the benchmark build writes benchmark-local task manifests so each final task remains self-contained and traceable to its source paper.

The optional raw markdown mirror at `data/ehrflowbench/raw/papers/markdowns/` is no longer required for canonical rebuilds. It can still be materialized on demand with `--include-markdowns` if someone wants a local audit copy of the extracted paper markdowns.

## Scoring

Deterministic expected outputs now live in `processed/expected/<qid>/`. The evaluator compares generated JSON/CSV/text outputs against those processed-side artifacts and falls back to direct answer comparison for text-only benchmarks.
