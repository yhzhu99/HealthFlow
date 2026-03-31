# CureBench

This folder stores the exact evaluation subset used in the paper.

## Raw

- `raw/curebench_valset.jsonl`
- `raw/curebench_valset_train.jsonl`
- `raw/source_manifest.json`

Source described in the paper:

- https://www.kaggle.com/competitions/cure-bench/data

## Scripts

- `python data/curebench/scripts/rebuild.py`
- `python data/curebench/scripts/evaluate.py --dataset-path <run_dir> --output-dir <eval_dir>`

## Processed

- `processed/eval.jsonl` (100 rows)
- `processed/train.jsonl` (10 rows)
- `processed/subset_manifest.json`
