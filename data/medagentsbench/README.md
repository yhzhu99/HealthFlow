# MedAgentsBench

This folder stores the exact hard-set subset used in the paper.

## Raw

- `raw/medagentsbench.jsonl`
- `raw/source_manifest.json`

Source described in the paper:

- https://github.com/gersteinlab/medagents-benchmark

## Scripts

- `python data/medagentsbench/scripts/rebuild.py`
- `python data/medagentsbench/scripts/evaluate.py --dataset-path <run_dir> --output-dir <eval_dir>`

## Processed

- `processed/eval.jsonl` (100 rows)
- `processed/subset_manifest.json`
