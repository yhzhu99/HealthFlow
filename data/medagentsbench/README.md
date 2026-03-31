# MedAgentsBench

This folder stores the exact hard-set subset used in the paper.

## Raw

- `raw/medagentsbench.jsonl`
- `raw/source_manifest.json`
- `raw/hf_snapshot_links.tsv`
- `raw/README.md`
- `raw/{AfrimedQA,MMLU,MMLU-Pro,MedBullets,MedExQA,MedMCQA,MedQA,MedXpertQA-R,MedXpertQA-U,PubMedQA}/*.parquet`

Source described in the paper and mirrored locally:

- https://github.com/gersteinlab/medagents-benchmark
- `~/.cache/huggingface/hub/datasets--super-dainiu--MedAgentsBench/snapshots/87e99bf48306788076dced215926e29c835fa892`

## Scripts

- `python data/medagentsbench/scripts/rebuild.py`
- `python data/medagentsbench/scripts/evaluate.py --dataset-path <run_dir> --output-dir <eval_dir>`

## Processed

- `processed/eval.jsonl` (100 rows)
- `processed/subset_manifest.json`
