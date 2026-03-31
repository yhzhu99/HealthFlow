# EHRFlowBench

This benchmark is rebuilt from extracted paper tasks into reproducible local-data proxy tasks.

## Raw

- `raw/papers/selected_ids.txt`
- `raw/papers/markdowns/`
- `raw/papers/extracted_tasks/`
- `raw/papers/paper_titles.csv`
- `raw/tjh/`
- `raw/mimic_iv_demo_meds/`

Dataset sources described in the paper:

- TJH: https://github.com/HAIRLAB/Pre_Surv_COVID_19
- MIMIC-IV demo MEDS: https://physionet.org/content/mimic-iv-demo-meds/0.0.1/
- Extracted-paper corpus: `scripts/filter_paper/results/`, `scripts/extract_task/tasks/`, and `scripts/extract_task/assets/markdowns/`

## Scripts

- `python data/ehrflowbench/scripts/prepare_raw.py`
- `python data/ehrflowbench/scripts/rebuild.py`

## Processed

- `processed/eval.jsonl`
- `processed/train.jsonl`
- `processed/paper_map.csv`
- `processed/rejected_tasks.csv`
- `processed/runtime/`
- `processed/expected/`
