# Deterministic Benchmark Evaluation

Canonical benchmark data now lives under `code/HealthFlow/data/<benchmark>/`.

## Canonical task files

- `data/ehrflowbench/processed/eval.jsonl`
- `data/ehrflowbench/processed/train.jsonl`
- `data/medagentboard/processed/eval.jsonl`
- `data/curebench/processed/eval.jsonl`
- `data/curebench/processed/train.jsonl`
- `data/hle/processed/eval.jsonl`
- `data/medagentsbench/processed/eval.jsonl`

Every final benchmark row contains only:

- `qid`
- `task`
- `answer`

## Layout

- `raw/`: committed or benchmark-local source inputs
- `scripts/`: benchmark-local preparation and rebuild entrypoints
- `processed/`: canonical JSONLs, deterministic expected artifacts, and runtime copies

For file-verified benchmarks, expected outputs live under `processed/expected/<qid>/`.

## Main scripts

- `rebuild_benchmarks.py`: rebuilds all canonical benchmark folders
- `deterministic_benchmark_eval.py`: compares generated outputs against `processed/expected/<qid>/`
- `ehrflowbencch_evaluation.py`: EHRFlowBench entrypoint
- `medagentboard_evaluation.py`: MedAgentBoard entrypoint

## Usage

Rebuild everything:

```bash
python scripts/evaluation/rebuild_benchmarks.py
```

Refresh EHRFlowBench raw paper inputs, then rebuild:

```bash
python data/ehrflowbench/scripts/prepare_raw.py
python data/ehrflowbench/scripts/rebuild.py
```

Evaluate EHRFlowBench outputs:

```bash
python scripts/evaluation/ehrflowbencch_evaluation.py \
  --dataset-path benchmark_results/ehrflowbench/my_model \
  --output-dir benchmark_results/ehrflowbench_eval/my_model
```

Evaluate MedAgentBoard outputs:

```bash
python scripts/evaluation/medagentboard_evaluation.py \
  --dataset-path benchmark_results/medagentboard/my_model \
  --output-dir benchmark_results/medagentboard_eval/my_model
```

Optional arguments:

- `--benchmark-file`: override the canonical JSONL
- `--answer-path`: override the `processed/expected/` root
- `--qid-range`: limit evaluation to selected QIDs
