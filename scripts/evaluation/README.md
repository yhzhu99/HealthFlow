# Deterministic Benchmark Evaluation

The canonical `EHRFlowBench` and `MedAgentBoard` benchmarks are now rebuilt from local, free-to-use data with committed ground-truth artifacts and deterministic scoring.

## Canonical datasets

- `data/ehrflowbench.jsonl`
- `data/ehrflowbench_train.jsonl`
- `data/medagentboard.jsonl`
- `data/TJH.csv`
- `data/mimic_iv_demo_meds/`

Each benchmark row includes:

- `required_files`
- `verification_spec`
- `ground_truth_ref`
- dataset provenance metadata

## Rebuild and provenance

- `rebuild_benchmarks.py`: regenerates the canonical JSONLs, ground-truth artifact folders, and paper-audit outputs.
- `fetch_mimic_demo_meds.py`: downloads the open PhysioNet `MIMIC-IV demo MEDS` snapshot and verifies checksums.
- `data/benchmark_ground_truth/`: per-QID expected files.
- `data/benchmark_provenance/`: audit tables and filter/markdown reconciliation manifests.

## Evaluation scripts

- `ehrflowbencch_evaluation.py`
- `medagentboard_evaluation.py`
- `deterministic_benchmark_eval.py`

These scripts no longer require LLM judges. They compare an agent's generated files against the committed ground-truth artifacts using exact or tolerance-bounded checks.

## Usage

Rebuild the benchmark:

```bash
python scripts/evaluation/rebuild_benchmarks.py
```

Re-download and verify the open MEDS demo:

```bash
python scripts/evaluation/fetch_mimic_demo_meds.py
```

Evaluate benchmark outputs deterministically:

```bash
python scripts/evaluation/ehrflowbencch_evaluation.py \
  --dataset-path benchmark_results/ehrflowbench/my_model \
  --output-dir benchmark_results/ehrflowbench_eval/my_model
```

```bash
python scripts/evaluation/medagentboard_evaluation.py \
  --dataset-path benchmark_results/medagentboard/my_model \
  --output-dir benchmark_results/medagentboard_eval/my_model
```

Optional arguments:

- `--benchmark-file`: override the canonical JSONL used to supply task metadata.
- `--answer-path`: override the ground-truth root directory.
- `--qid-range`: limit evaluation to a subset of QIDs.

## Validation expectations

- Canonical benchmark files must have no blank answers and no inaccessible paths.
- Every scored task must emit the machine-readable files listed in `required_files`.
- Plot tasks are scored from structured plot summaries and file existence, not from image pixel diffs.
