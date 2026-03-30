# HealthFlow: EHR-Specific Analysis Harness with Self-Evolving Memory

[![arXiv](https://img.shields.io/badge/arXiv-2508.02621-b31b1b.svg)](https://arxiv.org/abs/2508.02621)
[![Project Website](https://img.shields.io/badge/Project%20Website-HealthFlow-0066cc.svg)](https://healthflow-agent.netlify.app)

HealthFlow is a research framework for **EHR-focused analysis orchestration**. It is not a claim that one raw coding backend is universally strongest. The contribution in this codebase is the **EHR-aware harness** around execution:

- EHR task-family profiling and risk detection
- staged tool exposure instead of prompt dumping
- hierarchical memory with explicit strategy vs failure separation
- deterministic verifier gating before success is accepted
- reproducible workspace contracts and run manifests

The default execution backend is `healthflow_agent`, HealthFlow's integrated executor path. `claude_code` and `opencode` remain available as thin compatibility adapters.

## Core Runtime

HealthFlow runs a lean **Profile -> Plan -> Execute -> Verify -> Reflect** loop.

1. **Profile**: inspect uploaded files, classify the EHR task family, detect patient identifiers, target-like columns, time columns, and workflow hints.
2. **Plan**: retrieve relevant memory, separating reusable strategy from failure-avoidance memory.
3. **Execute**: run the selected backend inside a task workspace with an explicit output and verification contract.
4. **Verify**: apply deterministic artifact checks before success is allowed.
5. **Reflect**: write verified strategy/artifact memory from good runs and failure/verifier-rule memory from bad runs.

## What HealthFlow Contributes

- **EHR-specific harness**: cohort semantics, leakage checks, split expectations, temporal validation cues, and report contracts tuned for healthcare data science.
- **Inspectable memory**: dataset, strategy, failure, and artifact memories are stored in JSONL and retrieved with layer budgets, validation status, and conflict suppression.
- **Deterministic verifier**: success is gated by artifact checks such as cohort definition evidence, split evidence, audit artifacts, metrics files, and report sections.
- **Reproducibility contract**: every task workspace writes structured runtime artifacts instead of only human-readable logs.

## Workspace Artifacts

Each task creates a workspace under `workspace/<task_id>/` and writes:

- `executor_prompt.md`
- `<backend>_execution.log`
- `task_list_v*.md`
- `full_history.json`
- `memory_context.json`
- `verification.json`
- `run_manifest.json`
- `run_result.json`

These files are the main source of truth for rebuttal-oriented inspection.

## Evaluator vs Verifier

- **Verifier**: deterministic, file- and artifact-based. It checks whether required evidence exists and whether obvious execution failures happened.
- **Evaluator**: LLM-based quality scoring on top of the verifier output.

HealthFlow only marks a run successful when the execution succeeds, the evaluator score clears the threshold, and the verifier gate passes when verifier gating is required.

## Memory Behavior

HealthFlow uses four memory layers:

- `dataset`
- `strategy`
- `failure`
- `artifact`

Retrieval is auditable:

- verified memories are preferred for positive strategy layers
- failure memories keep their own retrieval budget
- contradictory memories are suppressed by `conflict_group`
- the retrieval audit is saved to `memory_context.json`

Writeback behavior:

- verified runs can produce `dataset`, `strategy`, and `artifact` memories
- failed or verifier-rejected runs are normalized into `failure` / `verifier_rule` style memory
- both successful and failed runs can teach future tasks, unless the memory mode is frozen

## Supported Execution Backends

HealthFlow keeps the executor layer backend-agnostic, but the public surface is intentionally small:

- `healthflow_agent` (default)
- `claude_code`
- `opencode`

You can still define additional CLI backends in `config.toml`, but the harness logic stays in HealthFlow rather than being baked into one external backend.

## OneEHR-Aware Workflows

HealthFlow can guide and verify OneEHR-style workflows without requiring a hard dependency on OneEHR internals. Modeling-task verifier checks accept artifacts such as:

- `manifest.json`
- `preprocess/split.json`
- `test/metrics.json`
- `analyze/*.json`

This lets HealthFlow act as the orchestration and audit harness around a reproducible EHR CLI workflow.

## Quick Start

### Prerequisites

- Python 3.12+
- `uv`
- one execution backend available in `PATH`
  - default: `healthflow-agent`
  - compatibility: `claude`, `opencode`

### Setup

```bash
uv sync
source .venv/bin/activate
cp config.toml.example config.toml
```

Then edit `config.toml` with the reasoning-model API credentials you want to use for planning, evaluation, and reflection.

### Web UI

```bash
streamlit run app.py
```

### Single Task

```bash
python run_healthflow.py run \
  "Analyze the uploaded patients.csv to identify the top 3 risk factors for readmission." \
  --active-llm deepseek-chat \
  --active-executor healthflow_agent
```

### Interactive Mode

```bash
python run_healthflow.py interactive \
  --active-llm deepseek-chat \
  --active-executor healthflow_agent
```

### Training

Training data must be JSONL with `qid`, `task`, and `answer`.

```bash
python run_training.py data/train_set.jsonl ehrflow_train \
  --active-llm deepseek-reasoner \
  --active-executor healthflow_agent
```

### Benchmarking

Benchmarking uses the same task JSONL shape, but **defaults to frozen memory behavior** for reproducibility.

```bash
python run_benchmark.py data/benchmark_set.jsonl ehrflow_eval \
  --active-llm deepseek-reasoner \
  --active-executor healthflow_agent
```

Results are written under `benchmark_results/` with per-task copies of the workspace artifacts and dataset-level summary JSON.

## Configuration

Main config sections:

- `[llm.*]`: reasoning model providers
- `[executor]`: default backend and CLI backend definitions
- `[memory]`: retrieval budgets and memory mode
- `[ehr]`: profiling controls
- `[verification]`: deterministic success gating
- `[evaluation]`: evaluator success threshold
- `[system]`: workspace and retry settings
- `[logging]`: log level and log file

## Repository Layout

- `app.py`: Streamlit UI
- `run_healthflow.py`: single-task and interactive CLI
- `run_training.py`: memory bootstrapping and training-style runs
- `run_benchmark.py`: reproducible benchmark runner with frozen memory default
- `healthflow/system.py`: orchestration loop
- `healthflow/execution/`: executor layer
- `healthflow/ehr/`: profiling, task-family logic, and risk checks
- `healthflow/verification/`: deterministic verifier
- `healthflow/experience/`: hierarchical memory and retrieval audit
- `healthflow/tools/`: staged tool-bundle selection

## Citation

```bibtex
@misc{zhu2025healthflow,
  title={HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research},
  author={Yinghao Zhu and Yifan Qi and Zixiang Wang and Lei Gu and Dehao Sui and Haoran Hu and Xichen Zhang and Ziyi He and and Junjun He and Liantao Ma and Lequan Yu},
  year={2025},
  eprint={2508.02621},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2508.02621},
}
```
