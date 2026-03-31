# HealthFlow: General Analysis Orchestration with EHR-Aware Self-Evolving Memory

[![arXiv](https://img.shields.io/badge/arXiv-2508.02621-b31b1b.svg)](https://arxiv.org/abs/2508.02621)
[![Project Website](https://img.shields.io/badge/Project%20Website-HealthFlow-0066cc.svg)](https://healthflow-agent.netlify.app)

HealthFlow is a research framework for **general analysis orchestration with healthcare-aware specialization** built around external coding executors. This repository serves the HealthFlow paper's study of autonomous EHR analysis and self-evolving planning, but the runtime is intentionally **general-first** rather than hard-locked to EHR-only tasks. EHR-specific safeguards are layered in only when the request, data profile, or workflow artifacts justify them.

- task-family profiling plus conditional domain-overlay detection
- staged tool exposure instead of prompt dumping
- hierarchical memory with explicit strategy vs failure separation
- deterministic verifier gating before success is accepted
- reproducible workspace contracts and run manifests

The default execution backend is `healthflow_agent`, HealthFlow's integrated executor path. `claude_code`, `opencode`, and `pi` remain available as thin compatibility adapters.

The current release surface is intentionally **backend and CLI only**. A frontend is not shipped in this repo at this stage.

## Core Runtime

HealthFlow runs a lean **Profile -> Plan -> Execute -> Verify -> Reflect** loop.

1. **Profile**: inspect uploaded files, classify the analysis task family, detect domain signals, and summarize identifiers, targets, time columns, and workflow hints.

The task-level self-correction budget is controlled by `system.max_attempts`, which counts total full attempts through the loop rather than "retries plus one".
2. **Plan**: retrieve relevant memory, separating reusable strategy from failure-avoidance memory.
3. **Execute**: run the selected backend inside a task workspace with soft deliverable guidance and auditable verification focus.
4. **Verify**: apply deterministic artifact checks before success is allowed, with stricter EHR checks only when the profiled context warrants them.
5. **Reflect**: write verified strategy/artifact memory from good runs and failure/verifier-rule memory from bad runs.

## What HealthFlow Contributes

- **General-first runtime with conditional EHR overlays**: the same loop handles ordinary analysis tasks, while cohort semantics, patient-aware split cues, leakage checks, and temporal validation hints activate only when supported by the profiled context.
- **Inspectable memory**: dataset, strategy, failure, and artifact memories are stored in JSONL and retrieved with layer budgets, validation status, and conflict suppression.
- **Deterministic verifier**: success is gated by artifact checks such as split evidence, audit artifacts, metrics files, figures, and optional report structure, with cohort-specific checks reserved for cohort or EHR-style workflows.
- **Reproducibility contract**: every task workspace writes structured runtime artifacts instead of only human-readable logs.
- **Executor telemetry**: run artifacts capture executor metadata, backend versions when available, LLM usage, and estimated LLM cost.
- **Role-specific internal models**: planner, evaluator, and reflector can be configured against different reasoning models to reduce single-model coupling.

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

For deterministic benchmarks such as EHRFlowBench and MedAgentBoard, the intended primary metric is file-verified pass rate. The LLM evaluator is secondary metadata for run inspection and error analysis.

## Memory Behavior

HealthFlow uses four memory layers:

- `dataset`
- `strategy`
- `failure`
- `artifact`

Retrieval is auditable:

- verified memories are preferred for positive strategy layers
- failure memories keep their own retrieval budget
- contradictory memories are tracked by `conflict_group`
- safety-critical failure memories suppress conflicting strategy memories before execution
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
- `pi`

You can still define additional CLI backends in `config.toml`, but the harness logic stays in HealthFlow rather than being baked into one external backend.
Executor-specific repository instruction files are intentionally avoided at the repo root so backend comparisons use the same injected prompt guidance.

## External CLI Workflows

HealthFlow does not hardcode compatibility for any specific domain package or external tool. Instead, it interacts with external systems through the same backend-agnostic execution layer used for all tasks.

When an EHR workflow is best handled by an external CLI, the agent can still invoke that CLI as part of its plan. HealthFlow itself remains responsible only for planning, memory, execution orchestration, and generic verification of the resulting analysis artifacts.

## Quick Start

### Prerequisites

- Python 3.12+
- `uv`
- one execution backend available in `PATH`
  - default: `healthflow-agent`
  - compatibility: `claude`, `opencode`, `pi`

### Setup

```bash
uv sync
source .venv/bin/activate
cp config.toml.example config.toml
```

Then edit `config.toml` with the reasoning-model API credentials you want to use for planning, evaluation, and reflection.

If you want estimated LLM cost summaries in run artifacts, set `input_cost_per_million_tokens` and `output_cost_per_million_tokens` for the active reasoning model in `config.toml`.

To decouple the internal roles, set:

```toml
[llm_roles]
planner = "deepseek-reasoner"
evaluator = "openai"
reflector = "deepseek-chat"
```

Any unset role falls back to `--active-llm`.

### Single Task

```bash
python run_healthflow.py run \
  "Analyze the uploaded sales.csv and summarize the top 3 drivers of revenue decline." \
  --active-llm deepseek-chat \
  --active-executor healthflow_agent
```

The same CLI can also run EHR-focused prompts used in the paper, benchmark rebuilds, and arbitrary external-CLI-driven workflows.

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
python run_benchmark.py data/ehrflowbench/processed/eval.jsonl ehrflow_eval \
  --active-llm deepseek-reasoner \
  --active-executor healthflow_agent
```

Results are written under `benchmark_results/<dataset>/<executor>/<reasoning_model>/` with per-task copies of the workspace artifacts and dataset-level summary JSON.

## Benchmark Framing

- **EHRFlowBench** is a paper-derived **proxy benchmark**. The canonical source of truth is the local task prompt plus `processed/expected/<qid>/`, not the original paper metric table.
- `data/ehrflowbench/processed/paper_map.csv` records provenance, proxy linkage mode, source-task eligibility, and review status for every canonical task.
- **MedAgentBoard** is a deterministic workflow benchmark grounded entirely in the committed TJH and MIMIC demo data under `data/medagentboard/`.

## Configuration

Main config sections:

- `[llm.*]`: reasoning model providers
- `[llm_roles]`: optional planner/evaluator/reflector model overrides
- `[executor]`: default backend and CLI backend definitions
- `[memory]`: retrieval budgets and memory mode
- `[ehr]`: optional EHR-overlay profiling and risk-check controls
- `[verification]`: deterministic success gating
- `[evaluation]`: evaluator success threshold
- `[system]`: workspace, shell, and task-attempt settings (`max_attempts`)
- `[logging]`: log level and log file

## Repository Layout

- `run_healthflow.py`: single-task and interactive CLI
- `run_training.py`: memory bootstrapping and training-style runs
- `run_benchmark.py`: reproducible benchmark runner with frozen memory default
- `healthflow/system.py`: orchestration loop
- `healthflow/execution/`: executor layer
- `healthflow/ehr/`: task-family profiling, domain overlays, and EHR-specific risk logic
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
