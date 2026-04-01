# HealthFlow: A Self-Evolving MERF Runtime for CodeAct Analysis

[![arXiv](https://img.shields.io/badge/arXiv-2508.02621-b31b1b.svg)](https://arxiv.org/abs/2508.02621)
[![Project Website](https://img.shields.io/badge/Project%20Website-HealthFlow-0066cc.svg)](https://healthflow-agent.netlify.app)

HealthFlow is a research framework for **self-evolving task execution with a four-stage Meta -> Executor -> Evaluator -> Reflector loop**. The core runtime is organized around planning, CodeAct-style execution, structured evaluation, per-task reporting, and long-term reflective memory. Dataset preparation and benchmark evaluation workflows can still live in the repository under `data/`, but they are intentionally decoupled from the `healthflow/` runtime package.

- structured `Meta` planning with explicit positive and avoidance memory
- `Executor` as a CodeAct runtime over pluggable tool surfaces
- `Evaluator`-driven retry and failure diagnosis
- `Reflector` writeback from both successful and failed trajectories
- inspectable workspace artifacts and run telemetry

HealthFlow compares external coding agents through a shared executor abstraction. The maintained built-in backends are `claude_code`, `codex`, `opencode`, and `pi`, with `opencode` as the default.

The current release surface is intentionally **backend and CLI only**. A frontend is not shipped in this repo at this stage.

## Core Runtime

HealthFlow runs a lean **Meta -> Executor -> Evaluator -> Reflector** loop.

1. **Meta**: retrieve relevant memory, separate recommended experience from avoidance memory, and emit a structured execution plan.
2. **Executor**: interpret the plan as a CodeAct brief and act through code, commands, workspace artifacts, and configured tool surfaces.
3. **Evaluator**: review the execution trace and produced artifacts, classify the outcome as `success`, `needs_retry`, or `failed`, and provide repair instructions for the next attempt.
4. **Reflector**: synthesize reusable positive or negative memories from the full trajectory after the task session ends.

The task-level self-correction budget is controlled by `system.max_attempts`, which counts total full attempts through the loop rather than "retries plus one".

## What HealthFlow Contributes

- **MERF core runtime**: the framework definition is the four-stage Meta, Executor, Evaluator, Reflector loop rather than an outer profiling or verification pipeline.
- **CodeAct execution surface**: the executor is framed around explicit actions and can work over pluggable tool surfaces such as CLI or MCP-backed tools.
- **Inspectable memory**: strategy, failure, dataset, and artifact memories are stored in JSONL, separated into recommendation vs avoidance guidance at retrieval time, and exposed through a saved retrieval audit.
- **Evaluator-centered recovery**: retries are driven by structured failure diagnosis and repair instructions instead of a single scalar score alone.
- **Reproducibility contract**: every task workspace writes structured runtime artifacts instead of only human-readable logs.
- **Executor telemetry**: run artifacts capture executor metadata, backend versions when available, LLM usage, executor usage, and stage-level estimated cost summaries.
- **Role-specific internal models**: planner, evaluator, and reflector can be configured against different reasoning models to reduce single-model coupling.

## Workspace Artifacts

Runtime state lives under `workspace/` by default:

- task artifacts: `workspace/tasks/<task_id>/`
- long-term memory: `workspace/memory/experience.jsonl`

Dataset preparation and benchmark evaluation assets remain under `data/`; they are outside the `healthflow/` package boundary.

Each task creates a workspace under `workspace/tasks/<task_id>/` and writes:

- `<backend>_execution.log`
- `task_list_v*.md`
- `full_history.json`
- `memory_context.json`
- `evaluation.json`
- `cost_analysis.json`
- `run_manifest.json`
- `run_result.json`

These files are the main source of truth for rebuttal-oriented inspection.

## Runtime Boundary

- **Core runtime**: the MERF loop in `healthflow/system.py`.
- **Task-level reporting and verification**: per-task report generation plus artifact/report checks under `healthflow/reporting/` and `healthflow/verification/`.
- **Domain specialization**: EHR-specific helpers under `healthflow/ehr/`.
- **Dataset prep and benchmark evaluation**: repository-level workflows under `data/`, intentionally decoupled from `healthflow/`.

The framework package is focused on taking a task, executing it, improving task success rate across attempts, and writing inspectable artifacts and reports for each task run.

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
- retrieved memories are split into recommendation vs avoidance guidance before they reach the planner
- the retrieval audit is saved to `memory_context.json`

Writeback behavior:

- verified runs can produce `dataset`, `strategy`, and `artifact` memories
- failed runs can still produce `failure` / `warning` style memory for future avoidance
- both successful and failed runs can teach future tasks, unless the memory mode is frozen

## Supported Execution Backends

HealthFlow keeps the executor layer backend-agnostic, but the public surface is intentionally small:

- `opencode` (default)
- `claude_code`
- `codex`
- `pi`

You can still define additional CLI backends in `config.toml`, but the harness logic stays in HealthFlow rather than being baked into one external backend.
Executor-specific repository instruction files are intentionally avoided at the repo root so backend comparisons use the same injected prompt guidance.

## External CLI Workflows

HealthFlow does not hardcode compatibility for any specific domain package or external tool. Instead, it interacts with external systems through the same CodeAct-style execution layer used for all tasks.

The executor can advertise tool surfaces through a unified catalog. The planner may recommend preferred tools, while the executor is still allowed to adapt at runtime.

Executor defaults are configured for normal text output. HealthFlow does not require external backends to finish in JSON. Structured event streams remain optional backend-specific telemetry modes.

## Quick Start

### Prerequisites

- Python 3.12+
- `uv`
- one execution backend available in `PATH`
  - default: `opencode`
  - alternatives: `claude`, `codex`, `pi`

### Setup

```bash
uv sync
source .venv/bin/activate
export ZENMUX_API_KEY="your_zenmux_key_here"
```

Then create `config.toml` with the reasoning models you want to expose to HealthFlow. Use `api_key_env` to keep secrets out of the file:

```toml
[llm."deepseek/deepseek-v3.2"]
api_key_env = "ZENMUX_API_KEY"
base_url = "https://zenmux.ai/api/v1"
model_name = "deepseek/deepseek-v3.2"
input_cost_per_million_tokens = 0.28
output_cost_per_million_tokens = 0.43

[llm."google/gemini-3-flash-preview"]
api_key_env = "ZENMUX_API_KEY"
base_url = "https://zenmux.ai/api/v1"
model_name = "google/gemini-3-flash-preview"
input_cost_per_million_tokens = 0.50
output_cost_per_million_tokens = 3.00
```

`api_key` still works for inline secrets, but `api_key_env` is the recommended path. Use quoted TOML table names for model keys that contain `/`.

If you want estimated LLM cost summaries in run artifacts, set `input_cost_per_million_tokens` and `output_cost_per_million_tokens` for the active reasoning model in `config.toml`. If those fields are omitted, HealthFlow skips cost estimation for that model. `opencode` executor runs also record per-step executor token usage and estimated executor cost when the CLI returns structured telemetry.

By default, the active executor inherits the same `model_name` as the selected `--active-llm`. Override the executor-side model only if you explicitly want the planner/evaluator model and the backend model to diverge for an experiment.

Example executor configuration with ZenMux-backed defaults:

```toml
[executor.backends.opencode]
binary = "opencode"
args = ["run"]
model_flag = "-m"
model_template = "$provider/$model"
provider = "zenmux"

[executor.backends.codex]
binary = "codex"
args = ["exec", "--skip-git-repo-check", "--color", "never", "--dangerously-bypass-approvals-and-sandbox"]
arg_templates = ["-c", "model_provider=\"$provider\"", "-c", "model_providers.$provider={name=\"ZenMux\", base_url=\"$provider_base_url\", env_key=\"$provider_api_key_env\", wire_api=\"responses\"}"]
model_flag = "-m"
provider = "zenmux"
provider_base_url = "https://zenmux.ai/api/v1"
provider_api_key_env = "ZENMUX_API_KEY"

[executor.backends.pi]
binary = "pi"
args = ["--print"]
provider_flag = "--provider"
model_flag = "--model"
provider = "zenmux"
provider_base_url = "https://zenmux.ai/api/v1"
provider_api = "openai-completions"
provider_api_key_env = "ZENMUX_API_KEY"

[executor.backends.claude_code]
binary = "claude"
args = ["--bare", "--setting-sources", "local", "--dangerously-skip-permissions", "--print", "--output-format", "text"]
env = { ANTHROPIC_BASE_URL = "https://zenmux.ai/api/anthropic", ANTHROPIC_API_KEY = "${ZENMUX_API_KEY}", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1" }
model_flag = "--model"
```

To decouple the internal roles, set:

```toml
[llm_roles]
planner = "deepseek/deepseek-v3.2"
evaluator = "openai/gpt-5.2"
reflector = "google/gemini-3-flash-preview"
```

Any unset role falls back to `--active-llm`.

Optional tool surfaces can be declared in `config.toml`:

```toml
[tools.python]
surface = "cli"
description = "Run Python scripts in the workspace."
invocation_hint = "python <script>.py"

[tools.local_mcp]
surface = "mcp"
description = "Domain-specific MCP connector exposed to the executor."
invocation_hint = "connector-defined"
```

### Single Task

```bash
python run_healthflow.py run \
  "Analyze the uploaded sales.csv and summarize the top 3 drivers of revenue decline." \
  --active-llm 'deepseek/deepseek-v3.2' \
  --active-executor opencode
```

The same CLI can also run EHR-focused prompts used in the paper and arbitrary external-CLI-driven workflows.

### Interactive Mode

```bash
python run_healthflow.py interactive \
  --active-llm 'deepseek/deepseek-v3.2' \
  --active-executor opencode
```

### Training

Training data must be JSONL with `qid`, `task`, and `answer`.

```bash
python run_training.py data/train_set.jsonl ehrflow_train \
  --active-llm 'deepseek/deepseek-v3.2' \
  --active-executor opencode
```

### Benchmarking

Benchmarking is just batch task execution over the same JSONL task shape used elsewhere in the runtime.
Dataset construction, benchmark-specific preparation, and benchmark-side evaluation are not part of the `healthflow/` package and should be handled under `data/` or other repo-level tooling.

```bash
python run_benchmark.py path/to/tasks.jsonl experiment_name \
  --active-llm 'deepseek/deepseek-v3.2' \
  --active-executor opencode
```

Results are written under `benchmark_results/<dataset>/<executor>/<reasoning_model>/` with per-task copies of the workspace artifacts and dataset-level summary JSON.

For a minimal executor smoke test, use [executor_smoke.jsonl](/home/yhzhu/projects/HealthFlow/data/examples/processed/executor_smoke.jsonl) with any built-in backend.

## Benchmark Framing

- **EHRFlowBench** is a paper-derived **proxy benchmark**. The canonical source of truth is the locally rebuilt task prompt plus `processed/expected/<qid>/`, not the original paper metric table.
- `data/ehrflowbench/processed/paper_map.csv` is a local rebuild artifact that records provenance, proxy linkage mode, source-task eligibility, and review status for every canonical task.
- **MedAgentBoard** is a deterministic workflow benchmark grounded in local TJH and MIMIC demo data prepared under `data/medagentboard/`.

## Configuration

Main config sections:

- `[llm.*]`: reasoning model providers, with either `api_key` or `api_key_env`
- `[llm_roles]`: optional planner/evaluator/reflector model overrides
- `[executor]`: default backend and CLI backend definitions
- `[tools.*]`: optional tool catalog entries exposed to the planner and executor, with `surface = "cli"` or `surface = "mcp"`
- `[memory]`: runtime write policy only (`append`, `freeze`, or `reset_before_run`)
- `[evaluation]`: evaluator success threshold
- `[system]`: workspace, shell, and task-attempt settings (`max_attempts`)
- `[logging]`: log level and log file

By default, `[system].workspace_dir` points to `workspace/tasks`, while CLI entrypoints use `workspace/memory/experience.jsonl` for shared long-term memory unless overridden.

## Repository Layout

- `run_healthflow.py`: single-task and interactive CLI
- `run_training.py`: dataset-style batch runner over task JSONL files
- `run_benchmark.py`: batch task runner over task JSONL files
- `healthflow/system.py`: orchestration loop
- `healthflow/execution/`: executor layer
- `healthflow/ehr/`: optional EHR specialization helpers kept outside the core loop
- `healthflow/verification/`: task-level artifact and report verification helpers
- `healthflow/reporting/`: post-run report generation helpers
- `healthflow/experience/`: hierarchical memory and retrieval audit
- `healthflow/tools/`: tool catalog and optional tool-surface metadata

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
