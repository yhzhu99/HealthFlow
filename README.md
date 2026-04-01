# HealthFlow: A Self-Evolving MERF Runtime for CodeAct Analysis

[![arXiv](https://img.shields.io/badge/arXiv-2508.02621-b31b1b.svg)](https://arxiv.org/abs/2508.02621)
[![Project Website](https://img.shields.io/badge/Project%20Website-HealthFlow-0066cc.svg)](https://healthflow-agent.netlify.app)

HealthFlow is a research framework for **self-evolving task execution with a four-stage Meta -> Executor -> Evaluator -> Reflector loop**. The core runtime is organized around planning, CodeAct-style execution, structured evaluation, per-task runtime artifacts, and long-term reflective memory. Dataset preparation and benchmark evaluation workflows can still live in the repository under `data/`, but they are intentionally decoupled from the `healthflow/` runtime package.

- structured `Meta` planning with EHR-adaptive memory retrieval
- `Executor` as a CodeAct runtime over external executor backends
- `Evaluator`-driven retry and failure diagnosis
- `Reflector` writeback from both successful and failed trajectories
- inspectable workspace artifacts and run telemetry

HealthFlow compares external coding agents through a shared executor abstraction. The maintained built-in backends are `claude_code`, `codex`, `opencode`, and `pi`, with `opencode` as the default.

The current release surface is intentionally **backend and CLI only**. A frontend is not shipped in this repo at this stage.

## Core Runtime

HealthFlow runs a lean **Meta -> Executor -> Evaluator -> Reflector** loop.

1. **Meta**: retrieve relevant safeguard, workflow, dataset, and execution memories, then emit a structured execution plan.
2. **Executor**: interpret the plan as a CodeAct brief and act through code, commands, and workspace artifacts using whatever tools are already configured in the outer executor.
3. **Evaluator**: review the execution trace and produced artifacts, classify the outcome as `success`, `needs_retry`, or `failed`, and provide repair instructions for the next attempt.
4. **Reflector**: synthesize reusable safeguard, workflow, dataset, or execution memories from the full trajectory after the task session ends.

The task-level self-correction budget is controlled by `system.max_attempts`, which counts total full attempts through the loop rather than "retries plus one".

## What HealthFlow Contributes

- **MERF core runtime**: the framework definition is the four-stage Meta, Executor, Evaluator, Reflector loop rather than an outer benchmark-evaluation pipeline.
- **Lean execution contract**: HealthFlow defines workspace rules, execution-environment defaults, and workflow recommendations without becoming a tool-hosting framework.
- **Inspectable memory**: safeguard, workflow, dataset, and execution memories are stored in JSONL, routed through adaptive retrieval lanes, and exposed through a saved retrieval audit.
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

When `healthflow run ... --report` is enabled, the same workspace also writes:

- `report.md`

These files are the main source of truth for rebuttal-oriented inspection.
`report.md` is a standard HealthFlow-generated markdown report that summarizes the run, links produced artifacts with relative paths, embeds a small number of images inline, and keeps runtime JSON/log files in a separate audit section.

## Runtime Boundary

- **Core runtime**: the MERF loop in `healthflow/system.py`.
- **Domain specialization**: EHR-specific helpers under `healthflow/ehr/`.
- **Dataset prep and benchmark evaluation**: repository-level workflows under `data/`, intentionally decoupled from `healthflow/`.

The framework package is focused on taking a task, executing it, improving task success rate across attempts, and writing inspectable artifacts and reports for each task run.

## Memory Behavior

HealthFlow uses four memory classes:

- `safeguard`
- `workflow`
- `dataset`
- `execution`

Retrieval is inspectable:

- retrieval is conditioned on task family, dataset signature, schema tags, and EHR risk tags
- safeguard memories are prioritized for elevated-risk EHR tasks
- contradictory memories are tracked by `conflict_slot`
- safeguard memories suppress conflicting workflow or execution memories before planning
- dataset memories act as anchors without replacing workflow guidance
- the retrieval audit is saved to `memory_context.json`

Writeback behavior:

- failed runs and near-miss recoveries can produce `safeguard` memory
- successful reusable procedures can produce `workflow` memory
- stable schema observations can produce `dataset` memory
- reusable task-completion habits can produce `execution` memory

## Supported Execution Backends

HealthFlow keeps the executor layer backend-agnostic, but the public surface is intentionally small:

- `opencode` (default)
- `claude_code`
- `codex`
- `pi`

You can still define additional CLI backends in `config.toml`, but the harness logic stays in HealthFlow rather than being baked into one external backend.
Executor-specific repository instruction files are intentionally avoided at the repo root so backend comparisons use the same injected prompt guidance.

## External CLI Workflows

HealthFlow does not implement an internal MCP registry, plugin framework, or large CLI catalog. Tool availability belongs to the outer executor layer such as Claude Code, OpenCode, Pi, or Codex.

HealthFlow only supplies:

- a lightweight execution-environment contract
- small workflow recommendations
- documentation recipes for selected external CLIs

When external CLIs are part of the supported workflow, prefer declaring them in this project's `pyproject.toml` and installing them into the shared repo `.venv`. Executor backends should use that same project environment rather than ad hoc global tool installs.

Executor defaults are configured for normal text output. HealthFlow does not require external backends to finish in JSON. Structured event streams remain optional backend-specific telemetry modes.

`run_benchmark.py` always forces `memory.write_policy = "freeze"` so benchmark evaluation remains decoupled from the framework's self-evolving writeback behavior.

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
export DEEPSEEK_API_KEY="your_deepseek_key_here"
```

The repo already ships a ready-to-edit [`config.toml`](/home/yhzhu/projects/HealthFlow/config.toml). Update that file with the reasoning models you want to expose to HealthFlow. If you prefer to write your own from scratch, use the same shape and keep secrets in `api_key_env`:

```toml
[llm."deepseek/deepseek-v3.2"]
api_key_env = "DEEPSEEK_API_KEY"
base_url = "https://api.deepseek.com"
model_name = "deepseek-chat"
executor_model_name = "deepseek/deepseek-chat"
input_cost_per_million_tokens = 0.28
output_cost_per_million_tokens = 0.43

[llm."deepseek/deepseek-reasoner"]
api_key_env = "DEEPSEEK_API_KEY"
base_url = "https://api.deepseek.com"
model_name = "deepseek-reasoner"
executor_model_name = "deepseek/deepseek-reasoner"

[llm."openai/gpt-5.4"]
api_key_env = "ZENMUX_API_KEY"
base_url = "https://zenmux.ai/api/v1"
model_name = "openai/gpt-5.4"
input_cost_per_million_tokens = 2.50
output_cost_per_million_tokens = 15.00

[llm."google/gemini-3-flash-preview"]
api_key_env = "ZENMUX_API_KEY"
base_url = "https://zenmux.ai/api/v1"
model_name = "google/gemini-3-flash-preview"
input_cost_per_million_tokens = 0.50
output_cost_per_million_tokens = 3.00
```

`api_key` still works for inline secrets, but `api_key_env` is the recommended path. Use quoted TOML table names for model keys that contain `/`.

If you want estimated LLM cost summaries in run artifacts, set `input_cost_per_million_tokens` and `output_cost_per_million_tokens` for the active reasoning model in `config.toml`. If those fields are omitted, HealthFlow skips cost estimation for that model. `opencode` executor runs also record per-step executor token usage and estimated executor cost when the CLI returns structured telemetry.

By default, the active executor inherits the same `model_name` as the selected `--active-llm`, except for `codex`, which is pinned to `openai/gpt-5.4` in the repo defaults because that is the only Codex model/provider path currently verified in this setup. Override the executor-side model only if you explicitly want the planner/evaluator model and the backend model to diverge for an experiment.

The built-in executor defaults also enable reasoning-oriented modes out of the box:
- `opencode`: `--variant high --thinking`
- `codex`: `model_reasoning_effort="high"` and `model_reasoning_summary="detailed"`
- `pi`: `--thinking high`
- `claude_code`: `--effort high`

These are still ordinary backend settings in `config.toml`, so you can override them per executor for large experiment sweeps.

Example executor configuration with ZenMux-backed defaults:

```toml
[executor.backends.opencode]
binary = "opencode"
args = ["run", "--variant", "high", "--thinking"]
model_flag = "-m"
model_template = "$provider/$model"
provider = "zenmux"

[executor.backends.codex]
binary = "codex"
args = ["exec", "--skip-git-repo-check", "--color", "never", "--dangerously-bypass-approvals-and-sandbox"]
arg_templates = ["-c", "model_provider=\"$provider\"", "-c", "model_providers.$provider={name=\"ZenMux\", base_url=\"$provider_base_url\", env_key=\"$provider_api_key_env\", wire_api=\"responses\"}", "-c", "model_reasoning_effort=\"high\"", "-c", "model_reasoning_summary=\"detailed\""]
model = "openai/gpt-5.4"
model_flag = "-m"
inherit_active_llm = false
provider = "zenmux"
provider_base_url = "https://zenmux.ai/api/v1"
provider_api_key_env = "ZENMUX_API_KEY"

[executor.backends.pi]
binary = "pi"
args = ["--print", "--thinking", "high"]
provider_flag = "--provider"
model_flag = "--model"
provider = "zenmux"
provider_base_url = "https://zenmux.ai/api/v1"
provider_api = "openai-completions"
provider_api_key_env = "ZENMUX_API_KEY"

[executor.backends.claude_code]
binary = "claude"
args = ["--bare", "--setting-sources", "local", "--dangerously-skip-permissions", "--print", "--output-format", "text", "--effort", "high"]
env = { ANTHROPIC_BASE_URL = "https://zenmux.ai/api/anthropic", ANTHROPIC_API_KEY = "${ZENMUX_API_KEY}", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1" }
model_flag = "--model"
```

HealthFlow also exposes a small execution-environment contract:

```toml
[environment]
python_version = "3.12"
package_manager = "uv"
install_command = "uv add"
run_prefix = "uv run"
```

To decouple the internal roles, set:

```toml
[llm_roles]
planner = "deepseek/deepseek-v3.2"
evaluator = "openai/gpt-5.4"
reflector = "google/gemini-3-flash-preview"
```

Any model named in `[llm_roles]` must also be declared under `[llm]`.

Any unset role falls back to `--active-llm`.

Legacy tool-registration sections are intentionally unsupported. If you previously configured CLI or MCP tools inside HealthFlow, move that setup into the outer executor and keep only the environment defaults above in HealthFlow.

### External CLI Recipes

HealthFlow may recommend selected external CLIs when the task warrants them, but it does not install, register, or invoke them directly.

ToolUniverse CLI examples:

```bash
uv run tu list
uv run tu find "pathway analysis"
uv run tu info <tool-name>
uv run tu run <tool-name> --help
```

ToolUniverse also supports a local `.tooluniverse/profile.yaml` workspace and can launch its own MCP server with `tu serve`, but HealthFlow does not manage that MCP surface.

OneEHR CLI examples:

```bash
uv run oneehr preprocess --help
uv run oneehr train --help
uv run oneehr test --help
uv run oneehr analyze --help
uv run oneehr plot --help
uv run oneehr convert --help
```

The OneEHR workflow is only surfaced as a recommendation for EHR modeling tasks, and only when the executor environment already provides the CLI.

### Single Task

```bash
python run_healthflow.py run \
  "Analyze the uploaded sales.csv and summarize the top 3 drivers of revenue decline." \
  --active-llm 'deepseek/deepseek-v3.2' \
  --active-executor opencode \
  --report
```

The same CLI can also run EHR-focused prompts used in the paper and arbitrary external-CLI-driven workflows.
When `--report` is enabled, HealthFlow writes `workspace/tasks/<task_id>/report.md` after the run finishes, even for failed runs, so a reviewer can inspect the task outcome from a single markdown artifact before exporting it to PDF or other formats.

### Interactive Mode

```bash
python run_healthflow.py interactive \
  --active-llm 'deepseek/deepseek-v3.2' \
  --active-executor opencode
```

Interactive mode now supports a command-aware shell:

- `/help`: show commands and keyboard hints
- `/clear`: clear the terminal and redraw the session banner
- `/new`: start a fresh local session while preserving `workspace/memory/experience.jsonl`
- `/exit`: exit interactive mode
- `exit` / `quit`: aliases for `/exit`
- Type `/` in column 1 to open slash-command suggestions
- `Tab`: complete slash commands
- `ESC ESC`: cancel the current run without leaving the shell

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
- `[environment]`: lightweight runtime defaults such as preferred Python version and `uv` command prefixes
- `[memory]`: runtime write policy only (`append`, `freeze`, or `reset_before_run`)
- `[evaluation]`: evaluator success threshold
- `[system]`: workspace and task-attempt settings (`workspace_dir`, `max_attempts`)
- `[logging]`: log level and log file

By default, `[system].workspace_dir` points to `workspace/tasks`, while CLI entrypoints use `workspace/memory/experience.jsonl` for shared long-term memory unless overridden.

## Repository Layout

- `run_healthflow.py`: single-task and interactive CLI
- `run_training.py`: dataset-style batch runner over task JSONL files
- `run_benchmark.py`: batch task runner over task JSONL files
- `healthflow/system.py`: orchestration loop
- `healthflow/execution/`: executor layer
- `healthflow/ehr/`: optional EHR specialization helpers kept outside the core loop
- `healthflow/experience/`: EHR-adaptive memory and retrieval audit

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
