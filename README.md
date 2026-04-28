# HealthFlow

[![Project Website](https://img.shields.io/badge/Project-HealthFlow-0066cc.svg)](https://healthflow.medx-pku.com/)
[![Online App](https://img.shields.io/badge/App-healthflow.medx--pku.com%2Fapp-0f766e.svg)](https://healthflow.medx-pku.com/app)
[![Datasets](https://img.shields.io/badge/Data-GitHub%20Releases-7c3aed.svg)](https://github.com/yhzhu99/HealthFlow/releases/tag/datasets)

HealthFlow is a strategically self-evolving multi-agent framework for automating electronic health record (EHR) analysis. It turns a clinical analysis request into a governed workflow that plans, executes, evaluates, repairs, and writes back reusable experience for later tasks.

The system is organized around four agents:

- **Meta agent**: profiles the task, retrieves bounded EHR-aware experience, and produces an attempt-specific plan.
- **Executor agent**: turns the plan into code, tool calls, and analytical artifacts through an external coding-agent backend.
- **Evaluator agent**: reviews execution traces and artifacts for runtime failures and EHR validity risks.
- **Reflector agent**: synthesizes reusable safeguards, workflows, dataset anchors, and code snippets from completed trajectories.

Useful links:

- Project website: <https://healthflow.medx-pku.com/>
- Public app: <https://healthflow.medx-pku.com/app>
- Human evaluation platform: <https://healthflow.medx-pku.com/evaluation>
- Code repository: <https://github.com/yhzhu99/HealthFlow>
- Dataset release: <https://github.com/yhzhu99/HealthFlow/releases/tag/datasets>

## Paper

**HealthFlow: Automating electronic health record analysis via a strategically self-evolving multi-agent framework**

Authors:
Yinghao Zhu, Zixiang Wang, Yifan Qi, Lei Gu, Dehao Sui, Haoran Hu, Xichen Zhang, Ziyi He, Yasha Wang, Junjun He, Liantao Ma, and Lequan Yu.

Affiliations:

1. National Engineering Research Center for Software Engineering, Peking University, Beijing, China, 100871
2. School of Computing and Data Science, The University of Hong Kong, Hong Kong SAR, China, 999077
3. Department of Computer Science and Engineering, The Hong Kong University of Science and Technology, Hong Kong SAR, China, 999077
4. Shanghai Artificial Intelligence Laboratory, Shanghai, China, 200232

Equal contribution: Yinghao Zhu, Zixiang Wang, and Yifan Qi.

Correspondence: `malt@pku.edu.cn`, `lqyu@hku.hk`.

The paper introduces HealthFlow and EHRFlowBench, a benchmark of realistic EHR analysis tasks derived from 51,280 peer-reviewed papers. Across EHRFlowBench and four established benchmarks, HealthFlow improves full-cycle EHR workflow completion by reusing prior analytical experience under dataset-specific and methodological constraints.

## Data Availability

The benchmark data are available in the GitHub Releases Zone:

- Release page: <https://github.com/yhzhu99/HealthFlow/releases/tag/datasets>
- Direct asset: <https://github.com/yhzhu99/HealthFlow/releases/download/datasets/healthflow_datasets.zip>

The release contains the curated benchmark assets for:

- EHRFlowBench
- MedAgentBoard
- MedAgentsBench
- HLE
- CureBench

Download and extract the release archive from the repository root:

```bash
curl -L -o healthflow_datasets.zip \
  https://github.com/yhzhu99/HealthFlow/releases/download/datasets/healthflow_datasets.zip

unzip -q healthflow_datasets.zip -d data -x "__MACOSX/*" "*/.DS_Store"
```

Dataset `raw/` and `processed/` directories are intentionally ignored by git. The committed `data/*/README.md` files and scripts document how each benchmark subset is prepared and evaluated.

Original public data sources used by the paper include the MIMIC-IV Public Demo Release on PhysioNet, the TJH dataset from `HAIRLAB/Pre_Surv_COVID_19`, MedAgentsBench, Humanity's Last Exam, and CureBench. Follow the usage terms of each upstream provider.

## Quick Start

Prerequisites:

- Python 3.12+
- `uv`
- one executor backend available in `PATH`
  - default: `opencode`
  - alternatives: `claude`, `codex`, `pi`

Install dependencies:

```bash
uv sync
source .venv/bin/activate
```

Set the API keys required by the default `config.toml`:

```bash
export ZENMUX_API_KEY="your_zenmux_key_here"
export DEEPSEEK_API_KEY="your_deepseek_key_here"
```

Run a one-shot task:

```bash
uv run healthflow run \
  "Analyze the uploaded EHR data and write a concise report with methods, results, and limitations." \
  --active-executor opencode \
  --report
```

HealthFlow writes each task workspace under `workspace/tasks/<task_id>/`. When `--report` is enabled, the final markdown report is written to `workspace/tasks/<task_id>/runtime/report.md`.

## Run HealthFlow

HealthFlow exposes three user-facing modes.

Non-interactive CLI:

```bash
uv run healthflow run "Your analysis task here" --active-executor opencode --report
```

Interactive terminal:

```bash
uv run healthflow interactive --active-executor opencode
```

Browser UI:

```bash
uv sync --extra web
uv run healthflow web --server-port 7860
```

The same entrypoints can also be run directly:

```bash
python run_healthflow.py run "Your analysis task here" --active-executor opencode --report
python run_healthflow.py interactive --active-executor opencode
python run_healthflow.py web --server-port 7860
```

Common runtime overrides:

- `--config`: path to `config.toml`
- `--experience-path`: path to the long-term memory JSONL file
- `--planner-llm`: override `runtime.planner_llm`
- `--evaluator-llm`: override `runtime.evaluator_llm`
- `--reflector-llm`: override `runtime.reflector_llm`
- `--executor-llm`: override `runtime.executor_llm`
- `--active-executor`: executor backend, such as `opencode`, `claude_code`, `codex`, or `pi`

## Benchmarks and Evaluation

HealthFlow was evaluated on five benchmark suites:

- **EHRFlowBench**: 100 open-ended EHR analysis tasks derived from peer-reviewed literature and instantiated on TJH and the MIMIC-IV Public Demo Release.
- **MedAgentBoard**: 100 executable workflow tasks spanning data extraction, predictive modeling, and visualization.
- **MedAgentsBench**: deterministic multiple-choice medical reasoning evaluation.
- **HLE**: selected Biology and Medicine questions from Humanity's Last Exam.
- **CureBench**: deterministic biomedical decision-making evaluation.

Run HealthFlow over a JSONL benchmark file:

```bash
python run_benchmark.py data/ehrflowbench/processed/test.jsonl ehrflowbench \
  --active-executor opencode
```

Results are written under:

```text
benchmark_results/<dataset>/<executor>/<runtime_selection>/
```

The benchmark runner freezes memory writeback during batch evaluation so benchmark execution remains separate from online self-evolving memory updates.

## Repository Layout

```text
healthflow/                 Core runtime package
  agents/                   Meta, evaluator, and reflector support
  core/                     Configuration, LLM provider, and direct responses
  ehr/                      EHR task profiling and domain helpers
  execution/                Executor backend adapters
  experience/               Governed experience memory and retrieval
  prompts/                  Prompt templates
data/                       Benchmark preparation and evaluation scripts
platform/                   Public website and blinded evaluation frontend
run_healthflow.py           CLI, interactive shell, and web app entrypoint
run_benchmark.py            Batch benchmark runner
run_training.py             Training-style batch runner
config.toml                 Default model, executor, memory, and runtime config
```

## Runtime Architecture

HealthFlow runs a four-stage loop for each task attempt:

1. **Planning**: the meta agent builds an EHR-aware task context, retrieves compatible memory, and emits a structured execution plan.
2. **Execution**: the executor backend implements the plan in a sandboxed workspace using code, shell commands, and available external tools.
3. **Evaluation**: the evaluator checks artifacts, traces, and methodological validity, then returns a verdict and targeted repair guidance.
4. **Reflection**: after task termination, the reflector updates long-term memory with validated safeguards, workflows, dataset anchors, and code snippets.

Long-term memory lives at `workspace/memory/experience.jsonl` by default. Retrieval is conditioned on task family, dataset signature, schema tags, and EHR risk tags. Dataset anchors require exact dataset match, and safeguards require risk-tag overlap.

Key runtime artifacts:

```text
workspace/healthflow.log
workspace/memory/experience.jsonl
workspace/tasks/<task_id>/sandbox/
workspace/tasks/<task_id>/runtime/index.json
workspace/tasks/<task_id>/runtime/events.jsonl
workspace/tasks/<task_id>/runtime/run/summary.json
workspace/tasks/<task_id>/runtime/run/trajectory.json
workspace/tasks/<task_id>/runtime/run/costs.json
workspace/tasks/<task_id>/runtime/run/final_evaluation.json
workspace/tasks/<task_id>/runtime/report.md
```

## Configuration

The default `config.toml` defines:

- `[llm.*]`: model registry entries and API-key environment variables
- `[runtime]`: planner, evaluator, reflector, and executor model choices
- `[executor]`: active backend and backend-specific CLI settings
- `[environment]`: lightweight execution environment defaults
- `[system]`: workspace directory and maximum attempts
- `[memory]`: write policy, such as `append`, `freeze`, or `reset_before_run`
- `[evaluation]`: evaluator success threshold
- `[logging]`: log level and log file

The default configuration uses DeepSeek for planning/execution, GPT-5.4 for evaluation, Gemini 3 Flash Preview for reflection, and OpenCode as the executor backend. You can swap these by editing `config.toml` or by passing the CLI override flags listed above.

External tools such as OneEHR and ToolUniverse are installed through the project environment and surfaced to the executor through HealthFlow's execution contract. HealthFlow does not host its own MCP registry or plugin catalog.

## Citation

Publication details are pending. For now, cite the manuscript as:

```bibtex
@misc{zhu2026healthflow,
  title = {HealthFlow: Automating electronic health record analysis via a strategically self-evolving multi-agent framework},
  author = {Zhu, Yinghao and Wang, Zixiang and Qi, Yifan and Gu, Lei and Sui, Dehao and Hu, Haoran and Zhang, Xichen and He, Ziyi and Wang, Yasha and He, Junjun and Ma, Liantao and Yu, Lequan},
  year = {2026},
  note = {Manuscript}
}
```
