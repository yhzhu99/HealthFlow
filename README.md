# HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research

[![arXiv](https://img.shields.io/badge/arXiv-2508.02621-b31b1b.svg)](https://arxiv.org/abs/2508.02621)
[![Project Website](https://img.shields.io/badge/Project%20Website-HealthFlow-0066cc.svg)](https://healthflow-agent.netlify.app)

**[📜 Read our ArXiv Paper](https://arxiv.org/abs/2508.02621)**

> **Authors:** Yinghao Zhu¹²*, Yifan Qi¹*, Zixiang Wang¹, Lei Gu¹, Dehao Sui¹, Haoran Hu¹, Xichen Zhang², Ziyi He², Liantao Ma¹†, Lequan Yu²†
>
> ¹Peking University, ²The University of Hong Kong
>
> *(\* Equal contribution, † Corresponding authors)*

---

HealthFlow is a research framework designed to orchestrate, evaluate, and learn from powerful, external agentic coders to solve complex healthcare research tasks. Its core innovation lies not in building a coding agent itself, but in creating a **self-evolving meta-system** that learns to become a better strategic planner.

The system treats every task as a scientific experiment, autonomously refining its own high-level problem-solving policies by distilling successes and failures into a durable, strategic knowledge base. This marks a shift from building better *tool-users* to designing smarter, self-evolving *task-managers*, paving the way for more autonomous and effective AI for scientific discovery.

## ✨ Core Features

-   **Meta-Level Evolution**: Goes beyond simple tool use by synthesizing successful task executions into a durable strategic knowledge base (`experience.jsonl`), allowing it to improve its high-level planning over time.
-   **Modular Multi-Agent System**: A robust architecture of specialized agents for Planning (`MetaAgent`), Execution (`ClaudeCodeExecutor`), Evaluation (`EvaluatorAgent`), and Reflection (`ReflectorAgent`).
-   **Knowledge Bootstrapping**: A `train_mode` to build an initial, high-quality experience base from curated problems with reference answers, addressing the "cold start" problem.
-   **Unified Workflow**: A consistent and powerful `Plan -> Execute -> Evaluate -> Reflect` cycle that handles all tasks, from simple questions to complex, multi-step data analysis.

## 🚀 How It Works: The Self-Evolving Loop

![HealthFlow Workflow](assets/healthflow_workflow.png)
*Figure: The self-evolving workflow of HealthFlow, which treats every task as a learning opportunity. The cycle consists of four key stages: **Plan**, **Execute**, **Evaluate**, and **Reflect**, with successful experiences synthesized and saved to a durable knowledge base to **Evolve** the agent's future planning capabilities.*

HealthFlow's novelty lies in its unified and automated **Plan -> Execute -> Evaluate -> Reflect -> Evolve** cycle. It treats every task as a learning opportunity, enabling it to continuously improve its own strategic capabilities.

1.  **Plan (MetaAgent)**: A user's request is analyzed by the `MetaAgent`. It queries the `ExperienceManager` for relevant past experiences and synthesizes them into a detailed, step-by-step markdown plan (`task_list.md`). This plan is context-aware, incorporating learned heuristics and warnings.

2.  **Execute (ClaudeCodeExecutor)**: The system delegates the execution of the plan to a powerful, external agentic coder (e.g., `claude`). It captures the entire terminal output, including commands, standard output, and errors, for analysis.

3.  **Evaluate (EvaluatorAgent)**: The `EvaluatorAgent` assesses the execution outcome against the original request and plan. It provides a quantitative score and qualitative feedback. If the task fails or quality is low, this feedback is used to generate a better plan in the next attempt.

4.  **Reflect (ReflectorAgent)**: Upon *successful* completion of any task, the `ReflectorAgent` analyzes the entire interaction (request, plan, logs, evaluation) to synthesize generalizable knowledge into structured **Experience Objects** (e.g., a `heuristic`, a `warning`, a `code_snippet`).

5.  **Evolve (ExperienceManager)**: These structured experiences are saved to a persistent `experience.jsonl` file. This growing knowledge base is used by the `MetaAgent` during future planning, enabling it to make smarter decisions and create better plans, thus closing the self-improvement loop.

## 🏁 Quick Start

### 1. Prerequisites

-   Python 3.12+
-   `uv` (a fast Python package installer and resolver)
-   [Anthropic's `claude` CLI](https://docs.anthropic.com/claude/docs/claude-code) installed and available in your `PATH`. This is the default execution agent.

### 2. Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/healthflow.git
cd healthflow

# 2. Install dependencies using uv
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Create your configuration file from the example
cp config.toml.example config.toml
```

**Next, edit `config.toml`** to add API keys for the LLMs you intend to use for *reasoning* (planning, evaluating, reflecting). You can configure multiple providers.

## 💻 Usage

HealthFlow is controlled via a powerful command-line interface. You must always specify which reasoning LLM to use with the `--active-llm` flag.

### Running a Single Task

To execute a single, specific task and then exit.

```bash
python run_healthflow.py run "Analyze the provided 'patients.csv' to identify the top 3 risk factors for readmission. Anonymize any patient identifiers in the output." --active-llm deepseek-v3
```

### Interactive Mode

For a chat-like session where you can run multiple tasks sequentially.

```bash
python run_healthflow.py interactive --active-llm deepseek-v3
```

### Training (Knowledge Bootstrapping)

Use this mode to populate the experience memory from a curated dataset with reference answers. This is key to bootstrapping the agent's strategic knowledge.

The training data should be a `.jsonl` file where each line is a JSON object with `qid`, `task`, and `answer` keys.

```bash
# Format: python run_training.py <training_file> <dataset_name> --active-llm <llm>
python run_training.py data/train_set.jsonl ehrflow_train --active-llm deepseek-r1
```

This will run each task, use the reference answer for evaluation, and save learned experiences to `workspace/experience.jsonl`. Detailed logs are saved to `benchmark_results/`.

### Benchmarking

Evaluate HealthFlow's performance on a benchmark dataset. The dataset format is the same as for training.

```bash
# Format: python run_benchmark.py <dataset_file> <dataset_name> --active-llm <llm>
python run_benchmark.py data/benchmark_set.jsonl ehrflow_eval --active-llm deepseek-r1
```

Results, including logs for each task and a final summary, will be saved in the `benchmark_results/` directory.

## 🏗️ Architecture

The project is designed to be modular and minimalist, serving as a clean research platform.

-   **`run_healthflow.py`, `run_training.py`, `run_benchmark.py`**: CLI entrypoints for different modes of operation.
-   **`healthflow/`**: The core library code.
    -   **`system.py`**: Contains `HealthFlowSystem`, the central orchestrator that manages the self-evolving workflow.
    -   **`agents/`**: LLM-powered agents for high-level reasoning (`MetaAgent`, `EvaluatorAgent`, `ReflectorAgent`).
    -   **`execution/`**: The `ClaudeCodeExecutor` wrapper for calling the external `claude` CLI tool.
    -   **`experience/`**: The heart of the self-evolution mechanism. `ExperienceManager` manages the `experience.jsonl` knowledge base, and `experience_models.py` defines its structure.
    -   **`prompts/`**: A centralized repository of prompt templates that guide the agents.
    -   **`core/`**: Core components like configuration loading (`config.py`) and the LLM provider wrapper (`llm_provider.py`).
-   **`workspace/`**: The default directory where all runtime artifacts are stored. Each task gets a unique subdirectory containing its plan, logs, and any generated files. The `experience.jsonl` file is also stored here.
-   **`benchmark_results/`**: The output directory for training and benchmarking runs, organized by dataset and model.
-   **`config.toml`**: The central configuration file for LLMs, system settings, and more.
-   **`pyproject.toml`**: Project metadata and dependencies, managed by `uv`.

## ⚙️ Configuration

All settings are managed in `config.toml`.

-   **`[llm.*]`**: Define connection details for different LLM providers (e.g., `[llm.deepseek-v3]`, `[llm.gemini]`). You must provide `base_url`, `api_key`, and `model_name`.
-   **`--active-llm <name>`**: This mandatory runtime flag tells HealthFlow which `[llm.*]` block from your `config.toml` to use for the reasoning agents.
-   **`[system]`**: Configure system-wide behavior like `max_retries` and the `workspace_dir`.
-   **`[evaluation]`**: Set the `success_threshold` score for a task to be considered successful.
-   **`[logging]`**: Control the log level and file path.

## 📜 Citation

If you use HealthFlow in your research, please cite our paper:

```bibtex
@misc{zhu2025healthflow,
      title={HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research}, 
      author={Yinghao Zhu and Yifan Qi and Zixiang Wang and Lei Gu and Dehao Sui and Haoran Hu and Xichen Zhang and Ziyi He and Liantao Ma and Lequan Yu},
      year={2025},
      eprint={2508.02621},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.02621}, 
}
```