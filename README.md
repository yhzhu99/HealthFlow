# HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research

[![arXiv](https://img.shields.io/badge/arXiv-2508.02621-b31b1b.svg)](https://arxiv.org/abs/2508.02621)
[![Project Website](https://img.shields.io/badge/Project%20Website-HealthFlow-0066cc.svg)](https://healthflow-agent.netlify.app)

**[📜 Read our ArXiv Paper](https://arxiv.org/abs/2508.02621)**

> **Authors:** Yinghao Zhu¹²*, Yifan Qi¹*, Zixiang Wang¹, Lei Gu¹, Dehao Sui¹, Haoran Hu¹, Xichen Zhang³, Ziyi He², Junjun He⁴, Liantao Ma¹†, Lequan Yu²†
>
> ¹Peking University, ²The University of Hong Kong, ³The Hong Kong University of Science and Technology, ⁴Shanghai Artificial Intelligence Laboratory
>
> *(\* Equal contribution, † Corresponding authors)*

---

HealthFlow is a research framework for **EHR-focused analysis orchestration**. Instead of trying to replace strong coding agents, it wraps them with an EHR-aware harness that profiles uploaded data, checks for leakage risks, retrieves validated memory, enforces deterministic verification, and writes back reusable strategy and failure memories.

The current runtime is **backend-agnostic** at the executor layer. Out of the box it can target `claude`, `opencode`, or `pi`, while keeping the planning, verification, and memory logic shared across backends.

## ✨ Core Features

-   **Backend-Agnostic Execution**: Switch between `claude`, `opencode`, and `pi` without changing the higher-level HealthFlow workflow.
-   **EHR-Aware Harness**: Task-family classification, schema profiling, leakage checks, staged tool exposure, and report contracts tuned for healthcare data science.
-   **Hierarchical Memory**: Dataset, strategy, failure, and artifact memories stored in one JSONL knowledge base with retrieval budgets and conflict handling.
-   **Verifier Gating**: Deterministic workspace checks run before a task is marked successful.
-   **Web-Based Interface**: A Streamlit UI for selecting the reasoning LLM, executor backend, and uploaded data files.

## 🚀 How It Works: The Self-Evolving Loop

![HealthFlow Workflow](assets/healthflow_workflow.png)
*Figure: The self-evolving workflow of HealthFlow, which treats every task as a learning opportunity. The cycle consists of four key stages: **Plan**, **Execute**, **Evaluate**, and **Reflect**, with successful experiences synthesized and saved to a durable knowledge base to **Evolve** the agent's future planning capabilities.*

HealthFlow's runtime centers on a lean **Profile -> Plan -> Execute -> Verify -> Reflect -> Evolve** loop.

1.  **Profile**: HealthFlow inspects uploaded inputs, classifies the task family, and generates EHR-specific risk checks.
2.  **Plan (MetaAgent)**: The planner retrieves relevant memories and writes a concrete execution plan with explicit artifacts.
3.  **Execute (Executor Adapter)**: The plan is handed to the selected backend CLI while HealthFlow captures logs and artifacts.
4.  **Verify + Evaluate**: Deterministic workspace checks run before LLM-based evaluation.
5.  **Reflect + Evolve**: Verified successes and failures are distilled into hierarchical memory for future runs.

## 🏁 Quick Start

### 1. Prerequisites

-   Python 3.12+
-   `uv` (a fast Python package installer and resolver)
-   At least one supported coding CLI installed and available in your `PATH`: `claude`, `opencode`, or `pi`.

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

### 3. Start the WebUI

For a user-friendly web interface, you can launch the Streamlit-based WebUI:

```bash
streamlit run app.py
```

This will start the web application, which you can access in your browser at the provided URL (typically `http://localhost:8501`).

### 4. CLI Usage

If you prefer the command-line interface, HealthFlow requires both a reasoning model (`--active-llm`) and optionally an executor backend (`--active-executor`).

#### Running a Single Task

To execute a single, specific task and then exit.

```bash
python run_healthflow.py run \
  "Analyze the provided patients.csv to identify the top 3 risk factors for readmission." \
  --active-llm deepseek-chat \
  --active-executor claude_code
```

#### Interactive Mode

For a chat-like session where you can run multiple tasks sequentially.

```bash
python run_healthflow.py interactive --active-llm deepseek-chat --active-executor opencode
```

#### Training (Knowledge Bootstrapping)

Use this mode to populate the experience memory from a curated dataset with reference answers. This is key to bootstrapping the agent's strategic knowledge.

The training data should be a `.jsonl` file where each line is a JSON object with `qid`, `task`, and `answer` keys.

```bash
# Format: python run_training.py <training_file> <dataset_name> --active-llm <llm>
python run_training.py data/train_set.jsonl ehrflow_train --active-llm deepseek-reasoner
```

This will run each task, use the reference answer for evaluation, and save learned experiences to `workspace/experience.jsonl`. Detailed logs are saved to `benchmark_results/`.

#### Benchmarking

Evaluate HealthFlow's performance on a benchmark dataset. The dataset format is the same as for training.

```bash
# Format: python run_benchmark.py <dataset_file> <dataset_name> --active-llm <llm>
python run_benchmark.py data/benchmark_set.jsonl ehrflow_eval --active-llm deepseek-reasoner
```

Results, including logs for each task and a final summary, will be saved in the `benchmark_results/` directory.

## 🏗️ Architecture

The project is designed to be modular and minimalist, serving as a clean research platform.

-   **`app.py`**: Streamlit-based WebUI entrypoint for a user-friendly browser interface.
-   **`run_healthflow.py`, `run_training.py`, `run_benchmark.py`**: CLI entrypoints for different modes of operation.
-   **`healthflow/`**: The core library code.
    -   **`system.py`**: The orchestrator tying together profiling, planning, execution, verification, and memory.
    -   **`agents/`**: LLM-powered planner, evaluator, and reflector agents.
    -   **`execution/`**: Backend adapters for `claude`, `opencode`, and `pi`.
    -   **`ehr/`**: Task-family classification, data profiling, and EHR-specific risk checks.
    -   **`verification/`**: Deterministic workspace verification before success is declared.
    -   **`experience/`**: Hierarchical memory models and retrieval logic.
    -   **`tools/`**: Small staged tool-bundle selection instead of large prompt-time tool dumps.
-   **`workspace/`**: The default directory where all runtime artifacts are stored. Each task gets a unique subdirectory containing its plan, logs, and any generated files. The `experience.jsonl` file is also stored here.
-   **`benchmark_results/`**: The output directory for training and benchmarking runs, organized by dataset and model.
-   **`config.toml`**: The central configuration file for LLMs, system settings, and more.
-   **`pyproject.toml`**: Project metadata and dependencies, managed by `uv`.

## ⚙️ Configuration

All settings are managed in `config.toml`.

-   **`[llm.*]`**: Define connection details for different LLM providers (e.g., `[llm.deepseek-chat]`, `[llm.gemini]`). You must provide `base_url`, `api_key`, and `model_name`.
-   **`--active-llm <name>`**: Select the reasoning model used by the planner, evaluator, and reflector.
-   **`--active-executor <name>`**: Select the execution backend used for coding and tool use.
-   **`[executor]`**: Configure supported CLIs and choose the default backend.
-   **`[memory]`**: Configure retrieval budgets and memory writeback mode.
-   **`[ehr]`**: Configure profiling and EHR-specific context building.
-   **`[verification]`**: Configure deterministic success gating.
-   **`[system]`**: Configure system-wide behavior like `max_retries` and `workspace_dir`.
-   **`[evaluation]`**: Set the score threshold for a task to be considered successful after verification.
   -   **`[logging]`**: Control the log level and file path.

## 📜 Citation

If you use HealthFlow in your research, please cite our paper:

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
