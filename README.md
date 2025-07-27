# HealthFlow: A Self-Evolving Meta-System for Agentic Healthcare AI

HealthFlow is a research framework designed to orchestrate, evaluate, and learn from powerful, external agentic coders like **Claude Code**, with a specific focus on healthcare tasks. Its core technical contribution is not in building a coding agent itself, but in creating a **self-evolving meta-system** that manages one to solve complex healthcare-related problems.

This project is designed with academic publication (e.g., NeurIPS) in mind, focusing on a novel approach to agentic AI: **experience-driven workflow evolution**. The system automates the process of turning a high-level user request into a successful outcome through an iterative feedback loop, while continuously capturing and synthesizing experience to improve its future performance.

## 🌟 Core Innovation: Unified, Experience-Driven Workflow

HealthFlow's novelty lies in its unified and automated **Plan -> Delegate -> Evaluate -> Reflect -> Evolve** cycle. It treats every task, from a simple question to a complex analysis, as a scientific experiment, learning from every success to continuously improve its own capabilities in the healthcare domain.

1.  **Plan (MetaAgent)**: A user's request (e.g., "Analyze patient dataset to find correlations" or "Who are you?") is analyzed by the `MetaAgent` in the context of relevant **experiences** retrieved from past tasks. The agent's key responsibility is to synthesize these experiences into actionable context and then generate a detailed, step-by-step markdown plan (`task_list.md`).
2.  **Delegate (ClaudeCodeExecutor)**: The system universally delegates the execution of the generated `task_list.md` to an external agentic coder (`Claude Code`) by invoking it via a shell command (e.g., `claude --dangerously-skip-permissions -p "..."`). It captures the entire terminal output for analysis.
3.  **Evaluate (EvaluatorAgent)**: A dedicated evaluator agent assesses the execution outcome, considering correctness, efficiency, and healthcare-specific best practices. If a task fails or the quality is low, the system can loop back to the Plan phase, using the feedback to refine the plan.
4.  **Reflect (ReflectorAgent)**: Upon successful completion of *any* task, a reflector agent analyzes the entire interaction (request, plan, logs) to synthesize key learnings into structured **experience objects** (e.g., a `heuristic`, a `warning`, a `code_snippet`).
5.  **Evolve (ExperienceManager)**: These structured experiences are stored in a persistent, searchable `experience.jsonl` file. This growing knowledge base is used by the `MetaAgent` during the planning step, enabling it to make smarter decisions and create better, more context-aware plans for future tasks, thus closing the self-improvement loop.

This unified architecture ensures that HealthFlow is both consistent in its approach and robust enough for complex challenges, while continuously improving its ability to manage all types of tasks over time.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- `uv` (for package management)
- [Claude Code CLI](https://docs.anthropic.com/claude/docs/claude-code) installed and available in your `PATH`.

### 2. Setup
```bash
# Clone the repository
git clone <repository-url>
cd healthflow-project

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Create your configuration file from the example
cp config.toml.example config.toml
```
Now, edit `config.toml`, set your desired `active_llm`, and add your LLM API key. This key is used for HealthFlow's internal reasoning agents, not for Claude Code.

### 3. Running HealthFlow

There are two ways to run the system:

#### Interactive Mode
For a chat-like session where you can run multiple healthcare-related tasks.
```bash
python run_healthflow.py interactive
```

#### Single-Task Mode
To execute a single, specific task and then exit.
```bash
python run_healthflow.py run "Analyze the provided 'patients.csv' to identify the top 3 risk factors for readmission. Anonymize any patient identifiers in the output."
```
HealthFlow will create a `workspace/` directory where all task artifacts, logs, and the `experience.jsonl` file will be stored.

#### Customizing Execution
You can customize the experience path and executor shell via CLI arguments:
```bash
python run_healthflow.py run "Who are you?" \
    --experience-path "my_custom_knowledge_base/experiences.jsonl" \
    --shell "/bin/zsh"
```

## 🏗️ Architecture

*   **`HealthFlowSystem` (`healthflow/system.py`)**: The central orchestrator that manages the unified workflow.
*   **Internal Agents (`healthflow/agents/`)**: A suite of LLM-powered agents for high-level reasoning:
    *   `MetaAgent`: **The core planner** that analyzes requests, synthesizes past experiences, and generates a detailed execution plan for every task.
    *   `EvaluatorAgent`: Assesses performance for all code execution outcomes with a healthcare context.
    *   `ReflectorAgent`: Synthesizes knowledge from all successful tasks.
*   **Execution Engine (`healthflow/execution/claude_executor.py`)**: A simple, robust wrapper for calling the external `claude` CLI tool, used for all tasks.
*   **Experience & Memory (`healthflow/experience/`)**: The heart of the self-evolution mechanism.
    *   `ExperienceManager`: Manages the `experience.jsonl` file for storing and retrieving structured experiences.
    *   `experience_models.py`: Pydantic models defining the structure of an `Experience`.
*   **Prompts (`healthflow/prompts/`)**: A centralized repository of prompt templates that guide the agents.
*   **CLI (`run_healthflow.py`)**: A user-friendly command-line interface powered by Typer and Rich.

This architecture is intentionally minimalist and modular to serve as a clean and effective research platform for studying self-improving AI systems in the healthcare domain.

## Project Structure & Conventions

*   **Configuration**: All settings are in `config.toml`.
*   **Dependencies**: Managed by `uv` via `pyproject.toml`.
*   **Logging**: Logs are written to `healthflow.log`.
*   **Data**: All generated data, including task artifacts and the experience database, is stored in the `workspace/` directory, organized by `task_id`.