# HealthFlow: A Self-Evolving Meta-System for Agentic Coding

HealthFlow is a research framework designed to orchestrate, evaluate, and learn from powerful, external agentic coders like **Claude Code**. Its core technical contribution is not in building a coding agent, but in creating a **self-evolving meta-system** that manages one.

This project is designed with academic publication (e.g., NeurIPS) in mind, focusing on a novel approach to agentic AI: **experience-driven workflow evolution**.

## 🌟 Core Innovation

HealthFlow's novelty lies in its automated **Plan -> Delegate -> Evaluate -> Reflect -> Evolve** cycle. It treats the process of instructing an agentic coder as a scientific experiment, learning from every success and failure to continuously improve its own capabilities.

1.  **Plan (TaskDecomposerAgent)**: A user's request is transformed into a detailed, step-by-step plan (`task_list.md`). This process is augmented by retrieving relevant, structured **experiences** from past tasks to inform the plan.
2.  **Delegate (ClaudeCodeExecutor)**: The system delegates the execution of the plan to an external agentic coder (`Claude Code`) by invoking it via a shell command.
3.  **Evaluate (EvaluatorAgent)**: A dedicated evaluator agent assesses the execution outcome. If the task fails or the quality is low, the system loops back to the Plan phase, using the feedback to refine the plan for a retry.
4.  **Reflect (ReflectorAgent)**: Upon successful completion, a reflector agent analyzes the entire interaction to synthesize the key learnings into structured **experience objects**.
5.  **Evolve (ExperienceManager)**: These structured experiences (e.g., `heuristics`, `code_snippets`, `warnings`) are stored in a persistent SQLite database. This growing knowledge base is used by the `TaskDecomposerAgent` to create better plans for future tasks, closing the self-improvement loop.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- `uv` (for package management)
- [Claude Code CLI](https://docs.anthropic.com/claude/docs/claude-code) installed and available.

### 2. Setup
```bash
# Clone the repository
git clone <repository-url>
cd healthflow

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Create your configuration file
cp config.toml.example config.toml
```
Now, edit `config.toml`, set your desired `active_llm`, and add your LLM API key. This key is used for HealthFlow's internal agents, not for Claude Code.

### 3. Running HealthFlow

There are two ways to run the system:

#### Interactive Mode
For a chat-like session where you can run multiple tasks.
```bash
python run_healthflow.py interactive
```

#### Single-Task Mode
To execute a single, specific task and then exit.
```bash
python run_healthflow.py run "Create a Python script that reads 'data.csv', calculates the average of the 'age' column, and prints the result."
```
HealthFlow will create a `workspace/` directory where all task artifacts, logs, and the `experience.db` will be stored.

## 🏗️ Architecture

-   **`HealthFlowSystem`**: The central orchestrator.
-   **Internal Agents (`healthflow/agents/`)**: LLM-powered agents for high-level reasoning (Decomposer, Evaluator, Reflector).
-   **Execution Engine (`healthflow/execution/`)**: A simple wrapper for calling the external `claude` CLI.
-   **Experience & Memory (`healthflow/experience/`)**: Manages the SQLite database for the system's long-term, evolving knowledge.

This architecture is intentionally minimalist and modular to serve as a clean and effective research platform for studying self-improving AI systems.