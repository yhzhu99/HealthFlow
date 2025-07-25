# HealthFlow: A Self-Evolving Meta-System for Agentic Healthcare AI

HealthFlow is a research framework designed to orchestrate, evaluate, and learn from powerful, external agentic coders like **Claude Code**, with a specific focus on healthcare tasks. Its core technical contribution is not in building a coding agent itself, but in creating a **self-evolving meta-system** that manages one to solve complex healthcare-related problems.

This project is designed with academic publication (e.g., NeurIPS) in mind, focusing on a novel approach to agentic AI: **experience-driven workflow evolution**.

## 🌟 Core Innovation

HealthFlow's novelty lies in its automated **Plan -> Delegate -> Evaluate -> Reflect -> Evolve** cycle. It treats the process of instructing an agentic coder as a scientific experiment, learning from every success and failure to continuously improve its own capabilities in the healthcare domain.

1.  **Plan (TaskDecomposerAgent)**: A user's request (e.g., "Analyze patient dataset to find correlations") is transformed into a detailed, step-by-step plan (`task_list.md`). This process is augmented by retrieving relevant, structured **experiences** from past tasks to inform the plan.
2.  **Delegate (ClaudeCodeExecutor)**: The system delegates the execution of the plan to an external agentic coder (`Claude Code`) by invoking it via a shell command.
3.  **Evaluate (EvaluatorAgent)**: A dedicated evaluator agent assesses the execution outcome, considering correctness, efficiency, and healthcare-specific best practices. If the task fails or quality is low, the system loops back to the Plan phase, using the feedback to refine the plan.
4.  **Reflect (ReflectorAgent)**: Upon successful completion, a reflector agent analyzes the entire interaction to synthesize key learnings into structured **experience objects**.
5.  **Evolve (ExperienceManager)**: These structured experiences (e.g., `heuristics`, `code_snippets`, `warnings`) are stored in a persistent `experience.jsonl` file. This growing knowledge base is used by the `TaskDecomposerAgent` to create better plans for future tasks, closing the self-improvement loop.

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

## 🏗️ Architecture

-   **`HealthFlowSystem`**: The central orchestrator of the entire process.
-   **Internal Agents (`healthflow/agents/`)**: LLM-powered agents for high-level reasoning (Decomposer, Evaluator, Reflector), guided by healthcare-specific prompts.
-   **Execution Engine (`healthflow/execution/`)**: A simple wrapper for calling the external `claude` CLI.
-   **Experience & Memory (`healthflow/experience/`)**: Manages the `experience.jsonl` file, forming the system's long-term, evolving knowledge base.

This architecture is intentionally minimalist and modular to serve as a clean and effective research platform for studying self-improving AI systems in the healthcare domain.