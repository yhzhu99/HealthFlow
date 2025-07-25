# CLAUDE.md

## Developer Notes
*   This project uses Python 3.12.
*   Activate the virtual environment by running `source ./.venv/bin/activate`.
*   Package Management: Use `uv` for dependency management.
*   Maintain version control with Git.

## Overview

HealthFlow is a research framework designed to orchestrate, evaluate, and learn from tasks of varying complexity, with a special focus on the healthcare domain. It can intelligently differentiate between simple informational queries and complex tasks that require an agentic coder like **Claude Code**. Its core technical contribution is a **self-evolving meta-system** that manages task fulfillment, whether through direct answers or delegated code execution.

The system automates the process of turning a high-level user request into a successful outcome through an iterative feedback loop, while continuously capturing and synthesizing experience to improve its future performance.

## 🌟 Core Innovation: Adaptive, Experience-Driven Workflow Evolution

HealthFlow's novelty lies in its adaptive **Triage -> Execute -> Evaluate -> Reflect -> Evolve** cycle. The system first analyzes the task to determine the most efficient path, treating every interaction as a scientific experiment to learn from.

1.  **Triage & Plan (TaskDecomposerAgent)**: This is the critical first step. A user's request is analyzed in the context of relevant **experiences** retrieved from past tasks. The agent then triages the task into one of two types:
    *   **`simple_qa`**: For informational questions (e.g., "who are you?"). The agent generates a direct, concise answer.
    *   **`code_execution`**: For complex requests requiring data analysis or file manipulation. The agent creates a detailed, step-by-step markdown file (`task_list.md`).

2.  **Execute (Conditional)**: The workflow adapts based on the triage decision.
    *   **For `simple_qa`**: The system uses the generated answer directly and proceeds to the evaluation phase.
    *   **For `code_execution` (Delegate)**: The system delegates the execution of `task_list.md` to the external `Claude Code` agent by invoking it via a shell command (`claude --dangerously-skip-permissions -p "..."`). It captures the entire terminal output for analysis.

3.  **Evaluate (EvaluatorAgent)**: A dedicated evaluator agent assesses the outcome. It uses specialized criteria to score either the direct answer's quality or the code execution's success against the plan, considering healthcare-specific contexts. If a task fails or the score is too low, the system can loop back, using the feedback to generate a better plan or answer.

4.  **Reflect (ReflectorAgent)**: Upon successful completion of *any* task, a reflector agent analyzes the entire interaction. It synthesizes key learnings into structured **experience objects**, whether it's a `heuristic` from a good answer or a `workflow_pattern` from a complex analysis.

5.  **Evolve (ExperienceManager)**: These structured experiences are stored in a persistent, searchable `experience.jsonl` file. This growing knowledge base is used by the `TaskDecomposerAgent` during the initial triage step, enabling it to make smarter decisions and create better plans/answers for future tasks, thus closing the self-improvement loop for all task types.

This adaptive architecture allows HealthFlow to be both efficient for simple queries and robust for complex ones, while continuously improving its ability to manage all types of tasks over time.

## 🏗️ Architecture

*   **`HealthFlowSystem` (`healthflow/system.py`)**: The central orchestrator that manages the new adaptive workflow.
*   **Internal Agents (`healthflow/agents/`)**: A suite of LLM-powered agents for high-level reasoning:
    *   `TaskDecomposerAgent`: **Triage agent** that analyzes requests, classifies them as `simple_qa` or `code_execution`, and generates either a direct answer or a detailed plan.
    *   `EvaluatorAgent`: Assesses performance for both direct answers and code execution outcomes with a healthcare context.
    *   `ReflectorAgent`: Synthesizes knowledge from all successful task types.
*   **Execution Engine (`healthflow/execution/claude_executor.py`)**: A simple, robust wrapper for calling the external `claude` CLI tool, used only for `code_execution` tasks.
*   **Experience & Memory (`healthflow/experience/`)**: The heart of the self-evolution mechanism.
    *   `ExperienceManager`: Manages the `experience.jsonl` file for storing and retrieving structured experiences from all task types.
    *   `experience_models.py`: Pydantic models defining the structure of an `Experience`.
*   **Prompts (`healthflow/prompts/`)**: A centralized repository of prompt templates. These prompts are now structured to support the task triage logic, with different templates for QA and code execution workflows.
*   **CLI (`run_healthflow.py`)**: A user-friendly command-line interface powered by Typer and Rich.

## Project Structure & Conventions

*   **Configuration**: All settings are in `config.toml`.
*   **Dependencies**: Managed by `uv` via `pyproject.toml`.
*   **Logging**: Logs are written to `healthflow.log`.
*   **Data**: All generated data, including task artifacts and the experience database, is stored in the `workspace/` directory, organized by `task_id`.