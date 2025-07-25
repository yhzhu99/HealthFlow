# CLAUDE.md

## Developer Notes
*   This project uses Python 3.12.
*   Activate the virtual environment by running `source ./.venv/bin/activate`.
*   Package Management: Use `uv` for dependency management. e.g., `uv sync`.
*   Maintain version control with Git.

## Overview

HealthFlow is a research framework designed to orchestrate, evaluate, and learn from powerful, external agentic coders like **Claude Code**. Its core technical contribution is not in building a coding agent, but in creating a **self-evolving meta-system** that manages one.

The system automates the process of turning a high-level user request into a successful software implementation through an iterative feedback loop, while continuously capturing and synthesizing experience to improve its future performance.

## 🌟 Core Innovation: Experience-Driven Workflow Evolution

HealthFlow's novelty lies in its **Plan -> Delegate -> Evaluate -> Reflect -> Evolve** cycle. It treats the process of instructing an agentic coder as a scientific experiment, learning from every success and failure.

1.  **Plan (TaskDecomposerAgent)**: A user's request is transformed into a detailed, step-by-step markdown file (`task_list.md`). This process is augmented by retrieving relevant, structured **experiences** from past tasks to inform the plan.
2.  **Delegate (ClaudeCodeExecutor)**: The system delegates the execution of the `task_list.md` to the external `Claude Code` agent by invoking it via a shell command (`claude -p "..."`). It captures the entire terminal output for analysis.
3.  **Evaluate (EvaluatorAgent)**: A dedicated evaluator agent assesses the execution outcome against the plan. It provides a structured critique, including a score and specific feedback for improvement. If the task fails or the score is too low, the system loops back to the Plan phase, using the feedback to refine the `task_list.md`.
4.  **Reflect (ReflectorAgent)**: Upon successful completion of a task, a reflector agent analyzes the entire interaction (initial request, final plan, execution log, evaluation). It synthesizes the key learnings into structured **experience objects**.
5.  **Evolve (ExperienceManager)**: These structured experiences (e.g., `heuristics`, `code_snippets`, `workflow_patterns`) are stored in a persistent, searchable database. This growing knowledge base is used by the `TaskDecomposerAgent` to create better plans for future tasks, thus closing the self-improvement loop.

This architecture allows HealthFlow to improve its ability to manage agentic coders over time, making its instructions more precise, its plans more robust, and its overall task success rate higher.

## 🏗️ Architecture

*   **`HealthFlowSystem` (`healthflow/system.py`)**: The central orchestrator that manages the entire workflow.
*   **Internal Agents (`healthflow/agents/`)**: A suite of LLM-powered agents for high-level reasoning:
    *   `TaskDecomposerAgent`: Creates detailed plans.
    *   `EvaluatorAgent`: Assesses performance.
    *   `ReflectorAgent`: Synthesizes knowledge.
*   **Execution Engine (`healthflow/execution/claude_executor.py`)**: A simple, robust wrapper for calling the external `claude` CLI tool and capturing its output.
*   **Experience & Memory (`healthflow/experience/`)**: The heart of the self-evolution mechanism.
    *   `ExperienceManager`: Manages the SQLite database for storing and retrieving structured experiences.
    *   `experience_models.py`: Pydantic models defining the structure of an `Experience`.
*   **Prompts (`healthflow/prompts/`)**: A centralized repository of prompt templates that guide the internal agents. These prompts are the "genes" of the system.
*   **CLI (`run_healthflow.py`)**: A user-friendly command-line interface powered by Typer and Rich.

## Project Structure & Conventions

*   **Configuration**: All settings are in `config.toml`.
*   **Dependencies**: Managed by `uv` via `pyproject.toml`.
*   **Logging**: Logs are written to `healthflow.log`.
*   **Data**: All generated data, including task artifacts and the experience database, is stored in the `workspace/` directory, organized by `task_id`.