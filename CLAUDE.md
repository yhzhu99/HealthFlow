# CLAUDE.md

## Developer Notes
*   This project uses Python 3.12.
*   Activate the virtual environment by running `source ./.venv/bin/activate`.
*   Package Management: Use `uv` for dependency management. e.g., `uv sync`.
*   Maintain version control with Git.

## Overview

HealthFlow is a **simple, effective, and self-evolving** multi-agent AI framework designed for complex, general-purpose healthcare tasks. Its architecture is built from first principles to be minimalist, readable, and highly extensible, making it an ideal platform for research into autonomous agent improvement.

The core philosophy of HealthFlow is **"simple yet powerful"**. It eschews complex, rigid frameworks in favor of a flexible, LLM-driven core that can adapt its strategies, prompts, and even its own toolset through experience.

## 🌟 Core Innovation: Evaluation-Driven Self-Evolution

HealthFlow's main contribution is its **integrated self-evolving mechanism**. This mechanism treats every component of the agent system—collaboration strategies, prompt templates, and the toolbank—as a dynamic, improvable asset. This is achieved through a powerful **Action -> Evaluation -> Evolution** feedback loop.

The operational cycle is as follows:

1.  **Plan & Act**: A user task is analyzed by an **Orchestrator Agent**, which devises a flexible plan and dynamically selects the best agent or combination of agents (e.g., `AnalystAgent` only, `ExpertAgent` only, or a collaboration) to execute it. For complex computational tasks, a specialized **`ReactAgent`** is employed to perform agentic coding: iteratively writing code, executing it, observing results, and debugging until the task is complete.
2.  **Evaluate**: A dedicated **LLM-based Evaluator** analyzes the complete execution trace. It provides a multi-dimensional critique, assessing not just the final outcome's accuracy but also the quality of reasoning, efficiency of tool usage, and effectiveness of the chosen collaboration strategy.
3.  **Reflect & Evolve**: The rich, structured feedback from the evaluator is stored as an **Experience**. This experience becomes a powerful supervision signal that drives the autonomous evolution of the system's core components, managed by the `EvolutionManager`:
    *   **Prompt Evolution**: Low-performing prompt templates are automatically refined based on evaluator suggestions. The system learns to select the best-performing prompt version for a given role.
    *   **Tool Evolution (Agentic Tool Creation)**: If the evaluation identifies a missing capability, or the `AnalystAgent` itself determines a new reusable function is needed, it can autonomously write, test, and integrate a new tool into its `ToolBank` using a dedicated `add_new_tool` function. This allows the system's capabilities to expand dynamically.
    *   **Strategy Evolution**: The system tracks the performance of different collaboration patterns (e.g., Analyst-only vs. Collaborative). The `Orchestrator`'s decision-making improves over time by learning which strategies work best for which types of tasks.

This cycle ensures that HealthFlow's task success rate improves over time as it learns from every interaction, becoming progressively more intelligent and capable.

## 🏗️ Simplified and Effective Architecture

HealthFlow's architecture is intentionally lean and modular, focusing on core agentic capabilities.

1.  **Core System (`healthflow/system.py`)**: The central `HealthFlowSystem` orchestrates the entire process. It manages the agent lifecycle, task execution flow, and the evaluation-evolution loop.
2.  **Flexible Agent Collaboration**: Instead of a fixed hierarchy, HealthFlow uses a dynamic approach. The `OrchestratorAgent` intelligently routes tasks to:
    *   **`ExpertAgent`**: For medical domain knowledge.
    *   **`AnalystAgent`**: For all computational tasks. This agent is wrapped in a **`ReactAgent`** (`healthflow/core/react_agent.py`) to enable iterative, agentic coding for complex data analysis and modeling.
3.  **Dynamic ToolBank (`healthflow/tools/tool_manager.py`)**: We've removed external dependencies like `MCP`. The `ToolManager` is a simple, in-process class that manages a set of callable Python functions. It includes a powerful `code_interpreter` and, critically, an `add_new_tool` function that allows the `AnalystAgent` to expand its own toolset on the fly.
4.  **Evolution & Memory (`healthflow/core/evolution.py`, `healthflow/core/memory.py`)**:
    *   The `EvolutionManager` is the brain of the self-improvement loop. It persists and manages evolving prompts and strategy performance data in simple JSON files.
    *   The `MemoryManager` logs all task experiences, including traces and evaluation results, creating a long-term institutional memory for the system.
5.  **Robust Interpreter (`healthflow/core/interpreter.py`)**: A custom `HealthFlowInterpreter` provides a secure and capable environment for code execution, pre-loaded with essential libraries (`pandas`, `numpy`, `torch`, `scikit-learn`), enabling the agent to perform serious data science tasks out-of-the-box.
6.  **CLI Interface (`run_healthflow.py`)**: A user-friendly and powerful command-line interface for interaction, task execution, and system monitoring.

## 🤖 Agent Roles & Flexible Interaction

*   **OrchestratorAgent**: The "thinking" center. It analyzes tasks and creates a high-level, flexible execution plan. It decides whether a task requires just the `ExpertAgent`, just the `AnalystAgent`, or a collaborative effort, making the system's collaboration patterns adaptable.
*   **ExpertAgent**: The medical reasoning engine. It handles tasks requiring deep clinical expertise. It is explicitly prompted to delegate all computational work.
*   **AnalystAgent**: The data and tool specialist, supercharged with **agentic coding** abilities via the `ReactAgent` wrapper. It handles all tool-heavy and coding operations. It can:
    *   Probe data files to understand their structure.
    *   Write and execute Python code for analysis and modeling.
    *   Reflect on code output and errors, and iteratively debug.
    *   **Create and add new tools to its own ToolBank** when it identifies a reusable piece of logic.

This flexible structure, guided by an evolving `Orchestrator`, ensures that the right agent (or agents) are used for the job, improving efficiency and success rates.