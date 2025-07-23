## Developer Notes
*   This project uses Python 3.12.
*   Activate the virtual environment by running `source ./.venv/bin/activate`.
*   Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Running tools: `uv run tool`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax
*   Maintain version control with Git, using `git commit` to manage different versions of the codebase.

## Overview

HealthFlow is a self-evolving, multi-agent AI system designed for complex medical reasoning. Its core innovation lies in a tight feedback loop of **action, evaluation, and evolution**, enabling the system to continuously improve its collaborative strategies, prompt templates, and even its own toolset through experience.

Built on the robust and minimalist **Camel AI framework**, HealthFlow prioritizes simplicity, readability, and extensibility. It serves as an ideal platform for research into autonomous agent improvement in the critical domain of healthcare.

## 🌟 Core Innovation: Evaluation-Driven Self-Evolution

HealthFlow's main contribution is its **self-evolving mechanism**, which treats every component of the agent system—prompts, tools, and collaboration patterns—as a dynamic, improvable asset. This is achieved through a powerful, LLM-driven evaluation and reflection cycle.

The core operational loop is:

1.  **Plan & Act**: A user task is processed by a society of agents (a Camel AI `Workforce`). The agents collaborate to solve the task, generating a detailed conversation trace that logs every reasoning step, tool call, and message.
2.  **Evaluate**: A dedicated **LLM-based Evaluator** analyzes the complete conversation trace. It provides a multi-dimensional critique, assessing not just the final outcome's accuracy but also the quality of reasoning, efficiency of tool usage, and effectiveness of agent collaboration.
3.  **Reflect & Evolve**: The rich, structured feedback from the evaluator is stored as an **Experience** in a shared `MemoryManager`. This experience becomes a powerful supervision signal that drives the autonomous evolution of the system's core components:
    *   **Prompt Evolution**: Low-performing prompt templates are automatically refined based on evaluator suggestions, and the system learns to select the best-performing prompt for a given task type.
    *   **Tool Evolution**: If the evaluation identifies a missing capability, the system can task an agent to write, test, and integrate a new tool into its `ToolBank`.
    *   **Strategy Evolution**: Collaboration patterns are improved by refining the prompts that guide the agents' planning and delegation logic.

This cycle ensures that HealthFlow's task success rate improves over time, as it learns from every interaction.

## 🏗️ Simplified and Effective Architecture

HealthFlow leverages modern AI frameworks to maintain a clean, modular, and research-focused architecture.

1.  **Core Agent System (`healthflow/agents/`)**: Built on **Camel AI**. We use a `Workforce` society to manage the three agent roles. This provides a clear, high-level abstraction for multi-agent collaboration, keeping the codebase simple and easy to follow.
2.  **MCP-Powered ToolBank (`healthflow/tools/`)**: A dynamic and extensible tool management system implemented as a **Model Context Protocol (MCP) server** using the `fastmcp` library. This decouples tools from the agent logic, allowing them to be added, removed, or updated independently and on-the-fly. The `AnalystAgent` acts as a client to this server.
3.  **LLM-Based Evaluator (`healthflow/evaluation/`)**: The engine of self-improvement. It analyzes conversation traces and provides structured, actionable feedback.
4.  **Self-Evolving Memory (`healthflow/core/memory.py`)**: The persistent knowledge layer. It stores not only experiences but also evolving prompt templates and metadata about tool performance, guiding the system's evolution.
5.  **CLI Interface (`run_healthflow.py`)**: A user-friendly command-line interface for interaction and system monitoring.

## 🤖 Streamlined Agent Roles within a Camel AI Workforce

HealthFlow uses a `Workforce` society from Camel AI, which provides a natural project-manager-and-workers collaboration model. This simplifies the orchestration logic significantly.

*   **OrchestratorAgent (Coordinator)**: Acts as the "project manager" of the `Workforce`. It receives user tasks, creates a high-level plan, and delegates sub-tasks to the appropriate specialist agents.
*   **ExpertAgent (Worker)**: The medical reasoning engine. It handles tasks requiring deep clinical expertise, such as differential diagnosis and evidence synthesis.
*   **AnalystAgent (Worker)**: The data and tool specialist. It executes all tool-heavy operations by communicating with the MCP ToolBank server. It is also responsible for creating new tools when the system identifies a need.

This structure, managed by Camel AI, fosters clear responsibilities and efficient, transparent collaboration.

## 🛠️ MCP-Powered ToolBank for Simplicity and Extensibility

Instead of a complex internal tool management system, HealthFlow exposes its tools via a lightweight **MCP server** built with `fastmcp`.

*   **Simplicity**: Defining a new tool is as easy as writing a Python function and adding a `@mcp.tool` decorator.
*   **Decoupling**: The ToolBank runs as a separate process, cleanly separating tool logic from agent reasoning.
*   **Dynamic Evolution**: The core of self-improvement. The `AnalystAgent` can programmatically generate Python code for a new tool and register it with the running MCP server *without a restart*. This allows the agent's capabilities to expand dynamically based on experience.
*   **Code Interpreter**: The ToolBank includes a powerful code interpreter tool, which leverages Camel AI's safe execution `Interpreter` for running arbitrary Python code.

This architecture is simple, robust, and perfectly suited for a self-evolving system.