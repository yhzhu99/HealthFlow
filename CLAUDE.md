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

HealthFlow is a streamlined, multi-agent AI system designed for complex medical reasoning. Its core innovation lies in a tight loop of **action, evaluation, and reflection**, enabling agents to continuously improve their performance through experience. The system is architected to be simple yet effective, focusing on research-oriented contributions in agent-based healthcare AI, making it an ideal framework for academic exploration and publication.

## 🌟 Core Innovation: Evaluation-Driven Self-Improvement

HealthFlow's primary contribution is its **self-improvement mechanism guided by a comprehensive, LLM-based evaluator**. Unlike traditional agent systems that learn only from task success or failure, HealthFlow learns from a deep, qualitative critique of its entire process.

The core operational loop is:

1.  **Plan & Act**: An agent formulates a plan and executes a task. During execution, it builds a detailed `ExecutionTrace`, logging every significant action—from reasoning steps to tool calls and inter-agent communication.
2.  **Evaluate**: A dedicated **LLM-based Evaluator** receives the complete `ExecutionTrace`. It assesses not just the final result, but also the efficiency of the collaboration, the quality of the reasoning steps, and the appropriateness of tool usage across multiple medical and operational criteria.
3.  **Reflect & Learn**: The rich, structured feedback from the evaluator is stored as an **Experience** in the agent's memory. This experience, containing a multi-faceted critique and actionable suggestions, becomes a powerful supervision signal to evolve all aspects of the system, including agent behavior, prompt templates, and even the tools themselves.

This cycle allows HealthFlow to autonomously refine its strategies, enhance its problem-solving capabilities, and build a robust, experience-grounded knowledge base.

## 🏗️ Simplified Architecture

HealthFlow's architecture is designed for clarity, modularity, and research focus. All agents share a single instance of the `ToolBank` and `Evaluator`, promoting resource efficiency and system-wide consistency.

1.  **Core Agent System (`healthflow/core/`)**: A compact multi-agent framework featuring a minimal set of powerful agent roles and an experience-based memory system. Agents are responsible for collecting detailed process traces (`ExecutionTrace`) for evaluation.
2.  **Shared ToolBank (`healthflow/tools/`)**: A centralized, dynamic tool management system with a hierarchical tagging structure for efficient, non-sequential retrieval.
3.  **LLM-Based Evaluator (`healthflow/evaluation/`)**: The heart of the self-improvement loop, providing deep, process-oriented feedback by analyzing `ExecutionTrace` objects.
4.  **CLI Interface (`healthflow/cli.py`)**: A user-friendly command-line interface for interaction, batch processing, and system monitoring.

## 🤖 Streamlined Agent Roles

To maximize impact and reduce complexity, HealthFlow is distilled to three essential agent roles:

*   **OrchestratorAgent**: The central coordinator. It receives user tasks, breaks them down into sub-tasks, and delegates them to the appropriate specialist agent. It manages the overall workflow and synthesizes the final response.
*   **ExpertAgent**: The core medical reasoning engine. It handles tasks requiring deep clinical expertise, such as differential diagnosis, interpreting complex medical queries, and synthesizing information into clinically relevant insights.
*   **AnalystAgent**: The data and tool specialist. It is responsible for all tool-heavy and data-intensive operations, such as performing data analysis, executing code, and generating visualizations. Its ability to provide evidence-based insights is directly tied to the capabilities of the tools it can access.

This streamlined structure fosters clear responsibilities and efficient collaboration.

## 🛠️ Hierarchical ToolBank for Efficient Retrieval

HealthFlow features a single, globally accessible `HierarchicalToolBank`. As the number of dynamic tools grows, finding the right one becomes a critical challenge. To solve this, HealthFlow implements a **hierarchical tagging system**.

*   **Hierarchical Tags**: Each tool is tagged with metadata across several categories defined in the `TagHierarchy` enum, such as:
    *   **Domain**: `medical`, `genomic`, `administrative`
    *   **Functionality**: `analysis`, `visualization`, `data_retrieval`
    *   **DataType**: `clinical_notes`, `lab_results`, `imaging_reports`
*   **Efficient Retrieval**: When an agent needs a tool, the `search_tools` method first filters the `ToolBank` by the most relevant tags (e.g., `Domain: medical`, `Functionality: analysis`), dramatically narrowing the search space. It then performs secondary scoring on this small subset, allowing for fast and accurate tool selection even as the system scales.
*   **Dynamic Creation**: Agents can programmatically generate new Python tools when needed. These new tools are automatically tagged and integrated into the hierarchical structure.

## 📊 The LLM-Based Evaluator: Engine of Evolution

The `LLMTaskEvaluator` is the cornerstone of HealthFlow's ability to learn and improve. It goes beyond simple outcome scoring by analyzing the entire task lifecycle via an `ExecutionTrace`.

*   **Process and Outcome Monitoring**: The evaluator assesses both the final result and the steps taken to achieve it. It reviews the agent's plan, the sequence of collaboration, tool selection, and reasoning traces.
*   **Multi-Dimensional Evaluation**: It provides feedback across critical criteria defined in `EvaluationCriteria`:
    *   `Medical Accuracy` & `Safety`: Correctness and adherence to safety protocols.
    *   `Reasoning Quality`: The logical soundness of the agent's plan and actions.
    *   `Tool Usage Efficiency`: The appropriateness and effectiveness of the tools used.
    *   `Collaboration Effectiveness`: The clarity and efficiency of inter-agent communication.
    *   `Completeness` & `Clarity`: How well the final output addresses the user's request.
*   **Rich Supervision Signal**: The evaluator provides a structured JSON output containing scores, detailed textual feedback, and concrete `improvement_suggestions`. This rich signal is the primary driver for evolution, providing actionable insights to:
    *   **Refine Agent Collaboration Patterns**: "The Orchestrator should have provided more context to the AnalystAgent."
    *   **Improve Prompt Templates**: "The planning prompt should explicitly ask for potential contraindications."
    *   **Evolve Tools**: "The data analysis tool should be modified to handle missing values."
