# HealthFlow: A Self-Evolving AI Agent Framework for Healthcare

HealthFlow is a streamlined, multi-agent AI framework designed for complex, general-purpose healthcare tasks. Its core innovation is a **self-evolving loop** driven by a comprehensive, process-oriented evaluator. By learning from a deep critique of its entire reasoning process—not just the final outcome—HealthFlow continuously refines its strategies, enhances its collaborative capabilities, and dynamically expands its own toolset via **agentic tool creation**.

Built on the **Camel AI framework** for its agent components, but with a custom, flexible orchestration and tool management system, its simple and modular architecture makes it an ideal framework for academic exploration and publication in agent-based AI.

## 🌟 Core Innovation: Evaluation-Driven Self-Evolution

HealthFlow's ability to learn is powered by a unique cycle of **action, evaluation, and evolution**:

1.  **Plan & Act**: An agent society, dynamically orchestrated, collaborates to solve a task. For computational tasks, a **ReactAgent** engages in an iterative **code-execute-debug** loop. This entire process creates a detailed execution trace.
2.  **Evaluate**: A dedicated **LLM-based Evaluator** analyzes the entire trace. It provides a multi-dimensional critique, assessing not just the outcome's accuracy but also the quality of the reasoning, the efficiency of tool usage, and the effectiveness of collaboration.
3.  **Reflect & Evolve**: The rich, structured feedback from the evaluator is stored as an **Experience**. This becomes a powerful supervision signal that drives the autonomous evolution of:
    *   **Prompt Templates**: Prompts are refined based on feedback to improve agent performance.
    *   **ToolBank**: The system can identify a missing capability and task an agent to write, test, and integrate a new tool on-the-fly.
    *   **Collaborative Strategies**: The logic guiding agent planning and delegation is improved through performance tracking.

## 🏗️ Simplified and Effective Architecture

Designed for clarity and research focus, HealthFlow's architecture is simple yet powerful.

-   **Core Agent System (`healthflow/system.py`)**: A custom orchestration logic that dynamically manages collaboration between an Orchestrator, an Expert, and a powerful Analyst.
-   **Agentic Coding with ReAct (`healthflow/core/react_agent.py`)**: A specialized agent wrapper that enables iterative problem-solving for coding and data analysis tasks.
-   **Dynamic ToolBank (`healthflow/tools/tool_manager.py`)**: A simple, in-process tool manager that is extensible and can be updated live by the agents themselves. No external dependencies.
-   **LLM-Based Evaluator (`healthflow/evaluation/`)**: The engine of self-improvement, providing deep, process-oriented feedback.
-   **Evolution & Memory (`healthflow/core/evolution.py`)**: The persistent knowledge layer that stores and manages evolving prompts and strategies.

## 🤖 Flexible Agent Roles

-   **Orchestrator**: The central coordinator that analyzes user tasks, creates a flexible plan, and dynamically delegates to the best-suited specialist(s).
-   **Expert**: The medical reasoning engine that handles tasks requiring deep clinical expertise.
-   **Analyst**: The data and tool specialist, capable of **agentic coding**. It handles all computational tasks, including data probing, analysis, modeling, and **creating new tools for itself**.

## 🚀 Quick Start

### 1. Environment Setup

This project requires Python 3.12.

```bash
# Install dependencies using uv
uv sync
# Then activate the virtual environment
source .venv/bin/activate
```

### 2. Configuration

Copy the example configuration file and add your LLM provider API key.

```bash
cp config.toml.example config.toml
```

Edit `config.toml` to add your credentials.

### 3. Run HealthFlow

#### Interactive Mode
```bash
python run_healthflow.py
```
This will start an interactive shell where you can provide tasks.

#### Single Task Execution
```bash
python run_healthflow.py --task "Analyze the provided EHR data file 'data/sample_ehr.csv' and build a simple predictive model for patient readmission."
```

## 🏥 Medical AI Safety

This system is designed for research and educational purposes. It can generate information that may be inaccurate or outdated. **Always consult qualified healthcare professionals for medical advice and decisions.** The safety features included are for research into building safer AI and are not a substitute for professional medical judgment.