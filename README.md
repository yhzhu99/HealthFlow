# HealthFlow: A Self-Evolving AI Agent System for Healthcare

HealthFlow is a streamlined, multi-agent AI system designed for complex medical reasoning and research. Its core innovation is a **self-evolving loop** driven by a comprehensive, process-oriented evaluator. By learning from a deep critique of its entire reasoning process—not just the final outcome—HealthFlow continuously refines its strategies, enhances its collaborative capabilities, and dynamically expands its own toolset.

Built on the **Camel AI framework** and using the **Model Context Protocol (MCP)** for tool management, its simple and modular architecture makes it an ideal framework for academic exploration and publication in agent-based healthcare AI.

## 🌟 Core Innovation: Evaluation-Driven Self-Evolution

HealthFlow's ability to learn is powered by a unique cycle of **action, evaluation, and evolution**:

1.  **Plan & Act**: An agent society (a Camel AI `Workforce`) collaborates to solve a task, creating a detailed **conversation trace** that logs every reasoning step, tool call, and message.
2.  **Evaluate**: A dedicated **LLM-based Evaluator** analyzes the entire trace. It provides a multi-dimensional critique, assessing not just the outcome's accuracy but also the quality of the reasoning, the efficiency of tool usage, and the effectiveness of collaboration.
3.  **Reflect & Evolve**: The rich, structured feedback from the evaluator is stored as an **Experience**. This becomes a powerful supervision signal that drives the autonomous evolution of:
    *   **Prompt Templates**: Prompts are refined based on feedback to improve agent performance.
    *   **ToolBank**: The system can identify a missing capability and task an agent to write, test, and integrate a new tool on-the-fly.
    *   **Collaborative Strategies**: The logic guiding agent planning and delegation is improved through prompt evolution.

## 🏗️ Simplified and Effective Architecture

Designed for clarity and research focus, HealthFlow's architecture leverages modern frameworks for simplicity and power.

-   **Core Agent System (`healthflow/agents/`)**: Built on **Camel AI**, using a `Workforce` society to manage collaboration between an Orchestrator, an Expert, and an Analyst.
-   **MCP-Powered ToolBank (`healthflow/tools/`)**: A dynamic tool server built with **`fastmcp`**. It's simple, extensible, and can be updated live by the agents themselves.
-   **LLM-Based Evaluator (`healthflow/evaluation/`)**: The engine of self-improvement, providing deep, process-oriented feedback.
-   **Self-Evolving Memory (`healthflow/core/memory.py`)**: The persistent knowledge layer that stores experiences and evolving prompts.
-   **CLI Interface (`run_healthflow.py`)**: The main entry point for user interaction.

## 🤖 Streamlined Agent Roles

HealthFlow uses a Camel AI `Workforce` to manage three essential agent roles:

-   **Orchestrator (Coordinator)**: The central coordinator that receives user tasks, creates a plan, and delegates sub-tasks to specialists.
-   **Expert (Worker)**: The medical reasoning engine that handles tasks requiring deep clinical expertise.
-   **Analyst (Worker)**: The data and tool specialist that executes all tool-heavy operations by communicating with the MCP ToolBank, including creating new tools.

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

Edit `config.toml` to add your credentials. For example:

```toml
# config.toml
active_llm = "deepseek-v3"

[llm.deepseek-v3]
base_url = "https://api.deepseek.com"
api_key = "YOUR_API_KEY_HERE" # Replace with your key
model_name = "deepseek-chat"
```

### 3. Run HealthFlow

#### Interactive Mode
```bash
python run_healthflow.py
```
This will start an interactive shell.
```
HealthFlow> What are the key considerations for prescribing statins to a 75-year-old female with a history of liver disease?
```

#### Single Task Execution
```bash
python run_healthflow.py --task "Analyze the symptoms: fever, cough, shortness of breath"
```

#### Process a Task File
Create a `tasks.jsonl` file with one JSON object per line, e.g., `{"task": "Explain the mechanism of action of ACE inhibitors"}`.

```bash
python run_healthflow.py --file path/to/tasks.jsonl
```

## 🏥 Medical AI Safety

This system is designed for research and educational purposes. It can generate information that may be inaccurate or outdated. **Always consult qualified healthcare professionals for medical advice and decisions.** The safety features included are for research into building safer AI and are not a substitute for professional medical judgment.