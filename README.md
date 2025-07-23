# HealthFlow: An Evaluation-Driven, Self-Improving AI Agent System for Healthcare

HealthFlow is a streamlined, multi-agent AI system designed for complex medical reasoning and research. Its core innovation is a self-improvement loop driven by a comprehensive, process-oriented evaluator. By learning from a deep critique of its entire reasoning process—not just the final outcome—HealthFlow continuously refines its strategies, enhances its collaborative capabilities, and builds a robust, experience-grounded knowledge base.

Its simple and modular architecture makes it an ideal framework for academic exploration and publication in agent-based healthcare AI.

## 🌟 Core Innovation: Evaluation-Driven Self-Improvement

HealthFlow's ability to learn is powered by a unique cycle of **action, evaluation, and reflection**:

1.  **Plan & Act**: An agent formulates a plan and executes a task, creating a detailed **ExecutionTrace** that logs every reasoning step, tool call, and collaboration message.
2.  **Evaluate**: A dedicated **LLM-based Evaluator** analyzes the entire `ExecutionTrace`. It provides a multi-dimensional critique, assessing not just the outcome's accuracy but also the quality of the reasoning, the efficiency of tool usage, and the effectiveness of collaboration.
3.  **Reflect & Learn**: The rich, structured feedback from the evaluator is stored as an **Experience** in the agent's memory. This becomes a powerful supervision signal that drives the autonomous evolution of agent behavior, prompts, and tools.

## 🏗️ Simplified Architecture

Designed for clarity and research focus, HealthFlow's architecture consists of three main components and a user-friendly CLI. All agents share a single `ToolBank` and `Evaluator` to ensure system-wide consistency and learning.

-   **Core Agent System (`healthflow/core/`)**: A compact framework with three specialized agent roles.
-   **Shared Hierarchical ToolBank (`healthflow/tools/`)**: A centralized tool registry with efficient, tag-based retrieval.
-   **LLM-Based Evaluator (`healthflow/evaluation/`)**: The engine of self-improvement, providing deep, process-oriented feedback.
-   **CLI Interface (`healthflow/cli.py`)**: The main entry point for interaction and batch processing.

## 🤖 Streamlined Agent Roles

HealthFlow is distilled to three essential agent roles for maximum efficiency and clarity:

-   **Orchestrator**: The central coordinator that receives user tasks, breaks them down, and delegates them to specialists.
-   **Expert**: The medical reasoning engine that handles tasks requiring deep clinical expertise, such as differential diagnosis and evidence synthesis.
-   **Analyst**: The data and tool specialist that executes all tool-heavy operations, including data analysis, code execution, and visualization.

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

```toml
# config.toml

# Select the active LLM provider for agent reasoning and evaluation
active_llm = "deepseekv3"

[llm.deepseekv3]
base_url = "https://api.deepseek.com"
api_key = "YOUR_API_KEY_HERE"
model_name = "deepseek-chat"

# ... other settings
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
```bash
# Create a tasks.jsonl file with one JSON object per line
# {"task": "Explain the mechanism of action of ACE inhibitors"}
# {"task": "What are the contraindications for MRI in patients with implants?"}

python run_healthflow.py --file path/to/tasks.jsonl
```

## 📁 Project Structure

```
HealthFlow/
├── healthflow/                 # Main package
│   ├── core/                  # Core agent and system logic
│   │   ├── agent.py          # The streamlined HealthFlowAgent class
│   │   ├── config.py         # Configuration management
│   │   ├── llm_provider.py   # LLM abstraction layer
│   │   └── memory.py         # Experience-based memory system
│   ├── tools/                 # Unified tool management
│   │   └── toolbank.py       # The Hierarchical ToolBank
│   ├── evaluation/            # Evaluation system
│   │   └── evaluator.py      # The core LLM-based Task Evaluator
│   └── cli.py               # Command line interface
├── data/                    # Runtime data (memory, tools, etc.)
├── .python-version          # Specifies Python 3.12
├── pyproject.toml           # Project dependencies
├── config.toml              # Your local configuration
├── run_healthflow.py        # Main execution script
└── README.md                # This file
```

## 🏥 Medical AI Safety

This system is designed for research and educational purposes. It can generate information that may be inaccurate or outdated. **Always consult qualified healthcare professionals for medical advice and decisions.** The safety features included are for research into building safer AI and are not a substitute for professional medical judgment.
