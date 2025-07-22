# HealthFlow: Self-Evolving Healthcare AI Agent System

HealthFlow is a state-of-the-art multi-agent healthcare AI system that features self-evolution, dynamic tool creation, and advanced medical reasoning capabilities. Built for academic research and healthcare applications, it surpasses traditional agent frameworks through continuous learning and improvement.

## 🌟 Key Features

### Core Capabilities
- **Self-Evolving Agents**: Agents continuously learn and improve from experience
- **Multi-Agent Collaboration**: 7 specialized agents working together
- **Dynamic Tool Creation**: Automatic creation and management of tools via ToolBank
- **Medical-Specific Rewards**: Mutual information rewards for diagnostic tasks
- **File-Based Persistence**: No database dependencies, uses JSONL/Parquet/Pickle
- **Multi-Provider LLM Support**: OpenAI, Anthropic, Gemini, and more

### Advanced Features
- **Memory Management**: Sophisticated memory with prompt evolution
- **Experience Accumulation**: Learn from successes and failures
- **Medical Safety**: Built-in safety checks and compliance monitoring
- **Code Execution**: Integrated Python code execution capabilities
- **Comprehensive Evaluation**: Medical-focused task evaluation metrics

## 🏗️ Architecture

HealthFlow consists of several key components:

1. **Core Agent System** (`healthflow/core/`)
   - Multi-role agents with self-evolution
   - Advanced memory management
   - LLM provider abstraction
   - Reward system with mutual information

2. **Tool Management** (`healthflow/tools/`)
   - Dynamic tool creation and execution
   - MCP (Model Context Protocol) support
   - Tool performance tracking

3. **Evaluation System** (`healthflow/evaluation/`)
   - Medical safety assessment
   - Evidence-based scoring
   - Performance tracking

4. **CLI Interface** (`healthflow/cli.py`)
   - Interactive and batch processing modes
   - Task file processing
   - System monitoring

## 🚀 Quick Start

### 1. Environment Setup

First, activate the virtual environment and install dependencies:

```bash
source ./.venv/bin/activate
uv sync
```

### 2. Configuration

Copy and configure the environment file:

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```bash
# LLM Configuration
BASE_URL=https://api.openai.com/v1  # or your preferred provider
API_KEY=your_api_key_here
MODEL_NAME=gpt-4-turbo-preview

# Optional: Other providers
# BASE_URL=https://api.anthropic.com
# MODEL_NAME=claude-3-sonnet-20240229

# Data directories (optional, will use defaults)
# DATA_DIR=./data
# MEMORY_DIR=./data/memory
# TOOLS_DIR=./data/tools
```

### 3. Run HealthFlow

#### Interactive Mode (Default)
```bash
python run_healthflow.py
```

#### Single Task Execution
```bash
python run_healthflow.py --task "Analyze the symptoms: fever, cough, shortness of breath"
```

#### Process Task File
```bash
python run_healthflow.py --file scripts/extract_task/tasks/1_tasks.jsonl --max-tasks 5
```

#### System Status
```bash
python run_healthflow.py --status
```

## 💡 Usage Examples

### Interactive Mode

```bash
$ python run_healthflow.py
HealthFlow> help
HealthFlow> What are the differential diagnoses for chest pain in a 45-year-old male?
HealthFlow> analyze patient data with symptoms: headache, nausea, photophobia
HealthFlow> status
HealthFlow> exit
```

### Batch Processing

Create a task file `my_tasks.jsonl`:
```json
{"task": "Explain the mechanism of action of ACE inhibitors"}
{"task": "What are the contraindications for MRI in patients with implants?"}
{"task": "Analyze drug interactions between warfarin and antibiotics"}
```

Run batch processing:
```bash
python run_healthflow.py --file my_tasks.jsonl
```

### Python API Usage

```python
import asyncio
from healthflow.cli import HealthFlowCLI

async def main():
    cli = HealthFlowCLI()
    await cli.initialize()
    
    result = await cli.execute_task(
        "Diagnose based on symptoms: fever, rash, joint pain",
        {"patient_age": 35, "patient_sex": "female"}
    )
    
    print(f"Result: {result['result']}")
    print(f"Success: {result['success']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🤖 Agent Roles

HealthFlow includes 7 specialized agents:

- **Coordinator**: Orchestrates multi-agent tasks and workflows
- **Medical Expert**: Provides specialized medical knowledge and clinical reasoning
- **Data Analyst**: Processes medical data and performs statistical analysis
- **Researcher**: Conducts literature reviews and evidence analysis
- **Diagnosis Specialist**: Excels at differential diagnosis and symptom analysis
- **Treatment Planner**: Develops treatment strategies and care plans
- **Code Executor**: Runs analysis code and creates visualizations

## 🛠️ Tool System

HealthFlow's ToolBank enables dynamic tool creation:

- **Automatic Tool Generation**: Agents create tools as needed
- **Tool Performance Tracking**: Success rates and usage statistics
- **Medical-Specific Tools**: Specialized for healthcare tasks
- **Code Generation**: Dynamic Python tool creation
- **Persistent Storage**: Tools saved for reuse across sessions

## 📊 Evaluation System

Comprehensive evaluation with medical-specific metrics:

- **Medical Safety**: Safety checks and contraindication detection
- **Evidence-Based Assessment**: Evaluation against medical literature
- **Completeness**: Comprehensive response analysis
- **Accuracy**: Medical knowledge accuracy assessment
- **Performance Tracking**: Success rates and improvement trends

## 🔬 Research Applications

HealthFlow is designed for cutting-edge healthcare AI research:

### Self-Evolution Features
- **Experience Accumulation**: Learn from task execution history
- **Prompt Evolution**: Automatically improve prompts based on performance
- **Tool Development**: Create and refine tools through usage
- **Memory Management**: Sophisticated short and long-term memory

### Medical AI Research
- **Diagnostic Accuracy**: Evaluate diagnostic reasoning capabilities
- **Treatment Planning**: Test treatment recommendation systems
- **Multi-Modal Analysis**: Process various medical data types
- **Safety Assessment**: Built-in medical safety evaluation

## 📁 Project Structure

```
HealthFlow/
├── healthflow/                 # Main package
│   ├── core/                  # Core agent system
│   │   ├── agent.py          # Multi-agent framework
│   │   ├── config.py         # Configuration management
│   │   ├── llm_provider.py   # LLM abstraction layer
│   │   ├── memory.py         # Memory management
│   │   └── rewards.py        # Reward functions
│   ├── tools/                # Tool management
│   │   └── toolbank.py       # Dynamic tool creation
│   ├── evaluation/           # Evaluation system
│   │   └── evaluator.py      # Medical evaluation metrics
│   └── cli.py               # Command line interface
├── baselines/               # Research baselines
├── scripts/                # Utility scripts
├── data/                   # Data storage (created at runtime)
├── pyproject.toml          # Dependencies and project config
├── .env.example           # Environment configuration template
└── README.md              # This file
```

## 🔧 Configuration Options

Environment variables for customization:

```bash
# LLM Settings
BASE_URL=https://api.openai.com/v1
API_KEY=your_key
MODEL_NAME=gpt-4-turbo-preview
MAX_TOKENS=4096
TEMPERATURE=0.7

# System Settings
MAX_ITERATIONS=10
MAX_AGENTS=7
MEMORY_WINDOW=1000
TOOL_TIMEOUT=30

# Storage
DATA_DIR=./data
MEMORY_DIR=./data/memory
TOOLS_DIR=./data/tools
EVALUATION_DIR=./data/evaluation

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/healthflow.log
```

## 🚀 Advanced Usage

### Custom Agent Creation

```python
from healthflow.core.agent import HealthFlowAgent, AgentRole
from healthflow.core.config import HealthFlowConfig

config = HealthFlowConfig.from_env()
agent = HealthFlowAgent(
    agent_id="custom_agent",
    role=AgentRole.MEDICAL_EXPERT,
    config=config
)
await agent.initialize()

result = await agent.execute_task("Custom medical task")
```

### Tool Development

```python
from healthflow.tools.toolbank import ToolBank

toolbank = ToolBank(Path("./data/tools"))
await toolbank.initialize()

tool_id = await toolbank.create_medical_analyzer_tool(
    name="Blood Pressure Analyzer",
    medical_domain="cardiology",
    analysis_type="blood_pressure_classification",
    implementation="def analyze_bp(systolic, diastolic): ...",
    input_schema={"systolic": "int", "diastolic": "int"},
    output_schema={"classification": "str", "risk": "str"}
)
```

## 🤝 Contributing

This project is designed for academic research. Key areas for contribution:

1. **Medical Knowledge Enhancement**: Improve medical reasoning capabilities
2. **Evaluation Metrics**: Develop better healthcare-specific evaluation
3. **Tool Development**: Create specialized medical analysis tools
4. **Safety Features**: Enhance medical safety checking
5. **Multi-Modal Support**: Add support for medical imaging, lab results

## 📝 License

This project is designed for academic and research purposes. Please respect medical AI safety guidelines and regulations when using for clinical applications.

## 🏥 Medical AI Safety

HealthFlow includes several safety features:

- **Safety Evaluation**: Built-in medical safety assessment
- **Evidence Requirements**: Emphasis on evidence-based recommendations
- **Uncertainty Handling**: Appropriate uncertainty acknowledgment
- **Compliance Checking**: Healthcare regulation compliance
- **Professional Guidance**: Recommendations to consult healthcare professionals

**Important**: This system is for research and educational purposes. Always consult qualified healthcare professionals for medical decisions.

## 📚 Research Citations

If you use HealthFlow in your research, please consider citing relevant medical AI and multi-agent system papers that inspired this work.

---

**HealthFlow**: Advancing Healthcare AI through Self-Evolution and Multi-Agent Collaboration 🚀🏥