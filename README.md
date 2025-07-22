# HealthFlow 🏥🤖

**Self-Evolving LLM Agent for Healthcare with Experience Accumulation and Sensitive Data Protection**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

HealthFlow is an advanced self-evolving multi-agent system designed specifically for healthcare applications. It combines cutting-edge LLM capabilities with sophisticated experience accumulation, memory management, and sensitive data protection to create continuously improving healthcare AI agents.

## 🌟 Key Features

### 🔄 Self-Evolution & Experience Accumulation
- **Prompt Evolution**: Continuously improves reasoning strategies based on past successes and failures
- **Experience-based Learning**: Accumulates insights from every task execution
- **Pattern Recognition**: Automatically identifies successful approaches and failure patterns
- **Performance Optimization**: Evolves tool usage and collaboration strategies over time

### 🔒 Healthcare Data Protection
- **HIPAA Compliance**: Built-in sensitive data detection and protection
- **Smart Anonymization**: Multi-level data anonymization (low, medium, high)
- **Schema-only Transmission**: Option to transmit data structure without actual values
- **Mock Data Generation**: Creates realistic test data while preserving structure
- **Audit Trails**: Complete logging of all data protection activities

### 🧠 Advanced Memory Management
- **Multi-level Memory**: Short-term, long-term, episodic, and semantic memory systems
- **Patient-specific Memory**: Privacy-preserving patient context retention
- **Memory Consolidation**: Automatic promotion of important memories
- **Contextual Retrieval**: Intelligent memory search based on task relevance

### 🛠️ Dynamic Tool Creation (MCP-driven)
- **Automatic Tool Generation**: Creates specialized healthcare tools on-demand
- **MCP Integration**: Model Context Protocol for standardized tool interfaces
- **Tool Validation**: Automatic testing and validation of generated tools
- **Tool Evolution**: Performance-based tool improvement and versioning

### 🤝 Multi-Agent Collaboration
- **Role-based Agents**: Specialized agents for different healthcare domains
- **Intelligent Collaboration**: Dynamic role assignment based on task requirements
- **Shared Context**: Secure information sharing between collaborating agents
- **Collective Learning**: Shared experience across agent teams

### 📊 Comprehensive Evaluation
- **Multi-dimensional Assessment**: Accuracy, safety, compliance, efficiency metrics
- **Clinical Relevance Scoring**: Healthcare-specific quality measures
- **Safety Monitoring**: Automatic detection of potentially harmful recommendations
- **Continuous Improvement**: Performance-based agent optimization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/healthflow/healthflow.git
cd healthflow

# Install with pip
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Environment Setup

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Basic Usage

```python
import asyncio
from healthflow.core.agent import HealthFlowAgent

async def main():
    # Create a diagnostic specialist agent
    agent = HealthFlowAgent(
        agent_id="diagnostic_specialist",
        openai_api_key="your-openai-api-key",
        specialized_domains=["diagnosis", "symptom_analysis"],
        config={"max_reasoning_steps": 10}
    )
    
    # Execute a diagnostic task
    result = await agent.execute_task(
        task_description="Analyze patient with fever, cough, and shortness of breath",
        task_type="diagnosis",
        data={
            "symptoms": ["fever", "cough", "shortness_of_breath"],
            "duration": "3 days",
            "severity": "moderate"
        }
    )
    
    print(f"Diagnosis completed: {result.success}")
    print(f"Response: {result.output}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Command Line Interface

```bash
# Create a new agent
healthflow create --agent-id my_agent --domains diagnosis analysis

# Execute a task
healthflow execute --agent-id my_agent --task "Analyze lab results" --type analysis

# Run demonstration
healthflow demo --scenario all

# View agent statistics
healthflow stats --agent-id my_agent
```

## 📋 Use Cases

### 🔍 Clinical Diagnosis Support
```python
diagnostic_agent = HealthFlowAgent(
    agent_id="diagnostician",
    specialized_domains=["differential_diagnosis", "symptom_analysis"]
)

result = await diagnostic_agent.execute_task(
    task_description="Patient presents with chest pain, analyze potential causes",
    task_type="diagnosis",
    context={"urgency": "high", "patient_age": 55}
)
```

### 💊 Drug Interaction Analysis
```python
pharma_agent = HealthFlowAgent(
    agent_id="pharmacist",
    specialized_domains=["drug_interactions", "medication_management"]
)

result = await pharma_agent.execute_task(
    task_description="Check interactions between warfarin, aspirin, and lisinopril",
    task_type="drug_interaction",
    data={"medications": ["warfarin", "aspirin", "lisinopril"]}
)
```

### 📊 Medical Data Analysis
```python
analyst_agent = HealthFlowAgent(
    agent_id="data_analyst", 
    specialized_domains=["statistical_analysis", "pattern_recognition"]
)

result = await analyst_agent.execute_task(
    task_description="Analyze EHR data for readmission risk factors",
    task_type="data_analysis",
    data=protected_ehr_data  # Automatically protected
)
```

### 🏥 Treatment Planning
```python
clinical_agent = HealthFlowAgent(
    agent_id="clinician",
    specialized_domains=["treatment_planning", "clinical_guidelines"]
)

result = await clinical_agent.execute_task(
    task_description="Develop treatment plan for Type 2 diabetes with hypertension",
    task_type="treatment_planning",
    data={"conditions": ["diabetes_t2", "hypertension"], "contraindications": []}
)
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     HealthFlow Agent                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Experience      │  │ Memory       │  │ Data Protection │   │
│  │ Accumulator     │  │ Manager      │  │ Layer           │   │
│  │ - Prompt Evol.  │  │ - Multi-level│  │ - Anonymization │   │
│  │ - Pattern Learn.│  │ - Contextual │  │ - Schema-only   │   │
│  │ - Performance   │  │ - Patient    │  │ - Audit Trail   │   │
│  └─────────────────┘  └──────────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ ToolBank        │  │ Evaluator    │  │ Collaboration   │   │
│  │ - MCP-driven    │  │ - Multi-dim. │  │ - Role Assignment│   │
│  │ - Auto Creation │  │ - Safety     │  │ - Shared Context │   │
│  │ - Validation    │  │ - Compliance │  │ - Collective    │   │
│  └─────────────────┘  └──────────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Core LLM Integration                       │
│                  (OpenAI GPT-4 + Custom Logic)                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Advanced Configuration

### Agent Configuration
```python
config = {
    "max_reasoning_steps": 15,
    "confidence_threshold": 0.8,
    "safety_checks": True,
    "memory_consolidation_hours": 24,
    "tool_creation_enabled": True,
    "collaboration_mode": "dynamic"
}

agent = HealthFlowAgent(
    agent_id="advanced_agent",
    openai_api_key=api_key,
    specialized_domains=["cardiology", "emergency_medicine"],
    config=config
)
```

### Data Protection Configuration
```python
from healthflow.core.security import DataProtector, ProtectionConfig

protector = DataProtector(ProtectionConfig(
    anonymization_level="high",
    preserve_structure=True,
    generate_mock_data=True,
    schema_only_mode=False
))

protected_data = await protector.protect_data(sensitive_patient_data)
```

### Memory Management
```python
from healthflow.core.memory import MemoryQuery

# Store important clinical insight
memory_id = await agent.memory_manager.store_memory(
    content={"insight": "Early antibiotic therapy improves outcomes in sepsis"},
    memory_type="semantic",
    importance_score=0.9,
    tags=["sepsis", "antibiotics", "clinical_protocol"],
    privacy_level="low"  # General medical knowledge
)

# Retrieve relevant memories
memories = await agent.memory_manager.retrieve_memories(
    MemoryQuery(
        query_text="sepsis treatment protocols",
        memory_types=["semantic", "episodic"],
        min_importance=0.7
    )
)
```

## 📊 Evaluation & Monitoring

HealthFlow provides comprehensive evaluation capabilities:

```python
# Get agent performance statistics
stats = agent.get_agent_statistics()
print(f"Success rate: {stats['task_history_summary']['successful_tasks']}")
print(f"Experience level: {stats['agent_state']['experience_level']}")

# Get evaluation metrics
eval_stats = agent.evaluator.get_evaluation_statistics()
print(f"Average performance: {eval_stats['average_score']}")
print(f"Performance trend: {eval_stats['performance_trend']}")

# Get memory system status
memory_stats = agent.memory_manager.get_memory_statistics()
print(f"Total memories: {memory_stats['total_memories']}")
print(f"Patient-specific memories: {memory_stats['unique_patients']}")
```

## 🛡️ Security & Compliance

HealthFlow is designed with healthcare security in mind:

- **HIPAA Compliance**: Automatic PII/PHI detection and protection
- **Data Anonymization**: Multiple levels of data de-identification
- **Audit Logging**: Complete trail of all data processing activities
- **Access Controls**: Role-based access to sensitive functionalities
- **Secure Communication**: Encrypted agent-to-agent communication

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m security      # Security-related tests
pytest -m memory        # Memory management tests

# Run with coverage
pytest --cov=healthflow --cov-report=html
```

## 📈 Performance Benchmarking

HealthFlow includes built-in benchmarking capabilities:

```python
# Run performance benchmark
from healthflow.evaluation.benchmark import HealthcareBenchmark

benchmark = HealthcareBenchmark()
results = await benchmark.run_comprehensive_evaluation(agent)

print(f"Diagnostic accuracy: {results['diagnostic_accuracy']}")
print(f"Treatment planning score: {results['treatment_planning']}")
print(f"Safety compliance: {results['safety_score']}")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run tests: `pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by recent advances in healthcare AI and multi-agent systems
- Built on top of OpenAI's powerful language models
- Incorporates best practices from medical informatics and AI safety research

## 📞 Support

- **Documentation**: [https://healthflow.readthedocs.io](https://healthflow.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/healthflow/healthflow/issues)
- **Discussions**: [GitHub Discussions](https://github.com/healthflow/healthflow/discussions)
- **Email**: support@healthflow.ai

---

**⚠️ Important**: HealthFlow is designed for research and development purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 📊 Data Curation (Research Context)

The project also includes comprehensive healthcare AI research data curation:

### Data Extraction Pipeline
- **Conference Papers**: KDD, WWW, ICLR, ICML, NeurIPS, AAAI, IJCAI
- **Journal Papers**: Nature series, NEJM AI
- **Extraction Methods**: Web scraping, BibTeX parsing, XML processing
- **Time Range**: 2020-2025 publications
- **Focus**: AI for Healthcare applications

### Directory Structure
- `title_extract/`: Conference/journal-specific extraction scripts
- `filter_paper/`: Healthcare AI paper filtering and classification  
- `extract_task/`: Task extraction from healthcare AI papers
- `baselines/`: Reference implementations (Biomni, Alita, OpenEvolve)
