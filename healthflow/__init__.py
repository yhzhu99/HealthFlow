"""
HealthFlow: Self-Evolving Healthcare AI Agent System

A comprehensive multi-agent healthcare AI system featuring:
- Self-evolving agents with experience accumulation
- Dynamic tool creation and management
- Medical-specific evaluation and safety checks
- Multi-provider LLM support
- File-based persistence (no databases required)
"""

__version__ = "1.0.0"
__author__ = "HealthFlow Team"
__email__ = "healthflow@example.com"

# Core imports
from .core.config import HealthFlowConfig
from .core.agent import HealthFlowAgent, AgentRole
from .core.llm_provider import LLMProvider, create_llm_provider
from .core.memory import MemoryManager, MemoryEntry, MemoryType
from .core.rewards import calculate_mi_reward, calculate_final_reward

# Tool system
from .tools.toolbank import ToolBank, Tool, ToolType

# Evaluation system
from .evaluation.evaluator import TaskEvaluator

# CLI interface
from .cli import HealthFlowCLI

__all__ = [
    # Core
    "HealthFlowConfig",
    "HealthFlowAgent",
    "AgentRole",
    "LLMProvider",
    "create_llm_provider",
    "MemoryManager",
    "MemoryEntry", 
    "MemoryType",
    "calculate_mi_reward",
    "calculate_final_reward",
    
    # Tools
    "ToolBank",
    "Tool",
    "ToolType",
    
    # Evaluation
    "TaskEvaluator",
    
    # CLI
    "HealthFlowCLI"
]