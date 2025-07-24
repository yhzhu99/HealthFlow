"""
Evolution Configuration System

Manages the evolving parameters of the HealthFlow system in external configuration files.
This allows the system to improve over time while maintaining transparency about what changes.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PromptEvolution:
    """Tracks the evolution of a single prompt."""
    version: str
    content: str
    score: float
    created_at: str
    feedback: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptEvolution':
        return cls(**data)


@dataclass
class AgentStrategy:
    """Defines collaboration and decision-making strategies."""
    name: str
    description: str
    success_rate: float
    usage_count: int
    last_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentStrategy':
        return cls(**data)


@dataclass
class ToolPerformance:
    """Tracks tool usage and performance metrics."""
    tool_name: str
    success_rate: float
    avg_execution_time: float
    usage_count: int
    error_patterns: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolPerformance':
        return cls(**data)


class EvolutionConfig:
    """
    Manages the evolution configuration for HealthFlow.
    
    Stores and tracks improvements to:
    - Prompt templates
    - Agent collaboration strategies  
    - Tool performance metrics
    - System parameters
    """
    
    def __init__(self, config_dir: Path):
        """Initialize with configuration directory."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.prompts_file = self.config_dir / "evolved_prompts.json"
        self.strategies_file = self.config_dir / "collaboration_strategies.json"
        self.tools_file = self.config_dir / "tool_performance.json"
        self.system_file = self.config_dir / "system_params.json"
        
        # Load existing configurations
        self.prompts: Dict[str, List[PromptEvolution]] = self._load_prompts()
        self.strategies: Dict[str, AgentStrategy] = self._load_strategies()
        self.tools: Dict[str, ToolPerformance] = self._load_tools()
        self.system_params: Dict[str, Any] = self._load_system_params()
    
    def _load_prompts(self) -> Dict[str, List[PromptEvolution]]:
        """Load evolved prompts from file."""
        if not self.prompts_file.exists():
            return {"orchestrator": [], "expert": [], "analyst": []}
        
        try:
            with open(self.prompts_file, 'r') as f:
                data = json.load(f)
            
            result = {}
            for role, prompt_list in data.items():
                result[role] = [PromptEvolution.from_dict(p) for p in prompt_list]
            
            return result
        except Exception as e:
            logger.warning(f"Failed to load prompts: {e}")
            return {"orchestrator": [], "expert": [], "analyst": []}
    
    def _load_strategies(self) -> Dict[str, AgentStrategy]:
        """Load collaboration strategies from file."""
        if not self.strategies_file.exists():
            return self._get_default_strategies()
        
        try:
            with open(self.strategies_file, 'r') as f:
                data = json.load(f)
            
            return {name: AgentStrategy.from_dict(strategy_data) 
                   for name, strategy_data in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load strategies: {e}")
            return self._get_default_strategies()
    
    def _load_tools(self) -> Dict[str, ToolPerformance]:
        """Load tool performance metrics from file."""
        if not self.tools_file.exists():
            return {}
        
        try:
            with open(self.tools_file, 'r') as f:
                data = json.load(f)
            
            return {name: ToolPerformance.from_dict(tool_data) 
                   for name, tool_data in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load tool performance: {e}")
            return {}
    
    def _load_system_params(self) -> Dict[str, Any]:
        """Load system parameters from file."""
        if not self.system_file.exists():
            return self._get_default_system_params()
        
        try:
            with open(self.system_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load system params: {e}")
            return self._get_default_system_params()
    
    def _get_default_strategies(self) -> Dict[str, AgentStrategy]:
        """Get default collaboration strategies."""
        return {
            "medical_routing": AgentStrategy(
                name="medical_routing",
                description="Route medical questions to expert agent",
                success_rate=0.85,
                usage_count=0,
                last_used=datetime.now().isoformat()
            ),
            "computational_routing": AgentStrategy(
                name="computational_routing", 
                description="Route computational tasks to analyst agent",
                success_rate=0.90,
                usage_count=0,
                last_used=datetime.now().isoformat()
            ),
            "collaborative_approach": AgentStrategy(
                name="collaborative_approach",
                description="Use both agents for complex healthcare AI tasks",
                success_rate=0.95,
                usage_count=0,
                last_used=datetime.now().isoformat()
            )
        }
    
    def _get_default_system_params(self) -> Dict[str, Any]:
        """Get default system parameters."""
        return {
            "max_react_rounds": 3,
            "success_threshold": 7.5,
            "evolution_frequency": 5,  # Evolve after every 5 tasks
            "prompt_version_limit": 10,  # Keep max 10 versions per role
            "tool_timeout": 120,
            "collaboration_timeout": 300
        }
    
    def save_all(self):
        """Save all configurations to files."""
        self._save_prompts()
        self._save_strategies()
        self._save_tools()
        self._save_system_params()
    
    def _save_prompts(self):
        """Save evolved prompts to file."""
        try:
            data = {}
            for role, prompt_list in self.prompts.items():
                data[role] = [p.to_dict() for p in prompt_list]
            
            with open(self.prompts_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save prompts: {e}")
    
    def _save_strategies(self):
        """Save collaboration strategies to file."""
        try:
            data = {name: strategy.to_dict() for name, strategy in self.strategies.items()}
            
            with open(self.strategies_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save strategies: {e}")
    
    def _save_tools(self):
        """Save tool performance metrics to file."""
        try:
            data = {name: tool.to_dict() for name, tool in self.tools.items()}
            
            with open(self.tools_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tool performance: {e}")
    
    def _save_system_params(self):
        """Save system parameters to file."""
        try:
            with open(self.system_file, 'w') as f:
                json.dump(self.system_params, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save system params: {e}")
    
    def add_prompt_evolution(self, role: str, new_prompt: str, score: float, feedback: str = ""):
        """Add a new evolved prompt for a role."""
        if role not in self.prompts:
            self.prompts[role] = []
        
        version = f"v{len(self.prompts[role]) + 1}"
        evolution = PromptEvolution(
            version=version,
            content=new_prompt,
            score=score,
            created_at=datetime.now().isoformat(),
            feedback=feedback
        )
        
        self.prompts[role].append(evolution)
        
        # Keep only the best N versions
        max_versions = self.system_params.get("prompt_version_limit", 10)
        if len(self.prompts[role]) > max_versions:
            # Sort by score and keep the best ones
            self.prompts[role].sort(key=lambda p: p.score, reverse=True)
            self.prompts[role] = self.prompts[role][:max_versions]
        
        self._save_prompts()
        logger.info(f"Added prompt evolution for {role}: {version} (score: {score})")
    
    def get_best_prompt(self, role: str) -> tuple[str, float]:
        """Get the best performing prompt for a role."""
        if role not in self.prompts or not self.prompts[role]:
            # Return simple default
            from .simple_prompts import get_simple_prompt
            return get_simple_prompt(role), 0.0
        
        # Get the highest scoring prompt
        best_prompt = max(self.prompts[role], key=lambda p: p.score)
        return best_prompt.content, best_prompt.score
    
    def update_strategy_performance(self, strategy_name: str, success: bool):
        """Update performance metrics for a collaboration strategy."""
        if strategy_name not in self.strategies:
            return
        
        strategy = self.strategies[strategy_name]
        strategy.usage_count += 1
        strategy.last_used = datetime.now().isoformat()
        
        # Update success rate using exponential moving average
        alpha = 0.1  # Learning rate
        new_success = 1.0 if success else 0.0
        strategy.success_rate = (1 - alpha) * strategy.success_rate + alpha * new_success
        
        self._save_strategies()
    
    def update_tool_performance(self, tool_name: str, success: bool, execution_time: float, error: str = ""):
        """Update performance metrics for a tool."""
        if tool_name not in self.tools:
            self.tools[tool_name] = ToolPerformance(
                tool_name=tool_name,
                success_rate=0.0,
                avg_execution_time=0.0,
                usage_count=0,
                error_patterns=[]
            )
        
        tool = self.tools[tool_name]
        tool.usage_count += 1
        
        # Update success rate
        alpha = 0.1
        new_success = 1.0 if success else 0.0
        tool.success_rate = (1 - alpha) * tool.success_rate + alpha * new_success
        
        # Update execution time
        tool.avg_execution_time = ((tool.avg_execution_time * (tool.usage_count - 1)) + execution_time) / tool.usage_count
        
        # Track error patterns
        if error and len(tool.error_patterns) < 50:  # Limit error pattern storage
            tool.error_patterns.append(error[:100])  # Truncate long errors
        
        self._save_tools()
    
    def get_system_param(self, param_name: str, default: Any = None) -> Any:
        """Get a system parameter value."""
        return self.system_params.get(param_name, default)
    
    def update_system_param(self, param_name: str, value: Any):
        """Update a system parameter."""
        self.system_params[param_name] = value
        self._save_system_params()
        logger.info(f"Updated system parameter {param_name} = {value}")
    
    def should_evolve(self, task_count: int) -> bool:
        """Check if the system should evolve based on task count."""
        frequency = self.get_system_param("evolution_frequency", 5)
        return task_count % frequency == 0