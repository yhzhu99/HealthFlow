"""
HealthFlow Configuration Management
Handles TOML configuration file parsing and settings validation.
"""

import toml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class HealthFlowConfig:
    """Configuration settings for HealthFlow"""

    # LLM Configuration
    base_url: str
    api_key: str
    model_name: str

    # Data Storage Configuration
    data_dir: Path
    memory_dir: Path
    tools_dir: Path
    cache_dir: Path
    evaluation_dir: Path

    # Agent Configuration
    max_iterations: int
    max_agents: int
    memory_window: int
    tool_timeout: int

    # Logging Configuration
    log_level: str
    log_file: Optional[Path]

    @classmethod
    def from_toml(cls, config_path: str = "config.toml") -> 'HealthFlowConfig':
        """Load configuration from a TOML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = toml.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}. Please create it from config.toml.example.")

        active_llm = config_data.get("active_llm", "openai")
        llm_config = config_data.get("llm", {}).get(active_llm, {})

        if not all(key in llm_config for key in ["base_url", "api_key", "model_name"]):
            raise ValueError(f"Missing required keys for active LLM provider '{active_llm}' in {config_path}")

        data_config = config_data.get("data", {})
        agent_config = config_data.get("agent", {})
        logging_config = config_data.get("logging", {})

        # Data Storage Configuration
        data_dir = Path(data_config.get('data_dir', './data'))
        memory_dir = Path(data_config.get('memory_dir', './data/memory'))
        tools_dir = Path(data_config.get('tools_dir', './data/tools'))
        cache_dir = Path(data_config.get('cache_dir', './data/cache'))
        evaluation_dir = Path(data_config.get('evaluation_dir', './data/evaluation'))

        # Logging Configuration
        log_file_str = logging_config.get('log_file')
        log_file = Path(log_file_str) if log_file_str else None

        # Create directories
        for dir_path in [data_dir, memory_dir, tools_dir, cache_dir, evaluation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)

        return cls(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            model_name=llm_config["model_name"],
            data_dir=data_dir,
            memory_dir=memory_dir,
            tools_dir=tools_dir,
            cache_dir=cache_dir,
            evaluation_dir=evaluation_dir,
            max_iterations=agent_config.get('max_iterations', 10),
            max_agents=agent_config.get('max_agents', 5),
            memory_window=agent_config.get('memory_window', 1000),
            tool_timeout=agent_config.get('tool_timeout', 30),
            log_level=logging_config.get('log_level', 'INFO'),
            log_file=log_file
        )

    def validate(self) -> bool:
        """Validate configuration settings"""
        if not self.api_key or "YOUR_" in self.api_key:
            raise ValueError("API_KEY must be provided and should not be a placeholder.")

        if self.max_iterations <= 0:
            raise ValueError("MAX_ITERATIONS must be positive")

        return True