import toml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class HealthFlowConfig:
    """Configuration settings for HealthFlow, loaded from a TOML file."""

    # LLM Configuration
    active_llm: str
    base_url: str
    api_key: str
    model_name: str

    # Data Storage Paths
    data_dir: Path
    memory_dir: Path
    tools_dir: Path
    evaluation_dir: Path

    # Agent System Settings
    max_iterations: int
    tool_timeout: int

    # Evaluation Settings
    success_threshold: float
    evaluation_timeout: int
    store_evaluation_traces: bool

    # Logging Settings
    log_level: str
    log_format: str
    log_file: Path

    @classmethod
    def from_toml(cls, config_path: str = "config.toml") -> 'HealthFlowConfig':
        """Loads configuration from a TOML file."""
        try:
            config_data = toml.load(config_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}. Please create it.")

        # LLM Config
        active_llm = config_data.get("active_llm", "deepseek-v3")
        llm_config = config_data.get("llm", {}).get(active_llm, {})
        if not all(k in llm_config for k in ["base_url", "api_key", "model_name"]):
            raise ValueError(f"Missing required keys for LLM provider '{active_llm}' in config.")

        # Data Config
        data_cfg = config_data.get("data", {})
        data_dir = Path(data_cfg.get("data_dir", "./data"))

        # Agent Config
        agent_cfg = config_data.get("agent", {})

        # Evaluation Config
        eval_cfg = config_data.get("evaluation", {})

        # Logging Config
        log_cfg = config_data.get("logging", {})

        # Create directories
        memory_dir = Path(data_cfg.get("memory_dir", data_dir / "memory"))
        tools_dir = Path(data_cfg.get("tools_dir", data_dir / "tools"))
        evaluation_dir = Path(data_cfg.get("evaluation_dir", data_dir / "evaluation"))
        for path in [memory_dir, tools_dir, evaluation_dir]:
            path.mkdir(parents=True, exist_ok=True)

        log_file = Path(log_cfg.get("log_file", "healthflow.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        return cls(
            active_llm=active_llm,
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            model_name=llm_config["model_name"],
            data_dir=data_dir,
            memory_dir=memory_dir,
            tools_dir=tools_dir,
            evaluation_dir=evaluation_dir,
            max_iterations=agent_cfg.get("max_iterations", 10),
            tool_timeout=agent_cfg.get("tool_timeout", 120),
            success_threshold=eval_cfg.get("success_threshold", 7.5),
            evaluation_timeout=eval_cfg.get("evaluation_timeout", 180),
            store_evaluation_traces=eval_cfg.get("store_evaluation_traces", True),
            log_level=log_cfg.get("log_level", "INFO"),
            log_format=log_cfg.get("log_format", "%(asctime)s - %(levelname)s - %(message)s"),
            log_file=log_file,
        )