import toml
from dataclasses import dataclass
from pathlib import Path

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
    evolution_dir: Path

    # Agent System Settings
    max_iterations: int
    tool_timeout: int
    max_react_rounds: int

    # Evaluation Settings
    success_threshold: float
    evaluation_timeout: int
    store_evaluation_traces: bool

    # Evolution Settings
    evolution_trigger_score: float
    evolution_frequency_tasks: int

    # Logging Settings
    log_level: str
    log_format: str
    log_file: Path

    @classmethod
    def from_toml(cls, config_path: str = "config.toml") -> 'HealthFlowConfig':
        """Loads configuration from a TOML file."""
        config_data = toml.load(config_path)

        # LLM Config
        active_llm = config_data["active_llm"]
        llm_config = config_data["llm"][active_llm]

        # Data, Agent, Eval, Evolution, Logging Configs
        data_cfg = config_data.get("data", {})
        agent_cfg = config_data.get("agent", {})
        eval_cfg = config_data.get("evaluation", {})
        evol_cfg = config_data.get("evolution", {})
        log_cfg = config_data.get("logging", {})

        # Create directories
        data_dir = Path(data_cfg.get("data_dir", "./data"))
        memory_dir = Path(data_cfg.get("memory_dir", data_dir / "memory"))
        tools_dir = Path(data_cfg.get("tools_dir", data_dir / "tools"))
        evolution_dir = Path(data_cfg.get("evolution_dir", data_dir / "evolution"))
        log_file = Path(log_cfg.get("log_file", "healthflow.log"))

        for path in [memory_dir, tools_dir, evolution_dir, log_file.parent]:
            path.mkdir(parents=True, exist_ok=True)

        return cls(
            active_llm=active_llm,
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            model_name=llm_config["model_name"],
            data_dir=data_dir,
            memory_dir=memory_dir,
            tools_dir=tools_dir,
            evolution_dir=evolution_dir,
            max_iterations=agent_cfg.get("max_iterations", 15),
            tool_timeout=agent_cfg.get("tool_timeout", 180),
            max_react_rounds=agent_cfg.get("max_react_rounds", 5),
            success_threshold=eval_cfg.get("success_threshold", 7.5),
            evaluation_timeout=eval_cfg.get("evaluation_timeout", 240),
            store_evaluation_traces=eval_cfg.get("store_evaluation_traces", True),
            evolution_trigger_score=evol_cfg.get("evolution_trigger_score", 7.0),
            evolution_frequency_tasks=evol_cfg.get("evolution_frequency_tasks", 3),
            log_level=log_cfg.get("log_level", "INFO"),
            log_format=log_cfg.get("log_format", "%(asctime)s - %(levelname)s - %(message)s"),
            log_file=log_file,
        )