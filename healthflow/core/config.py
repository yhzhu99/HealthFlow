import sys
from pydantic import BaseModel, Field
from pathlib import Path
import toml
from loguru import logger

class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    api_key: str = Field(..., description="API key for the LLM provider.")
    base_url: str = Field(..., description="Base URL for the LLM API.")
    model_name: str = Field(..., description="Model name to use.")
    timeout: int = Field(180, description="Request timeout in seconds.")

class SystemConfig(BaseModel):
    max_retries: int = Field(2, description="Maximum number of retries for a failing task.")
    workspace_dir: str = Field("workspace", description="Directory for task artifacts.")
    shell: str = Field("/usr/bin/zsh", description="Shell to use for subprocess execution.")

class EvaluationConfig(BaseModel):
    success_threshold: float = Field(8.0, description="Score (out of 10) to consider a task successful.")

class LoggingConfig(BaseModel):
    log_level: str = Field("INFO", description="Logging level.")
    log_file: str = Field("healthflow.log", description="Path to the log file.")

class HealthFlowConfig(BaseModel):
    """The main configuration object, assembled dynamically from the config file."""
    active_llm_name: str
    llm: LLMProviderConfig
    system: SystemConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

def get_config(config_path: Path, active_llm: str) -> HealthFlowConfig:
    """
    Loads configuration from a TOML file, validates it, selects the active LLM's
    settings, and returns a unified HealthFlowConfig object.
    """
    if not config_path.exists():
        example_path = Path("config.toml.example")
        if example_path.exists():
             raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please copy '{example_path}' to '{config_path}' and fill in your API key.")
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'.")

    try:
        config_data = toml.load(config_path)

        if not active_llm:
            raise ValueError("'active_llm' parameter is required")

        active_llm_config_data = config_data.get("llm", {}).get(active_llm)
        if not active_llm_config_data:
            raise ValueError(f"Configuration for LLM '{active_llm}' not found under the '[llm]' section.")

        llm_provider_config = LLMProviderConfig(**active_llm_config_data)
        system_config = SystemConfig(**config_data.get("system", {}))
        evaluation_config = EvaluationConfig(**config_data.get("evaluation", {}))
        logging_config = LoggingConfig(**config_data.get("logging", {}))

        config = HealthFlowConfig(
            active_llm_name=active_llm,
            llm=llm_provider_config,
            system=system_config,
            evaluation=evaluation_config,
            logging=logging_config,
        )
            
        logger.info(f"Configuration loaded successfully. Active LLM for reasoning: '{active_llm}'")
        return config

    except Exception as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}")
        raise ValueError(f"Error parsing configuration file '{config_path}': {e}") from e

def setup_logging(config: HealthFlowConfig):
    """Configures the Loguru logger based on the loaded configuration."""
    logger.remove()
    # Console logger
    logger.add(
        sys.stderr,
        level=config.logging.log_level.upper()
    )
    # File logger
    logger.add(
        config.logging.log_file,
        level=config.logging.log_level.upper(),
        rotation="10 MB",
        retention="7 days",
        enqueue=True, # Make logging non-blocking
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    logger.info("Logger configured.")