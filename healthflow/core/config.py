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
    workspace_dir: str = Field("workspace", description="Directory for task artifacts and experience DB.")

class EvaluationConfig(BaseModel):
    success_threshold: float = Field(7.5, description="Score (out of 10) to consider a task successful.")

class LoggingConfig(BaseModel):
    log_level: str = Field("INFO", description="Logging level.")
    log_file: str = Field("healthflow.log", description="Path to the log file.")

class HealthFlowConfig(BaseModel):
    """The main configuration object, assembled dynamically."""
    active_llm_name: str
    llm: LLMProviderConfig
    system: SystemConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

_config: HealthFlowConfig = None

def get_config(config_path: Path = Path("config.toml")) -> HealthFlowConfig:
    """
    Loads configuration, selects the active LLM, and returns a unified config object.
    """
    global _config
    if _config is not None:
        return _config

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please create it from 'config.toml.example'.")

    try:
        config_data = toml.load(config_path)

        # 1. Get the name of the active LLM
        active_llm_name = config_data.get("active_llm")
        if not active_llm_name:
            raise ValueError("'active_llm' key is missing in config.toml")

        # 2. Get the configuration for the active LLM
        active_llm_config_data = config_data.get("llm", {}).get(active_llm_name)
        if not active_llm_config_data:
            raise ValueError(f"Configuration for active_llm '{active_llm_name}' not found under the '[llm]' section.")

        # 3. Create the Pydantic models
        llm_provider_config = LLMProviderConfig(**active_llm_config_data)
        system_config = SystemConfig(**config_data.get("system", {}))
        evaluation_config = EvaluationConfig(**config_data.get("evaluation", {}))
        logging_config = LoggingConfig(**config_data.get("logging", {}))

        # 4. Assemble the final configuration object
        _config = HealthFlowConfig(
            active_llm_name=active_llm_name,
            llm=llm_provider_config,
            system=system_config,
            evaluation=evaluation_config,
            logging=logging_config,
        )
        logger.info(f"Configuration loaded successfully. Active LLM: '{active_llm_name}'")
        return _config

    except Exception as e:
        raise ValueError(f"Error parsing configuration file '{config_path}': {e}") from e

def setup_logging(config: HealthFlowConfig):
    """Configures the Loguru logger based on the loaded configuration."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=config.logging.log_level.upper()
    )
    logger.add(
        config.logging.log_file,
        level=config.logging.log_level.upper(),
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    logger.info("Logger configured.")