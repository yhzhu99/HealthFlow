import sys
from pathlib import Path
from typing import Dict, List, Literal

import toml
from loguru import logger
from pydantic import BaseModel, Field, model_validator


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    api_key: str = Field(..., description="API key for the LLM provider.")
    base_url: str = Field(..., description="Base URL for the LLM API.")
    model_name: str = Field(..., description="Model name to use.")
    timeout: int = Field(180, description="Request timeout in seconds.")


class BackendCLIConfig(BaseModel):
    """CLI settings for a single executor backend."""

    binary: str
    args: List[str] = Field(default_factory=list)
    prompt_mode: Literal["append", "stdin"] = "append"
    timeout_seconds: int = 900


class SystemConfig(BaseModel):
    max_retries: int = Field(2, description="Maximum number of retries for a failing task.")
    workspace_dir: str = Field("workspace", description="Directory for task artifacts.")
    shell: str = Field("/usr/bin/zsh", description="Shell to use for subprocess execution.")


def default_executor_backends() -> Dict[str, BackendCLIConfig]:
    return {
        "claude_code": BackendCLIConfig(
            binary="claude",
            args=["--dangerously-skip-permissions", "--print"],
            prompt_mode="append",
        ),
        "opencode": BackendCLIConfig(
            binary="opencode",
            args=[],
            prompt_mode="append",
        ),
        "pi": BackendCLIConfig(
            binary="pi",
            args=[],
            prompt_mode="append",
        ),
    }


class ExecutorConfig(BaseModel):
    active_backend: str = Field("claude_code", description="Executor backend to use for task execution.")
    prompt_file_name: str = Field("executor_prompt.md", description="Prompt file stored inside each workspace.")
    backends: Dict[str, BackendCLIConfig] = Field(
        default_factory=default_executor_backends
    )

    @model_validator(mode="after")
    def ensure_active_backend_exists(self):
        if not self.backends:
            self.backends = default_executor_backends()
        if self.active_backend not in self.backends:
            raise ValueError(f"Active backend '{self.active_backend}' is not defined under [executor.backends].")
        return self


class MemoryConfig(BaseModel):
    mode: Literal["accumulate_eval", "frozen_train", "reset"] = "accumulate_eval"
    retrieve_k: int = 6
    strategy_k: int = 3
    failure_k: int = 2
    dataset_k: int = 1
    writeback_on_failure: bool = True


class EHRConfig(BaseModel):
    enable_profile: bool = True
    enable_risk_checks: bool = True
    max_preview_rows: int = 3


class VerificationConfig(BaseModel):
    require_verifier_pass: bool = True
    required_report_sections: List[str] = Field(
        default_factory=lambda: [
            "Task Summary",
            "Data Profile",
            "Method",
            "Verification",
            "Limitations",
        ]
    )


class EvaluationConfig(BaseModel):
    success_threshold: float = Field(8.0, description="Score (out of 10) to consider a task successful.")


class LoggingConfig(BaseModel):
    log_level: str = Field("INFO", description="Logging level.")
    log_file: str = Field("healthflow.log", description="Path to the log file.")


class HealthFlowConfig(BaseModel):
    """The main configuration object, assembled dynamically from the config file."""

    active_llm_name: str
    active_executor_name: str
    llm: LLMProviderConfig
    system: SystemConfig
    executor: ExecutorConfig
    memory: MemoryConfig
    ehr: EHRConfig
    verification: VerificationConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

    @property
    def active_executor(self) -> BackendCLIConfig:
        return self.executor.backends[self.active_executor_name]


def get_config(config_path: Path, active_llm: str, active_executor: str | None = None) -> HealthFlowConfig:
    """
    Loads configuration from a TOML file, validates it, selects the active LLM and
    executor settings, and returns a unified HealthFlowConfig object.
    """
    if not config_path.exists():
        example_path = Path("config.toml.example")
        if example_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at '{config_path}'. "
                f"Please copy '{example_path}' to '{config_path}' and fill in your API key."
            )
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'.")

    try:
        config_data = toml.load(config_path)

        if not active_llm:
            raise ValueError("'active_llm' parameter is required")

        active_llm_config_data = config_data.get("llm", {}).get(active_llm)
        if not active_llm_config_data:
            raise ValueError(f"Configuration for LLM '{active_llm}' not found under the '[llm]' section.")

        executor_section = config_data.get("executor", {})
        executor_backends = executor_section.get("backends")
        executor_config = ExecutorConfig(
            active_backend=active_executor or executor_section.get("active_backend", "claude_code"),
            prompt_file_name=executor_section.get("prompt_file_name", "executor_prompt.md"),
            backends=executor_backends if executor_backends else default_executor_backends(),
        )

        config = HealthFlowConfig(
            active_llm_name=active_llm,
            active_executor_name=executor_config.active_backend,
            llm=LLMProviderConfig(**active_llm_config_data),
            system=SystemConfig(**config_data.get("system", {})),
            executor=executor_config,
            memory=MemoryConfig(**config_data.get("memory", {})),
            ehr=EHRConfig(**config_data.get("ehr", {})),
            verification=VerificationConfig(**config_data.get("verification", {})),
            evaluation=EvaluationConfig(**config_data.get("evaluation", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
        )

        logger.info(
            "Configuration loaded successfully. Active reasoning model: '{}'. Active executor: '{}'",
            active_llm,
            config.active_executor_name,
        )
        return config
    except Exception as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}")
        raise ValueError(f"Error parsing configuration file '{config_path}': {e}") from e


def setup_logging(config: HealthFlowConfig):
    """Configures the Loguru logger based on the loaded configuration."""
    logger.remove()
    logger.add(sys.stderr, level=config.logging.log_level.upper())
    logger.add(
        config.logging.log_file,
        level=config.logging.log_level.upper(),
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    logger.info("Logger configured.")
