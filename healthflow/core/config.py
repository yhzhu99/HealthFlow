import os
import sys
from pathlib import Path
from typing import Dict, List, Literal

import toml
from loguru import logger
from pydantic import BaseModel, Field, model_validator


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    api_key: str = Field(..., description="Resolved API key for the LLM provider.")
    api_key_env: str | None = Field(
        default=None,
        description="Optional environment variable name that stores the API key.",
    )
    base_url: str = Field(..., description="Base URL for the LLM API.")
    model_name: str = Field(..., description="Model name to use.")
    timeout: int = Field(180, description="Request timeout in seconds.")
    input_cost_per_million_tokens: float | None = Field(
        default=None,
        description="Optional estimated input-token price in USD per 1M tokens.",
    )
    output_cost_per_million_tokens: float | None = Field(
        default=None,
        description="Optional estimated output-token price in USD per 1M tokens.",
    )


class BackendCLIConfig(BaseModel):
    """CLI settings for a single executor backend."""

    binary: str
    args: List[str] = Field(default_factory=list)
    prompt_mode: Literal["append", "stdin"] = "append"
    timeout_seconds: int = 900
    version_args: List[str] = Field(default_factory=lambda: ["--version"])


class SystemConfig(BaseModel):
    max_attempts: int = Field(
        3,
        ge=1,
        description="Maximum number of full task attempts in the self-correction loop.",
    )
    workspace_dir: str = Field("workspace/tasks", description="Directory for task artifacts.")
    shell: str = Field("/usr/bin/zsh", description="Shell to use for subprocess execution.")


def default_executor_backends() -> Dict[str, BackendCLIConfig]:
    return {
        "claude_code": BackendCLIConfig(
            binary="claude",
            args=["--dangerously-skip-permissions", "--print"],
            prompt_mode="append",
        ),
        "codex": BackendCLIConfig(
            binary="codex",
            args=[
                "exec",
                "--skip-git-repo-check",
                "--color",
                "never",
                "--dangerously-bypass-approvals-and-sandbox",
            ],
            prompt_mode="stdin",
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
    active_backend: str = Field("opencode", description="Executor backend to use for task execution.")
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


class LLMRoleConfig(BaseModel):
    planner: str | None = Field(default=None, description="Optional model key for the planning agent.")
    evaluator: str | None = Field(default=None, description="Optional model key for the evaluator agent.")
    reflector: str | None = Field(default=None, description="Optional model key for the reflector agent.")


LEGACY_MEMORY_MODE_MAP = {
    "accumulate_eval": "append",
    "frozen_train": "freeze",
    "reset": "reset_before_run",
}
REMOVED_CONFIG_SECTIONS = {"ehr", "verification"}
REMOVED_MEMORY_KEYS = {"retrieve_k", "strategy_k", "failure_k", "dataset_k", "writeback_on_failure"}


class MemoryConfig(BaseModel):
    write_policy: Literal["append", "freeze", "reset_before_run"] = "append"


class EvaluationConfig(BaseModel):
    success_threshold: float = Field(8.0, description="Score (out of 10) to consider a task successful.")


class LoggingConfig(BaseModel):
    log_level: str = Field("INFO", description="Logging level.")
    log_file: str = Field("healthflow.log", description="Path to the log file.")


class HealthFlowConfig(BaseModel):
    """The main configuration object, assembled dynamically from the config file."""

    active_llm_name: str
    active_executor_name: str
    llm_registry: Dict[str, LLMProviderConfig]
    llm: LLMProviderConfig
    llm_roles: LLMRoleConfig
    system: SystemConfig
    executor: ExecutorConfig
    memory: MemoryConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

    @property
    def active_executor(self) -> BackendCLIConfig:
        return self.executor.backends[self.active_executor_name]

    def llm_config_for_role(self, role: str) -> LLMProviderConfig:
        configured_name = getattr(self.llm_roles, role, None) or self.active_llm_name
        return self.llm_registry[configured_name]


def _resolve_llm_provider_config(provider_name: str, provider_config: dict) -> LLMProviderConfig:
    resolved_config = dict(provider_config)
    api_key = resolved_config.get("api_key")
    api_key_env = resolved_config.get("api_key_env")

    if api_key is None:
        if api_key_env is None:
            raise ValueError(
                f"LLM '{provider_name}' must define either 'api_key' or 'api_key_env' under '[llm]'."
            )
        resolved_api_key = os.getenv(api_key_env)
        if not resolved_api_key:
            raise ValueError(
                f"LLM '{provider_name}' requires environment variable '{api_key_env}', but it is not set."
            )
        resolved_config["api_key"] = resolved_api_key

    return LLMProviderConfig(**resolved_config)


def _normalize_memory_config(memory_config: dict) -> dict:
    normalized = dict(memory_config)
    removed_keys = sorted(key for key in normalized if key in REMOVED_MEMORY_KEYS)
    if removed_keys:
        raise ValueError(
            "The following [memory] keys were removed because retrieval policy is now internal: "
            f"{', '.join(removed_keys)}. Keep only [memory].write_policy."
        )

    if "mode" in normalized and "write_policy" in normalized:
        raise ValueError(
            "Use either deprecated [memory].mode or [memory].write_policy, not both. "
            "Prefer [memory].write_policy."
        )

    if "mode" in normalized:
        legacy_mode = normalized.pop("mode")
        if legacy_mode not in LEGACY_MEMORY_MODE_MAP:
            raise ValueError(
                f"Unsupported legacy [memory].mode '{legacy_mode}'. "
                "Supported legacy values are: accumulate_eval, frozen_train, reset."
            )
        mapped_value = LEGACY_MEMORY_MODE_MAP[legacy_mode]
        logger.warning(
            "Deprecated [memory].mode='{}' detected. Map it to [memory].write_policy='{}' instead.",
            legacy_mode,
            mapped_value,
        )
        normalized["write_policy"] = mapped_value

    return normalized


def _validate_removed_sections(config_data: dict) -> None:
    removed_sections = sorted(section for section in REMOVED_CONFIG_SECTIONS if section in config_data)
    if removed_sections:
        raise ValueError(
            "The following config sections were removed because task policy is now internal to HealthFlow: "
            f"{', '.join(f'[{section}]' for section in removed_sections)}."
        )


def get_config(config_path: Path, active_llm: str, active_executor: str | None = None) -> HealthFlowConfig:
    """
    Loads configuration from a TOML file, validates it, selects the active LLM and
    executor settings, and returns a unified HealthFlowConfig object.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at '{config_path}'. "
            "Create it and supply API keys via 'api_key' or 'api_key_env' "
            "(for example, 'ZENMUX_API_KEY')."
        )

    try:
        config_data = toml.load(config_path)
        _validate_removed_sections(config_data)

        if not active_llm:
            raise ValueError("'active_llm' parameter is required")

        llm_section = config_data.get("llm", {})
        active_llm_config_data = llm_section.get(active_llm)
        if not active_llm_config_data:
            raise ValueError(f"Configuration for LLM '{active_llm}' not found under the '[llm]' section.")
        llm_registry = {
            name: _resolve_llm_provider_config(name, provider_config)
            for name, provider_config in llm_section.items()
        }
        llm_roles = LLMRoleConfig(**config_data.get("llm_roles", {}))
        for role_name in ["planner", "evaluator", "reflector"]:
            configured_name = getattr(llm_roles, role_name)
            if configured_name and configured_name not in llm_registry:
                raise ValueError(f"Configured llm_roles.{role_name}='{configured_name}' was not found under '[llm]'.")

        executor_section = config_data.get("executor", {})
        executor_backends = executor_section.get("backends")
        executor_config = ExecutorConfig(
            active_backend=active_executor or executor_section.get("active_backend", "opencode"),
            backends=executor_backends if executor_backends else default_executor_backends(),
        )

        config = HealthFlowConfig(
            active_llm_name=active_llm,
            active_executor_name=executor_config.active_backend,
            llm_registry=llm_registry,
            llm=llm_registry[active_llm],
            llm_roles=llm_roles,
            system=SystemConfig(**config_data.get("system", {})),
            executor=executor_config,
            memory=MemoryConfig(**_normalize_memory_config(config_data.get("memory", {}))),
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
