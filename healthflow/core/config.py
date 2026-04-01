import os
import sys
from pathlib import Path
from typing import Dict, List, Literal

import toml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    api_key: str = Field(..., description="Resolved API key for the LLM provider.")
    api_key_env: str | None = Field(
        default=None,
        description="Optional environment variable name that stores the API key.",
    )
    base_url: str = Field(..., description="Base URL for the LLM API.")
    model_name: str = Field(..., description="Model name to use.")
    executor_model_name: str | None = Field(
        default=None,
        description="Optional model name to inherit into executor backends when it differs from the internal LLM model ID.",
    )
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
    arg_templates: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    model_flag: str | None = None
    model_template: str = "$model"
    provider: str | None = None
    provider_flag: str | None = None
    provider_base_url: str | None = None
    provider_api: str | None = None
    provider_api_key_env: str | None = None
    output_mode: Literal["text", "json_events"] = "text"
    inherit_active_llm: bool = True
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


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python_version: str = Field("3.12", description="Preferred Python version for executor-side workflows.")
    package_manager: str = Field("uv", description="Preferred package manager available in the execution environment.")
    install_command: str = Field("uv add", description="Preferred dependency installation command when adding packages is necessary.")
    run_prefix: str = Field("uv run", description="Preferred command prefix for Python entrypoints.")

    def summary_lines(self) -> list[str]:
        return [
            f"Preferred Python version: {self.python_version}",
            f"Package manager: {self.package_manager}",
            f"Dependency install command: {self.install_command}",
            f"Python command prefix: {self.run_prefix}",
        ]


def default_executor_backends() -> Dict[str, BackendCLIConfig]:
    return {
        "claude_code": BackendCLIConfig(
            binary="claude",
            args=[
                "--bare",
                "--setting-sources",
                "local",
                "--dangerously-skip-permissions",
                "--print",
                "--output-format",
                "text",
                "--effort",
                "high",
            ],
            env={
                "ANTHROPIC_BASE_URL": "https://zenmux.ai/api/anthropic",
                "ANTHROPIC_API_KEY": "${ZENMUX_API_KEY}",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
            model_flag="--model",
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
            arg_templates=[
                "-c",
                'model_provider="$provider"',
                "-c",
                'model_providers.$provider={name="ZenMux", base_url="$provider_base_url", env_key="$provider_api_key_env", wire_api="responses"}',
                "-c",
                'model_reasoning_effort="high"',
                "-c",
                'model_reasoning_summary="detailed"',
            ],
            model="openai/gpt-5.4",
            model_flag="-m",
            inherit_active_llm=False,
            provider="zenmux",
            provider_base_url="https://zenmux.ai/api/v1",
            provider_api_key_env="ZENMUX_API_KEY",
            prompt_mode="stdin",
        ),
        "opencode": BackendCLIConfig(
            binary="opencode",
            args=["run", "--variant", "high", "--thinking"],
            model_flag="-m",
            model_template="$provider/$model",
            provider="zenmux",
            output_mode="text",
            prompt_mode="append",
        ),
        "pi": BackendCLIConfig(
            binary="pi",
            args=["--print", "--thinking", "high"],
            model_flag="--model",
            provider_flag="--provider",
            provider="zenmux",
            provider_base_url="https://zenmux.ai/api/v1",
            provider_api="openai-completions",
            provider_api_key_env="ZENMUX_API_KEY",
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


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    environment: EnvironmentConfig
    executor: ExecutorConfig
    memory: MemoryConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

    @property
    def active_executor(self) -> BackendCLIConfig:
        active_executor = self.executor.backends[self.active_executor_name]
        resolved_model = active_executor.model
        if resolved_model is None and active_executor.inherit_active_llm:
            resolved_model = self.llm.executor_model_name or self.llm.model_name
        return active_executor.model_copy(update={"model": resolved_model})

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

def _validate_top_level_sections(config_data: dict) -> None:
    if "tools" in config_data:
        raise ValueError(
            "Legacy tools configuration is no longer supported. "
            "HealthFlow does not host MCP or CLI tool integrations; configure tools in the outer executor instead. "
            "Use [environment] for lightweight runtime defaults."
        )

    allowed_sections = {"llm", "llm_roles", "system", "environment", "executor", "memory", "evaluation", "logging"}
    unexpected_sections = sorted(section for section in config_data if section not in allowed_sections)
    if unexpected_sections:
        raise ValueError(
            "Unsupported config sections: " + ", ".join(f"[{section}]" for section in unexpected_sections)
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
        _validate_top_level_sections(config_data)

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
            environment=EnvironmentConfig(**config_data.get("environment", {})),
            executor=executor_config,
            memory=MemoryConfig(**config_data.get("memory", {})),
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
