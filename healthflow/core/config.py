import os
import sys
from pathlib import Path
from typing import Dict, List, Literal

import toml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator


ReasoningEffort = Literal["low", "medium", "high"]


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
    executor_provider: str | None = Field(
        default=None,
        description="Optional executor provider name to inherit into backend-specific routing.",
    )
    executor_provider_base_url: str | None = Field(
        default=None,
        description="Optional executor provider base URL for backends that require explicit endpoint configuration.",
    )
    executor_provider_api: str | None = Field(
        default=None,
        description="Optional executor transport identifier such as anthropic-messages or openai-completions.",
    )
    executor_provider_api_key_env: str | None = Field(
        default=None,
        description="Optional environment variable name to use for executor-side provider authentication.",
    )
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description="Optional reasoning effort to request when the provider supports it.",
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

    @model_validator(mode="after")
    def validate_executor_model_name(self):
        if self.executor_provider == "deepseek" and self.executor_model_name and "/" in self.executor_model_name:
            raise ValueError(
                "DeepSeek executor_model_name must be a bare model id such as 'deepseek-chat', not a provider-prefixed value."
            )
        return self


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
    reasoning_effort: ReasoningEffort | None = None
    output_mode: Literal["text", "json_events"] = "text"
    inherit_executor_llm: bool = True
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
                "$reasoning_effort",
            ],
            env={
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
            model_flag="--model",
            provider="zenmux",
            provider_base_url="https://zenmux.ai/api/anthropic",
            provider_api="anthropic-messages",
            provider_api_key_env="ZENMUX_API_KEY",
            reasoning_effort="high",
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
                'model_reasoning_effort="$reasoning_effort"',
                "-c",
                'model_reasoning_summary="detailed"',
            ],
            model="openai/gpt-5.4",
            model_flag="-m",
            inherit_executor_llm=False,
            provider="zenmux",
            provider_base_url="https://zenmux.ai/api/v1",
            provider_api_key_env="ZENMUX_API_KEY",
            reasoning_effort="high",
            prompt_mode="stdin",
        ),
        "opencode": BackendCLIConfig(
            binary="opencode",
            args=["run", "--variant", "$reasoning_effort", "--format", "json"],
            model_flag="-m",
            model_template="$provider/$model",
            provider="zenmux",
            reasoning_effort="high",
            output_mode="json_events",
            prompt_mode="append",
        ),
        "pi": BackendCLIConfig(
            binary="pi",
            args=["--print", "--thinking", "$reasoning_effort"],
            model_flag="--model",
            provider_flag="--provider",
            provider="zenmux",
            provider_base_url="https://zenmux.ai/api/v1",
            provider_api="openai-completions",
            provider_api_key_env="ZENMUX_API_KEY",
            reasoning_effort="high",
            prompt_mode="append",
        ),
    }


class ExecutorConfig(BaseModel):
    active_backend: str = Field("opencode", description="Executor backend to use for task execution.")
    backends: Dict[str, BackendCLIConfig] = Field(default_factory=default_executor_backends)

    @model_validator(mode="after")
    def ensure_active_backend_exists(self):
        if not self.backends:
            self.backends = default_executor_backends()
        if self.active_backend not in self.backends:
            raise ValueError(f"Active backend '{self.active_backend}' is not defined under [executor.backends].")
        return self


class RuntimeLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner_llm: str | None = Field(default=None, description="LLM registry key for the planner agent.")
    evaluator_llm: str | None = Field(default=None, description="LLM registry key for the evaluator agent.")
    reflector_llm: str | None = Field(default=None, description="LLM registry key for the reflector agent.")
    executor_llm: str | None = Field(default=None, description="LLM registry key for the executor backend.")


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    write_policy: Literal["append", "freeze", "reset_before_run"] = "append"


class EvaluationConfig(BaseModel):
    success_threshold: float = Field(0.8, description="Score (0-1) required to consider a task successful.")


class LoggingConfig(BaseModel):
    log_level: str = Field("INFO", description="Logging level.")
    log_file: str = Field("healthflow.log", description="Path to the log file.")


class HealthFlowConfig(BaseModel):
    """The main configuration object, assembled dynamically from the config file."""

    planner_llm_name: str
    evaluator_llm_name: str
    reflector_llm_name: str
    executor_llm_name: str
    active_executor_name: str
    llm_registry: Dict[str, LLMProviderConfig]
    system: SystemConfig
    environment: EnvironmentConfig
    executor: ExecutorConfig
    memory: MemoryConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

    @property
    def planner_llm(self) -> LLMProviderConfig:
        return self.llm_registry[self.planner_llm_name]

    @property
    def evaluator_llm(self) -> LLMProviderConfig:
        return self.llm_registry[self.evaluator_llm_name]

    @property
    def reflector_llm(self) -> LLMProviderConfig:
        return self.llm_registry[self.reflector_llm_name]

    @property
    def executor_llm(self) -> LLMProviderConfig:
        return self.llm_registry[self.executor_llm_name]

    @property
    def runtime_llm_keys(self) -> dict[str, str]:
        return {
            "planner": self.planner_llm_name,
            "evaluator": self.evaluator_llm_name,
            "reflector": self.reflector_llm_name,
            "executor": self.executor_llm_name,
        }

    @property
    def active_executor(self) -> BackendCLIConfig:
        active_executor = self.executor.backends[self.active_executor_name]
        resolved_updates: dict[str, str | None] = {}
        executor_llm = self.executor_llm
        if active_executor.inherit_executor_llm:
            if active_executor.model is None:
                resolved_updates["model"] = executor_llm.executor_model_name or executor_llm.model_name
            if executor_llm.executor_provider is not None:
                resolved_updates["provider"] = executor_llm.executor_provider
            if executor_llm.executor_provider_base_url is not None:
                resolved_updates["provider_base_url"] = executor_llm.executor_provider_base_url
            if executor_llm.executor_provider_api is not None:
                resolved_updates["provider_api"] = executor_llm.executor_provider_api
            if executor_llm.executor_provider_api_key_env is not None:
                resolved_updates["provider_api_key_env"] = executor_llm.executor_provider_api_key_env
            if executor_llm.reasoning_effort is not None:
                resolved_updates["reasoning_effort"] = executor_llm.reasoning_effort
        return active_executor.model_copy(update=resolved_updates)

    def llm_config_for_role(self, role: str) -> LLMProviderConfig:
        mapping = {
            "planner": self.planner_llm_name,
            "evaluator": self.evaluator_llm_name,
            "reflector": self.reflector_llm_name,
            "executor": self.executor_llm_name,
        }
        if role not in mapping:
            raise ValueError(f"Unsupported LLM role '{role}'. Expected one of: {', '.join(mapping)}.")
        return self.llm_registry[mapping[role]]

    @property
    def workspace_root(self) -> Path:
        return Path(self.system.workspace_dir).parent

    @property
    def resolved_log_file(self) -> Path:
        log_path = Path(self.logging.log_file)
        if log_path.is_absolute():
            return log_path
        return self.workspace_root / log_path


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

    if "llm_roles" in config_data:
        raise ValueError(
            "Config section '[llm_roles]' is no longer supported. "
            "Move those selections to '[runtime]' using 'planner_llm', 'evaluator_llm', 'reflector_llm', and 'executor_llm'."
        )

    allowed_sections = {"llm", "runtime", "system", "environment", "executor", "memory", "evaluation", "logging"}
    unexpected_sections = sorted(section for section in config_data if section not in allowed_sections)
    if unexpected_sections:
        raise ValueError(
            "Unsupported config sections: " + ", ".join(f"[{section}]" for section in unexpected_sections)
        )


def _validate_system_section(system_data: dict) -> None:
    if "shell" in system_data:
        raise ValueError(
            "Config key 'system.shell' is no longer supported. "
            "HealthFlow executes configured backends directly rather than through a configurable shell. "
            "Remove 'shell' from the '[system]' section."
        )


def _validate_executor_section(executor_data: dict) -> None:
    for backend_name, backend_config in (executor_data.get("backends") or {}).items():
        if isinstance(backend_config, dict) and "inherit_active_llm" in backend_config:
            raise ValueError(
                f"Config key 'executor.backends.{backend_name}.inherit_active_llm' is no longer supported. "
                "Rename it to 'inherit_executor_llm'."
            )


def _resolve_runtime_llm_names(
    config_data: dict,
    llm_registry: dict[str, LLMProviderConfig],
    planner_llm: str | None,
    evaluator_llm: str | None,
    reflector_llm: str | None,
    executor_llm: str | None,
) -> dict[str, str]:
    runtime = RuntimeLLMConfig(**config_data.get("runtime", {}))
    resolved = {
        "planner_llm": planner_llm or runtime.planner_llm,
        "evaluator_llm": evaluator_llm or runtime.evaluator_llm,
        "reflector_llm": reflector_llm or runtime.reflector_llm,
        "executor_llm": executor_llm or runtime.executor_llm,
    }

    missing = [field for field, value in resolved.items() if not value]
    if missing:
        raise ValueError(
            "Missing runtime LLM selections for "
            + ", ".join(missing)
            + ". Define them under '[runtime]' or pass the matching CLI flags."
        )

    for field_name, llm_name in resolved.items():
        if llm_name not in llm_registry:
            raise ValueError(f"Configured runtime.{field_name}='{llm_name}' was not found under '[llm]'.")

    return {
        field_name: str(llm_name)
        for field_name, llm_name in resolved.items()
    }


def get_config(
    config_path: Path,
    planner_llm: str | None = None,
    active_executor: str | None = None,
    *,
    evaluator_llm: str | None = None,
    reflector_llm: str | None = None,
    executor_llm: str | None = None,
) -> HealthFlowConfig:
    """
    Loads configuration from a TOML file, validates it, resolves the runtime model
    selections, and returns a unified HealthFlowConfig object.
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
        _validate_system_section(config_data.get("system", {}))
        _validate_executor_section(config_data.get("executor", {}))

        llm_section = config_data.get("llm", {})
        llm_registry = {
            name: _resolve_llm_provider_config(name, provider_config)
            for name, provider_config in llm_section.items()
        }

        resolved_runtime_llms = _resolve_runtime_llm_names(
            config_data=config_data,
            llm_registry=llm_registry,
            planner_llm=planner_llm,
            evaluator_llm=evaluator_llm,
            reflector_llm=reflector_llm,
            executor_llm=executor_llm,
        )

        executor_section = config_data.get("executor", {})
        executor_backends = executor_section.get("backends")
        executor_config = ExecutorConfig(
            active_backend=active_executor or executor_section.get("active_backend", "opencode"),
            backends=executor_backends if executor_backends else default_executor_backends(),
        )

        config = HealthFlowConfig(
            planner_llm_name=resolved_runtime_llms["planner_llm"],
            evaluator_llm_name=resolved_runtime_llms["evaluator_llm"],
            reflector_llm_name=resolved_runtime_llms["reflector_llm"],
            executor_llm_name=resolved_runtime_llms["executor_llm"],
            active_executor_name=executor_config.active_backend,
            llm_registry=llm_registry,
            system=SystemConfig(**config_data.get("system", {})),
            environment=EnvironmentConfig(**config_data.get("environment", {})),
            executor=executor_config,
            memory=MemoryConfig(**config_data.get("memory", {})),
            evaluation=EvaluationConfig(**config_data.get("evaluation", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
        )

        logger.info(
            "Configuration loaded successfully. Runtime LLMs: planner='{}', evaluator='{}', reflector='{}', executor='{}'. Active executor: '{}'",
            config.planner_llm_name,
            config.evaluator_llm_name,
            config.reflector_llm_name,
            config.executor_llm_name,
            config.active_executor_name,
        )
        return config
    except Exception as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}")
        raise ValueError(f"Error parsing configuration file '{config_path}': {e}") from e


def setup_logging(config: HealthFlowConfig, console_log_level: str | None = None):
    """Configures the Loguru logger based on the loaded configuration."""
    log_path = config.resolved_log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=(console_log_level or "WARNING").upper())
    logger.add(
        log_path,
        level=config.logging.log_level.upper(),
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    logger.info("Logger configured.")
