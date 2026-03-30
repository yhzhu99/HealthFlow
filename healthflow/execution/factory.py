from __future__ import annotations

from ..core.config import BackendCLIConfig
from .base import ExecutorAdapter
from .cli_adapters import ClaudeCodeExecutor, OpenCodeExecutor, PiExecutor


def create_executor_adapter(backend_name: str, backend_config: BackendCLIConfig) -> ExecutorAdapter:
    if backend_name == "claude_code":
        return ClaudeCodeExecutor(backend_name, backend_config)
    if backend_name == "opencode":
        return OpenCodeExecutor(backend_name, backend_config)
    if backend_name == "pi":
        return PiExecutor(backend_name, backend_config)
    raise ValueError(f"Unsupported executor backend: {backend_name}")
