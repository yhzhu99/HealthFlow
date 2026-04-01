from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..core.config import ToolsConfig


class ToolSurface(str, Enum):
    CLI = "cli"
    MCP = "mcp"


@dataclass(frozen=True)
class ToolAdapter:
    name: str
    surface: ToolSurface
    description: str
    invocation_hint: str = ""

    def to_prompt_line(self) -> str:
        line = f"{self.name} [{self.surface.value}] - {self.description}"
        if self.invocation_hint:
            line += f" | Invocation hint: {self.invocation_hint}"
        return line


@dataclass
class ToolCatalog:
    tools: list[ToolAdapter] = field(default_factory=list)

    @classmethod
    def from_config(cls, executor_name: str, config: ToolsConfig) -> "ToolCatalog":
        tools = cls._default_tools(executor_name)
        configured = [
            ToolAdapter(
                name=name,
                surface=ToolSurface(definition.surface),
                description=definition.description or f"Configured {definition.surface} tool.",
                invocation_hint=definition.invocation_hint,
            )
            for name, definition in config.entries.items()
        ]
        merged = {tool.name: tool for tool in [*tools, *configured]}
        return cls(tools=list(merged.values()))

    @classmethod
    def _default_tools(cls, executor_name: str) -> list[ToolAdapter]:
        return [
            ToolAdapter(
                name="python",
                surface=ToolSurface.CLI,
                description="Run scripts or small analysis snippets when code execution is necessary.",
                invocation_hint="python <script>.py",
            ),
            ToolAdapter(
                name="shell",
                surface=ToolSurface.CLI,
                description="Use reproducible shell commands for inspection, file management, and automation.",
                invocation_hint="bash/zsh command",
            ),
            ToolAdapter(
                name="filesystem",
                surface=ToolSurface.CLI,
                description="Read and write artifacts inside the task workspace.",
                invocation_hint="workspace-local files only",
            ),
            ToolAdapter(
                name=f"{executor_name}_backend",
                surface=ToolSurface.CLI,
                description="Primary external coding backend used for CodeAct execution.",
                invocation_hint=executor_name,
            ),
        ]

    def names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def prompt_lines(self) -> list[str]:
        return [f"- {tool.to_prompt_line()}" for tool in self.tools]
