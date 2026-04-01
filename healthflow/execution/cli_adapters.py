from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from string import Template
from typing import Any, List

from loguru import logger

from ..core.config import BackendCLIConfig
from .base import ExecutionContext, ExecutionResult, ExecutorAdapter
from .opencode_parser import parse_opencode_json_events

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CLISubprocessExecutor(ExecutorAdapter):
    def __init__(self, backend_name: str, backend_config: BackendCLIConfig):
        super().__init__(backend_name)
        self.backend_config = backend_config

    async def execute(self, context: ExecutionContext, working_dir: Path) -> ExecutionResult:
        prompt_text = context.render_prompt()
        environment = self._build_environment(working_dir)
        command_args = self._build_command(prompt_text)
        backend_version = await self._capture_backend_version(environment)
        log_file_path = working_dir / f"{self.backend_name}_execution.log"
        logger.info(
            "Executing backend '{}' in '{}': {}",
            self.backend_name,
            working_dir,
            " ".join(command_args),
        )

        start_time = time.time()
        process = await asyncio.create_subprocess_exec(
            *command_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if self.backend_config.prompt_mode == "stdin" else None,
            cwd=working_dir,
            env=environment,
        )

        log_content = ""
        timed_out = False
        try:
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(
                        input=prompt_text.encode("utf-8") if self.backend_config.prompt_mode == "stdin" else None
                    ),
                    timeout=self.backend_config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error("Backend '{}' timed out after {} seconds", self.backend_name, self.backend_config.timeout_seconds)
                timed_out = True
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()

            log_content = self._format_combined_log(
                stdout_bytes.decode("utf-8", errors="replace"),
                stderr_bytes.decode("utf-8", errors="replace"),
                timed_out=timed_out,
            )
            log_file_path.write_text(log_content, encoding="utf-8")
            duration_seconds = round(time.time() - start_time, 2)

            return ExecutionResult(
                success=process.returncode == 0 and not timed_out,
                return_code=process.returncode,
                log=log_content,
                log_path=str(log_file_path),
                prompt_path=None,
                backend=self.backend_name,
                command=command_args,
                backend_version=backend_version,
                executor_metadata=self._executor_metadata(),
                duration_seconds=duration_seconds,
                timed_out=timed_out,
                usage=self._default_usage(
                    prompt_text=prompt_text,
                    stdout_bytes=stdout_bytes,
                    stderr_bytes=stderr_bytes,
                    duration_seconds=duration_seconds,
                    timed_out=timed_out,
                ),
            )
        except Exception as e:
            logger.error("Executor '{}' failed: {}", self.backend_name, e)
            if process.returncode is None:
                process.terminate()
            duration_seconds = round(time.time() - start_time, 2)
            return ExecutionResult(
                success=False,
                return_code=-1,
                log=f"HealthFlow Executor Error: {e}\n\n--- Captured Log Before Error ---\n{log_content}",
                log_path=str(log_file_path),
                prompt_path=None,
                backend=self.backend_name,
                command=command_args,
                backend_version=backend_version,
                executor_metadata=self._executor_metadata(),
                duration_seconds=duration_seconds,
                timed_out=timed_out,
                usage=self._default_usage(
                    prompt_text=prompt_text,
                    stdout_bytes=b"",
                    stderr_bytes=b"",
                    duration_seconds=duration_seconds,
                    timed_out=timed_out,
                ),
            )

    def _build_command(self, prompt_text: str) -> List[str]:
        command = [self.backend_config.binary, *self.backend_config.args]
        template_context = self._template_context()
        command.extend(self._render_arg_templates(template_context))

        provider = self._resolved_provider()
        if self.backend_config.provider_flag and provider:
            command.extend([self.backend_config.provider_flag, provider])

        model_argument = self._resolved_model_argument()
        if self.backend_config.model_flag and model_argument:
            command.extend([self.backend_config.model_flag, model_argument])

        if self.backend_config.prompt_mode == "append":
            command.append(prompt_text)
        return command

    def _build_environment(self, working_dir: Path) -> dict[str, str]:
        environment = os.environ.copy()
        for key, value in self.backend_config.env.items():
            environment[key] = self._expand_environment_value(key, value, environment)
        environment = self._ensure_project_venv_bin_on_path(environment)
        return self._prepare_environment(environment, working_dir)

    def _ensure_project_venv_bin_on_path(self, environment: dict[str, str]) -> dict[str, str]:
        venv_bin = _PROJECT_ROOT / ".venv" / "bin"
        if not venv_bin.exists():
            return environment

        path_entries = environment.get("PATH", "").split(os.pathsep) if environment.get("PATH") else []
        venv_bin_str = str(venv_bin)
        if venv_bin_str in path_entries:
            return environment

        updated_environment = dict(environment)
        updated_environment["PATH"] = os.pathsep.join([venv_bin_str, *path_entries]) if path_entries else venv_bin_str
        return updated_environment

    def _prepare_environment(self, environment: dict[str, str], working_dir: Path) -> dict[str, str]:
        return environment

    def _default_usage(
        self,
        prompt_text: str,
        stdout_bytes: bytes,
        stderr_bytes: bytes,
        duration_seconds: float,
        timed_out: bool,
    ) -> dict[str, Any]:
        return {
            "wall_time_seconds": duration_seconds,
            "timed_out": timed_out,
            "prompt_bytes": len(prompt_text.encode("utf-8")),
            "stdout_bytes": len(stdout_bytes),
            "stderr_bytes": len(stderr_bytes),
        }

    async def _capture_backend_version(self, environment: dict[str, str]) -> str | None:
        if not self.backend_config.version_args:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                self.backend_config.binary,
                *self.backend_config.version_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=5)
            version_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            if not version_text:
                version_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            return version_text or None
        except Exception:
            return None

    def _executor_metadata(self) -> dict:
        return {
            "binary": self.backend_config.binary,
            "args": list(self.backend_config.args),
            "arg_templates": list(self.backend_config.arg_templates),
            "output_mode": self.backend_config.output_mode,
            "provider": self._resolved_provider(),
            "provider_flag": self.backend_config.provider_flag,
            "model": self._resolved_model_name(),
            "model_argument": self._resolved_model_argument(),
            "model_flag": self.backend_config.model_flag,
            "model_template": self.backend_config.model_template,
            "env_keys": sorted(self.backend_config.env.keys()),
            "prompt_mode": self.backend_config.prompt_mode,
            "timeout_seconds": self.backend_config.timeout_seconds,
            "version_args": list(self.backend_config.version_args),
        }

    def _format_combined_log(self, stdout: str, stderr: str, timed_out: bool = False) -> str:
        lines: list[str] = []
        if timed_out:
            lines.append(
                f"STDERR: Process timed out after {self.backend_config.timeout_seconds} seconds.\n"
            )
        lines.extend(self._prefix_stream(stdout, "STDOUT: "))
        lines.extend(self._prefix_stream(stderr, "STDERR: "))
        return "".join(lines)

    def _prefix_stream(self, content: str, prefix: str) -> List[str]:
        if not content:
            return []
        chunks = content.splitlines(keepends=True)
        if content and not content.endswith(("\n", "\r")):
            chunks[-1] = chunks[-1] + "\n"
        return [f"{prefix}{chunk}" for chunk in chunks]

    def _resolved_provider(self) -> str | None:
        return self.backend_config.provider

    def _resolved_model_name(self) -> str | None:
        return self.backend_config.model

    def _resolved_model_argument(self) -> str | None:
        model_name = self._resolved_model_name()
        if not model_name:
            return None
        return self._render_template_value(
            self.backend_config.model_template,
            self._template_base_context(),
            "model_template",
        )

    def _template_base_context(self) -> dict[str, str]:
        return {
            "binary": self.backend_config.binary,
            "provider": self._resolved_provider() or "",
            "model": self._resolved_model_name() or "",
            "provider_base_url": self.backend_config.provider_base_url or "",
            "provider_api": self.backend_config.provider_api or "",
            "provider_api_key_env": self.backend_config.provider_api_key_env or "",
        }

    def _template_context(self) -> dict[str, str]:
        context = self._template_base_context()
        context["model_argument"] = self._resolved_model_argument() or ""
        return context

    def _render_arg_templates(self, template_context: dict[str, str]) -> list[str]:
        return [
            self._render_template_value(template, template_context, "arg_templates")
            for template in self.backend_config.arg_templates
        ]

    def _render_template_value(self, template: str, context: dict[str, str], field_name: str) -> str:
        try:
            return Template(template).substitute(context)
        except KeyError as exc:
            missing_key = exc.args[0]
            raise ValueError(
                f"Backend '{self.backend_name}' field '{field_name}' referenced unknown template key '{missing_key}'."
            ) from exc

    def _expand_environment_value(
        self,
        key: str,
        value: str,
        environment: dict[str, str],
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            env_key = match.group(1)
            if env_key not in environment:
                raise ValueError(
                    f"Backend '{self.backend_name}' environment variable '{key}' requires '{env_key}', but it is not set."
                )
            return environment[env_key]

        return _ENV_VAR_PATTERN.sub(replace, value)


class ClaudeCodeExecutor(CLISubprocessExecutor):
    pass


class CodexExecutor(CLISubprocessExecutor):
    pass


class OpenCodeExecutor(CLISubprocessExecutor):
    async def execute(self, context: ExecutionContext, working_dir: Path) -> ExecutionResult:
        if self.backend_config.output_mode != "json_events":
            return await super().execute(context, working_dir)

        prompt_text = context.render_prompt()
        environment = self._build_environment(working_dir)
        command_args = self._build_command(prompt_text)
        backend_version = await self._capture_backend_version(environment)
        log_file_path = working_dir / f"{self.backend_name}_execution.log"
        logger.info(
            "Executing backend '{}' in '{}': {}",
            self.backend_name,
            working_dir,
            " ".join(command_args),
        )

        start_time = time.time()
        process = await asyncio.create_subprocess_exec(
            *command_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=None,
            cwd=working_dir,
            env=environment,
        )

        log_content = ""
        timed_out = False
        stdout_bytes = b""
        stderr_bytes = b""
        try:
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.backend_config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error("Backend '{}' timed out after {} seconds", self.backend_name, self.backend_config.timeout_seconds)
                timed_out = True
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()

            duration_seconds = round(time.time() - start_time, 2)
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            parsed = parse_opencode_json_events(stdout_text)
            usage = self._default_usage(
                prompt_text=prompt_text,
                stdout_bytes=stdout_bytes,
                stderr_bytes=stderr_bytes,
                duration_seconds=duration_seconds,
                timed_out=timed_out,
            )
            telemetry = parsed.telemetry if parsed.telemetry.get("event_count") else {}
            if telemetry:
                configured_model = self._configured_model_hint(command_args)
                if configured_model and not telemetry.get("models"):
                    telemetry["models"] = [configured_model]
                usage.update(parsed.usage)

            log_content = self._render_opencode_log(
                parsed_log=parsed.log,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                timed_out=timed_out,
            )
            log_file_path.write_text(log_content, encoding="utf-8")

            return ExecutionResult(
                success=process.returncode == 0 and not timed_out,
                return_code=process.returncode,
                log=log_content,
                log_path=str(log_file_path),
                prompt_path=None,
                backend=self.backend_name,
                command=command_args,
                backend_version=backend_version,
                executor_metadata=self._executor_metadata(),
                duration_seconds=duration_seconds,
                timed_out=timed_out,
                usage=usage,
                telemetry=telemetry,
            )
        except Exception as e:
            logger.error("Executor '{}' failed: {}", self.backend_name, e)
            if process.returncode is None:
                process.terminate()
            duration_seconds = round(time.time() - start_time, 2)
            return ExecutionResult(
                success=False,
                return_code=-1,
                log=f"HealthFlow Executor Error: {e}\n\n--- Captured Log Before Error ---\n{log_content}",
                log_path=str(log_file_path),
                prompt_path=None,
                backend=self.backend_name,
                command=command_args,
                backend_version=backend_version,
                executor_metadata=self._executor_metadata(),
                duration_seconds=duration_seconds,
                timed_out=timed_out,
                usage=self._default_usage(
                    prompt_text=prompt_text,
                    stdout_bytes=stdout_bytes,
                    stderr_bytes=stderr_bytes,
                    duration_seconds=duration_seconds,
                    timed_out=timed_out,
                ),
            )

    def _render_opencode_log(
        self,
        parsed_log: str,
        stdout_text: str,
        stderr_text: str,
        timed_out: bool,
    ) -> str:
        lines: list[str] = []
        if timed_out:
            lines.append(f"STDERR: Process timed out after {self.backend_config.timeout_seconds} seconds.\n")
        if parsed_log:
            lines.extend(self._prefix_stream(parsed_log, ""))
        elif stdout_text:
            lines.extend(self._prefix_stream(stdout_text, "STDOUT: "))
        lines.extend(self._prefix_stream(stderr_text, "STDERR: "))
        return "".join(lines)

    def _configured_model_hint(self, command_args: list[str]) -> str | None:
        for index, value in enumerate(command_args):
            if value in {"--model", "-m"} and index + 1 < len(command_args):
                return command_args[index + 1]
        return None


class PiExecutor(CLISubprocessExecutor):
    def _prepare_environment(self, environment: dict[str, str], working_dir: Path) -> dict[str, str]:
        provider_name = self._resolved_provider() or "zenmux"
        model_name = self._resolved_model_name()
        if not model_name:
            raise ValueError(f"Backend '{self.backend_name}' requires a resolved model name.")

        agent_dir = working_dir / ".healthflow_pi_agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        models_path = agent_dir / "models.json"
        models_config = {
            "providers": {
                provider_name: {
                    "baseUrl": self.backend_config.provider_base_url or "https://zenmux.ai/api/v1",
                    "api": self.backend_config.provider_api or "openai-completions",
                    "apiKey": self.backend_config.provider_api_key_env or "ZENMUX_API_KEY",
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                    },
                    "models": [
                        {
                            "id": model_name,
                            "name": model_name,
                            "reasoning": True,
                            "input": ["text"],
                            "contextWindow": 128000,
                            "maxTokens": 8192,
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                        }
                    ],
                }
            }
        }
        models_path.write_text(json.dumps(models_config, indent=2), encoding="utf-8")

        environment = dict(environment)
        environment["PI_CODING_AGENT_DIR"] = str(agent_dir)
        return environment
