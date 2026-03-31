from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import List

from loguru import logger

from ..core.config import BackendCLIConfig
from .base import ExecutionContext, ExecutionResult, ExecutorAdapter


class CLISubprocessExecutor(ExecutorAdapter):
    def __init__(self, backend_name: str, backend_config: BackendCLIConfig):
        super().__init__(backend_name)
        self.backend_config = backend_config

    async def execute(self, context: ExecutionContext, working_dir: Path, prompt_file_name: str) -> ExecutionResult:
        prompt_text = context.render_prompt()
        prompt_file_path = working_dir / prompt_file_name
        prompt_file_path.write_text(prompt_text, encoding="utf-8")

        command_args = self._build_command(prompt_text)
        backend_version = await self._capture_backend_version()
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
                prompt_path=str(prompt_file_path),
                backend=self.backend_name,
                command=command_args,
                backend_version=backend_version,
                executor_metadata=self._executor_metadata(),
                duration_seconds=duration_seconds,
                timed_out=timed_out,
                usage={
                    "wall_time_seconds": duration_seconds,
                    "timed_out": timed_out,
                    "prompt_bytes": len(prompt_text.encode("utf-8")),
                    "stdout_bytes": len(stdout_bytes),
                    "stderr_bytes": len(stderr_bytes),
                },
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
                prompt_path=str(prompt_file_path),
                backend=self.backend_name,
                command=command_args,
                backend_version=backend_version,
                executor_metadata=self._executor_metadata(),
                duration_seconds=duration_seconds,
                timed_out=timed_out,
                usage={
                    "wall_time_seconds": duration_seconds,
                    "timed_out": timed_out,
                    "prompt_bytes": len(prompt_text.encode("utf-8")),
                },
            )

    def _build_command(self, prompt_text: str) -> List[str]:
        command = [self.backend_config.binary, *self.backend_config.args]
        if self.backend_config.prompt_mode == "append":
            command.append(prompt_text)
        return command

    async def _capture_backend_version(self) -> str | None:
        if not self.backend_config.version_args:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                self.backend_config.binary,
                *self.backend_config.version_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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


class ClaudeCodeExecutor(CLISubprocessExecutor):
    pass


class OpenCodeExecutor(CLISubprocessExecutor):
    pass


class PiExecutor(CLISubprocessExecutor):
    pass
