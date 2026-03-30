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
        try:
            with open(log_file_path, "w", encoding="utf-8") as log_file:

                async def read_stream(stream, prefix: str):
                    nonlocal log_content
                    while True:
                        line_bytes = await stream.readline()
                        if not line_bytes:
                            break
                        line = line_bytes.decode("utf-8", errors="replace")
                        log_file.write(f"{prefix}{line}")
                        log_content += f"{prefix}{line}"

                if self.backend_config.prompt_mode == "stdin" and process.stdin:
                    process.stdin.write(prompt_text.encode("utf-8"))
                    await process.stdin.drain()
                    process.stdin.close()

                await asyncio.gather(
                    read_stream(process.stdout, "STDOUT: "),
                    read_stream(process.stderr, "STDERR: "),
                )

            try:
                await asyncio.wait_for(process.wait(), timeout=self.backend_config.timeout_seconds)
            except asyncio.TimeoutError:
                logger.error("Backend '{}' timed out after {} seconds", self.backend_name, self.backend_config.timeout_seconds)
                process.terminate()
                await process.wait()
                return ExecutionResult(
                    success=False,
                    return_code=-1,
                    log=f"Process timed out after {self.backend_config.timeout_seconds} seconds\n\n--- Captured Log Before Timeout ---\n{log_content}",
                    log_path=str(log_file_path),
                    backend=self.backend_name,
                    command=command_args,
                    usage={"wall_time_seconds": round(time.time() - start_time, 2)},
                )

            return ExecutionResult(
                success=process.returncode == 0,
                return_code=process.returncode,
                log=log_content,
                log_path=str(log_file_path),
                backend=self.backend_name,
                command=command_args,
                usage={"wall_time_seconds": round(time.time() - start_time, 2)},
            )
        except Exception as e:
            logger.error("Executor '{}' failed: {}", self.backend_name, e)
            if process.returncode is None:
                process.terminate()
            return ExecutionResult(
                success=False,
                return_code=-1,
                log=f"HealthFlow Executor Error: {e}\n\n--- Captured Log Before Error ---\n{log_content}",
                log_path=str(log_file_path),
                backend=self.backend_name,
                command=command_args,
                usage={"wall_time_seconds": round(time.time() - start_time, 2)},
            )

    def _build_command(self, prompt_text: str) -> List[str]:
        command = [self.backend_config.binary, *self.backend_config.args]
        if self.backend_config.prompt_mode == "append":
            command.append(prompt_text)
        return command


class ClaudeCodeExecutor(CLISubprocessExecutor):
    pass


class OpenCodeExecutor(CLISubprocessExecutor):
    pass


class PiExecutor(CLISubprocessExecutor):
    pass
