import asyncio
from pathlib import Path
from loguru import logger

class ClaudeCodeExecutor:
    """
    A robust wrapper for executing tasks using the external Claude Code CLI tool.
    It captures stdout, stderr, and the return code of the execution.
    It can be configured with a specific shell.
    """
    def __init__(self, shell: str):
        self.shell = shell

    async def execute(self, user_request: str, task_list_path: Path, working_dir: Path) -> dict:
        """
        Runs `claude` in a subprocess with the original user request and a reference plan.
        """
        full_prompt = f'''Your task: {user_request}

I have prepared a detailed plan for reference in the file @{task_list_path.name}. You can use this plan as guidance, but feel free to adapt your approach as needed to best accomplish the original task. The plan is just a reference - you have autonomy to determine the best way to complete the user's request.'''

        command_args = [
            'claude',
            '--dangerously-skip-permissions',
            '--print',
            full_prompt
        ]

        log_file_path = working_dir / "execution.log"
        logger.info(f"Executing command in '{working_dir}': {' '.join(command_args)}")

        process = await asyncio.create_subprocess_exec(
            *command_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
            # executable=self.shell
        )

        log_content = ""
        try:
            # Open the log file once and stream output to it
            with open(log_file_path, "w", encoding="utf-8") as log_file:

                async def read_stream(stream, prefix):
                    nonlocal log_content
                    while True:
                        line_bytes = await stream.readline()
                        if not line_bytes:
                            break
                        line = line_bytes.decode('utf-8', errors='replace')
                        log_file.write(f"{prefix}{line}")
                        log_content += f"{prefix}{line}"

                # Concurrently read and log stdout and stderr
                await asyncio.gather(
                    read_stream(process.stdout, "STDOUT: "),
                    read_stream(process.stderr, "STDERR: ")
                )

            # Wait for process with a 10-minute timeout
            try:
                await asyncio.wait_for(process.wait(), timeout=600)
            except asyncio.TimeoutError:
                logger.error("Claude process timed out after 600 seconds")
                process.terminate()
                await process.wait()
                return {
                    "success": False,
                    "return_code": -1,
                    "log": f"Process timed out after 600 seconds\n\n--- Captured Log Before Timeout ---\n{log_content}",
                    "log_path": str(log_file_path)
                }

            logger.info(f"Claude process finished with return code: {process.returncode}")

            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "log": log_content,
                "log_path": str(log_file_path)
            }
        except Exception as e:
            logger.error(f"A critical error occurred while executing the Claude Code command: {e}")
            # Ensure process is terminated if it's still running
            if process.returncode is None:
                process.terminate()

            return {
                "success": False,
                "return_code": -1,
                "log": f"HealthFlow Executor Error: {e}\n\n--- Captured Log Before Error ---\n{log_content}",
                "log_path": str(log_file_path)
            }