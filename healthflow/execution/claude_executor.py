import asyncio
from pathlib import Path
from loguru import logger

class ClaudeCodeExecutor:
    """
    A robust wrapper for executing tasks using the external Claude Code CLI tool.
    It captures stdout, stderr, and the return code of the execution.
    """
    async def execute(self, task_list_path: Path, working_dir: Path) -> dict:
        """
        Runs `claude` in a subprocess with the given task list markdown file.

        Args:
            task_list_path: Path to the markdown file with step-by-step instructions.
            working_dir: The directory where the command should be executed and artifacts stored.

        Returns:
            A dictionary containing the success status, return code, a combined log,
            and the path to the log file.
        """
        # The command to execute. We use `@` to let Claude Code read the plan from a file.
        # The `--dangerously-skip-permissions` flag is used as requested to enable
        # non-interactive execution for automated workflows. This is a powerful
        # setting and should be used in controlled environments.
        command = f'claude --dangerously-skip-permissions --print "Carefully execute the tasks outlined in the file @{task_list_path.name}. Adhere strictly to the plan."'

        log_file_path = working_dir / "execution.log"
        logger.info(f"Executing command in '{working_dir}': {command}")

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir
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
                        # Optional: print to console for real-time user feedback
                        # print(f"[{prefix.strip()}] {line.strip()}", flush=True)

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