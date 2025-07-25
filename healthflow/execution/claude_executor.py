import asyncio
from pathlib import Path
from loguru import logger

class ClaudeCodeExecutor:
    """
    A wrapper for executing tasks using the external Claude Code CLI tool.
    """
    async def execute(self, task_list_path: Path, working_dir: Path) -> dict:
        """
        Runs `claude` in a subprocess with the given task list.

        Args:
            task_list_path: Path to the markdown file with instructions.
            working_dir: The directory where the command should be executed.

        Returns:
            A dictionary with success status, log, and any created artifacts.
        """
        # Command to execute. We use `@` to import the task list.
        # --dangerously-skip-permissions is used as requested for non-interactive execution.
        command = f'claude --dangerously-skip-permissions -p "Execute the tasks outlined in the file @{task_list_path.name}"'

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
            with open(log_file_path, "w") as log_file:
                # Asynchronously read stdout and stderr
                async for line in process.stdout:
                    decoded_line = line.decode('utf-8', errors='replace').strip()
                    log_file.write(decoded_line + '\n')
                    log_content += decoded_line + '\n'
                    print(f"[Claude Output] {decoded_line}") # Also print to console for real-time feedback

                async for line in process.stderr:
                    decoded_line = line.decode('utf-8', errors='replace').strip()
                    log_file.write(f"ERROR: {decoded_line}\n")
                    log_content += f"ERROR: {decoded_line}\n"
                    print(f"[Claude Error] {decoded_line}")

            await process.wait()

            logger.info(f"Claude process finished with return code: {process.returncode}")

            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "log": log_content,
                "log_path": str(log_file_path)
            }
        except Exception as e:
            logger.error(f"Failed to execute Claude Code command: {e}")
            return {
                "success": False,
                "return_code": -1,
                "log": f"HealthFlow Executor Error: {e}\n\n{log_content}",
                "log_path": str(log_file_path)
            }