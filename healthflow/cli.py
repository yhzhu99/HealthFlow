import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

# Assuming run_healthflow.py is in the parent directory
# This allows the CLI to call the interactive runner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_healthflow

from healthflow.core.config import HealthFlowConfig
from healthflow.system import HealthFlowSystem

app = typer.Typer(
    name="healthflow",
    help="A Simple, Effective, and Self-Evolving LLM Agent Framework for Healthcare.",
    add_completion=False,
)
console = Console()


def setup_logging(config: HealthFlowConfig):
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=config.log_level.upper(),
        format=config.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.log_file),
        ],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    task: Optional[str] = typer.Option(
        None, "--task", "-t", help="A single task to execute in non-interactive mode."
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Path to a JSONL file with tasks to execute."
    ),
    config_path: str = typer.Option(
        "config.toml", "--config", "-c", help="Path to the configuration file."
    ),
):
    """
    Initializes and runs the HealthFlow system.
    If no task or file is provided, it enters interactive mode.
    """
    if ctx.invoked_subcommand is not None:
        return

    # If no specific task or file, run the interactive main loop
    if not task and not file:
        asyncio.run(run_healthflow.main())
        return

    # Otherwise, run in batch mode
    try:
        config = HealthFlowConfig.from_toml(config_path)
        setup_logging(config)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]Error loading configuration: {e}[/bold red]")
        raise typer.Exit(code=1)

    system = HealthFlowSystem(config)

    async def run_batch():
        try:
            await system.start()
            if task:
                await execute_single_task(system, task)
            elif file:
                await execute_task_file(system, file)
        finally:
            await system.stop()

    asyncio.run(run_batch())


async def execute_single_task(system: HealthFlowSystem, task_description: str):
    console.print(f"\n[bold cyan]Executing task:[/] {task_description}")
    with console.status("[bold green]Processing...", spinner="dots"):
        result = await system.run_task(task_description)
    run_healthflow.display_task_result(console, result)


async def execute_task_file(system: HealthFlowSystem, file_path: Path):
    if not file_path.is_file():
        console.print(f"[bold red]Error: File not found at {file_path}[/bold red]")
        raise typer.Exit(code=1)

    with file_path.open("r") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    console.print(f"\n[bold cyan]Executing {len(tasks)} tasks from {file_path}...[/]")
    for i, task_data in enumerate(tasks):
        desc = task_data.get("task")
        if not desc:
            continue
        console.print(f"\n--- Task {i+1}/{len(tasks)} ---")
        console.print(f"[bold]Task:[/] {desc}")
        with console.status("[bold green]Processing...", spinner="dots"):
            result = await system.run_task(desc)
        run_healthflow.display_task_result(console, result)


if __name__ == "__main__":
    app()