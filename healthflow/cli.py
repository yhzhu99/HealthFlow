import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from healthflow.core.config import HealthFlowConfig
from healthflow.system import HealthFlowSystem

app = typer.Typer(
    name="healthflow",
    help="A Self-Evolving, Multi-Agent AI System for Healthcare.",
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
        None, "--task", "-t", help="A single task to execute."
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
    """
    if ctx.invoked_subcommand is not None:
        return

    try:
        config = HealthFlowConfig.from_toml(config_path)
        setup_logging(config)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]Error loading configuration: {e}[/bold red]")
        raise typer.Exit(code=1)

    system = HealthFlowSystem(config)

    async def run_async():
        try:
            await system.start()

            if task:
                await execute_single_task(system, task)
            elif file:
                await execute_task_file(system, file)
            else:
                await interactive_mode(system)
        finally:
            await system.stop()

    asyncio.run(run_async())


async def execute_single_task(system: HealthFlowSystem, task_description: str):
    console.print(f"\n[bold cyan]Executing task:[/] {task_description}")
    with console.status("[bold green]Processing...", spinner="dots"):
        result = await system.run_task(task_description)
    print_result(result)


async def execute_task_file(system: HealthFlowSystem, file_path: Path):
    if not file_path.is_file():
        console.print(f"[bold red]Error: File not found at {file_path}[/bold red]")
        raise typer.Exit(code=1)

    with file_path.open("r") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    console.print(f"\n[bold cyan]Executing {len(tasks)} tasks from {file_path}...[/]")
    results = []
    for i, task_data in enumerate(tasks):
        desc = task_data.get("task")
        console.print(f"\n--- Task {i+1}/{len(tasks)} ---")
        console.print(f"[bold]Task:[/] {desc}")
        with console.status("[bold green]Processing...", spinner="dots"):
            result = await system.run_task(desc)
        print_result(result)
        results.append(result)

    # Optionally save results
    output_path = file_path.parent / f"{file_path.stem}_results.jsonl"
    with output_path.open("w") as f:
        for res in results:
            f.write(json.dumps(res, default=str) + "\n")
    console.print(f"\n[bold green]All tasks complete. Results saved to {output_path}[/]")


async def interactive_mode(system: HealthFlowSystem):
    console.print(Panel("[bold green]Welcome to HealthFlow Interactive Mode[/]",
                        title="HealthFlow", subtitle="Type 'exit' or 'quit' to end."))
    while True:
        try:
            task_description = console.input("[bold magenta]HealthFlow> [/]")
            if task_description.lower() in ["exit", "quit"]:
                break
            if not task_description.strip():
                continue
            await execute_single_task(system, task_description)
        except (KeyboardInterrupt, EOFError):
            break
    console.print("\n[bold yellow]Exiting HealthFlow. Goodbye![/]")


def print_result(result: dict):
    """Prints a formatted result to the console."""
    success_emoji = "✅" if result["success"] else "❌"
    title = f"{success_emoji} Task Result"
    panel_color = "green" if result["success"] else "red"

    content = f"[bold]Final Answer:[/]\n{result['result']}\n\n"
    content += f"[bold]Execution Time:[/] {result['execution_time']:.2f}s\n"
    if result.get('tools_used'):
        content += f"[bold]Tools Used:[/] {', '.join(result['tools_used'])}\n"

    if result.get("evaluation"):
        eval_data = result["evaluation"]
        score = eval_data.get('overall_score', 'N/A')
        summary = eval_data.get('executive_summary', 'No summary.')
        content += f"\n[bold]--- Evaluation ---[/]\n"
        content += f"[bold]Score:[/] {score:.1f}/10.0\n"
        content += f"[bold]Summary:[/] {summary}\n"

    console.print(Panel(content, title=title, border_style=panel_color, expand=False))

if __name__ == "__main__":
    app()