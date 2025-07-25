import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner

# Add project root to path to allow direct execution
sys.path.insert(0, str(Path(__file__).parent))

from healthflow.system import HealthFlowSystem
from healthflow.core.config import get_config, setup_logging, HealthFlowConfig

app = typer.Typer(
    name="healthflow",
    help="A Self-Evolving Meta-System for Orchestrating Agentic Coders.",
    add_completion=False,
)
console = Console()

def _display_task_result(result: dict):
    """Helper function to display the final result of a task."""
    success = result.get("success", False)
    if success:
        panel_title = "[bold green]✅ Task Completed Successfully[/bold green]"
        panel_border = "green"
    else:
        panel_title = "[bold red]❌ Task Failed[/bold red]"
        panel_border = "red"

    final_report = f"""
[bold]Final Outcome:[/bold]
{result.get('final_summary', 'No summary available.')}

---
[bold]Workspace:[/bold] {result.get('workspace_path', 'N/A')}
[bold]Execution Time:[/bold] {result.get('execution_time', 0):.2f}s
"""
    console.print(Panel(final_report, title=panel_title, border_style=panel_border))

async def run_single_task(system: HealthFlowSystem, task: str):
    """Runs a single task and displays the result."""
    console.print(Panel(f"[bold cyan]Starting HealthFlow Task[/bold cyan]\n\n[dim]Task:[/dim] {task}", border_style="cyan"))
    with Live(Spinner("dots", text="[cyan]HealthFlow is thinking...[/cyan]"), console=console, transient=True, refresh_per_second=20):
        result = await system.run_task(task)
    _display_task_result(result)

async def main_interactive(system: HealthFlowSystem):
    """Runs the interactive mode loop."""
    console.print(Panel("[bold green]HealthFlow Interactive Mode[/bold green]", subtitle="Type 'exit' or 'quit' to end the session.", border_style="green"))
    while True:
        try:
            task_input = console.input("\n[bold magenta]HealthFlow > [/bold magenta]").strip()
            if not task_input:
                continue
            if task_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Exiting interactive mode.[/yellow]")
                break

            await run_single_task(system, task_input)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting interactive mode.[/yellow]")
            break

@app.command()
def run(
    task: str = typer.Argument(..., help="The high-level task for HealthFlow to accomplish."),
    config_path: Path = typer.Option("config.toml", "--config", "-c", help="Path to the configuration file."),
):
    """
    Run a single task through the HealthFlow system.
    """
    config = _initialize_system(config_path)
    system = HealthFlowSystem(config)
    asyncio.run(run_single_task(system, task))

@app.command()
def interactive(
    config_path: Path = typer.Option("config.toml", "--config", "-c", help="Path to the configuration file."),
):
    """
    Starts HealthFlow in an interactive mode for multiple tasks.
    """
    config = _initialize_system(config_path)
    system = HealthFlowSystem(config)
    asyncio.run(main_interactive(system))

def _initialize_system(config_path: Path) -> HealthFlowConfig:
    """Loads config and sets up logging, returns the config object."""
    try:
        config = get_config(config_path)
        setup_logging(config)
        return config
    except (ValueError, FileNotFoundError) as e:
        console.print(Panel(f"[bold red]Initialization Error:[/bold red] {e}", title="Error", border_style="red"))
        raise typer.Exit(code=1)

@app.callback(invoke_without_command=True)
def main_entry(ctx: typer.Context):
    """
    Main entry point for the CLI. Defaults to showing help.
    """
    if ctx.invoked_subcommand is None:
        console.print(Panel("[bold cyan]Welcome to HealthFlow V2[/bold cyan]",
                            subtitle="A Self-Evolving Meta-System for Agentic Coders",
                            border_style="cyan"))
        console.print("\nRun `[bold]python run_healthflow.py --help[/bold]` for commands.")
        console.print("  - `[bold]run \"<your task>\"[/bold]` to execute a single task.")
        console.print("  - `[bold]interactive[/bold]` to start a chat-like session.")

if __name__ == "__main__":
    app()