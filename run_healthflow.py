import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner

# Add project root to path to allow direct execution from the root directory
sys.path.insert(0, str(Path(__file__).parent))

from healthflow.system import HealthFlowSystem
from healthflow.core.config import get_config, setup_logging

app = typer.Typer(
    name="healthflow",
    help="A Self-Evolving Meta-System for Orchestrating Agentic Coders in Healthcare.",
    add_completion=False,
)
console = Console()

def _display_task_result(result: dict):
    """Helper function to display the final result of a task in a Rich panel."""
    success = result.get("success", False)
    report_requested = result.get("report_requested", False)
    report_generated = result.get("report_generated", False)
    if success:
        panel_title = "[bold green]✅ Task Completed Successfully[/bold green]"
        panel_border_style = "green"
    else:
        panel_title = "[bold red]❌ Task Failed[/bold red]"
        panel_border_style = "red"

    answer = result.get('answer', 'No answer available.')

    final_report = f"""
[bold cyan]ANSWER:[/bold cyan]
{answer}

---
[bold]Final Outcome:[/bold]
{result.get('final_summary', 'No summary available.')}

---
[bold]Backend:[/bold] {result.get('backend', 'N/A')}
[bold]Backend Version:[/bold] {result.get('backend_version', 'N/A')}
[bold]Reasoning Model:[/bold] {result.get('reasoning_model', 'N/A')}
[bold]Memory Write Policy:[/bold] {result.get('memory_write_policy', 'N/A')}
[bold]Evaluation Status:[/bold] {result.get('evaluation_status', 'N/A')}
[bold]Evaluation Score:[/bold] {result.get('evaluation_score', 'N/A')}
[bold]Usage Summary:[/bold] {result.get('usage_summary', {})}
[bold]Cost Summary:[/bold] {result.get('cost_summary', {})}
[bold]Cost Analysis JSON:[/bold] {result.get('cost_analysis_path', 'N/A')}
[bold]Log Path:[/bold] {result.get('log_path', 'N/A')}
[bold]Evaluation JSON:[/bold] {result.get('evaluation_path', 'N/A')}
[bold]Memory Context:[/bold] {result.get('memory_context_path', 'N/A')}
[bold]Run Result JSON:[/bold] {result.get('run_result_path', 'N/A')}
[bold]Report Requested:[/bold] {report_requested}
[bold]Report Generated:[/bold] {report_generated}
[bold]Report Path:[/bold] {result.get('report_path') or 'N/A'}
[bold]Report Error:[/bold] {result.get('report_error') or 'N/A'}

---
[bold]Workspace:[/bold] {result.get('workspace_path', 'N/A')}
[bold]Execution Time:[/bold] {result.get('execution_time', 0):.2f}s
"""
    console.print(Panel(final_report, title=panel_title, border_style=panel_border_style))

async def run_single_task_flow(system: HealthFlowSystem, task: str, report_requested: bool = False):
    """Runs a single task and displays the result with a live spinner."""
    console.print(Panel(f"[bold cyan]Starting HealthFlow Task[/bold cyan]\n\n[dim]Task:[/dim] {task}", border_style="cyan"))

    spinner = Spinner("dots", text="HealthFlow is orchestrating...")
    with Live(spinner, console=console, transient=True, refresh_per_second=20) as live:
        result = await system.run_task(task, live, spinner, report_requested=report_requested)

    _display_task_result(result)
    # Exit with a non-zero code if the task failed, useful for scripting/CI
    if not result.get("success", False):
        raise typer.Exit(code=1)
    if report_requested and not result.get("report_generated", False):
        raise typer.Exit(code=1)

async def main_interactive_loop(system: HealthFlowSystem):
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
            # In interactive mode, we don't exit the script on failure
            await run_single_task_flow(system, task_input)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting interactive mode.[/yellow]")
            break
        except typer.Exit: # Catch the exit from run_single_task_flow
            console.print("[yellow]Task failed. Ready for next command.[/yellow]")


def _initialize_system(config_path: Path, experience_path: Path, active_llm: str, active_executor: str | None) -> HealthFlowSystem:
    """Loads config, sets up logging, and initializes the HealthFlowSystem."""
    try:
        config = get_config(config_path, active_llm, active_executor)
        setup_logging(config)
        return HealthFlowSystem(
            config=config,
            experience_path=experience_path
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(Panel(f"[bold red]Initialization Error:[/bold red] {e}", title="Error", border_style="red"))
        raise typer.Exit(code=1)

@app.command()
def run(
    task: str = typer.Argument(..., help="The high-level analysis task for HealthFlow to accomplish."),
    config_path: Path = typer.Option("config.toml", "--config", "-c", help="Path to the configuration file."),
    experience_path: Path = typer.Option("workspace/memory/experience.jsonl", "--experience-path", help="Path to the experience knowledge base file."),
    active_llm: str = typer.Option(
        ...,
        "--active-llm",
        help="The active LLM key from config.toml (e.g., deepseek/deepseek-v3.2, openai/gpt-5.4).",
    ),
    active_executor: str = typer.Option(None, "--active-executor", help="The executor backend to use (e.g., claude_code, opencode, pi)."),
    report: bool = typer.Option(False, "--report", help="Generate a standard markdown report.md in the task workspace."),
):
    """
    Run a single task through the HealthFlow system.
    """
    system = _initialize_system(config_path, experience_path, active_llm, active_executor)
    asyncio.run(run_single_task_flow(system, task, report_requested=report))

@app.command()
def interactive(
    config_path: Path = typer.Option("config.toml", "--config", "-c", help="Path to the configuration file."),
    experience_path: Path = typer.Option("workspace/memory/experience.jsonl", "--experience-path", help="Path to the experience knowledge base file."),
    active_llm: str = typer.Option(
        ...,
        "--active-llm",
        help="The active LLM key from config.toml (e.g., deepseek/deepseek-v3.2, openai/gpt-5.4).",
    ),
    active_executor: str = typer.Option(None, "--active-executor", help="The executor backend to use (e.g., claude_code, opencode, pi)."),
):
    """
    Starts HealthFlow in an interactive, chat-like mode for multiple tasks.
    """
    system = _initialize_system(config_path, experience_path, active_llm, active_executor)
    asyncio.run(main_interactive_loop(system))

@app.callback(invoke_without_command=True)
def main_entry(ctx: typer.Context):
    """
    Main entry point for the CLI. Shows help by default.
    """
    if ctx.invoked_subcommand is None:
        console.print(Panel("[bold cyan]Welcome to HealthFlow[/bold cyan]",
                            subtitle="A Self-Evolving Meta-System for Agentic AI in Healthcare",
                            border_style="cyan"))
        console.print("\nRun `[bold]python run_healthflow.py --help[/bold]` for commands.")
        console.print("  - `[bold]run \"<your task>\"[/bold]` to execute a single task.")
        console.print("  - `[bold]interactive[/bold]` to start a chat-like session.")

if __name__ == "__main__":
    app()
