import asyncio
import sys
from pathlib import Path

import typer
from loguru import logger
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


def _chat_panel_style(result: dict) -> tuple[str, str]:
    success = result.get("success", False)
    cancelled = result.get("cancelled", False)
    if cancelled:
        return "[bold yellow]Run Cancelled[/bold yellow]", "yellow"
    if success:
        return "[bold green]HealthFlow[/bold green]", "green"
    return "[bold red]Run Failed[/bold red]", "red"


def _display_chat_result(result: dict) -> None:
    success = result.get("success", False)
    cancelled = result.get("cancelled", False)
    answer = str(result.get("answer") or "").strip() or "No answer available."
    title, border_style = _chat_panel_style(result)

    console.print(Panel(answer, title=title, border_style=border_style, expand=False))
    if cancelled:
        console.print(f"[yellow]{result.get('final_summary', 'Task cancelled.')}[/yellow]")
    elif not success:
        console.print(f"[red]{result.get('final_summary', 'Task failed.')}[/red]")

    status = "cancelled" if cancelled else ("success" if success else "failed")
    console.print(f"[dim]{status} · {result.get('execution_time', 0):.2f}s[/dim]")


def _display_task_result(result: dict, *, verbose: bool = False, chat_mode: bool = False):
    """Display a task result using either a concise or detailed layout."""
    success = result.get("success", False)
    cancelled = result.get("cancelled", False)
    answer = result.get("answer", "No answer available.")

    if not verbose:
        if chat_mode:
            _display_chat_result(result)
            return

        if answer:
            console.print(answer)
        status = "cancelled" if cancelled else ("success" if success else "failed")
        status_style = "yellow" if cancelled else ("green" if success else "red")
        console.print(f"[dim]Status: [{status_style}]{status}[/{status_style}] | Time: {result.get('execution_time', 0):.2f}s[/dim]")
        if cancelled:
            console.print(f"[yellow]{result.get('final_summary', 'Task cancelled.')}[/yellow]")
        elif not success:
            console.print(f"[red]{result.get('final_summary', 'Task failed.')}[/red]")
        return

    report_requested = result.get("report_requested", False)
    report_generated = result.get("report_generated", False)
    if cancelled:
        panel_title = "[bold yellow]⚠ Task Cancelled[/bold yellow]"
        panel_border_style = "yellow"
    elif success:
        panel_title = "[bold green]✅ Task Completed Successfully[/bold green]"
        panel_border_style = "green"
    else:
        panel_title = "[bold red]❌ Task Failed[/bold red]"
        panel_border_style = "red"

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
[bold]Cancelled:[/bold] {cancelled}
[bold]Cancel Reason:[/bold] {result.get('cancel_reason') or 'N/A'}
"""
    console.print(Panel(final_report, title=panel_title, border_style=panel_border_style))

async def execute_task_with_spinner(
    system: HealthFlowSystem,
    task: str,
    report_requested: bool = False,
    *,
    verbose: bool = False,
    chat_mode: bool = False,
):
    """Runs a single task and displays the result with a live spinner."""
    if verbose and not chat_mode:
        console.print(Panel(f"[bold cyan]Starting HealthFlow Task[/bold cyan]\n\n[dim]Task:[/dim] {task}", border_style="cyan"))

    spinner = Spinner("dots", text="HealthFlow is orchestrating...")
    with Live(spinner, console=console, transient=True, refresh_per_second=20) as live:
        result = await system.run_task(task, live, spinner, report_requested=report_requested)

    _display_task_result(result, verbose=verbose, chat_mode=chat_mode)
    return result


async def run_single_task_flow(
    system: HealthFlowSystem,
    task: str,
    report_requested: bool = False,
    *,
    verbose: bool = False,
    chat_mode: bool = False,
):
    """Runs a single task and raises a CLI exit code for scripting surfaces."""
    result = await execute_task_with_spinner(
        system,
        task,
        report_requested=report_requested,
        verbose=verbose,
        chat_mode=chat_mode,
    )
    # Exit with a non-zero code if the task failed, useful for scripting/CI
    if not result.get("success", False):
        raise typer.Exit(code=1)
    if report_requested and not result.get("report_generated", False):
        raise typer.Exit(code=1)
    return result


async def main_interactive_loop(system: HealthFlowSystem, *, verbose: bool = False):
    """Runs the basic interactive loop used for non-TTY or fallback sessions."""
    console.print(Panel("[bold green]HealthFlow Interactive Mode[/bold green]", subtitle="Type 'exit' or 'quit' to end the session.", border_style="green"))
    while True:
        try:
            task_input = console.input("\n[bold magenta]HealthFlow > [/bold magenta]").strip()
            if not task_input:
                continue
            if task_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Exiting interactive mode.[/yellow]")
                break
            await run_single_task_flow(system, task_input, verbose=verbose, chat_mode=True)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting interactive mode.[/yellow]")
            break
        except typer.Exit:
            console.print("[yellow]Task failed. Ready for next command.[/yellow]")


def _initialize_system(
    config_path: Path,
    experience_path: Path,
    active_llm: str,
    active_executor: str | None,
    *,
    verbose: bool = False,
) -> HealthFlowSystem:
    """Loads config, sets up logging, and initializes the HealthFlowSystem."""
    try:
        logger.remove()
        logger.add(sys.stderr, level="INFO" if verbose else "WARNING")
        config = get_config(config_path, active_llm, active_executor)
        setup_logging(config, console_log_level="INFO" if verbose else "WARNING")
        return HealthFlowSystem(
            config=config,
            experience_path=experience_path
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(Panel(f"[bold red]Initialization Error:[/bold red] {e}", title="Error", border_style="red"))
        raise typer.Exit(code=1)


def _build_system_factory(
    config_path: Path,
    experience_path: Path,
    active_llm: str,
    active_executor: str | None,
    *,
    verbose: bool = False,
):
    def _factory() -> HealthFlowSystem:
        return _initialize_system(
            config_path,
            experience_path,
            active_llm,
            active_executor,
            verbose=verbose,
        )

    return _factory

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
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed runtime metadata in the terminal output."),
):
    """
    Run a single task through the HealthFlow system.
    """
    system = _initialize_system(config_path, experience_path, active_llm, active_executor, verbose=verbose)
    asyncio.run(run_single_task_flow(system, task, report_requested=report, verbose=verbose))

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
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed runtime metadata in the terminal output."),
):
    """
    Starts HealthFlow in an interactive, chat-like mode for multiple tasks.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        system = _initialize_system(config_path, experience_path, active_llm, active_executor, verbose=verbose)
        asyncio.run(main_interactive_loop(system, verbose=verbose))
        return

    try:
        from healthflow.interactive_cli import InteractiveShell
    except ModuleNotFoundError:
        system = _initialize_system(config_path, experience_path, active_llm, active_executor, verbose=verbose)
        asyncio.run(main_interactive_loop(system, verbose=verbose))
        return

    system_factory = _build_system_factory(
        config_path,
        experience_path,
        active_llm,
        active_executor,
        verbose=verbose,
    )

    async def _interactive_runner(system: HealthFlowSystem, task: str) -> dict:
        return await execute_task_with_spinner(
            system,
            task,
            verbose=verbose,
            chat_mode=True,
        )

    shell = InteractiveShell(
        console=console,
        system_factory=system_factory,
        task_runner=_interactive_runner,
        verbose=verbose,
    )
    asyncio.run(shell.run())

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
