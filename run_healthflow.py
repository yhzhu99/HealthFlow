"""
HealthFlow Runner

A simple, effective, and self-evolving AI agent framework for healthcare.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from healthflow.core.config import HealthFlowConfig
from healthflow.system import HealthFlowSystem
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('healthflow.log'), logging.StreamHandler(sys.stdout)],
    force=True
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for HealthFlow."""
    console = Console()

    console.print(Panel(
        "[bold cyan]HealthFlow - A Simple, Effective, and Self-Evolving AI Agent Framework[/bold cyan]",
        title="Welcome",
        border_style="cyan"
    ))

    try:
        config = HealthFlowConfig.from_toml("config.toml")
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        console.print(f"[bold red]❌ Error loading configuration from 'config.toml': {e}[/bold red]")
        console.print("[dim]Please ensure the file exists and is correctly formatted. You can copy it from `config.toml.example`.[/dim]")
        return 1

    system = HealthFlowSystem(config)

    try:
        await system.start()
        console.print("[green]✅ System initialized and ready.[/green]")
        console.print("\nI am the [bold cyan]HealthFlow AI Agent[/bold cyan]. How can I help you today?")


        while True:
            console.print()
            task_input = console.input("[bold magenta]HealthFlow AI>[/bold magenta] ").strip()

            if not task_input:
                continue

            if task_input.lower() in ['exit', 'quit']:
                break

            if task_input.lower() == 'status':
                await display_system_status(console, system)
                continue

            console.print()
            spinner = Spinner("dots", text="[cyan]HealthFlow is thinking...")
            with Live(spinner, console=console, transient=True, refresh_per_second=20) as live:
                result = await system.run_task(task_input)

            display_task_result(console, result)

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]User interruption detected.[/yellow]")
    except Exception as e:
        logger.error(f"A fatal error occurred in the main loop: {e}", exc_info=True)
        console.print(f"[bold red]💥 An unexpected system error occurred: {e}[/bold red]")
    finally:
        console.print("\n[dim]Shutting down HealthFlow...[/dim]")
        await system.stop()
        console.print("[bold green]Goodbye![/bold green]")

    return 0

def display_task_result(console: Console, result: dict):
    """Display the final result of a task in a formatted panel."""
    success = result.get('success', False)
    panel_color = "green" if success else "red"
    title = f"[bold {panel_color}]✅ Task Succeeded[/bold {panel_color}]" if success else f"[bold {panel_color}]❌ Task Failed[/bold {panel_color}]"

    content = f"[bold]Final Answer:[/]\n{result.get('result', 'No result was generated.')}\n\n"
    content += f"[dim]Execution Time: {result.get('execution_time', 0):.2f}s | "
    content += f"Task ID: {result.get('task_id', 'N/A')[:8]}[/dim]\n"

    if "evaluation" in result and result["evaluation"]:
        eval_data = result["evaluation"]
        content += "\n--- [bold]Evaluation[/bold] ---\n"
        content += f"[bold]Score:[/] {eval_data.get('overall_score', 'N/A'):.1f}/10.0\n"
        content += f"[bold]Summary:[/] {eval_data.get('executive_summary', 'No summary.')}"

    console.print(Panel(content, title=title, border_style=panel_color, expand=False))


async def display_system_status(console: Console, system: HealthFlowSystem):
    """Display the current status and evolution metrics of the system."""
    status = await system.get_system_status()

    console.print(Panel(f"[bold blue]System Status[/bold blue] | Tasks Completed: {status['task_count']}", expand=False))

    # Prompt Evolution Table
    prompt_table = Table(title="📝 Prompt Evolution", show_header=True, header_style="bold magenta")
    prompt_table.add_column("Role", style="cyan")
    prompt_table.add_column("Versions", style="white")
    prompt_table.add_column("Best Score", style="green")

    for role, data in status['prompt_status'].items():
        prompt_table.add_row(role.title(), str(data['versions']), f"{data['best_score']:.2f}")

    # Strategy Performance Table
    strategy_table = Table(title="🧠 Strategy Performance", show_header=True, header_style="bold yellow")
    strategy_table.add_column("Strategy", style="cyan")
    strategy_table.add_column("Success Rate", style="green")
    strategy_table.add_column("Usage Count", style="white")

    for name, stats in status['strategy_performance'].items():
        strategy_table.add_row(name.replace("_", " ").title(), f"{stats['success_rate']:.1%}", str(stats['usage_count']))

    # ToolBank Status Table
    tool_table = Table(title="🛠️ ToolBank Status", show_header=True, header_style="bold green")
    tool_table.add_column("Tool Name", style="cyan")
    tool_table.add_column("Description", style="white")

    for tool_name, doc in status['tool_status'].items():
        tool_table.add_row(tool_name, doc.split('\n')[0]) # Show first line of docstring

    console.print(prompt_table)
    console.print(strategy_table)
    console.print(tool_table)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Application failed to run: {e}", exc_info=True)
        sys.exit(1)