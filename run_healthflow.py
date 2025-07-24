#!/usr/bin/env python3
"""
HealthFlow V2 Runner

Simple, self-evolving healthcare AI system with enhanced logging and clear workflow.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from healthflow.core.config import HealthFlowConfig
from healthflow.system import HealthFlowSystemV2
from healthflow.core.enhanced_logging import get_enhanced_logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


async def main():
    """Main entry point for HealthFlow V2."""
    console = Console()
    enhanced_logger = get_enhanced_logger()
    
    # Load configuration
    try:
        config = HealthFlowConfig.from_toml("config.toml")
    except Exception as e:
        console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
        console.print("[dim]Please ensure config.toml exists and is properly formatted.[/dim]")
        return 1
    
    # Initialize system
    system = HealthFlowSystemV2(config)
    
    try:
        # Start the system
        await system.start()
        
        # Display welcome message
        console.print(Panel(
            "[bold cyan]HealthFlow V2 - Simple & Self-Evolving Healthcare AI[/bold cyan]\n\n"
            "[white]Features:[/white]\n"
            "🧠 LLM-driven reasoning with ReAct loops\n"
            "🔄 Self-evolving prompts and strategies\n"
            "📊 Transparent evolution tracking\n"
            "🚀 Simple yet effective framework\n\n"
            "[dim]Type 'exit' or 'quit' to end. Type 'status' for system metrics.[/dim]",
            title="Welcome",
            border_style="cyan"
        ))
        
        # Interactive loop
        while True:
            try:
                console.print()
                task_input = console.input("[bold green]HealthFlow>[/bold green] ").strip()
                
                if not task_input:
                    continue
                
                if task_input.lower() in ['exit', 'quit']:
                    break
                
                if task_input.lower() == 'status':
                    display_system_status(console, system)
                    continue
                
                if task_input.lower() == 'help':
                    display_help(console)
                    continue
                
                # Execute the task
                console.print()
                result = await system.run_task(task_input)
                
                # Display result
                display_task_result(console, result)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️  Task interrupted by user[/yellow]")
                continue
            except Exception as e:
                enhanced_logger.error("Unexpected error during task execution", e)
                continue
        
        # Shutdown
        console.print("\n[dim]Shutting down HealthFlow V2...[/dim]")
        await system.stop()
        
        # Display final stats
        status = system.get_system_status()
        console.print(f"\n[green]✅ Completed {status['task_count']} tasks this session[/green]")
        console.print("[cyan]Thank you for using HealthFlow V2![/cyan]")
        
        return 0
        
    except Exception as e:
        enhanced_logger.error("System initialization failed", e)
        return 1


def display_system_status(console: Console, system: HealthFlowSystemV2):
    """Display current system status and metrics."""
    status = system.get_system_status()
    
    # Main status table
    status_table = Table(title="🔍 System Status", show_header=True, header_style="bold blue")
    status_table.add_column("Metric", style="cyan", no_wrap=True)
    status_table.add_column("Value", style="white")
    
    status_table.add_row("Tasks Completed", str(status['task_count']))
    status_table.add_row("System Running", "✅ Yes" if status['is_running'] else "❌ No")
    
    # Prompt evolution table
    prompt_table = Table(title="📝 Prompt Evolution", show_header=True, header_style="bold magenta")
    prompt_table.add_column("Role", style="cyan")
    prompt_table.add_column("Versions", style="white")
    prompt_table.add_column("Best Score", style="green")
    
    for role in ["orchestrator", "expert", "analyst"]:
        versions = status['prompt_versions'].get(role, 0)
        score = status['best_prompt_scores'].get(role, 0.0)
        prompt_table.add_row(role.title(), str(versions), f"{score:.2f}")
    
    # Strategy performance table
    strategy_table = Table(title="🤝 Strategy Performance", show_header=True, header_style="bold yellow")
    strategy_table.add_column("Strategy", style="cyan")
    strategy_table.add_column("Success Rate", style="green")
    
    for strategy, rate in status['strategy_performance'].items():
        strategy_table.add_row(strategy.replace("_", " ").title(), f"{rate:.1%}")
    
    console.print(status_table)
    console.print()
    console.print(prompt_table)
    console.print()
    console.print(strategy_table)


def display_help(console: Console):
    """Display help information."""
    help_text = """
[bold cyan]HealthFlow V2 Commands:[/bold cyan]

[white]Task Execution:[/white]
• Simply type your question or task to get started
• Examples:
  - "Calculate BMI for height 175cm, weight 70kg"
  - "Create a transformer model for EHR time series analysis"
  - "Explain hypertension treatment guidelines"

[white]System Commands:[/white]
• [green]status[/green]  - Show system metrics and evolution progress
• [green]help[/green]    - Show this help message
• [green]exit[/green]    - Exit HealthFlow V2

[white]Features:[/white]
• 🧠 Intelligent task routing between medical expert and data analyst
• 🔄 ReAct loops for iterative problem solving (max 3 rounds)
• 🧬 Self-evolving prompts that improve over time
• 📊 Transparent performance tracking
    """
    
    console.print(Panel(help_text, title="Help", border_style="blue"))


def display_task_result(console: Console, result: dict):
    """Display task execution result."""
    success_icon = "✅" if result['success'] else "❌"
    color = "green" if result['success'] else "red"
    
    # Create result panel
    result_text = f"[bold {color}]{success_icon} Task Result[/bold {color}]\n\n"
    result_text += f"[white]{result['result']}[/white]\n\n"
    result_text += f"[dim]Execution Time: {result['execution_time']:.1f}s | "
    result_text += f"Task #{result['task_count']}[/dim]"
    
    console.print(Panel(
        result_text,
        title="🏁 Result",
        border_style=color,
        expand=False
    ))


if __name__ == "__main__":
    try:
        # Set up basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('healthflow_v2.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Run the main function
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)