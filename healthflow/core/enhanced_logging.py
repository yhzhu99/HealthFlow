"""
Enhanced Logging System for HealthFlow

Provides clear, structured logging that shows the agent workflow step by step.
Makes it easy to understand what's happening in the terminal.
"""
import logging
from typing import Dict, Any
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from contextlib import contextmanager


class HealthFlowLogger:
    """
    Enhanced logger that provides clear visibility into agent workflows.
    
    Features:
    - Step-by-step workflow tracking
    - Agent interaction visualization
    - Performance metrics display
    - Real-time progress indicators
    """
    
    def __init__(self, name: str = "healthflow"):
        self.logger = logging.getLogger(name)
        self.console = Console()
        self.current_task_id = None
        self.current_step = 0
        self.task_start_time = None
        
    def start_task(self, task_id: str, task_description: str):
        """Start tracking a new task."""
        self.current_task_id = task_id
        self.current_step = 0
        self.task_start_time = datetime.now()
        
        # Create task header
        self.console.print(Panel(
            f"[bold cyan]Task {task_id[:8]}[/bold cyan]\n"
            f"[white]{task_description}[/white]",
            title="🎯 New Task",
            border_style="cyan",
            expand=False
        ))
        self.console.print()
    
    def step(self, step_name: str, description: str = ""):
        """Log a workflow step."""
        self.current_step += 1
        
        step_text = Text()
        step_text.append(f"Step {self.current_step}: ", style="bold blue")
        step_text.append(step_name, style="bold white")
        if description:
            step_text.append(f" - {description}", style="dim white")
        
        self.console.print(step_text)
        self.logger.info(f"Step {self.current_step}: {step_name} - {description}")
    
    def agent_thinking(self, agent_name: str, task: str):
        """Show agent thinking process."""
        self.console.print(f"🧠 [bold yellow]{agent_name}[/bold yellow] thinking about: [italic]{task}[/italic]")
    
    def agent_action(self, agent_name: str, action: str, details: str = ""):
        """Show agent taking action."""
        action_text = Text()
        action_text.append("⚡ ", style="bold yellow")
        action_text.append(f"{agent_name}", style="bold yellow")
        action_text.append(" → ", style="dim")
        action_text.append(action, style="bold green")
        if details:
            action_text.append(f" ({details})", style="dim")
        
        self.console.print(action_text)
    
    def agent_result(self, agent_name: str, success: bool, summary: str = ""):
        """Show agent result."""
        icon = "✅" if success else "❌"
        color = "green" if success else "red"
        
        result_text = Text()
        result_text.append(f"{icon} ", style=color)
        result_text.append(f"{agent_name}", style=f"bold {color}")
        result_text.append(" result: ", style="dim")
        result_text.append("Success" if success else "Failed", style=f"bold {color}")
        if summary:
            result_text.append(f" - {summary}", style="dim")
        
        self.console.print(result_text)
    
    def code_execution(self, code_preview: str, success: bool, result_preview: str = ""):
        """Show code execution status."""
        icon = "🐍" if success else "⚠️"
        color = "green" if success else "red"
        
        # Truncate long code/results for display
        code_display = code_preview[:100] + "..." if len(code_preview) > 100 else code_preview
        result_display = result_preview[:100] + "..." if len(result_preview) > 100 else result_preview
        
        exec_text = Text()
        exec_text.append(f"{icon} Code execution: ", style=f"bold {color}")
        exec_text.append(code_display, style="dim cyan")
        if result_display:
            exec_text.append(f" → {result_display}", style="dim")
        
        self.console.print(exec_text)
    
    def evolution_event(self, event_type: str, details: str):
        """Show system evolution events."""
        self.console.print(f"🧬 [bold magenta]Evolution[/bold magenta]: {event_type} - {details}")
    
    def performance_metrics(self, metrics: Dict[str, Any]):
        """Display performance metrics in a nice table."""
        table = Table(title="📊 Performance Metrics", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        
        for key, value in metrics.items():
            # Format different types appropriately
            if isinstance(value, float):
                display_value = f"{value:.2f}"
            elif isinstance(value, dict):
                display_value = f"{len(value)} items"
            else:
                display_value = str(value)
            
            table.add_row(key.replace("_", " ").title(), display_value)
        
        self.console.print(table)
        self.console.print()
    
    def finish_task(self, success: bool, execution_time: float, result_summary: str = ""):
        """Finish task tracking."""
        if self.task_start_time:
            elapsed = (datetime.now() - self.task_start_time).total_seconds()
        else:
            elapsed = execution_time
        
        icon = "✅" if success else "❌"
        color = "green" if success else "red"
        
        # Create completion panel
        completion_text = f"[bold {color}]{icon} Task {'Completed' if success else 'Failed'}[/bold {color}]\n"
        completion_text += f"[dim]Time: {elapsed:.1f}s[/dim]"
        if result_summary:
            completion_text += f"\n[white]{result_summary}[/white]"
        
        self.console.print(Panel(
            completion_text,
            title="🏁 Task Result",
            border_style=color,
            expand=False
        ))
        self.console.print()
        
        # Reset tracking
        self.current_task_id = None
        self.current_step = 0
        self.task_start_time = None
    
    @contextmanager
    def progress_context(self, description: str):
        """Context manager for showing progress during long operations."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            progress.add_task(description, total=None)
            try:
                yield progress
            finally:
                progress.stop()
    
    def error(self, message: str, exception: Exception = None):
        """Log error with nice formatting."""
        error_text = Text()
        error_text.append("❌ ERROR: ", style="bold red")
        error_text.append(message, style="red")
        
        self.console.print(error_text)
        if exception:
            self.console.print(f"[dim red]Details: {str(exception)}[/dim red]")
        
        self.logger.error(f"ERROR: {message}" + (f" - {exception}" if exception else ""))
    
    def warning(self, message: str):
        """Log warning with nice formatting."""
        warning_text = Text()
        warning_text.append("⚠️  WARNING: ", style="bold yellow")
        warning_text.append(message, style="yellow")
        
        self.console.print(warning_text)
        self.logger.warning(f"WARNING: {message}")
    
    def info(self, message: str):
        """Log info with nice formatting."""
        self.console.print(f"ℹ️  [dim]{message}[/dim]")
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug information."""
        self.logger.debug(message)


# Global enhanced logger instance
enhanced_logger = HealthFlowLogger()


def get_enhanced_logger() -> HealthFlowLogger:
    """Get the global enhanced logger instance."""
    return enhanced_logger