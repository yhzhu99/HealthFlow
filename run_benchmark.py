"""
HealthFlow Benchmarking Script

This script runs the HealthFlow system on a dataset of tasks and collects
results for performance evaluation.
"""

import json
import subprocess
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import typer
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.panel import Panel

app = typer.Typer(
    name="benchmark",
    help="Benchmark the HealthFlow system on datasets",
    add_completion=False,
)
console = Console()


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load tasks from a .jsonl file."""
    if not dataset_path.exists():
        console.print(f"[bold red]Error:[/bold red] Dataset file not found: {dataset_path}")
        raise typer.Exit(code=1)

    tasks = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    task_data = json.loads(line)
                    # Validate required fields
                    if not all(key in task_data for key in ['qid', 'task', 'answer']):
                        console.print(f"[bold yellow]Warning:[/bold yellow] Line {line_num} missing required fields (qid, task, answer)")
                        continue
                    tasks.append(task_data)
                except json.JSONDecodeError:
                    console.print(f"[bold yellow]Warning:[/bold yellow] Invalid JSON on line {line_num}")
                    continue
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to read dataset: {e}")
        raise typer.Exit(code=1)

    return tasks


def create_output_directory(dataset_name: str, qid: str) -> Path:
    """Create the output directory for a specific task."""
    output_dir = Path("benchmark_results") / dataset_name / str(qid)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_healthflow_task(task: str, output_dir: Path, config_path: str = None, experience_path: str = None, shell: str = None, active_llm: str = None) -> Dict[str, Any]:
    """Run a single HealthFlow task and capture the output."""
    try:
        # Build the command with optional arguments
        cmd = [sys.executable, "run_healthflow.py", "run", task]

        if config_path:
            cmd.extend(["--config", config_path])
        if experience_path:
            cmd.extend(["--experience-path", experience_path])
        if shell:
            cmd.extend(["--shell", shell])
        if active_llm:
            cmd.extend(["--active-llm", active_llm])

        # Run the HealthFlow system
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        # Save the raw output
        with open(output_dir / "stdout.txt", "w", encoding="utf-8") as f:
            f.write(result.stdout)

        with open(output_dir / "stderr.txt", "w", encoding="utf-8") as f:
            f.write(result.stderr)

        # Save execution info
        execution_info = {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }

        with open(output_dir / "execution_info.json", "w", encoding="utf-8") as f:
            json.dump(execution_info, f, indent=2)

        return execution_info

    except Exception as e:
        console.print(f"[bold red]Error running task:[/bold red] {e}")
        return {
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False
        }


def extract_answer_from_output(stdout: str) -> str:
    """Extract the generated answer from HealthFlow output."""
    lines = stdout.split('\n')

    # Look for the ANSWER section in the output
    in_answer_section = False
    answer_lines = []

    for line in lines:
        if "ANSWER:" in line:
            in_answer_section = True
            # Extract the answer part after "ANSWER:"
            answer_part = line.split("ANSWER:")[-1].strip()
            if answer_part:
                answer_lines.append(answer_part)
            continue

        if in_answer_section:
            if line.strip() == "---" or "Final Outcome:" in line:
                break
            if line.strip():
                answer_lines.append(line.strip())

    if answer_lines:
        return " ".join(answer_lines)

    # Fallback: look for common answer patterns
    for line in reversed(lines):
        line = line.strip()
        if any(indicator in line.lower() for indicator in ["answer:", "result:", "conclusion:", "priority:"]):
            return line

    return "No answer extracted"


def copy_workspace_files(workspace_path: str, output_dir: Path):
    """Copy generated files from the workspace to the output directory."""
    if not workspace_path or workspace_path == "N/A":
        return

    workspace = Path(workspace_path)
    if not workspace.exists():
        return

    # Copy all files from workspace to output directory
    try:
        import shutil
        for item in workspace.iterdir():
            if item.is_file():
                shutil.copy2(item, output_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
    except Exception as e:
        console.print(f"[bold yellow]Warning:[/bold yellow] Failed to copy workspace files: {e}")


@app.command()
def run(
    dataset_path: Path = typer.Argument(..., help="Path to the dataset .jsonl file"),
    dataset_name: str = typer.Argument(..., help="Name of the dataset (used for output directory)"),
    config_path: str = typer.Option("config.toml", "--config", "-c", help="Path to the configuration file"),
    experience_path: str = typer.Option("workspace/experience.jsonl", "--experience-path", help="Path to experience file for HealthFlow"),
    shell: str = typer.Option("/usr/bin/zsh", "--shell", help="Shell to use for command execution"),
    active_llm: str = typer.Option(None, "--active-llm", help="Override the active LLM from config.toml (e.g., deepseek-v3, deepseek-r1, kimi-k2, gemini)"),
):
    """
    Run HealthFlow benchmarking on a dataset.

    The dataset file should be in .jsonl format with each line containing:
    {"qid": <unique_id>, "task": "<task_description>", "answer": "<reference_answer>"}
    """
    console.print(Panel(f"[bold cyan]HealthFlow Benchmarking[/bold cyan]\n\nDataset: {dataset_path}\nName: {dataset_name}", border_style="cyan"))

    # Load the dataset
    tasks = load_dataset(dataset_path)
    console.print(f"[green]Loaded {len(tasks)} tasks from dataset[/green]")

    if not tasks:
        console.print("[bold red]No valid tasks found in dataset[/bold red]")
        raise typer.Exit(code=1)

    # Create main results directory
    results_dir = Path("benchmark_results") /dataset_name / active_llm
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Process each task with progress bar
    with Progress() as progress:
        task_progress = progress.add_task(f"[cyan]Processing {dataset_name}...", total=len(tasks))

        for task_data in tasks:
            qid = str(task_data['qid'])
            task_text = task_data['task']
            reference_answer = task_data['answer']

            progress.update(task_progress, description=f"[cyan]Processing task {qid}...")

            # Create output directory for this task
            output_dir = create_output_directory(dataset_name, qid)

            # Run the HealthFlow task
            execution_info = run_healthflow_task(task_text, output_dir, config_path, experience_path, shell, active_llm)

            # Extract the generated answer
            generated_answer = extract_answer_from_output(execution_info['stdout'])

            # Copy workspace files if available
            workspace_pattern = "Workspace: "
            for line in execution_info['stdout'].split('\n'):
                if workspace_pattern in line:
                    path_text = line.split(workspace_pattern)[-1].strip()
                    path_parts = path_text.split()
                    if path_parts:
                        workspace_path = path_parts[0]
                        copy_workspace_files(workspace_path, output_dir)
                    break

            # Create result entry
            result_entry = {
                "qid": qid,
                "task": task_text,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "success": execution_info['success'],
                "output_directory": str(output_dir)
            }

            results.append(result_entry)

            # Save individual result
            with open(output_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(result_entry, f, indent=2)

            progress.advance(task_progress)

    # Save aggregated results
    results_file = results_dir / "results.jsonl"
    with open(results_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Create summary
    successful_tasks = sum(1 for r in results if r['success'])
    summary = {
        "dataset_name": dataset_name,
        "total_tasks": len(tasks),
        "successful_tasks": successful_tasks,
        "success_rate": successful_tasks / len(tasks) if tasks else 0,
        "results_file": str(results_file),
        "output_directory": str(results_dir)
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Display final summary
    console.print(Panel(
        f"[bold green]Benchmarking Complete[/bold green]\n\n"
        f"Total tasks: {len(tasks)}\n"
        f"Successful: {successful_tasks}\n"
        f"Success rate: {summary['success_rate']:.1%}\n\n"
        f"Results saved to: {results_file}\n"
        f"Summary saved to: {results_dir / 'summary.json'}",
        border_style="green"
    ))


if __name__ == "__main__":
    app()