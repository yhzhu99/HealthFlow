"""
HealthFlow benchmarking script.

This script runs the HealthFlow system directly on a dataset of tasks and collects
structured results for evaluation and reproducibility reporting.
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).parent))

from healthflow.core.config import get_config, setup_logging
from healthflow.system import HealthFlowSystem

app = typer.Typer(
    name="benchmark",
    help="Benchmark the HealthFlow system on datasets",
    add_completion=False,
)
console = Console()


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load tasks from a JSONL file."""
    if not dataset_path.exists():
        console.print(f"[bold red]Error:[/bold red] Dataset file not found: {dataset_path}")
        raise typer.Exit(code=1)

    tasks = []
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                task_data = json.loads(line)
            except json.JSONDecodeError:
                console.print(f"[bold yellow]Warning:[/bold yellow] Invalid JSON on line {line_num}")
                continue
            if not all(key in task_data for key in ["qid", "task", "answer"]):
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Line {line_num} missing required fields (qid, task, answer)"
                )
                continue
            tasks.append(task_data)

    return tasks


def create_output_directory(dataset_name: str, active_executor: str, active_llm: str, qid: str) -> Path:
    output_dir = Path("benchmark_results") / dataset_name / active_executor / active_llm / str(qid)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def copy_workspace_files(workspace_path: str | None, output_dir: Path):
    if not workspace_path:
        return

    workspace = Path(workspace_path)
    if not workspace.exists():
        return

    try:
        for item in workspace.iterdir():
            if item.is_file():
                shutil.copy2(item, output_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
    except Exception as exc:
        console.print(f"[bold yellow]Warning:[/bold yellow] Failed to copy workspace files: {exc}")


def extract_last_attempt_score(result: Dict[str, Any]) -> float:
    workspace_path = result.get("workspace_path")
    if not workspace_path:
        return 0.0

    history_file = Path(workspace_path) / "full_history.json"
    if not history_file.exists():
        return 0.0

    try:
        history = json.loads(history_file.read_text(encoding="utf-8"))
        attempts = history.get("attempts", [])
        if not attempts:
            return 0.0
        return attempts[-1].get("evaluation", {}).get("score", 0.0)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return 0.0


def _initialize_system(
    config_path: Path,
    experience_path: Path,
    active_llm: str,
    active_executor: str | None,
    memory_mode_override: str | None = None,
) -> HealthFlowSystem:
    try:
        config = get_config(config_path, active_llm, active_executor)
        if memory_mode_override:
            config.memory.mode = memory_mode_override
        setup_logging(config)
        return HealthFlowSystem(config=config, experience_path=experience_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(Panel(f"[bold red]Initialization Error:[/bold red] {exc}", title="Error", border_style="red"))
        raise typer.Exit(code=1)


async def run_benchmark_async(
    dataset_path: Path,
    dataset_name: str,
    config_path: Path,
    experience_path: Path,
    active_llm: str,
    active_executor: str | None,
):
    console.print(
        Panel(
            f"[bold cyan]HealthFlow Benchmarking[/bold cyan]\n\n"
            f"Dataset: {dataset_path}\n"
            f"Name: {dataset_name}\n"
            f"Memory mode: frozen_train",
            border_style="cyan",
        )
    )

    tasks = load_dataset(dataset_path)
    console.print(f"[green]Loaded {len(tasks)} tasks from dataset[/green]")
    if not tasks:
        console.print("[bold red]No valid tasks found in dataset[/bold red]")
        raise typer.Exit(code=1)

    system = _initialize_system(
        config_path=config_path,
        experience_path=experience_path,
        active_llm=active_llm,
        active_executor=active_executor,
        memory_mode_override="frozen_train",
    )

    results_dir = Path("benchmark_results") / dataset_name / system.config.active_executor_name / active_llm
    results_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    with Progress() as progress:
        task_progress = progress.add_task(f"[cyan]Processing {dataset_name}...", total=len(tasks))
        for task_data in tasks:
            qid = str(task_data["qid"])
            task_text = task_data["task"]
            reference_answer = task_data["answer"]
            task_metadata = {
                key: value
                for key, value in task_data.items()
                if key not in {"qid", "task", "answer"}
            }
            output_dir = create_output_directory(dataset_name, system.config.active_executor_name, active_llm, qid)
            progress.update(task_progress, description=f"[cyan]Processing task {qid}...")

            try:
                result = await system.run_task(task_text)
            except Exception as exc:
                result = {
                    "success": False,
                    "answer": "",
                    "final_summary": str(exc),
                    "workspace_path": None,
                    "backend": system.config.active_executor_name,
                    "reasoning_model": system.config.llm_config_for_role("planner").model_name,
                    "memory_mode": system.config.memory.mode,
                    "verification_passed": False,
                    "execution_time": 0.0,
                    "log_path": None,
                    "verification_path": None,
                    "memory_context_path": None,
                    "run_result_path": None,
                    "error": str(exc),
                }

            copy_workspace_files(result.get("workspace_path"), output_dir)
            score = extract_last_attempt_score(result)
            benchmark_result = {
                "qid": qid,
                "task": task_text,
                "reference_answer": reference_answer,
                **task_metadata,
                "generated_answer": result.get("answer", ""),
                "success": result.get("success", False),
                "score": score,
                "verification_passed": result.get("verification_passed", False),
                "backend": result.get("backend"),
                "backend_version": result.get("backend_version"),
                "executor_metadata": result.get("executor_metadata"),
                "reasoning_model": result.get("reasoning_model"),
                "memory_mode": result.get("memory_mode"),
                "usage_summary": result.get("usage_summary"),
                "cost_summary": result.get("cost_summary"),
                "execution_time": result.get("execution_time", 0.0),
                "workspace_path": result.get("workspace_path"),
                "log_path": result.get("log_path"),
                "verification_path": result.get("verification_path"),
                "memory_context_path": result.get("memory_context_path"),
                "run_result_path": result.get("run_result_path"),
                "run_manifest_path": result.get("run_manifest_path"),
                "final_summary": result.get("final_summary"),
                "output_directory": str(output_dir),
            }
            results.append(benchmark_result)

            with open(output_dir / "result.json", "w", encoding="utf-8") as handle:
                json.dump(benchmark_result, handle, indent=2)

            progress.advance(task_progress)

    results_file = results_dir / "results.jsonl"
    with open(results_file, "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    successful_tasks = sum(1 for item in results if item["success"])
    verifier_passed = sum(1 for item in results if item["verification_passed"])
    average_score = sum(item["score"] for item in results) / len(results) if results else 0.0
    average_execution_time = sum(item["execution_time"] for item in results) / len(results) if results else 0.0
    llm_cost_total = sum(
        float(item.get("cost_summary", {}).get("llm_estimated_cost_usd") or 0.0)
        for item in results
    )
    summary = {
        "dataset_name": dataset_name,
        "total_tasks": len(tasks),
        "successful_tasks": successful_tasks,
        "verifier_passed_tasks": verifier_passed,
        "success_rate": successful_tasks / len(tasks) if tasks else 0,
        "verifier_pass_rate": verifier_passed / len(tasks) if tasks else 0,
        "average_score": average_score,
        "average_execution_time": average_execution_time,
        "backend": system.config.active_executor_name,
        "reasoning_model": system.config.llm_config_for_role("planner").model_name,
        "memory_mode": system.config.memory.mode,
        "total_llm_estimated_cost_usd": round(llm_cost_total, 8),
        "experience_path": str(experience_path),
        "results_file": str(results_file),
        "output_directory": str(results_dir),
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    console.print(
        Panel(
            f"[bold green]Benchmarking Complete[/bold green]\n\n"
            f"Total tasks: {len(tasks)}\n"
            f"Successful: {successful_tasks}\n"
            f"Verifier passed: {verifier_passed}\n"
            f"Success rate: {summary['success_rate']:.1%}\n"
            f"Average score: {average_score:.2f}\n"
            f"Average execution time: {average_execution_time:.2f}s\n\n"
            f"Estimated LLM cost: ${summary['total_llm_estimated_cost_usd']:.4f}\n\n"
            f"Results saved to: {results_file}\n"
            f"Summary saved to: {results_dir / 'summary.json'}",
            border_style="green",
        )
    )


@app.command()
def run(
    dataset_path: Path = typer.Argument(..., help="Path to the dataset .jsonl file"),
    dataset_name: str = typer.Argument(..., help="Name of the dataset (used for output directory)"),
    config_path: Path = typer.Option("config.toml", "--config", "-c", help="Path to the configuration file"),
    experience_path: Path = typer.Option("workspace/memory/experience.jsonl", "--experience-path", help="Path to experience file for HealthFlow"),
    active_llm: str = typer.Option(
        ...,
        "--active-llm",
        help="The active LLM key from config.toml (e.g., deepseek/deepseek-v3.2, openai/gpt-5.2)",
    ),
    active_executor: str = typer.Option(None, "--active-executor", help="The executor backend to use (e.g., claude_code, opencode, pi)"),
):
    asyncio.run(
        run_benchmark_async(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            config_path=config_path,
            experience_path=experience_path,
            active_llm=active_llm,
            active_executor=active_executor,
        )
    )


if __name__ == "__main__":
    app()
