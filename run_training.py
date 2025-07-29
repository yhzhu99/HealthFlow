#!/usr/bin/env python3
"""
Training script for HealthFlow system.
Processes training data from JSONL files and saves experiences.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse
from dataclasses import dataclass
import shutil

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from healthflow.system import HealthFlowSystem
from healthflow.core.config import get_config, setup_logging

console = Console()

@dataclass
class TrainingExample:
    """Represents a single training example."""
    qid: str
    task: str
    answer: str

class TrainingRunner:
    """Handles the training process for HealthFlow."""
    
    def __init__(self, system: HealthFlowSystem, experience_path: Path):
        self.system = system
        self.experience_path = experience_path
        self.results: List[Dict[str, Any]] = []
    
    async def run_training(self, training_file: Path) -> Dict[str, Any]:
        """Run training on all examples in the training file."""
        training_examples = self._load_training_data(training_file)
        
        console.print(Panel(
            f"[bold cyan]Starting HealthFlow Training[/bold cyan]\n\n"
            f"[dim]Training file:[/dim] {training_file}\n"
            f"[dim]Total examples:[/dim] {len(training_examples)}\n"
            f"[dim]Experience path:[/dim] {self.experience_path}",
            border_style="cyan"
        ))
        
        successful_tasks = 0
        total_tasks = len(training_examples)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            main_task = progress.add_task("Training Progress", total=total_tasks)
            
            for i, example in enumerate(training_examples):
                progress.update(main_task, description=f"Processing example {i+1}/{total_tasks} (QID: {example.qid})")
                
                try:
                    result = await self.system.run_task(
                        user_request=example.task,
                        train_mode=True,
                        reference_answer=example.answer
                    )
                    
                    if result.get("success", False):
                        successful_tasks += 1
                    
                    # Store result with training metadata
                    training_result = {
                        "qid": example.qid,
                        "task": example.task,
                        "reference_answer": example.answer,
                        "result": result,
                        "success": result.get("success", False),
                        "score": self._extract_score_from_result(result),
                        "workspace_path": result.get("workspace_path")
                    }
                    self.results.append(training_result)
                    
                    logger.info(f"Training example {example.qid} completed. Success: {result.get('success', False)}")
                    
                except Exception as e:
                    logger.error(f"Error processing training example {example.qid}: {e}")
                    error_result = {
                        "qid": example.qid,
                        "task": example.task,
                        "reference_answer": example.answer,
                        "result": {"success": False, "error": str(e)},
                        "success": False,
                        "score": 0.0,
                        "workspace_path": None
                    }
                    self.results.append(error_result)
                
                progress.advance(main_task)
        
        # Calculate summary statistics
        summary = self._calculate_summary(successful_tasks, total_tasks)
        self._display_summary(summary)
        
        return summary
    
    def _load_training_data(self, training_file: Path) -> List[TrainingExample]:
        """Load training examples from JSONL file."""
        examples = []
        
        if not training_file.exists():
            raise FileNotFoundError(f"Training file not found: {training_file}")
        
        with open(training_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # Validate required keys
                    required_keys = ["qid", "task", "answer"]
                    missing_keys = [key for key in required_keys if key not in data]
                    if missing_keys:
                        logger.warning(f"Skipping line {line_num}: missing keys {missing_keys}")
                        continue
                    
                    examples.append(TrainingExample(
                        qid=str(data["qid"]),
                        task=data["task"],
                        answer=data["answer"]
                    ))
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
        
        if not examples:
            raise ValueError(f"No valid training examples found in {training_file}")
        
        logger.info(f"Loaded {len(examples)} training examples from {training_file}")
        return examples
    
    def _extract_score_from_result(self, result: Dict[str, Any]) -> float:
        """Extract evaluation score from result."""
        try:
            # The result structure should be checked from the actual logs
            # Let's try to find the score from the workspace full_history.json
            workspace_path = result.get("workspace_path")
            if workspace_path:
                history_file = Path(workspace_path) / "full_history.json"
                if history_file.exists():
                    with open(history_file, 'r') as f:
                        history = json.loads(f.read())
                        attempts = history.get("attempts", [])
                        if attempts and "evaluation" in attempts[-1]:
                            return attempts[-1]["evaluation"].get("score", 0.0)
            return 0.0
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, FileNotFoundError):
            return 0.0
    
    def _calculate_summary(self, successful_tasks: int, total_tasks: int) -> Dict[str, Any]:
        """Calculate training summary statistics."""
        success_rate = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Calculate average score
        scores = [r["score"] for r in self.results if isinstance(r["score"], (int, float))]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        summary = {
            "total_examples": total_tasks,
            "successful_examples": successful_tasks,
            "failed_examples": total_tasks - successful_tasks,
            "success_rate": success_rate,
            "average_score": avg_score,
            "results": self.results
        }
        
        return summary
    
    def _display_summary(self, summary: Dict[str, Any]):
        """Display training summary."""
        table = Table(title="Training Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Examples", str(summary["total_examples"]))
        table.add_row("Successful", str(summary["successful_examples"]))
        table.add_row("Failed", str(summary["failed_examples"]))
        table.add_row("Success Rate", f"{summary['success_rate']:.1f}%")
        table.add_row("Average Score", f"{summary['average_score']:.2f}/10.0")
        
        console.print("\n")
        console.print(table)
    
    def save_results(self, dataset_name: str, active_llm: str):
        """Save training results using the benchmark-style directory structure."""
        # Create main results directory using the same structure as benchmark
        results_dir = Path("benchmark_results") / dataset_name / active_llm
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual results in qid directories (same as benchmark)
        for result in self.results:
            qid = result["qid"]
            qid_dir = results_dir / str(qid)
            qid_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual result JSON
            with open(qid_dir / "training_result.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            # Copy workspace files if available
            workspace_path = result.get("workspace_path")
            if workspace_path and Path(workspace_path).exists():
                self._copy_workspace_files(Path(workspace_path), qid_dir)
        
        # Save aggregated results
        results_file = results_dir / "training_results.jsonl"
        with open(results_file, 'w', encoding='utf-8') as f:
            for result in self.results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        console.print(f"\n[green]Training results saved to: {results_file}[/green]")
        logger.info(f"Training results saved to {results_file}")
        
        return results_dir
    
    def _copy_workspace_files(self, workspace_path: Path, output_dir: Path):
        """Copy generated files from the workspace to the output directory."""
        if not workspace_path.exists():
            return
        
        try:
            for item in workspace_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, output_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to copy workspace files: {e}")

def _initialize_system(config_path: Path, experience_path: Path, shell: str, active_llm: str) -> HealthFlowSystem:
    """Initialize the HealthFlow system."""
    try:
        config = get_config(config_path, active_llm)
        setup_logging(config)
        return HealthFlowSystem(
            config=config,
            experience_path=experience_path,
            shell=shell
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(Panel(f"[bold red]Initialization Error:[/bold red] {e}", title="Error", border_style="red"))
        raise typer.Exit(code=1)

async def main_async(
    training_file: Path,
    dataset_name: str,
    config_path: Path,
    experience_path: Path,
    shell: str,
    active_llm: str = None
):
    """Main async function to run training."""
    system = _initialize_system(config_path, experience_path, shell, active_llm)
    
    trainer = TrainingRunner(system, experience_path)
    summary = await trainer.run_training(training_file)
    results_dir = trainer.save_results(dataset_name, active_llm)
    
    # Create summary file (similar to benchmark)
    summary_data = {
        "dataset_name": dataset_name,
        "total_examples": summary["total_examples"],
        "successful_examples": summary["successful_examples"],
        "failed_examples": summary["failed_examples"], 
        "success_rate": summary["success_rate"],
        "average_score": summary["average_score"],
        "results_directory": str(results_dir)
    }
    
    with open(results_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    
    # Exit with non-zero code if success rate is too low
    if summary["success_rate"] < 50:
        console.print(f"\n[bold red]Training completed with low success rate: {summary['success_rate']:.1f}%[/bold red]")
        raise typer.Exit(code=1)
    else:
        console.print(f"\n[bold green]Training completed successfully! Success rate: {summary['success_rate']:.1f}%[/bold green]")

def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Run HealthFlow training on a dataset")
    parser.add_argument("training_file", type=Path, help="Path to the training JSONL file")
    parser.add_argument("dataset_name", help="Name of the dataset (used for output directory structure)")
    parser.add_argument("--config", "-c", type=Path, default="config.toml", help="Path to the configuration file")
    parser.add_argument("--experience-path", type=Path, default="workspace/experience.jsonl", help="Path to the experience knowledge base file")
    parser.add_argument("--shell", default="/usr/bin/zsh", help="Shell to use for subprocess execution")
    parser.add_argument("--active-llm", required=True, help="The active LLM to use (e.g., deepseek-v3, deepseek-r1, kimi-k2, gemini)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main_async(
            training_file=args.training_file,
            dataset_name=args.dataset_name,
            config_path=args.config,
            experience_path=args.experience_path,
            shell=args.shell,
            active_llm=args.active_llm
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Training failed: {e}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()