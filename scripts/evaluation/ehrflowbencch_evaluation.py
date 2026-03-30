from pathlib import Path

from deterministic_benchmark_eval import run_cli


if __name__ == "__main__":
    run_cli(default_benchmark_file=str(Path(__file__).resolve().parents[4] / "data" / "ehrflowbench.jsonl"))
