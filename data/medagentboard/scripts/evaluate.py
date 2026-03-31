from __future__ import annotations

import sys
from pathlib import Path


def find_healthflow_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "run_benchmark.py").exists():
            return parent
    raise FileNotFoundError("Could not locate HealthFlow root")


HEALTHFLOW_ROOT = find_healthflow_root()
if str(HEALTHFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(HEALTHFLOW_ROOT))

from healthflow.benchmarks.deterministic_eval import run_cli  # noqa: E402


if __name__ == "__main__":
    run_cli(default_benchmark_file=str(HEALTHFLOW_ROOT / "data" / "medagentboard" / "processed" / "eval.jsonl"))
