from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def find_healthflow_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "run_benchmark.py").exists():
            return parent
    raise FileNotFoundError("Could not locate HealthFlow root")


def main() -> None:
    healthflow_root = find_healthflow_root()
    benchmark_names = ["curebench", "hle", "medagentsbench", "medagentboard", "ehrflowbench"]
    counts: dict[str, int] = {}
    for benchmark_name in benchmark_names:
        script_path = healthflow_root / "data" / benchmark_name / "scripts" / "rebuild.py"
        subprocess.run([sys.executable, str(script_path)], check=True)
        processed_dir = healthflow_root / "data" / benchmark_name / "processed"
        counts[benchmark_name] = len(list(processed_dir.glob("*.jsonl")))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
