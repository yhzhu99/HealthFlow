from __future__ import annotations

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
    subprocess.run([sys.executable, str(healthflow_root / "scripts" / "evaluation" / "rebuild_benchmarks.py")], check=True)


if __name__ == "__main__":
    main()
