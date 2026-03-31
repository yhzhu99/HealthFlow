from __future__ import annotations

import json
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

from healthflow.benchmarks.ehr_benchmark_builders import rebuild_medagentboard  # noqa: E402


def main() -> None:
    print(json.dumps(rebuild_medagentboard(), indent=2))


if __name__ == "__main__":
    main()
