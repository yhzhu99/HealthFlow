from __future__ import annotations

import argparse
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


HEALTHFLOW_ROOT = find_healthflow_root()
if str(HEALTHFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(HEALTHFLOW_ROOT))

from healthflow.benchmarks.ehr_benchmark_builders import rebuild_ehrflowbench  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh EHRFlowBench raw inputs and rebuild the processed benchmark.")
    parser.add_argument(
        "--include-markdowns",
        action="store_true",
        help="Also materialize raw/papers/markdowns before rebuilding.",
    )
    args = parser.parse_args()
    prepare_raw_args = [sys.executable, str(Path(__file__).with_name("prepare_raw.py"))]
    if args.include_markdowns:
        prepare_raw_args.append("--include-markdowns")
    subprocess.run(prepare_raw_args, check=True)
    print(json.dumps(rebuild_ehrflowbench(), indent=2))


if __name__ == "__main__":
    main()
