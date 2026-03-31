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

from healthflow.benchmarks.simple import build_simple_processed_split, write_subset_manifest  # noqa: E402


def main() -> None:
    eval_result = build_simple_processed_split(
        benchmark_name="medagentsbench",
        source_filename="medagentsbench.jsonl",
        split_name="eval",
    )
    write_subset_manifest(
        "medagentsbench",
        {
            "benchmark": "medagentsbench",
            "selection_policy": "Use the committed benchmark-local hard-set subset as the canonical deterministic evaluation split.",
            "splits": [
                {
                    "split": eval_result.split,
                    "row_count": eval_result.row_count,
                    "qids": eval_result.qids,
                }
            ],
        },
    )
    print(json.dumps({"eval": eval_result.row_count}, indent=2))


if __name__ == "__main__":
    main()
