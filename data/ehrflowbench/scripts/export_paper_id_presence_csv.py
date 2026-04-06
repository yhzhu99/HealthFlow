from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "config.toml").exists() and (parent / "data").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")


PROJECT_ROOT = find_project_root()
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "ehrflowbench" / "processed" / "test.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "ehrflowbench" / "processed" / "test_paper_id_presence.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper_id presence as a CSV indicator table.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=162)
    return parser.parse_args()


def load_paper_ids(path: Path) -> set[int]:
    paper_ids: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            paper_ids.add(int(payload["paper_id"]))
    return paper_ids


def export_presence_csv(input_path: Path, output_path: Path, start_id: int, end_id: int) -> None:
    paper_ids = load_paper_ids(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paper_id", "present"])
        for paper_id in range(start_id, end_id + 1):
            writer.writerow([paper_id, 1 if paper_id in paper_ids else 0])


def main() -> None:
    args = parse_args()
    export_presence_csv(
        input_path=args.input_path,
        output_path=args.output_path,
        start_id=args.start_id,
        end_id=args.end_id,
    )


if __name__ == "__main__":
    main()
