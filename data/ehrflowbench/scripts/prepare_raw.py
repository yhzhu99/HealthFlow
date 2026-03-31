from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def find_healthflow_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "run_benchmark.py").exists():
            return parent
    raise FileNotFoundError("Could not locate HealthFlow root")


HEALTHFLOW_ROOT = find_healthflow_root()
RAW_ROOT = HEALTHFLOW_ROOT / "data" / "ehrflowbench" / "raw" / "papers"
UPSTREAM_ROOT = HEALTHFLOW_ROOT / "data" / "ehrflowbench" / "scripts" / "upstream"
SOURCE_SELECTED_IDS = UPSTREAM_ROOT / "filter_paper" / "results" / "final_selected_ID.txt"
SOURCE_TASKS = UPSTREAM_ROOT / "extract_task" / "tasks"
SOURCE_MARKDOWNS = UPSTREAM_ROOT / "extract_task" / "assets" / "markdowns"


def title_from_markdown_dir(name: str) -> tuple[str, str]:
    paper_id, raw_title = name.split("_", 1)
    title = raw_title.split(".pdf", 1)[0].replace("_", " ").strip()
    return paper_id, " ".join(title.split())


def refresh_selected_ids() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_SELECTED_IDS, RAW_ROOT / "selected_ids.txt")


def refresh_extracted_tasks() -> None:
    destination = RAW_ROOT / "extracted_tasks"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(SOURCE_TASKS.glob("*_tasks.jsonl")):
        shutil.copy2(source_path, destination / source_path.name)


def refresh_paper_titles() -> None:
    if not SOURCE_MARKDOWNS.exists():
        if not (RAW_ROOT / "paper_titles.csv").exists():
            raise FileNotFoundError("Missing paper_titles.csv and no upstream markdown mirror is available to regenerate it.")
        return
    rows = [title_from_markdown_dir(path.name) for path in sorted(SOURCE_MARKDOWNS.iterdir()) if path.is_dir()]
    if not rows:
        if not (RAW_ROOT / "paper_titles.csv").exists():
            raise FileNotFoundError("Upstream markdown mirror is empty and no committed paper_titles.csv is available.")
        return
    rows.sort(key=lambda item: int(item[0]))
    with (RAW_ROOT / "paper_titles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paper_id", "paper_title"])
        writer.writerows(rows)


def refresh_markdown_mirror() -> int:
    if not SOURCE_MARKDOWNS.exists():
        raise FileNotFoundError("Cannot materialize raw markdowns because the upstream markdown mirror is not available locally.")
    destination = RAW_ROOT / "markdowns"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    shutil.copytree(SOURCE_MARKDOWNS, destination)
    return len([path for path in destination.iterdir() if path.is_dir()])


def prepare_raw(include_markdowns: bool = False) -> dict[str, int]:
    refresh_selected_ids()
    refresh_extracted_tasks()
    refresh_paper_titles()
    markdown_dirs = 0
    if include_markdowns:
        markdown_dirs = refresh_markdown_mirror()
    elif (RAW_ROOT / "markdowns").exists():
        markdown_dirs = len([path for path in (RAW_ROOT / "markdowns").iterdir() if path.is_dir()])
    with (RAW_ROOT / "paper_titles.csv").open("r", encoding="utf-8") as handle:
        paper_title_rows = sum(1 for _ in csv.DictReader(handle))
    return {
        "selected_ids": sum(1 for line in (RAW_ROOT / "selected_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()),
        "task_files": len(list((RAW_ROOT / "extracted_tasks").glob("*_tasks.jsonl"))),
        "paper_titles": paper_title_rows,
        "markdown_dirs": markdown_dirs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh benchmark-local paper raw inputs from the vendored EHRFlowBench upstream curation assets."
    )
    parser.add_argument(
        "--include-markdowns",
        action="store_true",
        help="Also copy the large markdown mirror into raw/papers/markdowns.",
    )
    args = parser.parse_args()
    print(json.dumps(prepare_raw(include_markdowns=args.include_markdowns), indent=2))


if __name__ == "__main__":
    main()
