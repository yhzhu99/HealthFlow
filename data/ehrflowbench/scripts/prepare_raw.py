from __future__ import annotations

import csv
import os
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
SOURCE_SELECTED_IDS = HEALTHFLOW_ROOT / "scripts" / "filter_paper" / "results" / "final_selected_ID.txt"
SOURCE_TASKS = HEALTHFLOW_ROOT / "scripts" / "extract_task" / "tasks"
SOURCE_MARKDOWNS = HEALTHFLOW_ROOT / "scripts" / "extract_task" / "assets" / "markdowns"


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
    rows = [title_from_markdown_dir(path.name) for path in sorted(SOURCE_MARKDOWNS.iterdir()) if path.is_dir()]
    rows.sort(key=lambda item: int(item[0]))
    with (RAW_ROOT / "paper_titles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["paper_id", "paper_title"])
        writer.writerows(rows)


def refresh_markdown_link() -> None:
    destination = RAW_ROOT / "markdowns"
    if destination.is_symlink():
        if destination.resolve() == SOURCE_MARKDOWNS.resolve():
            return
        destination.unlink()
    elif destination.exists():
        if any(destination.iterdir()):
            return
        destination.rmdir()

    relative_target = os.path.relpath(SOURCE_MARKDOWNS, RAW_ROOT)
    destination.symlink_to(relative_target, target_is_directory=True)


def main() -> None:
    refresh_selected_ids()
    refresh_extracted_tasks()
    refresh_paper_titles()
    refresh_markdown_link()
    print(
        {
            "selected_ids": sum(1 for line in (RAW_ROOT / "selected_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()),
            "task_files": len(list((RAW_ROOT / "extracted_tasks").glob("*_tasks.jsonl"))),
            "markdown_dirs": len([path for path in (RAW_ROOT / "markdowns").iterdir() if path.is_dir()]),
        }
    )


if __name__ == "__main__":
    main()
