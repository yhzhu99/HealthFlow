from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from summarize_focus_areas import PRIMARY_CATEGORY_META, analyze_tasks


SEED = 42
DATASET_ORDER = ("TJH", "MIMIC-IV-demo")
SELECT_COUNT_PER_DATASET = 55
TRAIN_COUNT_PER_DATASET = 5
CONTRACT_VERSION = "ehrflowbench_report_generation_v1"
SUBSET_SCHEMA_VERSION = "ehrflowbench_subset_v2"


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "config.toml").exists() and (parent / "data").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")


PROJECT_ROOT = find_project_root()
DATASET_ROOT = PROJECT_ROOT / "data" / "ehrflowbench"
PROCESSED_ROOT = DATASET_ROOT / "processed"
PAPERS_ROOT = PROCESSED_ROOT / "papers"
DEFAULT_INPUT_PATH = PAPERS_ROOT / "final_220_tasks.json"
DEFAULT_TRAIN_PATH = PROCESSED_ROOT / "train.jsonl"
DEFAULT_TEST_PATH = PROCESSED_ROOT / "test.jsonl"
DEFAULT_COMBINED_PATH = PROCESSED_ROOT / "ehrflowbench.jsonl"
DEFAULT_SUBSET_MANIFEST_PATH = PROCESSED_ROOT / "subset_manifest.json"
DEFAULT_DISTRIBUTION_REPORT_PATH = PROCESSED_ROOT / "subset_distribution.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a balanced 110-task EHRFlowBench subset.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--combined-path", type=Path, default=DEFAULT_COMBINED_PATH)
    parser.add_argument("--subset-manifest-path", type=Path, default=DEFAULT_SUBSET_MANIFEST_PATH)
    parser.add_argument("--distribution-report-path", type=Path, default=DEFAULT_DISTRIBUTION_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--select-count-per-dataset", type=int, default=SELECT_COUNT_PER_DATASET)
    parser.add_argument("--train-count-per-dataset", type=int, default=TRAIN_COUNT_PER_DATASET)
    return parser.parse_args()


def load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["tasks"])


def stable_seed(*parts: object) -> int:
    joined = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "big")


def task_sort_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        DATASET_ORDER.index(task["dataset"]),
        task["paper_id"],
        task["source_task_idx"],
        task["task_idx"],
    )


def infer_dataset(required_inputs: list[str]) -> str:
    if any("/processed/tjh/" in value for value in required_inputs):
        return "TJH"
    if any("/processed/mimic_iv_demo/" in value for value in required_inputs):
        return "MIMIC-IV-demo"
    raise ValueError(f"Unable to infer dataset from required_inputs={required_inputs!r}")


def largest_remainder_quotas(counts: Counter[str], target_total: int) -> tuple[dict[str, float], dict[str, int]]:
    total = sum(counts.values())
    if total < target_total:
        raise ValueError(f"Cannot allocate {target_total} items from only {total} tasks")
    exact = {key: counts[key] * target_total / total for key in counts}
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = target_total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (
            exact[key] - quotas[key],
            counts[key],
            key,
        ),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return exact, quotas


def choose_tasks(candidates: list[dict[str, Any]], count: int, seed: int, *seed_parts: object) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=task_sort_key)
    if count > len(ordered):
        raise ValueError(f"Cannot select {count} tasks from {len(ordered)} candidates")
    rng = random.Random(stable_seed(seed, *seed_parts))
    return sorted(rng.sample(ordered, count), key=task_sort_key)


def compute_source_stats(tasks: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    dataset_counts: Counter[str] = Counter()
    category_counts_by_dataset: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASET_ORDER}
    for task in tasks:
        dataset_counts[task["dataset"]] += 1
        category_counts_by_dataset[task["dataset"]][task["primary_category"]] += 1
    return dataset_counts, category_counts_by_dataset


def ensure_global_category_coverage(
    quotas_by_dataset: dict[str, dict[str, int]],
    exact_by_dataset: dict[str, dict[str, float]],
    source_counts_by_dataset: dict[str, Counter[str]],
) -> None:
    all_categories = set()
    for counts in source_counts_by_dataset.values():
        all_categories.update(counts.keys())

    for category in sorted(all_categories):
        total_quota = sum(quotas_by_dataset[dataset].get(category, 0) for dataset in DATASET_ORDER)
        if total_quota > 0:
            continue

        target_dataset = max(
            DATASET_ORDER,
            key=lambda dataset: (
                source_counts_by_dataset[dataset].get(category, 0),
                dataset,
            ),
        )
        donor_category = max(
            (
                key
                for key, quota in quotas_by_dataset[target_dataset].items()
                if quota > 0 and key != category
            ),
            key=lambda key: (
                quotas_by_dataset[target_dataset][key] - exact_by_dataset[target_dataset].get(key, 0.0),
                quotas_by_dataset[target_dataset][key],
                source_counts_by_dataset[target_dataset][key],
                key,
            ),
        )
        quotas_by_dataset[target_dataset][donor_category] -= 1
        quotas_by_dataset[target_dataset][category] = quotas_by_dataset[target_dataset].get(category, 0) + 1


def select_balanced_subset(
    tasks: list[dict[str, Any]],
    *,
    seed: int,
    select_count_per_dataset: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, dict[str, float]], dict[str, Counter[str]]]:
    tasks_by_dataset_and_category: dict[str, dict[str, list[dict[str, Any]]]] = {
        dataset: defaultdict(list) for dataset in DATASET_ORDER
    }
    source_counts_by_dataset: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASET_ORDER}

    for task in tasks:
        dataset = task["dataset"]
        category = task["primary_category"]
        tasks_by_dataset_and_category[dataset][category].append(task)
        source_counts_by_dataset[dataset][category] += 1

    exact_by_dataset: dict[str, dict[str, float]] = {}
    quotas_by_dataset: dict[str, dict[str, int]] = {}
    for dataset in DATASET_ORDER:
        exact, quotas = largest_remainder_quotas(source_counts_by_dataset[dataset], select_count_per_dataset)
        exact_by_dataset[dataset] = exact
        quotas_by_dataset[dataset] = quotas

    ensure_global_category_coverage(quotas_by_dataset, exact_by_dataset, source_counts_by_dataset)

    selected: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for category, quota in quotas_by_dataset[dataset].items():
            if quota == 0:
                continue
            category_tasks = tasks_by_dataset_and_category[dataset][category]
            selected.extend(
                choose_tasks(
                    category_tasks,
                    quota,
                    seed,
                    *("subset", dataset, category),
                )
            )

    return (
        sorted(selected, key=task_sort_key),
        quotas_by_dataset,
        exact_by_dataset,
        source_counts_by_dataset,
    )


def select_train_split(
    selected_tasks: list[dict[str, Any]],
    *,
    seed: int,
    train_count_per_dataset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASET_ORDER}
    category_counts_by_dataset: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASET_ORDER}
    category_counts_total: Counter[str] = Counter()

    for task in selected_tasks:
        dataset = task["dataset"]
        selected_by_dataset[dataset].append(task)
        category_counts_by_dataset[dataset][task["primary_category"]] += 1
        category_counts_total[task["primary_category"]] += 1

    dataset_remaining = {dataset: train_count_per_dataset for dataset in DATASET_ORDER}
    train_ids: set[tuple[int, int]] = set()
    train_tasks: list[dict[str, Any]] = []

    categories_by_rarity = sorted(
        category_counts_total,
        key=lambda category: (
            category_counts_total[category],
            PRIMARY_CATEGORY_META[category]["label"],
        ),
    )

    for category in categories_by_rarity:
        if sum(dataset_remaining.values()) == 0:
            break
        candidate_datasets = [
            dataset
            for dataset in DATASET_ORDER
            if dataset_remaining[dataset] > 0 and category_counts_by_dataset[dataset].get(category, 0) > 0
        ]
        if not candidate_datasets:
            continue
        chosen_dataset = min(
            candidate_datasets,
            key=lambda dataset: (
                category_counts_by_dataset[dataset][category],
                -dataset_remaining[dataset],
                DATASET_ORDER.index(dataset),
            ),
        )
        candidates = [
            task
            for task in selected_by_dataset[chosen_dataset]
            if task["primary_category"] == category and (task["paper_id"], task["source_task_idx"]) not in train_ids
        ]
        chosen = choose_tasks(candidates, 1, seed, *("train-cover", chosen_dataset, category))[0]
        train_tasks.append(chosen)
        train_ids.add((chosen["paper_id"], chosen["source_task_idx"]))
        dataset_remaining[chosen_dataset] -= 1

    train_counts_by_dataset: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASET_ORDER}
    for task in train_tasks:
        train_counts_by_dataset[task["dataset"]][task["primary_category"]] += 1

    for dataset in DATASET_ORDER:
        _, target_train_quotas = largest_remainder_quotas(category_counts_by_dataset[dataset], train_count_per_dataset)
        while dataset_remaining[dataset] > 0:
            remaining_candidates = [
                task
                for task in selected_by_dataset[dataset]
                if (task["paper_id"], task["source_task_idx"]) not in train_ids
            ]
            if not remaining_candidates:
                raise ValueError(f"Not enough remaining {dataset} tasks to fill train split")

            available_categories = Counter(task["primary_category"] for task in remaining_candidates)
            positive_deficits = {
                category: target_train_quotas.get(category, 0) - train_counts_by_dataset[dataset].get(category, 0)
                for category in available_categories
            }
            deficit_categories = [category for category, deficit in positive_deficits.items() if deficit > 0]
            if deficit_categories:
                chosen_category = max(
                    deficit_categories,
                    key=lambda category: (
                        positive_deficits[category],
                        category_counts_by_dataset[dataset][category],
                        -train_counts_by_dataset[dataset][category],
                        PRIMARY_CATEGORY_META[category]["label"],
                    ),
                )
            else:
                chosen_category = max(
                    available_categories,
                    key=lambda category: (
                        category_counts_by_dataset[dataset][category],
                        -train_counts_by_dataset[dataset][category],
                        PRIMARY_CATEGORY_META[category]["label"],
                    ),
                )

            candidates = [
                task
                for task in remaining_candidates
                if task["primary_category"] == chosen_category
            ]
            chosen = choose_tasks(
                candidates,
                1,
                seed,
                *("train-fill", dataset, chosen_category, dataset_remaining[dataset]),
            )[0]
            train_tasks.append(chosen)
            train_ids.add((chosen["paper_id"], chosen["source_task_idx"]))
            train_counts_by_dataset[dataset][chosen_category] += 1
            dataset_remaining[dataset] -= 1

    train_tasks = sorted(train_tasks, key=task_sort_key)
    test_tasks = sorted(
        [
            task
            for task in selected_tasks
            if (task["paper_id"], task["source_task_idx"]) not in train_ids
        ],
        key=task_sort_key,
    )
    return train_tasks, test_tasks


def guess_media_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        return "image"
    if suffix in {".md", ".txt"}:
        return "text"
    return "binary"


def build_dataset_row(task: dict[str, Any], *, qid: int, split: str) -> dict[str, Any]:
    return {
        "qid": qid,
        "task": task["task"],
        "task_brief": task["task_brief"],
        "dataset": task["dataset"],
        "task_type": task["task_type"],
        "reference_answer": f"reference_answers/{split}/{qid}/answer_manifest.json",
        "paper_id": task["paper_id"],
        "paper_title": task["paper_title"],
        "source_task_idx": task["source_task_idx"],
    }


def build_answer_manifest(task: dict[str, Any], *, qid: int, split: str) -> dict[str, Any]:
    required_outputs = [
        {
            "file_name": file_name,
            "reference_path": f"reference_answers/{split}/{qid}/{file_name}",
            "media_type": guess_media_type(file_name),
        }
        for file_name in task["deliverables"]
    ]
    return {
        "qid": qid,
        "dataset": task["dataset"],
        "task_type": task["task_type"],
        "required_inputs": list(task["required_inputs"]),
        "required_outputs": required_outputs,
        "primary_category": PRIMARY_CATEGORY_META[task["primary_category"]]["label"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reset_outputs(output_root: Path) -> None:
    for file_name in ("train.jsonl", "test.jsonl", "ehrflowbench.jsonl", "subset_manifest.json", "subset_distribution.md"):
        path = output_root / file_name
        if path.exists():
            path.unlink()
    reference_root = output_root / "reference_answers"
    if reference_root.exists():
        shutil.rmtree(reference_root)


def category_count_table(tasks: list[dict[str, Any]]) -> Counter[str]:
    return Counter(task["primary_category"] for task in tasks)


def category_count_table_by_dataset(tasks: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    counts = {dataset: Counter() for dataset in DATASET_ORDER}
    for task in tasks:
        counts[task["dataset"]][task["primary_category"]] += 1
    return counts


def dataset_count_table(tasks: list[dict[str, Any]]) -> Counter[str]:
    return Counter(task["dataset"] for task in tasks)


def percent(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%"


def build_distribution_report(
    *,
    source_tasks: list[dict[str, Any]],
    selected_tasks: list[dict[str, Any]],
    train_tasks: list[dict[str, Any]],
    test_tasks: list[dict[str, Any]],
    quotas_by_dataset: dict[str, dict[str, int]],
    exact_by_dataset: dict[str, dict[str, float]],
    seed: int,
) -> str:
    source_category_counts = category_count_table(source_tasks)
    selected_category_counts = category_count_table(selected_tasks)
    train_category_counts = category_count_table(train_tasks)
    test_category_counts = category_count_table(test_tasks)

    source_dataset_counts = dataset_count_table(source_tasks)
    selected_dataset_counts = dataset_count_table(selected_tasks)
    train_dataset_counts = dataset_count_table(train_tasks)
    test_dataset_counts = dataset_count_table(test_tasks)

    selected_counts_by_dataset = category_count_table_by_dataset(selected_tasks)
    train_counts_by_dataset = category_count_table_by_dataset(train_tasks)
    test_counts_by_dataset = category_count_table_by_dataset(test_tasks)

    lines: list[str] = []
    lines.append("# EHRFlowBench Selected Subset Distribution")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Seed: `{seed}`")
    lines.append(f"- Source tasks: `{len(source_tasks)}`")
    lines.append(f"- Selected tasks: `{len(selected_tasks)}`")
    lines.append(f"- Train tasks: `{len(train_tasks)}`")
    lines.append(f"- Test tasks: `{len(test_tasks)}`")
    lines.append("- Selection strategy: deterministic stratified sampling by `dataset x primary_category`, followed by a balanced 5/5 train split that prioritizes category coverage and rarity.")
    lines.append(f"- Category coverage: selected subset covers `{len(selected_category_counts)}/{len(source_category_counts)}` primary categories; train covers `{len(train_category_counts)}/{len(source_category_counts)}` primary categories.")
    lines.append("- Note: with only 10 train tasks, covering all 11 primary categories in train is impossible; the current split reaches the maximum possible coverage of 10 categories.")
    lines.append("")
    lines.append("## Dataset Balance")
    lines.append("")
    lines.append("| Split | TJH | MIMIC-IV-demo | Total |")
    lines.append("| --- | ---: | ---: | ---: |")
    for split_name, counts, total in [
        ("Source", source_dataset_counts, len(source_tasks)),
        ("Selected", selected_dataset_counts, len(selected_tasks)),
        ("Train", train_dataset_counts, len(train_tasks)),
        ("Test", test_dataset_counts, len(test_tasks)),
    ]:
        lines.append(f"| {split_name} | {counts['TJH']} | {counts['MIMIC-IV-demo']} | {total} |")
    lines.append("")
    lines.append("## Primary Category Distribution")
    lines.append("")
    lines.append("| Category | Source | Selected | Exact Half Target | Train | Test |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for category in sorted(source_category_counts, key=lambda key: (-source_category_counts[key], PRIMARY_CATEGORY_META[key]["label"])):
        label = PRIMARY_CATEGORY_META[category]["label"]
        exact_half = source_category_counts[category] / 2
        lines.append(
            f"| {label} | {source_category_counts[category]} ({percent(source_category_counts[category], len(source_tasks))}) "
            f"| {selected_category_counts[category]} ({percent(selected_category_counts[category], len(selected_tasks))}) "
            f"| {exact_half:.1f} | {train_category_counts[category]} | {test_category_counts[category]} |"
        )
    lines.append("")
    lines.append("## Selected Category Allocation by Dataset")
    lines.append("")
    lines.append("| Category | TJH Source | TJH Exact | TJH Selected | MIMIC Source | MIMIC Exact | MIMIC Selected |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    all_categories = sorted(source_category_counts, key=lambda key: PRIMARY_CATEGORY_META[key]["label"])
    for category in all_categories:
        label = PRIMARY_CATEGORY_META[category]["label"]
        lines.append(
            f"| {label} | "
            f"{category_count_table_by_dataset(source_tasks)['TJH'][category]} | {exact_by_dataset['TJH'].get(category, 0.0):.1f} | {selected_counts_by_dataset['TJH'][category]} | "
            f"{category_count_table_by_dataset(source_tasks)['MIMIC-IV-demo'][category]} | {exact_by_dataset['MIMIC-IV-demo'].get(category, 0.0):.1f} | {selected_counts_by_dataset['MIMIC-IV-demo'][category]} |"
        )
    lines.append("")
    lines.append("## Train Split Tasks")
    lines.append("")
    lines.append("| QID | Dataset | Category | paper_id | source_task_idx | Task Brief |")
    lines.append("| ---: | --- | --- | ---: | ---: | --- |")
    for qid, task in enumerate(train_tasks, start=1):
        lines.append(
            f"| {qid} | {task['dataset']} | {PRIMARY_CATEGORY_META[task['primary_category']]['label']} | "
            f"{task['paper_id']} | {task['source_task_idx']} | {task['task_brief']} |"
        )
    lines.append("")
    lines.append("## Train Category Counts by Dataset")
    lines.append("")
    lines.append("| Category | TJH Train | MIMIC Train |")
    lines.append("| --- | ---: | ---: |")
    for category in all_categories:
        lines.append(
            f"| {PRIMARY_CATEGORY_META[category]['label']} | {train_counts_by_dataset['TJH'][category]} | {train_counts_by_dataset['MIMIC-IV-demo'][category]} |"
        )
    lines.append("")
    lines.append("## Test Category Counts by Dataset")
    lines.append("")
    lines.append("| Category | TJH Test | MIMIC Test |")
    lines.append("| --- | ---: | ---: |")
    for category in all_categories:
        lines.append(
            f"| {PRIMARY_CATEGORY_META[category]['label']} | {test_counts_by_dataset['TJH'][category]} | {test_counts_by_dataset['MIMIC-IV-demo'][category]} |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_subset_manifest(
    *,
    seed: int,
    source_tasks: list[dict[str, Any]],
    selected_tasks: list[dict[str, Any]],
    train_tasks: list[dict[str, Any]],
    test_tasks: list[dict[str, Any]],
    quotas_by_dataset: dict[str, dict[str, int]],
) -> dict[str, Any]:
    source_category_counts = category_count_table(source_tasks)
    selected_category_counts = category_count_table(selected_tasks)
    train_category_counts = category_count_table(train_tasks)
    test_category_counts = category_count_table(test_tasks)

    return {
        "dataset": "ehrflowbench",
        "schema_version": SUBSET_SCHEMA_VERSION,
        "selection_seed": seed,
        "source_task_count": len(source_tasks),
        "selected_task_count": len(selected_tasks),
        "train_task_count": len(train_tasks),
        "test_task_count": len(test_tasks),
        "dataset_counts": {
            "source": dict(dataset_count_table(source_tasks)),
            "selected": dict(dataset_count_table(selected_tasks)),
            "train": dict(dataset_count_table(train_tasks)),
            "test": dict(dataset_count_table(test_tasks)),
        },
        "primary_category_counts": {
            "source": {PRIMARY_CATEGORY_META[key]["label"]: source_category_counts[key] for key in sorted(source_category_counts)},
            "selected": {PRIMARY_CATEGORY_META[key]["label"]: selected_category_counts[key] for key in sorted(selected_category_counts)},
            "train": {PRIMARY_CATEGORY_META[key]["label"]: train_category_counts[key] for key in sorted(train_category_counts)},
            "test": {PRIMARY_CATEGORY_META[key]["label"]: test_category_counts[key] for key in sorted(test_category_counts)},
        },
        "selected_quotas_by_dataset": {
            dataset: {PRIMARY_CATEGORY_META[key]["label"]: value for key, value in sorted(counts.items())}
            for dataset, counts in quotas_by_dataset.items()
        },
        "selected_tasks": [
            {
                "dataset": task["dataset"],
                "paper_id": task["paper_id"],
                "source_task_idx": task["source_task_idx"],
                "primary_category": PRIMARY_CATEGORY_META[task["primary_category"]]["label"],
            }
            for task in selected_tasks
        ],
    }


def main() -> None:
    args = parse_args()
    if args.train_count_per_dataset > args.select_count_per_dataset:
        raise ValueError("train_count_per_dataset cannot exceed select_count_per_dataset")

    raw_tasks = load_tasks(args.input_path)
    enriched_tasks, _, _, _ = analyze_tasks(raw_tasks)
    for task in enriched_tasks:
        task["dataset"] = infer_dataset(task["required_inputs"])

    reset_outputs(PROCESSED_ROOT)

    selected_tasks, quotas_by_dataset, exact_by_dataset, _ = select_balanced_subset(
        enriched_tasks,
        seed=args.seed,
        select_count_per_dataset=args.select_count_per_dataset,
    )
    train_tasks, test_tasks = select_train_split(
        selected_tasks,
        seed=args.seed,
        train_count_per_dataset=args.train_count_per_dataset,
    )

    combined_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for qid, task in enumerate(train_tasks, start=1):
        train_rows.append(build_dataset_row(task, qid=qid, split="train"))
        manifest = build_answer_manifest(task, qid=qid, split="train")
        write_json(PROCESSED_ROOT / f"reference_answers/train/{qid}/answer_manifest.json", manifest)

    for qid, task in enumerate(test_tasks, start=1):
        test_rows.append(build_dataset_row(task, qid=qid, split="test"))
        manifest = build_answer_manifest(task, qid=qid, split="test")
        write_json(PROCESSED_ROOT / f"reference_answers/test/{qid}/answer_manifest.json", manifest)

    for qid, row in enumerate(train_rows + test_rows, start=1):
        combined_row = dict(row)
        combined_row["qid"] = qid
        combined_rows.append(combined_row)

    write_jsonl(args.train_path, train_rows)
    write_jsonl(args.test_path, test_rows)
    write_jsonl(args.combined_path, combined_rows)
    write_json(
        args.subset_manifest_path,
        build_subset_manifest(
            seed=args.seed,
            source_tasks=enriched_tasks,
            selected_tasks=selected_tasks,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            quotas_by_dataset=quotas_by_dataset,
        ),
    )
    args.distribution_report_path.write_text(
        build_distribution_report(
            source_tasks=enriched_tasks,
            selected_tasks=selected_tasks,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            quotas_by_dataset=quotas_by_dataset,
            exact_by_dataset=exact_by_dataset,
            seed=args.seed,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "seed": args.seed,
                "selected_task_count": len(selected_tasks),
                "train_task_count": len(train_tasks),
                "test_task_count": len(test_tasks),
                "selected_dataset_counts": dict(dataset_count_table(selected_tasks)),
                "train_dataset_counts": dict(dataset_count_table(train_tasks)),
                "test_dataset_counts": dict(dataset_count_table(test_tasks)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
