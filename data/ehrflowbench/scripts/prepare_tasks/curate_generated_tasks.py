from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEED = 42
DEFAULT_SAMPLE_COUNT_PER_DATASET = 55
DEFAULT_TRAIN_COUNT_PER_DATASET = 5
TASK_TYPE = "report_generation"
CONTRACT_VERSION = "ehrflowbench_report_generation_v1"
SUBSET_SCHEMA_VERSION = "ehrflowbench_subset_v1"
DATASET_ORDER = ("tjh", "mimic_iv_demo")
DATASET_DISPLAY_NAMES = {
    "tjh": "TJH",
    "mimic_iv_demo": "MIMIC-IV-demo",
}
DATASET_CORE_REQUIRED_INPUTS = {
    "tjh": {
        "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
        "data/ehrflowbench/processed/tjh/split_metadata.json",
    },
    "mimic_iv_demo": {
        "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet",
        "data/ehrflowbench/processed/mimic_iv_demo/split_metadata.json",
    },
}
DATASET_OPTIONAL_REQUIRED_INPUTS = {
    "tjh": set(),
    "mimic_iv_demo": {
        "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_value_reference.md",
    },
}


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "config.toml").exists() and (parent / "data").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")


PROJECT_ROOT = find_project_root()
DATASET_ROOT = PROJECT_ROOT / "data" / "ehrflowbench"
RAW_PAPERS_ROOT = DATASET_ROOT / "raw" / "papers"
PAPER_TITLES_PATH = RAW_PAPERS_ROOT / "paper_titles.csv"
GENERATED_TASKS_ROOT = DATASET_ROOT / "processed" / "papers" / "generated_tasks"
PROCESSED_ROOT = DATASET_ROOT / "processed"


@dataclass(frozen=True)
class GeneratedTaskCandidate:
    paper_id: int
    paper_title: str
    task_idx: int
    task_brief: str
    task_type: str
    task: str
    required_inputs: tuple[str, ...]
    deliverables: tuple[str, ...]
    dataset_key: str

    @property
    def dataset(self) -> str:
        return DATASET_DISPLAY_NAMES[self.dataset_key]


@dataclass
class SampledTask:
    source: GeneratedTaskCandidate
    split: str | None = None
    split_qid: int | None = None
    global_qid: int | None = None

    @property
    def dataset(self) -> str:
        return self.source.dataset

    @property
    def dataset_key(self) -> str:
        return self.source.dataset_key

    @property
    def paper_id(self) -> int:
        return self.source.paper_id

    @property
    def paper_title(self) -> str:
        return self.source.paper_title

    @property
    def task_idx(self) -> int:
        return self.source.task_idx

    @property
    def reference_answer(self) -> str:
        if self.split is None or self.split_qid is None:
            raise ValueError("split qid has not been assigned")
        return f"reference_answers/{self.split}/{self.split_qid}/answer_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and export the EHRFlowBench task subset.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-count-per-dataset", type=int, default=DEFAULT_SAMPLE_COUNT_PER_DATASET)
    parser.add_argument("--train-count-per-dataset", type=int, default=DEFAULT_TRAIN_COUNT_PER_DATASET)
    parser.add_argument("--output-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--generated-tasks-root", type=Path, default=GENERATED_TASKS_ROOT)
    parser.add_argument("--paper-titles-path", type=Path, default=PAPER_TITLES_PATH)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_paper_titles(path: Path) -> dict[int, str]:
    titles: dict[int, str] = {}
    if not path.exists():
        return titles
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            titles[int(row["paper_id"])] = str(row["paper_title"]).strip()
    return titles


def paper_id_from_name(name: str) -> int:
    match = re.match(r"(\d+)_", name)
    if not match:
        raise ValueError(f"Could not parse paper id from {name!r}")
    return int(match.group(1))


def classify_dataset(required_inputs: tuple[str, ...]) -> str | None:
    input_set = set(required_inputs)
    matches: list[str] = []
    for dataset_key, core_inputs in DATASET_CORE_REQUIRED_INPUTS.items():
        allowed = core_inputs | DATASET_OPTIONAL_REQUIRED_INPUTS.get(dataset_key, set())
        if core_inputs.issubset(input_set) and input_set.issubset(allowed):
            matches.append(dataset_key)
    if len(matches) == 1:
        return matches[0]
    return None


def candidate_sort_key(candidate: GeneratedTaskCandidate) -> tuple[Any, ...]:
    return (
        DATASET_ORDER.index(candidate.dataset_key),
        candidate.paper_id,
        candidate.task_idx,
    )


def sampled_sort_key(task: SampledTask) -> tuple[Any, ...]:
    return (
        DATASET_ORDER.index(task.dataset_key),
        task.paper_id,
        task.task_idx,
    )


def collect_generated_candidates(
    generated_tasks_root: Path,
    paper_titles: dict[int, str],
) -> tuple[list[GeneratedTaskCandidate], dict[str, Any]]:
    candidates: list[GeneratedTaskCandidate] = []
    skipped_invalid_candidates = 0
    source_bundle_count = 0

    for tasks_path in sorted(generated_tasks_root.glob("*_tasks.json"), key=lambda path: paper_id_from_name(path.name)):
        paper_id = paper_id_from_name(tasks_path.name)
        payload = read_json(tasks_path)
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            continue
        source_bundle_count += 1
        paper_title = paper_titles.get(paper_id, f"Paper {paper_id}")

        for task_idx, item in enumerate(tasks, start=1):
            task_type = str(item.get("task_type", "")).strip()
            if task_type != TASK_TYPE:
                skipped_invalid_candidates += 1
                continue

            required_inputs = tuple(str(value).strip() for value in item.get("required_inputs", []))
            dataset_key = classify_dataset(required_inputs)
            if dataset_key is None:
                skipped_invalid_candidates += 1
                continue

            candidates.append(
                GeneratedTaskCandidate(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    task_idx=task_idx,
                    task_brief=str(item["task_brief"]).strip(),
                    task_type=task_type,
                    task=str(item["task"]).strip(),
                    required_inputs=required_inputs,
                    deliverables=tuple(str(value).strip() for value in item.get("deliverables", [])),
                    dataset_key=dataset_key,
                )
            )

    summary = {
        "source_bundle_count": source_bundle_count,
        "candidate_count": len(candidates),
        "skipped_invalid_candidates": skipped_invalid_candidates,
        "dataset_candidate_counts": {
            DATASET_DISPLAY_NAMES[key]: sum(1 for candidate in candidates if candidate.dataset_key == key)
            for key in DATASET_ORDER
        },
    }
    return candidates, summary


def sample_candidates_by_dataset(
    candidates: list[GeneratedTaskCandidate],
    *,
    sample_count_per_dataset: int,
    seed: int,
) -> list[GeneratedTaskCandidate]:
    selected: list[GeneratedTaskCandidate] = []
    for offset, dataset_key in enumerate(DATASET_ORDER):
        dataset_candidates = sorted(
            [candidate for candidate in candidates if candidate.dataset_key == dataset_key],
            key=candidate_sort_key,
        )
        if len(dataset_candidates) < sample_count_per_dataset:
            raise ValueError(
                f"Not enough {DATASET_DISPLAY_NAMES[dataset_key]} tasks: "
                f"need {sample_count_per_dataset}, found {len(dataset_candidates)}"
            )
        rng = random.Random(seed + offset)
        selected.extend(rng.sample(dataset_candidates, sample_count_per_dataset))
    return sorted(selected, key=candidate_sort_key)


def split_sampled_tasks(
    selected: list[GeneratedTaskCandidate],
    *,
    train_count_per_dataset: int,
    seed: int,
) -> tuple[list[SampledTask], list[SampledTask]]:
    train: list[SampledTask] = []
    test: list[SampledTask] = []

    for offset, dataset_key in enumerate(DATASET_ORDER):
        dataset_candidates = sorted(
            [candidate for candidate in selected if candidate.dataset_key == dataset_key],
            key=candidate_sort_key,
        )
        if len(dataset_candidates) < train_count_per_dataset:
            raise ValueError(
                f"Not enough sampled {DATASET_DISPLAY_NAMES[dataset_key]} tasks for train split: "
                f"need {train_count_per_dataset}, found {len(dataset_candidates)}"
            )

        rng = random.Random(seed + 100 + offset)
        train_indexes = set(rng.sample(range(len(dataset_candidates)), train_count_per_dataset))
        for index, candidate in enumerate(dataset_candidates):
            sampled_task = SampledTask(source=candidate)
            if index in train_indexes:
                train.append(sampled_task)
            else:
                test.append(sampled_task)

    return sorted(train, key=sampled_sort_key), sorted(test, key=sampled_sort_key)


def assign_qids(train: list[SampledTask], test: list[SampledTask]) -> None:
    for qid, task in enumerate(train, start=1):
        task.split = "train"
        task.split_qid = qid

    for qid, task in enumerate(test, start=1):
        task.split = "test"
        task.split_qid = qid

    for qid, task in enumerate(train + test, start=1):
        task.global_qid = qid


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


def build_dataset_row(task: SampledTask, *, use_split_qid: bool) -> dict[str, Any]:
    qid = task.split_qid if use_split_qid else task.global_qid
    if qid is None:
        raise ValueError("qid has not been assigned")
    return {
        "qid": qid,
        "task": task.source.task,
        "task_brief": task.source.task_brief,
        "dataset": task.dataset,
        "task_type": task.source.task_type,
        "reference_answer": task.reference_answer,
        "paper_id": task.paper_id,
        "paper_title": task.paper_title,
        "source_task_idx": task.task_idx,
    }


def build_required_outputs(task: SampledTask) -> list[dict[str, Any]]:
    if task.split is None or task.split_qid is None:
        raise ValueError("split qid has not been assigned")
    required_outputs: list[dict[str, Any]] = []
    for file_name in task.source.deliverables:
        required_outputs.append(
            {
                "file_name": file_name,
                "reference_path": f"reference_answers/{task.split}/{task.split_qid}/{file_name}",
                "media_type": guess_media_type(file_name),
            }
        )
    return required_outputs


def build_answer_manifest(task: SampledTask) -> dict[str, Any]:
    required_outputs = build_required_outputs(task)
    return {
        "contract_version": CONTRACT_VERSION,
        "qid": task.split_qid,
        "dataset": task.dataset,
        "task_type": task.source.task_type,
        "required_inputs": list(task.source.required_inputs),
        "required_outputs": required_outputs,
        "all_outputs": [item["reference_path"] for item in required_outputs],
        "paper_id": task.paper_id,
        "paper_title": task.paper_title,
        "source_task_idx": task.task_idx,
    }


def count_by_dataset(tasks: list[SampledTask]) -> dict[str, int]:
    return {
        DATASET_DISPLAY_NAMES[key]: sum(1 for task in tasks if task.dataset_key == key)
        for key in DATASET_ORDER
    }


def build_subset_manifest(
    *,
    collection_summary: dict[str, Any],
    train: list[SampledTask],
    test: list[SampledTask],
    seed: int,
    sample_count_per_dataset: int,
    train_count_per_dataset: int,
) -> dict[str, Any]:
    all_tasks = train + test
    return {
        "dataset": "ehrflowbench",
        "seed": seed,
        "schema_version": SUBSET_SCHEMA_VERSION,
        "sample_count_per_dataset": sample_count_per_dataset,
        "train_count_per_dataset": train_count_per_dataset,
        "test_count_per_dataset": sample_count_per_dataset - train_count_per_dataset,
        "source_pool": collection_summary,
        "combined": {
            "count": len(all_tasks),
            "dataset_counts": count_by_dataset(all_tasks),
        },
        "train": {
            "count": len(train),
            "dataset_counts": count_by_dataset(train),
            "qids": [task.split_qid for task in train],
        },
        "test": {
            "count": len(test),
            "dataset_counts": count_by_dataset(test),
            "qids": [task.split_qid for task in test],
        },
        "qid_remap": [
            {
                "global_qid": task.global_qid,
                "split": task.split,
                "split_qid": task.split_qid,
                "dataset": task.dataset,
                "paper_id": task.paper_id,
                "source_task_idx": task.task_idx,
            }
            for task in all_tasks
        ],
    }


def reset_outputs(output_root: Path) -> None:
    for file_name in ("ehrflowbench.jsonl", "train.jsonl", "test.jsonl", "subset_manifest.json"):
        path = output_root / file_name
        if path.exists():
            path.unlink()
    reference_root = output_root / "reference_answers"
    if reference_root.exists():
        shutil.rmtree(reference_root)


def run_curation(
    *,
    seed: int = SEED,
    sample_count_per_dataset: int = DEFAULT_SAMPLE_COUNT_PER_DATASET,
    train_count_per_dataset: int = DEFAULT_TRAIN_COUNT_PER_DATASET,
    output_root: Path = PROCESSED_ROOT,
    generated_tasks_root: Path = GENERATED_TASKS_ROOT,
    paper_titles_path: Path = PAPER_TITLES_PATH,
) -> dict[str, Any]:
    if train_count_per_dataset > sample_count_per_dataset:
        raise ValueError("train_count_per_dataset cannot exceed sample_count_per_dataset")

    paper_titles = load_paper_titles(paper_titles_path)
    candidates, collection_summary = collect_generated_candidates(generated_tasks_root, paper_titles)
    selected = sample_candidates_by_dataset(
        candidates,
        sample_count_per_dataset=sample_count_per_dataset,
        seed=seed,
    )
    train, test = split_sampled_tasks(
        selected,
        train_count_per_dataset=train_count_per_dataset,
        seed=seed,
    )
    assign_qids(train, test)

    output_root.mkdir(parents=True, exist_ok=True)
    reset_outputs(output_root)

    write_jsonl(output_root / "ehrflowbench.jsonl", [build_dataset_row(task, use_split_qid=False) for task in train + test])
    write_jsonl(output_root / "train.jsonl", [build_dataset_row(task, use_split_qid=True) for task in train])
    write_jsonl(output_root / "test.jsonl", [build_dataset_row(task, use_split_qid=True) for task in test])

    for task in train + test:
        write_json(output_root / task.reference_answer, build_answer_manifest(task))

    write_json(
        output_root / "subset_manifest.json",
        build_subset_manifest(
            collection_summary=collection_summary,
            train=train,
            test=test,
            seed=seed,
            sample_count_per_dataset=sample_count_per_dataset,
            train_count_per_dataset=train_count_per_dataset,
        ),
    )

    return {
        "seed": seed,
        "sample_count_per_dataset": sample_count_per_dataset,
        "train_count_per_dataset": train_count_per_dataset,
        "test_count_per_dataset": sample_count_per_dataset - train_count_per_dataset,
        "candidate_count": len(candidates),
        "selected_count": len(train) + len(test),
        "train_count": len(train),
        "test_count": len(test),
    }


def main() -> None:
    args = parse_args()
    summary = run_curation(
        seed=args.seed,
        sample_count_per_dataset=args.sample_count_per_dataset,
        train_count_per_dataset=args.train_count_per_dataset,
        output_root=args.output_root,
        generated_tasks_root=args.generated_tasks_root,
        paper_titles_path=args.paper_titles_path,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
