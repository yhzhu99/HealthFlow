from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEED = 42
DEFAULT_SAMPLE_COUNT_PER_DATASET = 55
DEFAULT_TRAIN_COUNT_PER_DATASET = 5
DEFAULT_PASS_THRESHOLD = 7.0
CONTRACT_VERSION = "ehrflowbench_report_generation_v1"
SUBSET_SCHEMA_VERSION = "ehrflowbench_report_generation_subset_v1"
TASK_TYPE = "report_generation"
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
TITLE_BUCKET_PATTERNS = {
    "causal": ("causal", "counterfactual", "treatment effect", "dynamic treatment", "confound"),
    "fairness": ("fair", "bias", "equity", "debiased"),
    "unsupervised": ("cluster", "phenotyp", "subgroup"),
    "interpretability": ("explain", "interpretable", "interpretability"),
    "graph": ("graph", "graphical", "knowledge graph"),
    "temporal": ("temporal", "time", "trajectory", "sequence", "visit", "longitudinal", "hazard"),
    "representation": ("representation", "embedding", "transformer", "attention", "language model"),
    "prediction": ("prediction", "predictive", "forecast", "risk", "deterioration", "outcome"),
}
TEXT_BUCKET_PATTERNS = {
    "causal": ("causal", "counterfactual", "treatment effect", "dynamic treatment", "confound"),
    "fairness": ("fairness", "bias mitigation", "equity"),
    "unsupervised": ("cluster", "clustering", "phenotype", "patient similarity", "subgroup"),
    "interpretability": ("interpretability", "interpretable", "feature attribution", "explain"),
    "graph": ("graph", "graphical", "knowledge graph", "node", "edge"),
    "temporal": ("temporal", "trajectory", "horizon", "longitudinal", "time-aware", "event order", "next-visit"),
    "representation": ("representation", "embedding", "feature engineering", "representation learning"),
    "prediction": ("prediction", "predictive", "risk", "readmission", "outcome", "forecast"),
}
BUCKET_PRIORITY = (
    "causal",
    "fairness",
    "unsupervised",
    "interpretability",
    "graph",
    "temporal",
    "representation",
    "prediction",
    "other",
)
REPORT_SECTION_ORDER = (
    "Objective",
    "Data",
    "Method",
    "Results",
    "Evidence",
    "Conclusion",
)


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
    focus_areas: tuple[str, ...]
    task: str
    required_inputs: tuple[str, ...]
    deliverables: tuple[str, ...]
    report_requirements: tuple[str, ...]
    tasks_path: Path
    response_path: Path
    dataset_key: str
    primary_bucket: str

    @property
    def dataset(self) -> str:
        return DATASET_DISPLAY_NAMES[self.dataset_key]

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.paper_id, self.task_idx, self.dataset_key)


@dataclass
class CuratedTask:
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
    def primary_bucket(self) -> str:
        return self.source.primary_bucket

    @property
    def task_type(self) -> str:
        return self.source.task_type

    @property
    def reference_answer(self) -> str:
        if self.split is None or self.split_qid is None:
            raise ValueError("split and split_qid must be assigned before building reference paths")
        return f"reference_answers/{self.split}/{self.split_qid}/answer_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a seeded 110-task EHRFlowBench subset in a MedAgentBoard-like processed format."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-count-per-dataset", type=int, default=DEFAULT_SAMPLE_COUNT_PER_DATASET)
    parser.add_argument("--train-count-per-dataset", type=int, default=DEFAULT_TRAIN_COUNT_PER_DATASET)
    parser.add_argument("--output-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--generated-tasks-root", type=Path, default=GENERATED_TASKS_ROOT)
    parser.add_argument("--paper-titles-path", type=Path, default=PAPER_TITLES_PATH)
    parser.add_argument(
        "--allow-missing-responses",
        action="store_true",
        help="Allow task bundles whose sibling *_response.json file is missing.",
    )
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


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def contains_any(text: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def paper_id_from_name(name: str) -> int:
    match = re.match(r"(\d+)_", name)
    if not match:
        raise ValueError(f"Could not parse paper id from {name!r}")
    return int(match.group(1))


def load_paper_titles(path: Path) -> dict[int, str]:
    titles: dict[int, str] = {}
    if not path.exists():
        return titles
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            paper_id = int(row["paper_id"])
            titles[paper_id] = str(row["paper_title"]).strip()
    return titles


def classify_required_input_set(required_inputs: tuple[str, ...]) -> tuple[str | None, bool, bool, bool]:
    input_set = set(required_inputs)
    touched_datasets = {
        dataset_key
        for dataset_key in DATASET_CORE_REQUIRED_INPUTS
        if input_set
        & (
            DATASET_CORE_REQUIRED_INPUTS[dataset_key]
            | DATASET_OPTIONAL_REQUIRED_INPUTS.get(dataset_key, set())
        )
    }
    has_mixed_dataset_inputs = len(touched_datasets) > 1

    matched_dataset: str | None = None
    for dataset_key, core_inputs in DATASET_CORE_REQUIRED_INPUTS.items():
        allowed_inputs = core_inputs | DATASET_OPTIONAL_REQUIRED_INPUTS.get(dataset_key, set())
        if core_inputs.issubset(input_set) and input_set.issubset(allowed_inputs):
            matched_dataset = dataset_key
            break

    has_valid_single_dataset_inputs = matched_dataset is not None
    has_split_inputs = bool(
        matched_dataset
        and any(path.endswith("/split_metadata.json") for path in DATASET_CORE_REQUIRED_INPUTS[matched_dataset] & input_set)
    )
    return matched_dataset, has_valid_single_dataset_inputs, has_split_inputs, has_mixed_dataset_inputs


def classify_primary_bucket(paper_title: str, task_brief: str, focus_areas: tuple[str, ...], task_text: str) -> str:
    title_text = normalize_text(paper_title)
    for bucket in BUCKET_PRIORITY:
        if contains_any(title_text, TITLE_BUCKET_PATTERNS.get(bucket, ())):
            return bucket

    task_blob = normalize_text(" ".join([task_brief, " ".join(focus_areas), task_text]))
    for bucket in BUCKET_PRIORITY:
        if contains_any(task_blob, TEXT_BUCKET_PATTERNS.get(bucket, ())):
            return bucket
    return "other"


def collect_generated_candidates(
    generated_tasks_root: Path,
    paper_titles: dict[int, str],
    *,
    allow_missing_responses: bool,
) -> tuple[list[GeneratedTaskCandidate], dict[str, Any]]:
    candidates: list[GeneratedTaskCandidate] = []
    skipped_invalid_candidates = 0
    skipped_missing_responses = 0
    source_bundle_count = 0

    for tasks_path in sorted(generated_tasks_root.glob("*_tasks.json"), key=lambda path: paper_id_from_name(path.name)):
        paper_id = paper_id_from_name(tasks_path.name)
        response_path = tasks_path.with_name(f"{paper_id}_response.json")
        if not response_path.exists() and not allow_missing_responses:
            skipped_missing_responses += 1
            continue

        payload = read_json(tasks_path)
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            continue
        source_bundle_count += 1

        paper_title = paper_titles.get(paper_id, f"Paper {paper_id}")
        for task_idx, item in enumerate(tasks, start=1):
            required_inputs = tuple(str(value).strip() for value in item.get("required_inputs", []))
            dataset_key, is_valid_single_dataset, has_split_inputs, has_mixed_inputs = classify_required_input_set(required_inputs)
            if not is_valid_single_dataset or not has_split_inputs or has_mixed_inputs:
                skipped_invalid_candidates += 1
                continue

            task_type = str(item.get("task_type", "")).strip()
            if task_type != TASK_TYPE:
                skipped_invalid_candidates += 1
                continue

            focus_areas = tuple(str(value).strip() for value in item.get("focus_areas", []))
            task_brief = str(item["task_brief"]).strip()
            task_text = str(item["task"]).strip()
            candidate = GeneratedTaskCandidate(
                paper_id=paper_id,
                paper_title=paper_title,
                task_idx=task_idx,
                task_brief=task_brief,
                task_type=task_type,
                focus_areas=focus_areas,
                task=task_text,
                required_inputs=required_inputs,
                deliverables=tuple(str(value).strip() for value in item.get("deliverables", [])),
                report_requirements=tuple(str(value).strip() for value in item.get("report_requirements", [])),
                tasks_path=tasks_path,
                response_path=response_path,
                dataset_key=dataset_key,
                primary_bucket=classify_primary_bucket(paper_title, task_brief, focus_areas, task_text),
            )
            candidates.append(candidate)

    summary = {
        "source_bundle_count": source_bundle_count,
        "candidate_count": len(candidates),
        "skipped_invalid_candidates": skipped_invalid_candidates,
        "skipped_missing_responses": skipped_missing_responses,
        "dataset_candidate_counts": {
            DATASET_DISPLAY_NAMES[key]: sum(1 for candidate in candidates if candidate.dataset_key == key)
            for key in DATASET_ORDER
        },
    }
    return candidates, summary


def candidate_sort_key(candidate: GeneratedTaskCandidate) -> tuple[Any, ...]:
    return (
        DATASET_ORDER.index(candidate.dataset_key),
        candidate.paper_id,
        candidate.task_idx,
    )


def curated_sort_key(task: CuratedTask) -> tuple[Any, ...]:
    return (
        DATASET_ORDER.index(task.dataset_key),
        task.paper_id,
        task.task_idx,
    )


def sample_candidates_by_dataset(
    candidates: list[GeneratedTaskCandidate],
    *,
    sample_count_per_dataset: int,
    seed: int,
) -> list[GeneratedTaskCandidate]:
    selected: list[GeneratedTaskCandidate] = []
    for dataset_offset, dataset_key in enumerate(DATASET_ORDER):
        dataset_candidates = sorted(
            [candidate for candidate in candidates if candidate.dataset_key == dataset_key],
            key=candidate_sort_key,
        )
        if len(dataset_candidates) < sample_count_per_dataset:
            raise ValueError(
                f"Not enough {DATASET_DISPLAY_NAMES[dataset_key]} tasks for sampling: "
                f"need {sample_count_per_dataset}, found {len(dataset_candidates)}"
            )
        dataset_rng = random.Random(seed + dataset_offset)
        picked = dataset_rng.sample(dataset_candidates, sample_count_per_dataset)
        selected.extend(sorted(picked, key=candidate_sort_key))
    return sorted(selected, key=candidate_sort_key)


def split_curated_candidates(
    selected: list[GeneratedTaskCandidate],
    *,
    train_count_per_dataset: int,
    seed: int,
) -> tuple[list[CuratedTask], list[CuratedTask]]:
    train: list[CuratedTask] = []
    test: list[CuratedTask] = []

    for dataset_offset, dataset_key in enumerate(DATASET_ORDER):
        dataset_candidates = sorted(
            [candidate for candidate in selected if candidate.dataset_key == dataset_key],
            key=candidate_sort_key,
        )
        if len(dataset_candidates) < train_count_per_dataset:
            raise ValueError(
                f"Not enough sampled {DATASET_DISPLAY_NAMES[dataset_key]} tasks for train split: "
                f"need {train_count_per_dataset}, found {len(dataset_candidates)}"
            )

        train_rng = random.Random(seed + 100 + dataset_offset)
        train_indexes = set(train_rng.sample(range(len(dataset_candidates)), train_count_per_dataset))
        for index, candidate in enumerate(dataset_candidates):
            curated = CuratedTask(source=candidate)
            if index in train_indexes:
                train.append(curated)
            else:
                test.append(curated)

    return sorted(train, key=curated_sort_key), sorted(test, key=curated_sort_key)


def assign_qids(train: list[CuratedTask], test: list[CuratedTask]) -> None:
    for split_qid, task in enumerate(train, start=1):
        task.split = "train"
        task.split_qid = split_qid

    for split_qid, task in enumerate(test, start=1):
        task.split = "test"
        task.split_qid = split_qid

    global_qid = 1
    for task in train + test:
        task.global_qid = global_qid
        global_qid += 1


def guess_media_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
        return "image"
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".md", ".txt"}:
        return "text"
    return "binary"


def build_task_prompt(task: CuratedTask) -> str:
    lines = [
        "As an expert AI agent, your goal is to accurately perform the requested task.",
        "Please use only the repository-local dataset information below to understand the format and scope of the task.",
        "",
        "--- Dataset Information ---",
        f'Dataset Name: "{task.dataset}"',
        "Required Input Paths:",
    ]
    lines.extend(f"- `{value}`" for value in task.source.required_inputs)
    lines.extend(
        [
            "",
            "--- Specific Task to Perform ---",
            task.source.task,
            "",
            "--- Required Output Files ---",
            "Save the following final output files with exactly these names:",
        ]
    )
    lines.extend(f"- `{value}`" for value in task.source.deliverables)
    lines.extend(
        [
            "",
            "--- Report Requirements ---",
        ]
    )
    lines.extend(f"- {value}" for value in task.source.report_requirements)
    lines.extend(
        [
            "",
            "--- Recommended Report Section Order ---",
        ]
    )
    lines.extend(f"- `{value}`" for value in REPORT_SECTION_ORDER)
    lines.extend(
        [
            "",
            "--- Hard Constraints ---",
            "- Use only the listed repository-local processed inputs.",
            "- Do not use any code under `healthflow/`.",
            "- Do not call external APIs, external services, or LLM endpoints.",
            "- Do not rely on the original paper text beyond the task content frozen here.",
            "- Keep the workflow executable on the assigned dataset only.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_dataset_row(task: CuratedTask, *, use_split_qid: bool) -> dict[str, Any]:
    qid = task.split_qid if use_split_qid else task.global_qid
    if qid is None:
        raise ValueError("qid is not assigned")
    return {
        "qid": qid,
        "task": build_task_prompt(task),
        "task_brief": task.source.task_brief,
        "dataset": task.dataset,
        "task_type": task.task_type,
        "reference_answer": task.reference_answer,
        "paper_id": task.paper_id,
        "paper_title": task.paper_title,
        "source_task_idx": task.task_idx,
        "focus_areas": list(task.source.focus_areas),
        "primary_bucket": task.primary_bucket,
    }


def build_required_outputs(task: CuratedTask) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for file_name in task.source.deliverables:
        reference_path = f"reference_answers/{task.split}/{task.split_qid}/{file_name}"
        outputs.append(
            {
                "file_name": file_name,
                "reference_path": reference_path,
                # Compatibility extra for local runners that still expect expected_path.
                "expected_path": reference_path,
                "media_type": guess_media_type(file_name),
            }
        )
    return outputs


def build_answer_manifest(task: CuratedTask) -> dict[str, Any]:
    required_outputs = build_required_outputs(task)
    return {
        "contract_version": CONTRACT_VERSION,
        "qid": task.split_qid,
        "dataset": task.dataset,
        "task_type": task.task_type,
        "judge_prompt_type": TASK_TYPE,
        "score_scale": {
            "min": 0,
            "max": 10.0,
            "pass_threshold": DEFAULT_PASS_THRESHOLD,
        },
        "required_inputs": list(task.source.required_inputs),
        "required_outputs": required_outputs,
        "all_outputs": [item["reference_path"] for item in required_outputs],
        "task_brief": task.source.task_brief,
        "paper_id": task.paper_id,
        "paper_title": task.paper_title,
        "source_task_idx": task.task_idx,
        "focus_areas": list(task.source.focus_areas),
        "primary_bucket": task.primary_bucket,
        "report_requirements": list(task.source.report_requirements),
        "origin": "generated_tasks_random_sample",
        "reference_outputs_ready": False,
        "reference_outputs_state": "manifest_only",
        "source_paths": {
            "generated_tasks": relative_path(task.source.tasks_path),
            "generation_response": relative_path(task.source.response_path),
        },
    }


def reset_output_targets(output_root: Path) -> None:
    for file_name in ("ehrflowbench.jsonl", "train.jsonl", "test.jsonl", "subset_manifest.json"):
        path = output_root / file_name
        if path.exists():
            path.unlink()

    reference_root = output_root / "reference_answers"
    for split in ("train", "test"):
        split_dir = reference_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)


def count_rows(rows: list[CuratedTask]) -> dict[str, Any]:
    dataset_counts = {
        DATASET_DISPLAY_NAMES[dataset_key]: 0
        for dataset_key in DATASET_ORDER
    }
    for row in rows:
        dataset_counts[row.dataset] += 1
    return {
        "task_type_counts": {TASK_TYPE: len(rows)},
        "dataset_counts": dataset_counts,
    }


def build_subset_manifest(
    *,
    candidates: list[GeneratedTaskCandidate],
    selected_train: list[CuratedTask],
    selected_test: list[CuratedTask],
    seed: int,
    sample_count_per_dataset: int,
    train_count_per_dataset: int,
    collection_summary: dict[str, Any],
) -> dict[str, Any]:
    selected_all = selected_train + selected_test
    test_count_per_dataset = sample_count_per_dataset - train_count_per_dataset
    return {
        "dataset": "ehrflowbench",
        "seed": seed,
        "schema_version": SUBSET_SCHEMA_VERSION,
        "task_types": [TASK_TYPE],
        "reference_answer_field": "reference_answer",
        "reference_answer_contract": "Train/test rows point to reference_answers/<split>/<qid>/answer_manifest.json.",
        "global_qid_policy": "ehrflowbench.jsonl keeps global qids; train.jsonl and test.jsonl are renumbered within each split.",
        "source_files": {
            DATASET_DISPLAY_NAMES["tjh"]: next(iter(sorted(DATASET_CORE_REQUIRED_INPUTS["tjh"]))),
            DATASET_DISPLAY_NAMES["mimic_iv_demo"]: next(
                iter(sorted(DATASET_CORE_REQUIRED_INPUTS["mimic_iv_demo"]))
            ),
        },
        "selection_policy": {
            "sample_count_per_dataset": sample_count_per_dataset,
            "train_count_per_dataset": train_count_per_dataset,
            "test_count_per_dataset": test_count_per_dataset,
            "sampling_mode": "random_without_replacement_per_dataset",
        },
        "source_pool": collection_summary,
        "combined": {
            "count": len(selected_all),
            **count_rows(selected_all),
        },
        "train": {
            "count": len(selected_train),
            **count_rows(selected_train),
            "qids": [task.split_qid for task in selected_train],
        },
        "test": {
            "count": len(selected_test),
            **count_rows(selected_test),
            "qids": [task.split_qid for task in selected_test],
        },
        "qid_remap": [
            {
                "global_qid": task.global_qid,
                "split": task.split,
                "split_qid": task.split_qid,
                "paper_id": task.paper_id,
                "paper_title": task.paper_title,
                "source_task_idx": task.task_idx,
                "dataset": task.dataset,
                "primary_bucket": task.primary_bucket,
            }
            for task in selected_all
        ],
        "selected_rows": [
            {
                "global_qid": task.global_qid,
                "split": task.split,
                "split_qid": task.split_qid,
                "dataset": task.dataset,
                "paper_id": task.paper_id,
                "source_task_idx": task.task_idx,
                "reference_answer": task.reference_answer,
            }
            for task in selected_all
        ],
        "candidate_dataset_counts": {
            DATASET_DISPLAY_NAMES[key]: sum(1 for candidate in candidates if candidate.dataset_key == key)
            for key in DATASET_ORDER
        },
    }


def run_curation(
    *,
    seed: int = SEED,
    sample_count_per_dataset: int = DEFAULT_SAMPLE_COUNT_PER_DATASET,
    train_count_per_dataset: int = DEFAULT_TRAIN_COUNT_PER_DATASET,
    output_root: Path = PROCESSED_ROOT,
    generated_tasks_root: Path = GENERATED_TASKS_ROOT,
    paper_titles_path: Path = PAPER_TITLES_PATH,
    allow_missing_responses: bool = False,
) -> dict[str, Any]:
    if train_count_per_dataset > sample_count_per_dataset:
        raise ValueError("train_count_per_dataset cannot exceed sample_count_per_dataset")

    paper_titles = load_paper_titles(paper_titles_path)
    candidates, collection_summary = collect_generated_candidates(
        generated_tasks_root,
        paper_titles,
        allow_missing_responses=allow_missing_responses,
    )
    selected_candidates = sample_candidates_by_dataset(
        candidates,
        sample_count_per_dataset=sample_count_per_dataset,
        seed=seed,
    )
    train, test = split_curated_candidates(
        selected_candidates,
        train_count_per_dataset=train_count_per_dataset,
        seed=seed,
    )
    assign_qids(train, test)

    output_root.mkdir(parents=True, exist_ok=True)
    reset_output_targets(output_root)

    combined_rows = [build_dataset_row(task, use_split_qid=False) for task in train + test]
    train_rows = [build_dataset_row(task, use_split_qid=True) for task in train]
    test_rows = [build_dataset_row(task, use_split_qid=True) for task in test]
    write_jsonl(output_root / "ehrflowbench.jsonl", combined_rows)
    write_jsonl(output_root / "train.jsonl", train_rows)
    write_jsonl(output_root / "test.jsonl", test_rows)

    for task in train + test:
        manifest_path = output_root / task.reference_answer
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, build_answer_manifest(task))

    subset_manifest = build_subset_manifest(
        candidates=candidates,
        selected_train=train,
        selected_test=test,
        seed=seed,
        sample_count_per_dataset=sample_count_per_dataset,
        train_count_per_dataset=train_count_per_dataset,
        collection_summary=collection_summary,
    )
    write_json(output_root / "subset_manifest.json", subset_manifest)

    return {
        "seed": seed,
        "sample_count_per_dataset": sample_count_per_dataset,
        "train_count_per_dataset": train_count_per_dataset,
        "test_count_per_dataset": sample_count_per_dataset - train_count_per_dataset,
        "candidate_count": len(candidates),
        "selected_count": len(train) + len(test),
        "train_count": len(train),
        "test_count": len(test),
        "output_root": relative_path(output_root),
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
        allow_missing_responses=args.allow_missing_responses,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
