from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


DEFAULT_TRAIN_SIZE = 10
DEFAULT_TEST_SIZE = 100
DEFAULT_MIN_QUALITY_SCORE = 7
DEFAULT_DUPLICATE_THRESHOLD = 0.55

SOURCE_TASK_LINKAGE_MODE = "task_proxy"
SOURCE_TASK_ELIGIBILITY = "proxy_candidate"
REVIEW_STATUS = "seeded_for_human_review"
TASK_MANIFEST_VERSION = "ehrflowbench_task_manifest_v1"

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

TRAIN_COVERAGE_BUCKETS = (
    "graph",
    "temporal",
    "representation",
    "prediction",
    "causal",
    "unsupervised",
    "interpretability",
)

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

NUMERIC_SUFFIXES = {".json", ".csv", ".parquet", ".tsv", ".xlsx"}
VISUAL_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}

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

FALLBACK_BUCKET_PRIORITY = (
    "causal",
    "fairness",
    "unsupervised",
    "graph",
    "temporal",
    "representation",
    "interpretability",
)

METHOD_FAMILY_PATTERNS = (
    ("causal", ("causal", "counterfactual", "treatment effect", "dynamic treatment")),
    ("fairness", ("fairness", "bias", "equity", "debiased")),
    ("graph", ("graph", "network", "node", "edge")),
    ("unsupervised", ("cluster", "clustering", "phenotype", "subgroup", "retrieval", "similarity")),
    ("text", ("clinical notes", "event text", "tf-idf", "bag-of-words", "text representations", "medical text")),
    ("temporal", ("temporal", "trajectory", "horizon", "sequence", "time-aware", "next-visit", "longitudinal")),
    ("feature_selection", ("feature selection", "feature acquisition", "importance", "mask", "attention-like")),
    ("robustness", ("robustness", "perturb", "dataset shift", "generalization", "transfer")),
    ("representation", ("representation", "embedding", "representation learning", "context model")),
)

TARGET_FAMILY_PATTERNS = (
    ("readmission", ("readmission",)),
    ("length_of_stay", ("length-of-stay", "length of stay", "`los`", " high-los", "high los")),
    ("mortality", ("mortality", "death", "deterioration")),
    ("next_visit", ("next-visit", "next visit")),
    ("utilization", ("utilization", "revisit", "inter-visit", "icu revisit")),
    ("retrieval", ("retrieval", "similarity search")),
    ("clustering", ("cluster", "phenotype")),
    ("text_extraction", ("extraction", "verification from unstructured text")),
    ("outcome_prediction", ("outcome", "risk prediction", "prediction", "forecast")),
)

EXPLICIT_MODEL_PATTERNS = (
    "logistic regression",
    "naive bayes",
    "gradient boosting",
    "random forest",
    "linear model",
    "sklearn",
    "tf-idf",
    "bag-of-words",
    "permutation importance",
)

METRIC_PATTERNS = (
    "auroc",
    "auprc",
    "accuracy",
    "f1",
    "brier score",
    "rmse",
    "mae",
    "calibration",
    "silhouette score",
)

LIGHTWEIGHT_PATTERNS = (
    "lightweight",
    "deterministic",
    "cpu-friendly",
    "sklearn",
    "small tree-based model",
    "small validation-based hyperparameter selection",
)

TARGET_FALLBACK_PATTERNS = (
    "if a candidate target is unavailable",
    "fall back to another directly available target",
    "if multiple feasible targets exist",
    "if split files only define patient partitions",
)

EXPLICIT_COMMON_TARGET_PATTERNS = (
    "choose one binary target that is present in both tjh and mimic-iv-demo",
    "common executable utilization-oriented endpoint available or derivable in both datasets",
    "prioritizing `outcome`, `readmission`, or",
    "target that is present in both tjh and mimic-iv-demo",
)

TEXT_DEPENDENCY_PATTERNS = (
    "clinical notes",
    "event text",
    "medical text",
    "bag-of-words",
    "tf-idf",
    "unstructured text",
)

FORBIDDEN_DEPENDENCY_PATTERNS = {
    "depends_on_healthflow_runtime": (
        "use code under `healthflow/`",
        "use code from `healthflow/`",
        "rely on code under `healthflow/`",
        "depends on `healthflow/`",
    ),
    "depends_on_raw_rebuild": (
        "prepare_tjh.py",
        "prepare_mimic_iv_demo.py",
        "rerun raw-data preprocessing",
        "rebuild the processed ehr",
        "regenerate raw data",
        "raw inputs",
    ),
    "depends_on_external_data": ("external dataset", "outside this repository"),
    "depends_on_manual_annotation": ("manual annotation", "human annotation", "annotator"),
}

HEAVY_EXECUTION_PATTERNS = (
    r"\btrain a gan\b",
    r"\breinforcement learning\b",
    r"\bfrom-scratch pretraining\b",
    r"\bgpu-only\b",
    r"\bhours-long experiments\b",
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
MARKDOWN_ROOT = RAW_PAPERS_ROOT / "markdowns"
PAPER_TITLES_PATH = RAW_PAPERS_ROOT / "paper_titles.csv"
GENERATED_TASKS_ROOT = DATASET_ROOT / "processed" / "papers" / "generated_tasks"
PROCESSED_ROOT = DATASET_ROOT / "processed"
EXPECTED_ROOT = PROCESSED_ROOT / "expected"


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

    @property
    def key(self) -> tuple[int, int]:
        return (self.paper_id, self.task_idx)


@dataclass
class ReviewedTask:
    source: GeneratedTaskCandidate
    primary_bucket: str
    method_family: str
    target_family: str
    flags: list[str] = field(default_factory=list)
    hard_reject_reasons: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    similarity_text: str = ""
    feasibility_score: int = 0
    specificity_score: int = 0
    evaluability_score: int = 0
    practicality_score: int = 0
    novelty_score: int = 0
    quality_score: int = 0
    explicit_common_target: bool = False
    has_target_fallback: bool = False
    has_numeric_artifact: bool = False
    has_figure_or_table_artifact: bool = False
    has_text_dependency: bool = False
    input_dataset: str | None = None
    has_valid_single_dataset_inputs: bool = False
    has_mixed_dataset_inputs: bool = False
    has_split_metadata_inputs: bool = False
    lightweight_hint: bool = False
    selected: bool = False
    split: str | None = None
    qid: int | None = None

    @property
    def key(self) -> tuple[int, int]:
        return self.source.key

    @property
    def paper_id(self) -> int:
        return self.source.paper_id

    @property
    def task_idx(self) -> int:
        return self.source.task_idx

    @property
    def paper_title(self) -> str:
        return self.source.paper_title

    @property
    def task_brief(self) -> str:
        return self.source.task_brief

    @property
    def deliverables(self) -> tuple[str, ...]:
        return self.source.deliverables

    @property
    def required_inputs(self) -> tuple[str, ...]:
        return self.source.required_inputs

    @property
    def review_status(self) -> str:
        return REVIEW_STATUS if self.selected else "rejected_by_heuristic_curation"

    @property
    def reference_answer(self) -> str:
        if self.qid is None:
            raise ValueError("qid is not assigned yet")
        return f"expected/{self.qid}/task_manifest.json"

    def quality_breakdown(self) -> dict[str, int]:
        return {
            "feasibility": self.feasibility_score,
            "specificity": self.specificity_score,
            "evaluability": self.evaluability_score,
            "practicality": self.practicality_score,
            "novelty": self.novelty_score,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate EHRFlowBench generated tasks into train/test splits.")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--min-quality-score", type=int, default=DEFAULT_MIN_QUALITY_SCORE)
    parser.add_argument("--duplicate-threshold", type=float, default=DEFAULT_DUPLICATE_THRESHOLD)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow curation to run before every markdown paper has a completed task bundle.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROCESSED_ROOT,
        help="Processed output root. Defaults to data/ehrflowbench/processed.",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def paper_id_from_name(name: str) -> int:
    match = re.match(r"(\d+)_", name)
    if not match:
        raise ValueError(f"Could not parse paper id from {name!r}")
    return int(match.group(1))


def discover_markdown_dirs(markdown_root: Path) -> dict[int, Path]:
    if not markdown_root.exists():
        return {}
    result: dict[int, Path] = {}
    for path in sorted(markdown_root.iterdir()):
        if path.is_dir():
            result[paper_id_from_name(path.name)] = path
    return result


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


def title_from_dir_name(path: Path | None) -> str:
    if path is None:
        return ""
    tail = re.sub(r"^\d+_", "", path.name)
    tail = re.sub(r"\.pdf-[a-f0-9-]+$", "", tail, flags=re.IGNORECASE)
    return tail.replace("_", " ").strip()


def collect_generated_candidates(
    output_root: Path,
    paper_titles: dict[int, str],
    markdown_dirs: dict[int, Path],
    *,
    allow_incomplete: bool,
) -> tuple[list[GeneratedTaskCandidate], set[int], list[str]]:
    task_files = {
        int(path.stem.split("_")[0]): path
        for path in output_root.glob("*_tasks.json")
    }
    response_files = {
        int(path.stem.split("_")[0]): path
        for path in output_root.glob("*_response.json")
    }

    completed_ids = sorted(set(task_files) & set(response_files))
    warnings: list[str] = []
    candidates: list[GeneratedTaskCandidate] = []
    parseable_ids: set[int] = set()

    for paper_id in completed_ids:
        try:
            task_payload = read_json(task_files[paper_id])
            _ = read_json(response_files[paper_id])
        except Exception as exc:
            if not allow_incomplete:
                raise ValueError(f"Failed to parse generated bundle for paper {paper_id}: {exc}") from exc
            warnings.append(f"Skipped partially written bundle for paper {paper_id}: {exc}")
            continue

        tasks = task_payload.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 2:
            if not allow_incomplete:
                raise ValueError(f"Paper {paper_id} does not contain exactly 2 tasks")
            warnings.append(f"Skipped incomplete bundle for paper {paper_id}: expected 2 tasks")
            continue

        paper_title = paper_titles.get(paper_id) or title_from_dir_name(markdown_dirs.get(paper_id)) or f"Paper {paper_id}"
        parseable_ids.add(paper_id)
        for task_idx, item in enumerate(tasks, start=1):
            candidates.append(
                GeneratedTaskCandidate(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    task_idx=task_idx,
                    task_brief=str(item["task_brief"]).strip(),
                    task_type=str(item["task_type"]).strip(),
                    focus_areas=tuple(str(value).strip() for value in item.get("focus_areas", [])),
                    task=str(item["task"]).strip(),
                    required_inputs=tuple(str(value).strip() for value in item.get("required_inputs", [])),
                    deliverables=tuple(str(value).strip() for value in item.get("deliverables", [])),
                    report_requirements=tuple(str(value).strip() for value in item.get("report_requirements", [])),
                    tasks_path=task_files[paper_id],
                    response_path=response_files[paper_id],
                )
            )

    return candidates, parseable_ids, warnings


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def contains_any(text: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def classify_primary_bucket(paper_title: str, task_blob: str) -> str:
    title_text = normalize_text(paper_title)
    for bucket in BUCKET_PRIORITY:
        if contains_any(title_text, TITLE_BUCKET_PATTERNS.get(bucket, ())):
            return bucket

    task_text = normalize_text(task_blob)
    for bucket in FALLBACK_BUCKET_PRIORITY:
        if contains_any(task_text, TEXT_BUCKET_PATTERNS.get(bucket, ())):
            return bucket
    return "other"


def classify_family(task_blob: str, family_patterns: tuple[tuple[str, tuple[str, ...]], ...], fallback: str) -> str:
    text = normalize_text(task_blob)
    for family, patterns in family_patterns:
        if contains_any(text, patterns):
            return family
    return fallback


def classify_method_family(task_blob: str) -> str:
    return classify_family(task_blob, METHOD_FAMILY_PATTERNS, "prediction")


def classify_target_family(task_blob: str) -> str:
    return classify_family(task_blob, TARGET_FAMILY_PATTERNS, "generic")


def looks_like_numeric_artifact(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in NUMERIC_SUFFIXES


def looks_like_figure_or_table(path: str) -> bool:
    if path.startswith("tables/") or path.startswith("figures/"):
        return True
    suffix = Path(path).suffix.lower()
    return suffix in NUMERIC_SUFFIXES or suffix in VISUAL_SUFFIXES


def classify_required_input_set(required_inputs: tuple[str, ...]) -> tuple[str | None, bool, bool, bool]:
    input_set = set(required_inputs)
    touched_datasets = {
        dataset_key
        for dataset_key in DATASET_CORE_REQUIRED_INPUTS
        if input_set & (
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


def contains_forbidden_dependency(text: str) -> list[str]:
    negative_prefix_markers = ("do not", "don't", "avoid", "without", "rather than", "not require", "must not")
    reasons: list[str] = []
    for reason, patterns in FORBIDDEN_DEPENDENCY_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(re.escape(pattern), text):
                prefix = text[max(0, match.start() - 60):match.start()]
                if any(marker in prefix for marker in negative_prefix_markers):
                    continue
                reasons.append(reason)
                break
            if reason in reasons:
                break
    return reasons


def requires_heavy_execution(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in HEAVY_EXECUTION_PATTERNS)


def build_similarity_text(candidate: GeneratedTaskCandidate) -> str:
    prefix = candidate.task[:400]
    parts = [candidate.task_brief, " ".join(candidate.focus_areas), prefix]
    return " ".join(part for part in parts if part).strip()


def review_candidate(candidate: GeneratedTaskCandidate) -> ReviewedTask:
    classification_blob = " ".join(
        [
            candidate.task_brief,
            " ".join(candidate.focus_areas),
            candidate.task,
        ]
    )
    task_blob = " ".join(
        [
            candidate.paper_title,
            candidate.task_brief,
            " ".join(candidate.focus_areas),
            candidate.task,
            " ".join(candidate.deliverables),
            " ".join(candidate.report_requirements),
        ]
    )
    normalized_blob = normalize_text(task_blob)
    bucket = classify_primary_bucket(candidate.paper_title, classification_blob)
    method_family = classify_method_family(task_blob)
    target_family = classify_target_family(task_blob)

    reviewed = ReviewedTask(
        source=candidate,
        primary_bucket=bucket,
        method_family=method_family,
        target_family=target_family,
        similarity_text=build_similarity_text(candidate),
    )

    reviewed.has_target_fallback = contains_any(normalized_blob, TARGET_FALLBACK_PATTERNS)
    reviewed.explicit_common_target = contains_any(normalized_blob, EXPLICIT_COMMON_TARGET_PATTERNS)
    reviewed.has_text_dependency = contains_any(normalized_blob, TEXT_DEPENDENCY_PATTERNS)
    reviewed.lightweight_hint = contains_any(normalized_blob, LIGHTWEIGHT_PATTERNS)
    reviewed.has_numeric_artifact = any(looks_like_numeric_artifact(path) for path in candidate.deliverables)
    reviewed.has_figure_or_table_artifact = any(looks_like_figure_or_table(path) for path in candidate.deliverables)
    (
        reviewed.input_dataset,
        reviewed.has_valid_single_dataset_inputs,
        reviewed.has_split_metadata_inputs,
        reviewed.has_mixed_dataset_inputs,
    ) = classify_required_input_set(candidate.required_inputs)

    if reviewed.explicit_common_target:
        reviewed.flags.append("explicit_common_target")
    if reviewed.has_target_fallback:
        reviewed.flags.append("target_fallback")
    if reviewed.has_text_dependency:
        reviewed.flags.append("text_dependency")
    if reviewed.lightweight_hint:
        reviewed.flags.append("lightweight")
    if reviewed.input_dataset:
        reviewed.flags.append(f"input_dataset:{reviewed.input_dataset}")
    if reviewed.method_family == "graph":
        reviewed.flags.append("graph_method")
    if reviewed.method_family == "unsupervised":
        reviewed.flags.append("unsupervised_method")

    if candidate.task_type != "report_generation":
        reviewed.hard_reject_reasons.append("invalid_task_type")
    if "report.md" not in candidate.deliverables:
        reviewed.hard_reject_reasons.append("missing_report_deliverable")
    if not reviewed.has_numeric_artifact:
        reviewed.hard_reject_reasons.append("missing_numeric_artifact")
    if reviewed.has_mixed_dataset_inputs:
        reviewed.hard_reject_reasons.append("mixed_dataset_inputs")
    if not reviewed.has_valid_single_dataset_inputs:
        reviewed.hard_reject_reasons.append("invalid_single_dataset_inputs")
    reviewed.hard_reject_reasons.extend(contains_forbidden_dependency(normalized_blob))

    if requires_heavy_execution(normalized_blob):
        reviewed.flags.append("heavy_execution_reference")

    reviewed.feasibility_score = 0
    if reviewed.has_valid_single_dataset_inputs:
        reviewed.feasibility_score += 1
    if reviewed.has_split_metadata_inputs:
        reviewed.feasibility_score += 1
    if reviewed.explicit_common_target or (
        "`outcome`" in normalized_blob or "`readmission`" in normalized_blob or "`los`" in normalized_blob
    ):
        reviewed.feasibility_score += 1

    reviewed.specificity_score = 0
    if contains_any(normalized_blob, EXPLICIT_MODEL_PATTERNS):
        reviewed.specificity_score += 1
    if contains_any(normalized_blob, METRIC_PATTERNS) or "split_metadata.json" in normalized_blob:
        reviewed.specificity_score += 1

    reviewed.evaluability_score = 0
    if "report.md" in candidate.deliverables and reviewed.has_numeric_artifact:
        reviewed.evaluability_score += 1
    if reviewed.has_figure_or_table_artifact and len(candidate.report_requirements) >= 6:
        reviewed.evaluability_score += 1

    reviewed.practicality_score = 0
    if reviewed.lightweight_hint:
        reviewed.practicality_score += 1
    if not requires_heavy_execution(normalized_blob) and len(candidate.deliverables) <= 14:
        reviewed.practicality_score += 1

    reviewed.quality_score = (
        reviewed.feasibility_score
        + reviewed.specificity_score
        + reviewed.evaluability_score
        + reviewed.practicality_score
    )
    return reviewed


def assign_novelty_scores(reviewed_tasks: list[ReviewedTask]) -> None:
    bucket_sizes = Counter(task.primary_bucket for task in reviewed_tasks if not task.hard_reject_reasons)
    bucket_method_counts = Counter(
        (task.primary_bucket, task.method_family) for task in reviewed_tasks if not task.hard_reject_reasons
    )
    bucket_target_counts = Counter(
        (task.primary_bucket, task.target_family) for task in reviewed_tasks if not task.hard_reject_reasons
    )

    for task in reviewed_tasks:
        novelty = 0
        if not task.hard_reject_reasons:
            method_count = bucket_method_counts[(task.primary_bucket, task.method_family)]
            target_count = bucket_target_counts[(task.primary_bucket, task.target_family)]
            bucket_size = bucket_sizes[task.primary_bucket]
            rarity_threshold = max(1, math.ceil(bucket_size / 5))
            if method_count <= rarity_threshold or target_count <= rarity_threshold:
                novelty = 1
            if task.primary_bucket in {"causal", "fairness"} or task.has_text_dependency:
                novelty = 1
        task.novelty_score = novelty
        task.quality_score += novelty


def selection_sort_key(task: ReviewedTask) -> tuple[Any, ...]:
    return (
        -task.quality_score,
        task.has_target_fallback,
        -int(task.explicit_common_target),
        -int(task.has_figure_or_table_artifact),
        BUCKET_PRIORITY.index(task.primary_bucket),
        task.paper_id,
        task.task_idx,
    )


def select_best_task_per_paper(reviewed_tasks: list[ReviewedTask], min_quality_score: int) -> tuple[list[ReviewedTask], dict[tuple[int, int], list[str]]]:
    grouped: dict[int, list[ReviewedTask]] = defaultdict(list)
    for task in reviewed_tasks:
        grouped[task.paper_id].append(task)

    selected: list[ReviewedTask] = []
    rejection_reasons: dict[tuple[int, int], list[str]] = defaultdict(list)

    for paper_id, tasks in sorted(grouped.items()):
        eligible = [task for task in tasks if not task.hard_reject_reasons and task.quality_score >= min_quality_score]
        if not eligible:
            for task in tasks:
                if task.hard_reject_reasons:
                    rejection_reasons[task.key].extend(task.hard_reject_reasons)
                else:
                    rejection_reasons[task.key].append("quality_below_threshold")
            continue

        best = sorted(eligible, key=selection_sort_key)[0]
        selected.append(best)
        for task in tasks:
            if task.key == best.key:
                continue
            if task.hard_reject_reasons:
                rejection_reasons[task.key].extend(task.hard_reject_reasons)
            elif task.quality_score < min_quality_score:
                rejection_reasons[task.key].append("quality_below_threshold")
            else:
                rejection_reasons[task.key].append(f"paper_peer_lower_rank:selected_task_idx={best.task_idx}")

    return selected, rejection_reasons


def allocate_bucket_quotas(bucket_counts: Counter[str], total: int) -> dict[str, int]:
    non_empty_buckets = [bucket for bucket in BUCKET_PRIORITY if bucket_counts.get(bucket, 0) > 0]
    if total < len(non_empty_buckets):
        raise ValueError("Total selection size is smaller than the number of non-empty buckets")

    quotas = {bucket: 1 for bucket in non_empty_buckets}
    remaining = total - len(non_empty_buckets)
    total_candidates = sum(bucket_counts[bucket] for bucket in non_empty_buckets)
    if remaining == 0:
        return quotas

    remainders: list[tuple[float, str]] = []
    for bucket in non_empty_buckets:
        exact_extra = remaining * bucket_counts[bucket] / total_candidates
        extra_floor = math.floor(exact_extra)
        quotas[bucket] += extra_floor
        remainders.append((exact_extra - extra_floor, bucket))

    assigned = sum(quotas.values())
    for _, bucket in sorted(remainders, key=lambda item: (-item[0], BUCKET_PRIORITY.index(item[1]))):
        if assigned >= total:
            break
        quotas[bucket] += 1
        assigned += 1

    if sum(quotas.values()) != total:
        raise ValueError("Bucket quota allocation failed to preserve the total")
    return quotas


def build_similarity_matrices(tasks: list[ReviewedTask]) -> tuple[dict[str, Any], dict[tuple[int, int], int]]:
    matrices: dict[str, Any] = {}
    positions: dict[tuple[int, int], int] = {}
    grouped: dict[str, list[ReviewedTask]] = defaultdict(list)
    for task in tasks:
        grouped[task.primary_bucket].append(task)

    for bucket, bucket_tasks in grouped.items():
        documents = [task.similarity_text for task in bucket_tasks]
        if len(documents) == 1:
            matrices[bucket] = None
            positions[bucket_tasks[0].key] = 0
            continue
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(documents)
        matrices[bucket] = matrix
        for index, task in enumerate(bucket_tasks):
            positions[task.key] = index
    return matrices, positions


def find_near_duplicate(
    candidate: ReviewedTask,
    selected_in_bucket: list[ReviewedTask],
    matrices: dict[str, Any],
    positions: dict[tuple[int, int], int],
    threshold: float,
) -> tuple[ReviewedTask | None, float]:
    if not selected_in_bucket:
        return None, 0.0

    matrix = matrices.get(candidate.primary_bucket)
    if matrix is None:
        return None, 0.0

    candidate_index = positions[candidate.key]
    selected_indexes = [positions[task.key] for task in selected_in_bucket]
    similarities = linear_kernel(matrix[candidate_index], matrix[selected_indexes]).ravel()
    best_index = int(similarities.argmax())
    best_similarity = float(similarities[best_index])
    best_task = selected_in_bucket[best_index]

    if best_similarity <= threshold:
        return None, best_similarity
    if candidate.method_family != best_task.method_family or candidate.target_family != best_task.target_family:
        return None, best_similarity
    return best_task, best_similarity


def select_canonical_tasks(
    paper_best_tasks: list[ReviewedTask],
    total_target: int,
    duplicate_threshold: float,
) -> tuple[list[ReviewedTask], dict[str, int], dict[tuple[int, int], list[str]], bool]:
    if len(paper_best_tasks) < total_target:
        raise ValueError(f"Not enough eligible papers to select {total_target} tasks: found {len(paper_best_tasks)}")

    bucket_counts = Counter(task.primary_bucket for task in paper_best_tasks)
    quotas = allocate_bucket_quotas(bucket_counts, total_target)

    grouped: dict[str, list[ReviewedTask]] = defaultdict(list)
    for task in paper_best_tasks:
        grouped[task.primary_bucket].append(task)
    for bucket in grouped:
        grouped[bucket].sort(key=selection_sort_key)

    matrices, positions = build_similarity_matrices(paper_best_tasks)
    selected: list[ReviewedTask] = []
    selected_keys: set[tuple[int, int]] = set()
    rejections: dict[tuple[int, int], list[str]] = defaultdict(list)

    for bucket in BUCKET_PRIORITY:
        bucket_tasks = grouped.get(bucket, [])
        bucket_selected: list[ReviewedTask] = []
        quota = quotas.get(bucket, 0)
        for task in bucket_tasks:
            if len(bucket_selected) >= quota:
                rejections[task.key].append("bucket_quota_exhausted")
                continue
            duplicate_with, similarity = find_near_duplicate(
                task,
                bucket_selected,
                matrices,
                positions,
                duplicate_threshold,
            )
            if duplicate_with is not None:
                rejections[task.key].append(
                    f"near_duplicate_of_qid_pending:paper={duplicate_with.paper_id}:task={duplicate_with.task_idx}:similarity={similarity:.3f}"
                )
                continue
            selected.append(task)
            selected_keys.add(task.key)
            bucket_selected.append(task)

    leftovers = [task for task in paper_best_tasks if task.key not in selected_keys]
    leftovers.sort(key=selection_sort_key)

    for task in leftovers:
        if len(selected) >= total_target:
            break
        selected_in_bucket = [item for item in selected if item.primary_bucket == task.primary_bucket]
        duplicate_with, similarity = find_near_duplicate(
            task,
            selected_in_bucket,
            matrices,
            positions,
            duplicate_threshold,
        )
        if duplicate_with is not None:
            rejections[task.key].append(
                f"global_fill_near_duplicate_of_paper={duplicate_with.paper_id}:task={duplicate_with.task_idx}:similarity={similarity:.3f}"
            )
            continue
        selected.append(task)
        selected_keys.add(task.key)

    duplicate_guard_relaxed = False
    if len(selected) < total_target:
        duplicate_guard_relaxed = True
        for task in leftovers:
            if len(selected) >= total_target:
                break
            if task.key in selected_keys:
                continue
            rejections[task.key] = [reason for reason in rejections[task.key] if not reason.startswith("global_fill_near_duplicate")]
            selected.append(task)
            selected_keys.add(task.key)

    if len(selected) != total_target:
        raise ValueError(f"Could not select exactly {total_target} canonical tasks; selected {len(selected)}")

    final_rejections = {
        key: reasons
        for key, reasons in rejections.items()
        if key not in selected_keys and reasons
    }
    return selected, quotas, final_rejections, duplicate_guard_relaxed


def eligible_for_train(task: ReviewedTask) -> bool:
    if task.primary_bucket == "fairness" and task.quality_score >= 8:
        return False
    return True


def assign_splits(selected_tasks: list[ReviewedTask], train_size: int, test_size: int) -> tuple[list[ReviewedTask], list[ReviewedTask]]:
    if len(selected_tasks) != train_size + test_size:
        raise ValueError("Split sizes do not sum to the number of selected tasks")

    grouped: dict[str, list[ReviewedTask]] = defaultdict(list)
    for task in selected_tasks:
        grouped[task.primary_bucket].append(task)
    for bucket in grouped:
        grouped[bucket].sort(key=selection_sort_key)

    train: list[ReviewedTask] = []
    used: set[tuple[int, int]] = set()

    for bucket in TRAIN_COVERAGE_BUCKETS:
        candidates = [task for task in grouped.get(bucket, []) if task.key not in used and eligible_for_train(task)]
        if not candidates:
            continue
        chosen = candidates[0]
        train.append(chosen)
        used.add(chosen.key)
        if len(train) == train_size:
            break

    if len(train) < train_size:
        remaining = [task for task in selected_tasks if task.key not in used and eligible_for_train(task)]
        remaining_bucket_counts = Counter(task.primary_bucket for task in remaining)
        remaining.sort(
            key=lambda task: (
                -remaining_bucket_counts[task.primary_bucket],
                *selection_sort_key(task),
            )
        )
        for task in remaining:
            if len(train) >= train_size:
                break
            train.append(task)
            used.add(task.key)

    if len(train) < train_size:
        fallback_remaining = [task for task in selected_tasks if task.key not in used]
        fallback_remaining.sort(key=selection_sort_key)
        for task in fallback_remaining:
            if len(train) >= train_size:
                break
            train.append(task)
            used.add(task.key)

    if len(train) != train_size:
        raise ValueError(f"Could not assign {train_size} train tasks; assigned {len(train)}")

    test_tasks = [task for task in selected_tasks if task.key not in used]
    test_tasks.sort(key=selection_sort_key)
    if len(test_tasks) != test_size:
        raise ValueError(f"Could not assign {test_size} test tasks; assigned {len(test_tasks)}")

    train = train[:]
    return train, test_tasks


def assign_qids(train_tasks: list[ReviewedTask], test_tasks: list[ReviewedTask]) -> None:
    qid = 1
    for split, rows in (("train", train_tasks), ("test", test_tasks)):
        for task in rows:
            task.selected = True
            task.split = split
            task.qid = qid
            qid += 1


def guess_media_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".json"}:
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix in {".parquet"}:
        return "parquet"
    if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "image"
    if suffix in {".pdf"}:
        return "pdf"
    if suffix in {".md"}:
        return "markdown"
    return "text"


def dataset_display_name(dataset_key: str | None) -> str:
    if dataset_key is None:
        return "the assigned dataset"
    return DATASET_DISPLAY_NAMES.get(dataset_key, dataset_key)


def build_task_prompt(task: ReviewedTask) -> str:
    input_dataset_label = dataset_display_name(task.input_dataset)
    lines = [
        "Complete this EHRFlowBench report-generation project using only the repository-local processed EHR assets.",
        "",
        f"Task brief: {task.source.task_brief}",
        "",
        "Project task:",
        task.source.task,
        "",
        "Required inputs:",
    ]
    lines.extend(f"- `{value}`" for value in task.source.required_inputs)
    lines.append("")
    lines.append("Required deliverables:")
    lines.extend(f"- `{value}`" for value in task.source.deliverables)
    lines.append("")
    lines.append("Report requirements:")
    lines.extend(f"- {value}" for value in task.source.report_requirements)
    lines.append("")
    lines.append("Execution constraints:")
    lines.append("- Use `uv run python` for Python execution.")
    lines.append("- Do not use any code under `healthflow/`.")
    lines.append(f"- Keep the workflow executable on {input_dataset_label} only.")
    return "\n".join(lines).strip()


def build_dataset_row(task: ReviewedTask) -> dict[str, Any]:
    return {
        "qid": task.qid,
        "task": build_task_prompt(task),
        "task_brief": task.source.task_brief,
        "task_type": task.source.task_type,
        "input_dataset": task.input_dataset,
        "focus_areas": list(task.source.focus_areas),
        "paper_id": task.paper_id,
        "paper_title": task.paper_title,
        "source_task_idx": task.task_idx,
        "primary_bucket": task.primary_bucket,
        "quality_score": task.quality_score,
        "reference_answer": task.reference_answer,
    }


def build_task_manifest(task: ReviewedTask) -> dict[str, Any]:
    return {
        "contract_version": TASK_MANIFEST_VERSION,
        "qid": task.qid,
        "split": task.split,
        "dataset": "ehrflowbench",
        "task_type": task.source.task_type,
        "input_dataset": task.input_dataset,
        "task_brief": task.source.task_brief,
        "paper_id": task.paper_id,
        "paper_title": task.paper_title,
        "source_task_idx": task.task_idx,
        "primary_bucket": task.primary_bucket,
        "method_family": task.method_family,
        "target_family": task.target_family,
        "quality_score": task.quality_score,
        "quality_breakdown": task.quality_breakdown(),
        "required_inputs": list(task.source.required_inputs),
        "deliverables": list(task.source.deliverables),
        "report_requirements": list(task.source.report_requirements),
        "required_outputs": [
            {
                "file_name": file_name,
                "expected_path": f"expected/{task.qid}/{file_name}",
                "media_type": guess_media_type(file_name),
            }
            for file_name in task.source.deliverables
        ],
        "source_task_linkage_mode": SOURCE_TASK_LINKAGE_MODE,
        "source_task_eligibility": SOURCE_TASK_ELIGIBILITY,
        "review_status": REVIEW_STATUS,
        "reference_outputs_ready": False,
        "source_paths": {
            "generated_tasks": relative_path(task.source.tasks_path),
            "generation_response": relative_path(task.source.response_path),
        },
    }


def build_manifest_index_row(task: ReviewedTask) -> dict[str, Any]:
    return {
        "qid": task.qid,
        "split": task.split,
        "paper_id": task.paper_id,
        "paper_title": task.paper_title,
        "source_task_idx": task.task_idx,
        "task_brief": task.source.task_brief,
        "primary_bucket": task.primary_bucket,
        "method_family": task.method_family,
        "target_family": task.target_family,
        "quality_score": task.quality_score,
        "quality_breakdown": task.quality_breakdown(),
        "flags": task.flags,
        "reference_answer": task.reference_answer,
    }


def build_paper_map_row(task: ReviewedTask) -> dict[str, Any]:
    return {
        "qid": task.qid,
        "split": task.split,
        "paper_id": task.paper_id,
        "paper_title": task.paper_title,
        "selected_task_idx": task.task_idx,
        "task_brief": task.source.task_brief,
        "primary_bucket": task.primary_bucket,
        "method_family": task.method_family,
        "target_family": task.target_family,
        "quality_score": task.quality_score,
        "feasibility_score": task.feasibility_score,
        "specificity_score": task.specificity_score,
        "evaluability_score": task.evaluability_score,
        "practicality_score": task.practicality_score,
        "novelty_score": task.novelty_score,
        "source_task_linkage_mode": SOURCE_TASK_LINKAGE_MODE,
        "source_task_eligibility": SOURCE_TASK_ELIGIBILITY,
        "review_status": REVIEW_STATUS,
        "reference_answer": task.reference_answer,
        "generated_tasks_path": relative_path(task.source.tasks_path),
        "generation_response_path": relative_path(task.source.response_path),
    }


def build_rejected_row(task: ReviewedTask) -> dict[str, Any]:
    reasons = task.rejection_reasons or task.hard_reject_reasons or ["unspecified_rejection"]
    return {
        "paper_id": task.paper_id,
        "paper_title": task.paper_title,
        "task_idx": task.task_idx,
        "task_brief": task.source.task_brief,
        "primary_bucket": task.primary_bucket,
        "method_family": task.method_family,
        "target_family": task.target_family,
        "quality_score": task.quality_score,
        "feasibility_score": task.feasibility_score,
        "specificity_score": task.specificity_score,
        "evaluability_score": task.evaluability_score,
        "practicality_score": task.practicality_score,
        "novelty_score": task.novelty_score,
        "hard_reject_reasons": "|".join(task.hard_reject_reasons),
        "rejection_reasons": "|".join(reasons),
        "generated_tasks_path": relative_path(task.source.tasks_path),
        "generation_response_path": relative_path(task.source.response_path),
    }


def reset_output_files(output_root: Path) -> None:
    expected_root = output_root / "expected"
    if expected_root.exists():
        shutil.rmtree(expected_root)


def build_subset_manifest(
    *,
    raw_markdown_ids: list[int],
    parseable_ids: list[int],
    missing_ids: list[int],
    reviewed_tasks: list[ReviewedTask],
    selected_tasks: list[ReviewedTask],
    train_tasks: list[ReviewedTask],
    test_tasks: list[ReviewedTask],
    bucket_quotas: dict[str, int],
    duplicate_guard_relaxed: bool,
    warnings: list[str],
    allow_incomplete: bool,
    min_quality_score: int,
    duplicate_threshold: float,
) -> dict[str, Any]:
    selected_bucket_counts = Counter(task.primary_bucket for task in selected_tasks)
    reviewed_bucket_counts = Counter(task.primary_bucket for task in reviewed_tasks if not task.hard_reject_reasons)
    return {
        "dataset": "ehrflowbench",
        "schema_version": TASK_MANIFEST_VERSION,
        "selection_unit": "paper_grouped_single_task",
        "reference_answer_field": "reference_answer",
        "reference_answer_contract": "Rows point to expected/<qid>/task_manifest.json.",
        "qid_policy": "Global qids are assigned once across train then test and reused in every manifest.",
        "split_counts": {
            "train": len(train_tasks),
            "test": len(test_tasks),
        },
        "selection_policy": {
            "min_quality_score": min_quality_score,
            "duplicate_threshold": duplicate_threshold,
            "allow_incomplete": allow_incomplete,
            "quality_axes": ["feasibility", "specificity", "evaluability", "practicality", "novelty"],
        },
        "source_pool": {
            "raw_markdown_paper_count": len(raw_markdown_ids),
            "completed_bundle_count": len(parseable_ids),
            "missing_paper_ids": missing_ids,
            "freeze_ready": len(missing_ids) == 0,
            "warnings": warnings,
        },
        "selection_summary": {
            "reviewed_task_count": len(reviewed_tasks),
            "eligible_paper_count": len({task.paper_id for task in reviewed_tasks if not task.hard_reject_reasons and task.quality_score >= min_quality_score}),
            "selected_task_count": len(selected_tasks),
            "duplicate_guard_relaxed": duplicate_guard_relaxed,
            "bucket_quotas": bucket_quotas,
            "reviewed_bucket_counts": {bucket: reviewed_bucket_counts.get(bucket, 0) for bucket in BUCKET_PRIORITY if reviewed_bucket_counts.get(bucket, 0) > 0},
            "selected_bucket_counts": {bucket: selected_bucket_counts.get(bucket, 0) for bucket in BUCKET_PRIORITY if selected_bucket_counts.get(bucket, 0) > 0},
        },
        "train_qids": [task.qid for task in train_tasks],
        "test_qids": [task.qid for task in test_tasks],
        "selected_rows": [
            {
                "qid": task.qid,
                "split": task.split,
                "paper_id": task.paper_id,
                "paper_title": task.paper_title,
                "source_task_idx": task.task_idx,
                "primary_bucket": task.primary_bucket,
                "quality_score": task.quality_score,
                "reference_answer": task.reference_answer,
            }
            for task in selected_tasks
        ],
    }


def run_curation(
    *,
    train_size: int,
    test_size: int,
    min_quality_score: int,
    duplicate_threshold: float,
    allow_incomplete: bool,
    output_root: Path,
) -> dict[str, Any]:
    markdown_dirs = discover_markdown_dirs(MARKDOWN_ROOT)
    raw_markdown_ids = sorted(markdown_dirs)
    paper_titles = load_paper_titles(PAPER_TITLES_PATH)
    candidates, parseable_ids, warnings = collect_generated_candidates(
        GENERATED_TASKS_ROOT,
        paper_titles,
        markdown_dirs,
        allow_incomplete=allow_incomplete,
    )

    missing_ids = sorted(set(raw_markdown_ids) - set(parseable_ids))
    if missing_ids and not allow_incomplete:
        raise ValueError(
            "Refusing to curate an incomplete source pool. Missing completed bundles for papers: "
            + ", ".join(str(value) for value in missing_ids)
        )

    reviewed_tasks = [review_candidate(candidate) for candidate in candidates]
    assign_novelty_scores(reviewed_tasks)

    paper_best_tasks, initial_rejections = select_best_task_per_paper(reviewed_tasks, min_quality_score)
    selected_tasks, bucket_quotas, quota_rejections, duplicate_guard_relaxed = select_canonical_tasks(
        paper_best_tasks,
        train_size + test_size,
        duplicate_threshold,
    )
    train_tasks, test_tasks = assign_splits(selected_tasks, train_size, test_size)
    assign_qids(train_tasks, test_tasks)

    rejection_lookup = defaultdict(list)
    for key, reasons in initial_rejections.items():
        rejection_lookup[key].extend(reasons)
    for key, reasons in quota_rejections.items():
        rejection_lookup[key].extend(reasons)

    selected_lookup = {task.key for task in train_tasks + test_tasks}
    for task in reviewed_tasks:
        if task.key in selected_lookup:
            continue
        task.rejection_reasons = sorted(set(rejection_lookup.get(task.key) or task.hard_reject_reasons or ["not_selected"]))

    selected_tasks_ordered = train_tasks + test_tasks
    dataset_rows_train = [build_dataset_row(task) for task in train_tasks]
    dataset_rows_test = [build_dataset_row(task) for task in test_tasks]
    manifest_rows_train = [build_manifest_index_row(task) for task in train_tasks]
    manifest_rows_test = [build_manifest_index_row(task) for task in test_tasks]
    paper_map_rows = [build_paper_map_row(task) for task in selected_tasks_ordered]
    rejected_rows = [build_rejected_row(task) for task in reviewed_tasks if task.key not in selected_lookup]

    output_root.mkdir(parents=True, exist_ok=True)
    reset_output_files(output_root)
    write_jsonl(output_root / "train.jsonl", dataset_rows_train)
    write_jsonl(output_root / "test.jsonl", dataset_rows_test)
    write_jsonl(output_root / "task_manifest_train.jsonl", manifest_rows_train)
    write_jsonl(output_root / "task_manifest_test.jsonl", manifest_rows_test)
    write_csv(output_root / "paper_map.csv", paper_map_rows)
    write_csv(output_root / "rejected_tasks.csv", rejected_rows)

    for task in selected_tasks_ordered:
        expected_dir = output_root / "expected" / str(task.qid)
        expected_dir.mkdir(parents=True, exist_ok=True)
        write_json(expected_dir / "task_manifest.json", build_task_manifest(task))

    subset_manifest = build_subset_manifest(
        raw_markdown_ids=raw_markdown_ids,
        parseable_ids=sorted(parseable_ids),
        missing_ids=missing_ids,
        reviewed_tasks=reviewed_tasks,
        selected_tasks=selected_tasks_ordered,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        bucket_quotas=bucket_quotas,
        duplicate_guard_relaxed=duplicate_guard_relaxed,
        warnings=warnings,
        allow_incomplete=allow_incomplete,
        min_quality_score=min_quality_score,
        duplicate_threshold=duplicate_threshold,
    )
    write_json(output_root / "subset_manifest.json", subset_manifest)

    return {
        "raw_markdown_paper_count": len(raw_markdown_ids),
        "completed_bundle_count": len(parseable_ids),
        "missing_paper_ids": missing_ids,
        "selected_count": len(selected_tasks_ordered),
        "train_count": len(train_tasks),
        "test_count": len(test_tasks),
        "bucket_quotas": bucket_quotas,
        "duplicate_guard_relaxed": duplicate_guard_relaxed,
        "output_root": relative_path(output_root),
    }


def main() -> None:
    args = parse_args()
    summary = run_curation(
        train_size=args.train_size,
        test_size=args.test_size,
        min_quality_score=args.min_quality_score,
        duplicate_threshold=args.duplicate_threshold,
        allow_incomplete=args.allow_incomplete,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
