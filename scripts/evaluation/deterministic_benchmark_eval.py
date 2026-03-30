from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def find_workspace_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "data" / "TJH.csv").exists():
            return parent
    raise FileNotFoundError("Could not locate workspace root containing data/TJH.csv")


WORKSPACE_ROOT = find_workspace_root()


def parse_qid_range(spec: str | None) -> set[str] | None:
    if not spec:
        return None
    selected: set[str] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start, end = int(start_text), int(end_text)
            for value in range(start, end + 1):
                selected.add(str(value))
        else:
            selected.add(str(int(chunk)))
    return selected


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_benchmark_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["qid"])] = row
    return rows


def compare_json(expected: Any, actual: Any, tolerance: float) -> tuple[bool, str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"expected dict, found {type(actual).__name__}"
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return False, f"key mismatch: expected {sorted(expected_keys)}, found {sorted(actual_keys)}"
        for key in sorted(expected):
            matched, details = compare_json(expected[key], actual[key], tolerance)
            if not matched:
                return False, f"{key}: {details}"
        return True, "json matches"
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False, f"expected list, found {type(actual).__name__}"
        if len(expected) != len(actual):
            return False, f"length mismatch: expected {len(expected)}, found {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            matched, details = compare_json(expected_item, actual_item, tolerance)
            if not matched:
                return False, f"index {index}: {details}"
        return True, "json matches"
    if expected is None or actual is None:
        return (expected is None and actual is None), f"expected {expected!r}, found {actual!r}"
    if isinstance(expected, bool) or isinstance(actual, bool):
        return (expected == actual), f"expected {expected!r}, found {actual!r}"
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if abs(float(expected) - float(actual)) <= tolerance:
            return True, "numeric values within tolerance"
        return False, f"expected {expected}, found {actual}"
    if str(expected) == str(actual):
        return True, "string values match"
    return False, f"expected {expected!r}, found {actual!r}"


def normalized_csv(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    normalized = normalized.loc[:, columns]
    normalized = normalized.fillna("")
    sort_keys = list(columns)
    string_keys = normalized.astype(str)
    order = string_keys.sort_values(by=sort_keys).index
    normalized = normalized.loc[order].reset_index(drop=True)
    return normalized


def compare_csv(expected_path: Path, actual_path: Path, tolerance: float) -> tuple[bool, str]:
    expected = pd.read_csv(expected_path)
    actual = pd.read_csv(actual_path)
    if list(expected.columns) != list(actual.columns):
        return False, f"column mismatch: expected {list(expected.columns)}, found {list(actual.columns)}"
    expected_norm = normalized_csv(expected, list(expected.columns))
    actual_norm = normalized_csv(actual, list(actual.columns))
    if expected_norm.shape != actual_norm.shape:
        return False, f"shape mismatch: expected {expected_norm.shape}, found {actual_norm.shape}"
    for column in expected_norm.columns:
        expected_series = expected_norm[column]
        actual_series = actual_norm[column]
        expected_numeric = pd.to_numeric(expected_series, errors="coerce")
        actual_numeric = pd.to_numeric(actual_series, errors="coerce")
        numeric_mask = expected_numeric.notna() & actual_numeric.notna()
        if numeric_mask.any():
            if ((expected_numeric[numeric_mask] - actual_numeric[numeric_mask]).abs() > tolerance).any():
                return False, f"numeric mismatch in column {column}"
        text_mask = ~numeric_mask
        if text_mask.any():
            if (expected_series[text_mask].astype(str) != actual_series[text_mask].astype(str)).any():
                return False, f"text mismatch in column {column}"
    return True, "csv matches"


def compare_file(expected_path: Path, actual_path: Path, spec: dict[str, Any]) -> tuple[bool, str]:
    compare_mode = spec.get("compare", "exists")
    tolerance = float(spec.get("float_tolerance", 1e-6))
    if not actual_path.exists():
        return False, "missing file"
    if compare_mode == "exists":
        return True, "file exists"
    if not expected_path.exists():
        return False, f"missing ground truth file: {expected_path}"
    if compare_mode == "json":
        return compare_json(load_json(expected_path), load_json(actual_path), tolerance)
    if compare_mode == "csv":
        return compare_csv(expected_path, actual_path, tolerance)
    if compare_mode == "text":
        expected_text = expected_path.read_text(encoding="utf-8")
        actual_text = actual_path.read_text(encoding="utf-8")
        return (expected_text == actual_text), ("text matches" if expected_text == actual_text else "text differs")
    return False, f"unsupported compare mode: {compare_mode}"


def resolve_metadata(result_data: dict[str, Any], benchmark_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    benchmark_row = benchmark_rows.get(str(result_data.get("qid", "")), {})
    merged = benchmark_row.copy()
    merged.update(result_data)
    return merged


def evaluate_single_task(
    qid_path: Path,
    result_data: dict[str, Any],
    benchmark_rows: dict[str, dict[str, Any]],
    answer_path: Path | None,
) -> dict[str, Any]:
    merged = resolve_metadata(result_data, benchmark_rows)
    verification_spec = merged.get("verification_spec", {}) or {}
    file_specs = verification_spec.get("files", {})
    required_files = merged.get("required_files", list(file_specs))
    if not required_files:
        raise ValueError(f"No required files found for QID {merged.get('qid')}")

    ground_truth_ref = merged.get("ground_truth_ref")
    if ground_truth_ref:
        ground_truth_dir = WORKSPACE_ROOT / ground_truth_ref
    elif answer_path is not None:
        ground_truth_dir = answer_path / str(merged["qid"])
    else:
        raise ValueError(f"No ground truth reference for QID {merged.get('qid')}")

    file_results: list[dict[str, Any]] = []
    passed_files = 0
    for relative_name in required_files:
        expected_path = ground_truth_dir / relative_name
        actual_path = qid_path / relative_name
        spec = file_specs.get(relative_name, {"compare": "exists"})
        matched, details = compare_file(expected_path, actual_path, spec)
        passed_files += int(matched)
        file_results.append(
            {
                "file": relative_name,
                "matched": matched,
                "details": details,
                "expected_path": str(expected_path),
                "actual_path": str(actual_path),
            }
        )

    pass_rate = passed_files / len(required_files)
    return {
        "qid": str(merged["qid"]),
        "dataset": merged.get("dataset"),
        "task_family": merged.get("task_family"),
        "task_type": merged.get("task_type"),
        "ground_truth_ref": str(ground_truth_dir),
        "required_files": required_files,
        "matched_files": passed_files,
        "required_file_count": len(required_files),
        "pass_rate": pass_rate,
        "passed": passed_files == len(required_files),
        "file_results": file_results,
    }


def iter_qid_dirs(dataset_path: Path, selected_qids: set[str] | None) -> list[Path]:
    qid_dirs = [path for path in dataset_path.iterdir() if path.is_dir() and (path / "result.json").exists()]
    qid_dirs.sort(key=lambda path: int(path.name) if path.name.isdigit() else path.name)
    if selected_qids is None:
        return qid_dirs
    return [path for path in qid_dirs if path.name in selected_qids]


def run_cli(default_benchmark_file: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run deterministic evaluation for benchmark task outputs.")
    parser.add_argument("--config-file", type=str, default=None, help="Unused legacy argument for compatibility.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to per-QID output directories.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save evaluation JSON outputs.")
    parser.add_argument("--benchmark-file", type=Path, default=Path(default_benchmark_file) if default_benchmark_file else None)
    parser.add_argument("--answer-path", type=Path, default=None, help="Optional override for the ground-truth directory root.")
    parser.add_argument("--qid-range", type=str, default=None, help="Optional QID filter like 1-10,15,18.")
    args = parser.parse_args()

    selected_qids = parse_qid_range(args.qid_range)
    benchmark_rows = load_benchmark_rows(args.benchmark_file)
    qid_dirs = iter_qid_dirs(args.dataset_path, selected_qids)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for qid_dir in qid_dirs:
        result_data = load_json(qid_dir / "result.json")
        evaluation = evaluate_single_task(qid_dir, result_data, benchmark_rows, args.answer_path)
        qid_output_dir = args.output_dir / qid_dir.name
        qid_output_dir.mkdir(parents=True, exist_ok=True)
        with (qid_output_dir / "deterministic_eval.json").open("w", encoding="utf-8") as handle:
            json.dump(evaluation, handle, indent=2, ensure_ascii=False)
        results.append(evaluation)

    summary = {
        "total_tasks": len(results),
        "passed_tasks": sum(1 for item in results if item["passed"]),
        "task_pass_rate": sum(1 for item in results if item["passed"]) / len(results) if results else 0.0,
        "average_file_pass_rate": sum(item["pass_rate"] for item in results) / len(results) if results else 0.0,
    }
    with (args.output_dir / "_final_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_cli()
