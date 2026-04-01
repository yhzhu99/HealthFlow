from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import random
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


DEFAULT_JUDGE_LLM = "openai/gpt-5.4"
DEFAULT_PASS_THRESHOLD = 7.0
CSV_SAMPLE_ROWS = 8
TEXT_CHAR_LIMIT = 12000
JSON_CHAR_LIMIT = 12000


@dataclass(frozen=True)
class JudgeConfig:
    llm_key: str
    api_key: str
    base_url: str | None
    model_name: str


@dataclass(frozen=True)
class RequiredOutput:
    file_name: str
    reference_path: str
    media_type: str


@dataclass(frozen=True)
class SubmissionArtifact:
    file_name: str
    submission_path: Path
    reference_path: Path
    media_type: str


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def default_output_path(submission_root: Path) -> Path:
    return submission_root / "medagentboard.eval.json"


def guess_media_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix == ".txt":
        return "text"
    if suffix == ".parquet":
        return "parquet"
    return "binary"


def load_manifest_outputs(
    benchmark_root: Path,
    row: dict[str, Any],
) -> tuple[dict[str, Any], list[RequiredOutput]]:
    manifest_path = benchmark_root / row["reference_answer"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "required_outputs" in manifest:
        required = [
            RequiredOutput(
                file_name=item["file_name"],
                reference_path=item["reference_path"],
                media_type=item.get("media_type") or guess_media_type(item["file_name"]),
            )
            for item in manifest["required_outputs"]
        ]
        return manifest, required

    required = []
    for path_text in manifest.get("primary_outputs", []):
        file_name = Path(path_text).name
        required.append(
            RequiredOutput(
                file_name=file_name,
                reference_path=path_text,
                media_type=guess_media_type(file_name),
            )
        )
    return manifest, required


def resolve_judge_config(config_path: Path, judge_llm: str | None) -> JudgeConfig:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    llm_table = config.get("llm")
    if not isinstance(llm_table, dict) or not llm_table:
        raise ValueError(f"missing [llm] table in {config_path}")

    candidate_key = judge_llm or DEFAULT_JUDGE_LLM
    if candidate_key not in llm_table:
        available = ", ".join(sorted(llm_table))
        raise ValueError(f"unknown judge llm {candidate_key!r}; available: {available}")

    raw = llm_table[candidate_key]
    if not isinstance(raw, dict):
        raise ValueError(f"invalid llm entry for {candidate_key!r}")

    api_key = raw.get("api_key")
    api_key_env = raw.get("api_key_env")
    if api_key is None and api_key_env:
        api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(
            f"LLM {candidate_key!r} requires a resolved API key. "
            f"Check config.toml or export {api_key_env!r}."
        )

    model_name = raw.get("model_name") or candidate_key
    return JudgeConfig(
        llm_key=candidate_key,
        api_key=api_key,
        base_url=raw.get("base_url"),
        model_name=model_name,
    )


def build_artifacts(
    benchmark_root: Path,
    submission_root: Path,
    required_outputs: list[RequiredOutput],
    qid: int,
) -> list[SubmissionArtifact]:
    artifacts: list[SubmissionArtifact] = []
    for item in required_outputs:
        artifacts.append(
            SubmissionArtifact(
                file_name=item.file_name,
                submission_path=submission_root / str(qid) / item.file_name,
                reference_path=benchmark_root / item.reference_path,
                media_type=item.media_type,
            )
        )
    return artifacts


def summarize_csv(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    payload: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
    }
    if frame.empty:
        payload["sample_rows"] = []
        return payload

    sample_parts = [frame.head(CSV_SAMPLE_ROWS // 2)]
    if len(frame) > CSV_SAMPLE_ROWS // 2:
        sample_parts.append(frame.tail(CSV_SAMPLE_ROWS - len(sample_parts[0])))
    sample = pd.concat(sample_parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
    payload["sample_rows"] = json.loads(sample.to_json(orient="records", force_ascii=False))

    numeric = frame.select_dtypes(include=["number"])
    if not numeric.empty:
        summary: dict[str, Any] = {}
        for column in numeric.columns[:12]:
            series = numeric[column].dropna()
            if series.empty:
                continue
            summary[str(column)] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
            }
        if summary:
            payload["numeric_summary"] = summary
    return payload


def summarize_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) <= JSON_CHAR_LIMIT:
        return {"content": payload}
    return {
        "summary": text[:JSON_CHAR_LIMIT] + "\n...[truncated]",
        "top_level_type": type(payload).__name__,
    }


def summarize_text(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= TEXT_CHAR_LIMIT:
        return {"content": text}
    return {"content": text[:TEXT_CHAR_LIMIT] + "\n...[truncated]"}


def summarize_parquet(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    payload: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns[:32]],
    }
    if not frame.empty:
        sample = frame.head(CSV_SAMPLE_ROWS).copy()
        payload["sample_rows"] = json.loads(sample.to_json(orient="records", force_ascii=False))
    return payload


def summarize_binary(path: Path) -> dict[str, Any]:
    return {"size_bytes": path.stat().st_size}


def summarize_file(path: Path, media_type: str) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True}

    if media_type == "csv":
        return summarize_csv(path)
    if media_type == "json":
        return summarize_json(path)
    if media_type == "text":
        return summarize_text(path)
    if media_type == "parquet":
        return summarize_parquet(path)
    return summarize_binary(path)


def encode_image_as_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def load_prompt(prompt_root: Path, task_type: str) -> str:
    prompt_path = prompt_root / f"{task_type}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"missing prompt template: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def build_text_context(
    row: dict[str, Any],
    manifest: dict[str, Any],
    artifacts: list[SubmissionArtifact],
    pass_threshold: float,
) -> str:
    payload = {
        "evaluation_goal": "Score whether the submission solves the benchmark task correctly.",
        "score_scale": {"min": 0, "max": 10, "pass_threshold": pass_threshold},
        "qid": row["qid"],
        "dataset": row["dataset"],
        "task_type": row["task_type"],
        "task_brief": row["task_brief"],
        "task": row["task"],
        "manifest": {
            "contract_version": manifest.get("contract_version", "legacy"),
            "origin": manifest.get("origin"),
            "protocol_notes": manifest.get("protocol_notes", []),
        },
        "required_outputs": [],
    }
    for artifact in artifacts:
        payload["required_outputs"].append(
            {
                "file_name": artifact.file_name,
                "media_type": artifact.media_type,
                "submission_exists": artifact.submission_path.exists(),
                "submission_summary": summarize_file(artifact.submission_path, artifact.media_type),
                "reference_summary": summarize_file(artifact.reference_path, artifact.media_type),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_messages(
    prompt_text: str,
    row: dict[str, Any],
    manifest: dict[str, Any],
    artifacts: list[SubmissionArtifact],
    pass_threshold: float,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_text_context(row, manifest, artifacts, pass_threshold),
        }
    ]
    for artifact in artifacts:
        if artifact.media_type != "image":
            continue
        if artifact.reference_path.exists():
            content.append({"type": "text", "text": f"Reference image: {artifact.file_name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_as_data_url(artifact.reference_path)},
                }
            )
        if artifact.submission_path.exists():
            content.append({"type": "text", "text": f"Submission image: {artifact.file_name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_as_data_url(artifact.submission_path)},
                }
            )

    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": content},
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"judge response does not contain a JSON object: {text[:400]!r}")


def normalize_judge_payload(
    payload: dict[str, Any],
    *,
    qid: int,
    task_type: str,
    dataset: str,
    pass_threshold: float,
    judge_model: str,
) -> dict[str, Any]:
    score_value = payload.get("score", 0)
    try:
        score = float(score_value)
    except (TypeError, ValueError):
        score = 0.0
    if math.isnan(score):
        score = 0.0
    score = max(0.0, min(10.0, score))
    passed = payload.get("passed")
    if not isinstance(passed, bool):
        passed = score >= pass_threshold
    file_level_notes = payload.get("file_level_notes", [])
    if not isinstance(file_level_notes, list):
        file_level_notes = []
    return {
        "qid": qid,
        "task_type": task_type,
        "dataset": dataset,
        "score": score,
        "passed": passed,
        "summary": str(payload.get("summary", "")).strip(),
        "reason": str(payload.get("reason", "")).strip(),
        "file_level_notes": file_level_notes,
        "judge_model": judge_model,
        "raw_judge_payload": payload,
    }


def call_judge(
    client: OpenAI,
    judge_config: JudgeConfig,
    prompt_text: str,
    row: dict[str, Any],
    manifest: dict[str, Any],
    artifacts: list[SubmissionArtifact],
    pass_threshold: float,
) -> dict[str, Any]:
    messages = build_messages(prompt_text, row, manifest, artifacts, pass_threshold)
    completion = client.chat.completions.create(
        model=judge_config.model_name,
        messages=messages,
        temperature=0,
    )
    message = completion.choices[0].message
    content = message.content or ""
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    payload = extract_json_object(str(content))
    return normalize_judge_payload(
        payload,
        qid=int(row["qid"]),
        task_type=row["task_type"],
        dataset=row["dataset"],
        pass_threshold=pass_threshold,
        judge_model=judge_config.model_name,
    )


def evaluate_predictions(
    benchmark_file: Path,
    submission_root: Path,
    *,
    config_path: Path,
    judge_llm: str | None,
    prompt_root: Path,
    pass_threshold: float,
    qids: set[int] | None = None,
) -> dict[str, Any]:
    benchmark_rows = load_jsonl(benchmark_file)
    benchmark_root = benchmark_file.parent
    judge_config = resolve_judge_config(config_path, judge_llm)
    client = OpenAI(api_key=judge_config.api_key, base_url=judge_config.base_url)

    results: list[dict[str, Any]] = []
    for row in benchmark_rows:
        qid = int(row["qid"])
        if qids and qid not in qids:
            continue
        manifest, required_outputs = load_manifest_outputs(benchmark_root, row)
        prompt_text = load_prompt(prompt_root, manifest.get("judge_prompt_type", row["task_type"]))
        artifacts = build_artifacts(benchmark_root, submission_root, required_outputs, qid)
        results.append(
            call_judge(
                client,
                judge_config,
                prompt_text,
                row,
                manifest,
                artifacts,
                pass_threshold,
            )
        )

    total_questions = len(results)
    passed_questions = sum(int(item["passed"]) for item in results)
    average_score = sum(item["score"] for item in results) / total_questions if total_questions else 0.0
    return {
        "benchmark_file": str(benchmark_file),
        "submission_root": str(submission_root),
        "config_file": str(config_path),
        "judge_llm": judge_config.llm_key,
        "judge_model": judge_config.model_name,
        "score_scale": {"min": 0, "max": 10, "pass_threshold": pass_threshold},
        "total_questions": total_questions,
        "passed_questions": passed_questions,
        "average_score": average_score,
        "results": results,
    }


def run_cli(
    *,
    default_benchmark_file: str | None = None,
    default_prompt_root: str | None = None,
) -> None:
    parser = argparse.ArgumentParser(description="Evaluate MedAgentBoard submissions with an LLM judge.")
    parser.add_argument("--submission-root", type=Path, required=True, help="Directory laid out as <root>/<qid>/<files>.")
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=Path(default_benchmark_file) if default_benchmark_file else None,
        help="Optional override for the reference benchmark jsonl.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("config.toml"),
        help="Path to the repo config.toml file with [llm] entries.",
    )
    parser.add_argument(
        "--judge-llm",
        type=str,
        default=None,
        help=f"Judge LLM key from config.toml. Defaults to {DEFAULT_JUDGE_LLM!r}.",
    )
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=Path(default_prompt_root) if default_prompt_root else None,
        help="Directory containing task-type-specific judge prompt markdown files.",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=DEFAULT_PASS_THRESHOLD,
        help="Pass threshold on the 0-10 score scale.",
    )
    parser.add_argument(
        "--qid",
        type=int,
        action="append",
        default=None,
        help="Optional repeated qid filter for targeted judging.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output json path. Defaults to <submission-root>/medagentboard.eval.json.",
    )
    args = parser.parse_args()

    if args.benchmark_file is None:
        raise ValueError("--benchmark-file is required when no default benchmark file is configured")
    if args.prompt_root is None:
        raise ValueError("--prompt-root is required when no default prompt root is configured")

    output_file = args.output_file or default_output_path(args.submission_root)
    payload = evaluate_predictions(
        args.benchmark_file,
        args.submission_root,
        config_path=args.config_file,
        judge_llm=args.judge_llm,
        prompt_root=args.prompt_root,
        pass_threshold=args.pass_threshold,
        qids=set(args.qid or []),
    )
    write_json(output_file, payload)

    print(
        json.dumps(
            {
                "total_questions": payload["total_questions"],
                "passed_questions": payload["passed_questions"],
                "average_score": payload["average_score"],
                "output_file": str(output_file),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
