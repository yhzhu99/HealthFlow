from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from data.ehrflowbench.scripts import generate_tasks
from data.ehrflowbench.scripts.curate_generated_tasks import BUCKET_PRIORITY
from data.ehrflowbench.scripts.curate_generated_tasks import DATASET_DISPLAY_NAMES
from data.ehrflowbench.scripts.curate_generated_tasks import FALLBACK_BUCKET_PRIORITY
from data.ehrflowbench.scripts.curate_generated_tasks import MARKDOWN_ROOT
from data.ehrflowbench.scripts.curate_generated_tasks import PAPER_TITLES_PATH
from data.ehrflowbench.scripts.curate_generated_tasks import TEXT_BUCKET_PATTERNS
from data.ehrflowbench.scripts.curate_generated_tasks import TITLE_BUCKET_PATTERNS
from data.ehrflowbench.scripts.curate_generated_tasks import classify_family
from data.ehrflowbench.scripts.curate_generated_tasks import classify_method_family
from data.ehrflowbench.scripts.curate_generated_tasks import classify_primary_bucket
from data.ehrflowbench.scripts.curate_generated_tasks import classify_required_input_set
from data.ehrflowbench.scripts.curate_generated_tasks import classify_target_family
from data.ehrflowbench.scripts.curate_generated_tasks import contains_any
from data.ehrflowbench.scripts.curate_generated_tasks import discover_markdown_dirs
from data.ehrflowbench.scripts.curate_generated_tasks import load_paper_titles
from data.ehrflowbench.scripts.curate_generated_tasks import looks_like_figure_or_table
from data.ehrflowbench.scripts.curate_generated_tasks import looks_like_numeric_artifact
from data.ehrflowbench.scripts.curate_generated_tasks import normalize_text
from data.ehrflowbench.scripts.curate_generated_tasks import paper_id_from_name
from data.ehrflowbench.scripts.curate_generated_tasks import relative_path
from data.ehrflowbench.scripts.curate_generated_tasks import title_from_dir_name


PROMPT_TASK_CANDIDATE_PLACEHOLDER = "{{TASK_CANDIDATE_JSON}}"
PROMPT_DERIVED_CONTEXT_PLACEHOLDER = "{{DERIVED_CONTEXT_JSON}}"
PROMPT_PATH = generate_tasks.DATASET_ROOT / "scripts" / "prompt_score_ehrflowbench.md"
DEFAULT_INPUT_ROOT = generate_tasks.DATASET_ROOT / "processed" / "papers" / "generated_tasks"
DEFAULT_OUTPUT_ROOT = generate_tasks.DATASET_ROOT / "processed" / "papers" / "task_scores"
RAW_RESPONSE_DIRNAME = "raw_responses"
PROMPT_OUTPUT_DIRNAME = "prompts"
DEFAULT_RANK_STATUS = "eligible"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_THINKING_MIN_BUDGET_TOKENS = 1024
ANTHROPIC_THINKING_MIN_OUTPUT_HEADROOM = 256
ANTHROPIC_THINKING_EFFORT_RATIOS = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
}


@dataclass(frozen=True)
class TaskScoreCandidate:
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

    @property
    def key(self) -> tuple[int, int]:
        return (self.paper_id, self.task_idx)


@dataclass(frozen=True)
class ScoringAPIResponse:
    id: str | None
    status: str | None
    error: Any
    incomplete_details: Any
    usage: Any
    output_text: str | None
    output: Any


class AnthropicMessagesRequestError(Exception):
    def __init__(self, cause: Exception, attempts: int):
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts


class TaskScoreBreakdown(BaseModel):
    feasibility: int = Field(ge=0, le=5)
    specificity: int = Field(ge=0, le=5)
    evaluability: int = Field(ge=0, le=5)
    practicality: int = Field(ge=0, le=5)
    novelty: int = Field(ge=0, le=5)


class LLMTaskScorePayload(BaseModel):
    hard_reject: bool
    hard_reject_reasons: list[str] = Field(default_factory=list)
    scores: TaskScoreBreakdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score EHRFlowBench generated tasks via the OpenAI-compatible Responses API.")
    parser.add_argument("--paper-id", type=int, help="Paper id whose <paper_id>_tasks.json should be scored.")
    parser.add_argument("--tasks-path", type=Path, help="Explicit path to a generated tasks JSON file.")
    parser.add_argument("--model-key", default=generate_tasks.DEFAULT_MODEL_KEY, help="LLM key under config.toml.")
    parser.add_argument("--max-output-tokens", type=int, default=generate_tasks.DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-prompt-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", file=sys.stderr)


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: to_jsonable(item) for key, item in vars(value).items()}
    return value


def resolve_tasks_path(input_root: Path, paper_id: int | None, tasks_path: Path | None) -> Path:
    if tasks_path is not None:
        return tasks_path.resolve()
    if paper_id is None:
        raise ValueError("Either --paper-id or --tasks-path must be provided unless --aggregate-only is used")
    return (input_root / f"{paper_id}_tasks.json").resolve()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_task_candidates(tasks_path: Path) -> list[TaskScoreCandidate]:
    if not tasks_path.exists():
        raise FileNotFoundError(f"Generated tasks file does not exist: {tasks_path}")

    payload = read_json(tasks_path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Tasks file does not contain a non-empty tasks list: {tasks_path}")

    paper_id = paper_id_from_name(tasks_path.name)
    markdown_dirs = discover_markdown_dirs(MARKDOWN_ROOT)
    paper_titles = load_paper_titles(PAPER_TITLES_PATH)
    paper_title = paper_titles.get(paper_id) or title_from_dir_name(markdown_dirs.get(paper_id)) or f"Paper {paper_id}"

    candidates: list[TaskScoreCandidate] = []
    for task_idx, item in enumerate(tasks, start=1):
        candidates.append(
            TaskScoreCandidate(
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
                tasks_path=tasks_path,
            )
        )
    return candidates


def build_candidate_prompt_payload(candidate: TaskScoreCandidate) -> dict[str, Any]:
    return {
        "paper_id": candidate.paper_id,
        "task": candidate.task,
        "deliverables": list(candidate.deliverables),
    }


def build_candidate_contexts(candidates: list[TaskScoreCandidate]) -> dict[tuple[int, int], dict[str, Any]]:
    prelim: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in candidates:
        classification_blob = " ".join([candidate.task_brief, " ".join(candidate.focus_areas), candidate.task])
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
        input_dataset, has_valid_single_dataset_inputs, has_split_metadata_inputs, has_mixed_dataset_inputs = classify_required_input_set(
            candidate.required_inputs
        )
        prelim[candidate.key] = {
            "primary_bucket": classify_primary_bucket(candidate.paper_title, classification_blob),
            "method_family": classify_method_family(task_blob),
            "target_family": classify_target_family(task_blob),
            "input_dataset": input_dataset,
            "input_dataset_display_name": DATASET_DISPLAY_NAMES.get(input_dataset, input_dataset) if input_dataset else None,
            "has_valid_single_dataset_inputs": has_valid_single_dataset_inputs,
            "has_split_metadata_inputs": has_split_metadata_inputs,
            "has_mixed_dataset_inputs": has_mixed_dataset_inputs,
            "has_numeric_artifact": any(looks_like_numeric_artifact(path) for path in candidate.deliverables),
            "has_figure_or_table_artifact": any(looks_like_figure_or_table(path) for path in candidate.deliverables),
        }

    bucket_sizes: dict[str, int] = {}
    bucket_method_counts: dict[tuple[str, str], int] = {}
    bucket_target_counts: dict[tuple[str, str], int] = {}
    for context in prelim.values():
        bucket = str(context["primary_bucket"])
        method = str(context["method_family"])
        target = str(context["target_family"])
        bucket_sizes[bucket] = bucket_sizes.get(bucket, 0) + 1
        bucket_method_counts[(bucket, method)] = bucket_method_counts.get((bucket, method), 0) + 1
        bucket_target_counts[(bucket, target)] = bucket_target_counts.get((bucket, target), 0) + 1

    contexts: dict[tuple[int, int], dict[str, Any]] = {}
    for key, context in prelim.items():
        bucket = str(context["primary_bucket"])
        method = str(context["method_family"])
        target = str(context["target_family"])
        bucket_size = bucket_sizes[bucket]
        contexts[key] = {
            **context,
            "bucket_size": bucket_size,
            "bucket_method_count": bucket_method_counts[(bucket, method)],
            "bucket_target_count": bucket_target_counts[(bucket, target)],
            "novelty_rarity_threshold": max(1, math.ceil(bucket_size / 5)),
        }
    return contexts


def build_score_prompt(prompt_body: str, candidate: TaskScoreCandidate, derived_context: dict[str, Any]) -> str:
    prompt = prompt_body.replace(
        PROMPT_TASK_CANDIDATE_PLACEHOLDER,
        json.dumps(build_candidate_prompt_payload(candidate), ensure_ascii=False, indent=2),
    )
    prompt = prompt.replace(
        PROMPT_DERIVED_CONTEXT_PLACEHOLDER,
        json.dumps(derived_context, ensure_ascii=False, indent=2),
    )
    return prompt.strip()


def build_scoring_request_input(prompt_text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
            ],
        }
    ]


def parse_score_payload_from_response(response: Any) -> LLMTaskScorePayload:
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("Response output_text was empty")
    payload = generate_tasks.extract_json_payload(output_text)
    return LLMTaskScorePayload.model_validate(payload)


def raw_response_output_path(output_dir: Path, candidate: TaskScoreCandidate) -> Path:
    return output_dir / RAW_RESPONSE_DIRNAME / f"{candidate.paper_id}_task_{candidate.task_idx}.json"


def prompt_output_path(output_dir: Path, candidate: TaskScoreCandidate) -> Path:
    return output_dir / PROMPT_OUTPUT_DIRNAME / f"{candidate.paper_id}_task_{candidate.task_idx}_prompt.md"


def paper_scores_output_path(output_dir: Path, paper_id: int) -> Path:
    return output_dir / f"{paper_id}_scores.json"


def response_index_output_path(output_dir: Path, paper_id: int) -> Path:
    return output_dir / f"{paper_id}_score_response_index.json"


def existing_output_paths(output_dir: Path, paper_id: int, task_count: int, output_prompt_only: bool) -> list[Path]:
    if output_prompt_only:
        candidate_paths = [
            output_dir / PROMPT_OUTPUT_DIRNAME / f"{paper_id}_task_{task_idx}_prompt.md"
            for task_idx in range(1, task_count + 1)
        ]
    else:
        candidate_paths = [
            paper_scores_output_path(output_dir, paper_id),
            response_index_output_path(output_dir, paper_id),
        ]
    return [path for path in candidate_paths if path.exists()]


def response_debug_payload(
    *,
    response: Any,
    llm_config: generate_tasks.LLMConfig,
    candidate: TaskScoreCandidate,
    prompt_text: str,
    derived_context: dict[str, Any],
    max_output_tokens: int,
    reasoning_config: dict[str, Any] | None,
    thinking_config: dict[str, Any] | None,
    request_attempts: int,
    api_name: str,
    fallback_chain: list[dict[str, Any]] | None = None,
    thinking_status: str | None = None,
    thinking_text: str | None = None,
    thinking_blocks: list[dict[str, Any]] | None = None,
    provider_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "response_id": getattr(response, "id", None),
        "model": llm_config.model_name,
        "paper_id": candidate.paper_id,
        "paper_title": candidate.paper_title,
        "task_idx": candidate.task_idx,
        "task_brief": candidate.task_brief,
        "source_paths": {
            "generated_tasks": relative_path(candidate.tasks_path),
        },
        "request": {
            "max_output_tokens": max_output_tokens,
            "api": api_name,
            "reasoning": reasoning_config,
            "thinking": thinking_config,
            "timeout_seconds": generate_tasks.DEFAULT_REQUEST_TIMEOUT_SECONDS,
            "attempts": request_attempts,
            "fallback_chain": fallback_chain or [],
        },
        "prompt_text": prompt_text,
        "derived_context": derived_context,
        "status": getattr(response, "status", None),
        "error": to_jsonable(getattr(response, "error", None)),
        "incomplete_details": to_jsonable(getattr(response, "incomplete_details", None)),
        "usage": to_jsonable(getattr(response, "usage", None)),
        "thinking_status": thinking_status,
        "thinking_text": thinking_text,
        "thinking_blocks": thinking_blocks,
        "output_text": getattr(response, "output_text", None),
        "output": to_jsonable(getattr(response, "output", None)),
        "provider_response": provider_response,
    }


def is_anthropic_model(llm_config: generate_tasks.LLMConfig) -> bool:
    return llm_config.model_name.startswith("anthropic/")


def resolve_anthropic_messages_url(base_url: str) -> str | None:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/api/v1"):
        return f"{normalized[:-len('/api/v1')]}/api/anthropic/v1/messages"
    return None


def build_thinking_config(reasoning_effort: str, max_output_tokens: int) -> dict[str, Any] | None:
    if max_output_tokens < ANTHROPIC_THINKING_MIN_BUDGET_TOKENS + ANTHROPIC_THINKING_MIN_OUTPUT_HEADROOM:
        return None

    effort_ratio = ANTHROPIC_THINKING_EFFORT_RATIOS.get(reasoning_effort, ANTHROPIC_THINKING_EFFORT_RATIOS["medium"])
    max_budget = max_output_tokens - ANTHROPIC_THINKING_MIN_OUTPUT_HEADROOM
    budget_tokens = int(max_output_tokens * effort_ratio)
    budget_tokens = max(ANTHROPIC_THINKING_MIN_BUDGET_TOKENS, min(budget_tokens, max_budget))
    return {"type": "enabled", "budget_tokens": budget_tokens}


def join_nonempty_text(parts: list[str]) -> str | None:
    cleaned = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    if not cleaned:
        return None
    return "\n\n".join(cleaned)


def normalize_anthropic_response(
    payload: dict[str, Any],
) -> tuple[ScoringAPIResponse, str | None, list[dict[str, Any]], str]:
    content = payload.get("content")
    if not isinstance(content, list):
        content = []

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    thinking_blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type == "text":
            text_value = block.get("text")
            if isinstance(text_value, str):
                text_parts.append(text_value)
            continue
        if block_type == "thinking":
            thinking_blocks.append(block)
            thinking_value = block.get("thinking")
            if isinstance(thinking_value, str):
                thinking_parts.append(thinking_value)

    stop_reason = payload.get("stop_reason")
    incomplete_details = None
    if stop_reason not in (None, "end_turn"):
        incomplete_details = {"reason": stop_reason}

    response = ScoringAPIResponse(
        id=str(payload.get("id")) if payload.get("id") is not None else None,
        status="completed" if payload.get("type") == "message" else str(payload.get("type") or "unknown"),
        error=payload.get("error"),
        incomplete_details=incomplete_details,
        usage=payload.get("usage"),
        output_text=join_nonempty_text(text_parts),
        output=content,
    )
    thinking_text = join_nonempty_text(thinking_parts)
    thinking_status = "captured" if thinking_text else "missing"
    return response, thinking_text, thinking_blocks, thinking_status


def build_anthropic_request_payload(
    llm_config: generate_tasks.LLMConfig,
    prompt_text: str,
    max_output_tokens: int,
    thinking_config: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": llm_config.model_name,
        "max_tokens": max_output_tokens,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
    }
    if thinking_config is not None:
        payload["thinking"] = thinking_config
    return payload


def describe_httpx_error(exc: Exception) -> str:
    if isinstance(exc, AnthropicMessagesRequestError):
        return describe_httpx_error(exc.cause)
    if isinstance(exc, httpx.HTTPStatusError):
        message = str(exc)
        try:
            payload = exc.response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                error_message = error_payload.get("message")
                if isinstance(error_message, str) and error_message.strip():
                    message = error_message.strip()
        return f"HTTP {exc.response.status_code}: {message}"
    return str(exc)


def should_retry_httpx_exception(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in generate_tasks.TRANSIENT_STATUS_CODES
    return False


def should_fallback_without_thinking(exc: Exception) -> bool:
    if isinstance(exc, AnthropicMessagesRequestError):
        exc = exc.cause
    if should_retry_httpx_exception(exc):
        return False
    message = describe_httpx_error(exc).lower()
    if "thinking" not in message and "budget_tokens" not in message:
        return False
    return True


def append_fallback_step(
    fallback_chain: list[dict[str, Any]],
    *,
    api_name: str,
    thinking_config: dict[str, Any] | None,
    outcome: str,
    attempts: int,
    error: str | None = None,
) -> None:
    step = {
        "api": api_name,
        "thinking": thinking_config,
        "outcome": outcome,
        "attempts": attempts,
    }
    if error is not None:
        step["error"] = error
    fallback_chain.append(step)


def call_responses_scoring_api(
    *,
    client: Any,
    llm_config: generate_tasks.LLMConfig,
    candidate: TaskScoreCandidate,
    request_input: list[dict[str, Any]],
    max_output_tokens: int,
    reasoning_config: dict[str, Any] | None,
) -> tuple[Any, int]:
    response = None
    last_exc: Exception | None = None
    for attempt in range(1, generate_tasks.DEFAULT_REQUEST_MAX_ATTEMPTS + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": llm_config.model_name,
                "input": request_input,
                "max_output_tokens": max_output_tokens,
                "store": False,
                "timeout": generate_tasks.DEFAULT_REQUEST_TIMEOUT_SECONDS,
            }
            if reasoning_config is not None:
                request_kwargs["reasoning"] = reasoning_config
            response = client.responses.create(
                **request_kwargs,
            )
            return response, attempt
        except Exception as exc:
            last_exc = exc
            if not generate_tasks.should_retry_api_exception(exc) or attempt >= generate_tasks.DEFAULT_REQUEST_MAX_ATTEMPTS:
                raise
            delay_seconds = generate_tasks.retry_sleep_seconds(attempt)
            log(
                "transient API error for "
                f"paper={candidate.paper_id} task={candidate.task_idx} "
                f"attempt={attempt}/{generate_tasks.DEFAULT_REQUEST_MAX_ATTEMPTS}: {exc}; "
                f"retrying in {delay_seconds:.1f}s"
            )
            time.sleep(delay_seconds)

    assert last_exc is not None
    raise last_exc


def call_anthropic_messages_scoring_api(
    *,
    llm_config: generate_tasks.LLMConfig,
    candidate: TaskScoreCandidate,
    prompt_text: str,
    max_output_tokens: int,
    thinking_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], int]:
    api_key = os.environ.get(llm_config.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Environment variable {llm_config.api_key_env} is not set")

    api_url = resolve_anthropic_messages_url(llm_config.base_url)
    if api_url is None:
        raise ValueError(f"Could not derive Anthropic Messages API URL from base URL: {llm_config.base_url}")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    request_payload = build_anthropic_request_payload(
        llm_config=llm_config,
        prompt_text=prompt_text,
        max_output_tokens=max_output_tokens,
        thinking_config=thinking_config,
    )

    last_exc: Exception | None = None
    for attempt in range(1, generate_tasks.DEFAULT_REQUEST_MAX_ATTEMPTS + 1):
        try:
            with httpx.Client(timeout=generate_tasks.DEFAULT_REQUEST_TIMEOUT_SECONDS) as http_client:
                response = http_client.post(api_url, headers=headers, json=request_payload)
                response.raise_for_status()
                return response.json(), attempt
        except Exception as exc:
            last_exc = exc
            if not should_retry_httpx_exception(exc) or attempt >= generate_tasks.DEFAULT_REQUEST_MAX_ATTEMPTS:
                raise AnthropicMessagesRequestError(exc, attempt) from exc
            delay_seconds = generate_tasks.retry_sleep_seconds(attempt)
            log(
                "transient API error for "
                f"paper={candidate.paper_id} task={candidate.task_idx} "
                f"attempt={attempt}/{generate_tasks.DEFAULT_REQUEST_MAX_ATTEMPTS}: {describe_httpx_error(exc)}; "
                f"retrying in {delay_seconds:.1f}s"
            )
            time.sleep(delay_seconds)

    assert last_exc is not None
    raise AnthropicMessagesRequestError(last_exc, generate_tasks.DEFAULT_REQUEST_MAX_ATTEMPTS) from last_exc


def call_scoring_api(
    *,
    client: Any,
    llm_config: generate_tasks.LLMConfig,
    candidate: TaskScoreCandidate,
    prompt_text: str,
    derived_context: dict[str, Any],
    max_output_tokens: int,
) -> tuple[Any, dict[str, Any]]:
    request_input = build_scoring_request_input(prompt_text)
    fallback_chain: list[dict[str, Any]] = []
    reasoning_config: dict[str, Any] | None = {"effort": llm_config.reasoning_effort}
    thinking_config: dict[str, Any] | None = None
    thinking_status: str | None = None
    thinking_text: str | None = None
    thinking_blocks: list[dict[str, Any]] | None = None
    provider_response: dict[str, Any] | None = None
    api_name = "responses"
    request_attempts = 0

    if is_anthropic_model(llm_config):
        thinking_config = build_thinking_config(llm_config.reasoning_effort, max_output_tokens)
        reasoning_config = None
        messages_response: dict[str, Any] | None = None
        messages_attempts = 0
        allow_messages_without_thinking = True

        if thinking_config is None:
            thinking_status = "disabled_budget_too_small"
            append_fallback_step(
                fallback_chain,
                api_name="anthropic-messages",
                thinking_config=None,
                outcome="thinking_disabled_budget_too_small",
                attempts=0,
            )
        else:
            try:
                messages_response, messages_attempts = call_anthropic_messages_scoring_api(
                    llm_config=llm_config,
                    candidate=candidate,
                    prompt_text=prompt_text,
                    max_output_tokens=max_output_tokens,
                    thinking_config=thinking_config,
                )
            except Exception as exc:
                attempts = exc.attempts if isinstance(exc, AnthropicMessagesRequestError) else 1
                append_fallback_step(
                    fallback_chain,
                    api_name="anthropic-messages",
                    thinking_config=thinking_config,
                    outcome="error",
                    attempts=attempts,
                    error=describe_httpx_error(exc),
                )
                if should_fallback_without_thinking(exc):
                    log(
                        f"paper={candidate.paper_id} task={candidate.task_idx} Anthropic thinking failed; "
                        "retrying without thinking"
                    )
                    thinking_status = "fallback_without_thinking"
                else:
                    allow_messages_without_thinking = False
                    messages_attempts = 0
                    log(
                        f"paper={candidate.paper_id} task={candidate.task_idx} Anthropic Messages API failed; "
                        "falling back to Responses API"
                    )
            else:
                append_fallback_step(
                    fallback_chain,
                    api_name="anthropic-messages",
                    thinking_config=thinking_config,
                    outcome="success",
                    attempts=messages_attempts,
                )

        if messages_response is None and allow_messages_without_thinking:
            try:
                messages_response, messages_attempts_without_thinking = call_anthropic_messages_scoring_api(
                    llm_config=llm_config,
                    candidate=candidate,
                    prompt_text=prompt_text,
                    max_output_tokens=max_output_tokens,
                    thinking_config=None,
                )
            except Exception as exc:
                attempts = exc.attempts if isinstance(exc, AnthropicMessagesRequestError) else 1
                append_fallback_step(
                    fallback_chain,
                    api_name="anthropic-messages",
                    thinking_config=None,
                    outcome="error",
                    attempts=attempts,
                    error=describe_httpx_error(exc),
                )
                log(
                    f"paper={candidate.paper_id} task={candidate.task_idx} Anthropic Messages API fallback failed; "
                    "falling back to Responses API"
                )
            else:
                messages_attempts += messages_attempts_without_thinking
                append_fallback_step(
                    fallback_chain,
                    api_name="anthropic-messages",
                    thinking_config=None,
                    outcome="success",
                    attempts=messages_attempts_without_thinking,
                )
                if thinking_status is None:
                    thinking_status = "disabled_budget_too_small"

        if messages_response is not None:
            api_name = "anthropic-messages"
            request_attempts = sum(int(step.get("attempts", 0)) for step in fallback_chain)
            provider_response = messages_response
            response, thinking_text, extracted_thinking_blocks, extracted_thinking_status = normalize_anthropic_response(
                messages_response
            )
            thinking_blocks = extracted_thinking_blocks
            if thinking_status is None:
                thinking_status = extracted_thinking_status
            sent_thinking_config = None
            if thinking_status not in {"fallback_without_thinking", "disabled_budget_too_small"}:
                sent_thinking_config = thinking_config
            metadata = response_debug_payload(
                response=response,
                llm_config=llm_config,
                candidate=candidate,
                prompt_text=prompt_text,
                derived_context=derived_context,
                max_output_tokens=max_output_tokens,
                reasoning_config=reasoning_config,
                thinking_config=sent_thinking_config,
                request_attempts=request_attempts,
                api_name=api_name,
                fallback_chain=fallback_chain,
                thinking_status=thinking_status,
                thinking_text=thinking_text,
                thinking_blocks=thinking_blocks,
                provider_response=provider_response,
            )
            return response, metadata

        api_name = "responses"
        thinking_status = "fallback_to_responses"
        response, responses_attempts = call_responses_scoring_api(
            client=client,
            llm_config=llm_config,
            candidate=candidate,
            request_input=request_input,
            max_output_tokens=max_output_tokens,
            reasoning_config=None,
        )
        request_attempts = responses_attempts
        append_fallback_step(
            fallback_chain,
            api_name="responses",
            thinking_config=None,
            outcome="success",
            attempts=responses_attempts,
        )
        request_attempts = sum(int(step.get("attempts", 0)) for step in fallback_chain)
        metadata = response_debug_payload(
            response=response,
            llm_config=llm_config,
            candidate=candidate,
            prompt_text=prompt_text,
            derived_context=derived_context,
            max_output_tokens=max_output_tokens,
            reasoning_config=None,
            thinking_config=None,
            request_attempts=request_attempts,
            api_name=api_name,
            fallback_chain=fallback_chain,
            thinking_status=thinking_status,
        )
        return response, metadata

    response, request_attempts = call_responses_scoring_api(
        client=client,
        llm_config=llm_config,
        candidate=candidate,
        request_input=request_input,
        max_output_tokens=max_output_tokens,
        reasoning_config=reasoning_config,
    )
    metadata = response_debug_payload(
        response=response,
        llm_config=llm_config,
        candidate=candidate,
        prompt_text=prompt_text,
        derived_context=derived_context,
        max_output_tokens=max_output_tokens,
        reasoning_config=reasoning_config,
        thinking_config=None,
        request_attempts=request_attempts,
        api_name=api_name,
    )
    return response, metadata


def task_record_from_payload(
    *,
    candidate: TaskScoreCandidate,
    derived_context: dict[str, Any],
    payload: LLMTaskScorePayload,
    raw_response_path: Path,
) -> dict[str, Any]:
    scores = payload.scores.model_dump()
    final_score = sum(int(value) for value in scores.values())
    rank_status = "rejected" if payload.hard_reject else DEFAULT_RANK_STATUS
    return {
        "paper_id": candidate.paper_id,
        "paper_title": candidate.paper_title,
        "task_idx": candidate.task_idx,
        "task_brief": candidate.task_brief,
        "task_type": candidate.task_type,
        "focus_areas": list(candidate.focus_areas),
        "task": candidate.task,
        "required_inputs": list(candidate.required_inputs),
        "deliverables": list(candidate.deliverables),
        "report_requirements": list(candidate.report_requirements),
        "generated_tasks_path": relative_path(candidate.tasks_path),
        "input_dataset": derived_context.get("input_dataset"),
        "primary_bucket": derived_context["primary_bucket"],
        "method_family": derived_context["method_family"],
        "target_family": derived_context["target_family"],
        "bucket_size": derived_context["bucket_size"],
        "bucket_method_count": derived_context["bucket_method_count"],
        "bucket_target_count": derived_context["bucket_target_count"],
        "novelty_rarity_threshold": derived_context["novelty_rarity_threshold"],
        "hard_reject": payload.hard_reject,
        "hard_reject_reasons": payload.hard_reject_reasons,
        "scores": scores,
        "final_score": final_score,
        "rank_status": rank_status,
        "raw_response_path": relative_path(raw_response_path),
        "parse_status": "parsed",
    }


def task_record_for_parse_failure(
    *,
    candidate: TaskScoreCandidate,
    derived_context: dict[str, Any],
    raw_response_path: Path,
    parse_error: Exception,
) -> dict[str, Any]:
    return {
        "paper_id": candidate.paper_id,
        "paper_title": candidate.paper_title,
        "task_idx": candidate.task_idx,
        "task_brief": candidate.task_brief,
        "task_type": candidate.task_type,
        "focus_areas": list(candidate.focus_areas),
        "task": candidate.task,
        "required_inputs": list(candidate.required_inputs),
        "deliverables": list(candidate.deliverables),
        "report_requirements": list(candidate.report_requirements),
        "generated_tasks_path": relative_path(candidate.tasks_path),
        "input_dataset": derived_context.get("input_dataset"),
        "primary_bucket": derived_context["primary_bucket"],
        "method_family": derived_context["method_family"],
        "target_family": derived_context["target_family"],
        "bucket_size": derived_context["bucket_size"],
        "bucket_method_count": derived_context["bucket_method_count"],
        "bucket_target_count": derived_context["bucket_target_count"],
        "novelty_rarity_threshold": derived_context["novelty_rarity_threshold"],
        "hard_reject": None,
        "hard_reject_reasons": [],
        "scores": None,
        "final_score": None,
        "rank_status": "parse_failed",
        "parse_error": str(parse_error),
        "raw_response_path": relative_path(raw_response_path),
        "parse_status": "parse_failed",
    }


def score_single_candidate(
    *,
    client: Any,
    llm_config: generate_tasks.LLMConfig,
    prompt_body: str,
    candidate: TaskScoreCandidate,
    derived_context: dict[str, Any],
    output_dir: Path,
    max_output_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_text = build_score_prompt(prompt_body, candidate, derived_context)
    log(f"paper={candidate.paper_id} task={candidate.task_idx} scoring started")
    response, raw_payload = call_scoring_api(
        client=client,
        llm_config=llm_config,
        candidate=candidate,
        prompt_text=prompt_text,
        derived_context=derived_context,
        max_output_tokens=max_output_tokens,
    )

    raw_path = raw_response_output_path(output_dir, candidate)
    write_json(raw_path, raw_payload)
    log(f"paper={candidate.paper_id} task={candidate.task_idx} raw response saved path={relative_path(raw_path)}")

    try:
        payload = parse_score_payload_from_response(response)
    except Exception as exc:
        log(f"paper={candidate.paper_id} task={candidate.task_idx} parse failed error={exc}")
        record = task_record_for_parse_failure(
            candidate=candidate,
            derived_context=derived_context,
            raw_response_path=raw_path,
            parse_error=exc,
        )
        index_row = {
            "paper_id": candidate.paper_id,
            "task_idx": candidate.task_idx,
            "task_brief": candidate.task_brief,
            "raw_response_path": relative_path(raw_path),
            "parse_status": "parse_failed",
            "response_id": raw_payload.get("response_id"),
        }
        return record, index_row

    record = task_record_from_payload(
        candidate=candidate,
        derived_context=derived_context,
        payload=payload,
        raw_response_path=raw_path,
    )
    log(
        f"paper={candidate.paper_id} task={candidate.task_idx} parsed successfully "
        f"hard_reject={record['hard_reject']} final_score={record['final_score']}"
    )
    index_row = {
        "paper_id": candidate.paper_id,
        "task_idx": candidate.task_idx,
        "task_brief": candidate.task_brief,
        "raw_response_path": relative_path(raw_path),
        "parse_status": "parsed",
        "response_id": raw_payload.get("response_id"),
        "hard_reject": record["hard_reject"],
        "final_score": record["final_score"],
    }
    return record, index_row


def write_prompt_outputs(
    output_dir: Path,
    prompt_body: str,
    candidates: list[TaskScoreCandidate],
    contexts: dict[tuple[int, int], dict[str, Any]],
    overwrite: bool,
) -> list[Path]:
    prompt_paths = [prompt_output_path(output_dir, candidate) for candidate in candidates]
    if not overwrite and any(path.exists() for path in prompt_paths):
        raise FileExistsError(f"Prompt output already exists for paper {candidates[0].paper_id}; use --overwrite")

    written_paths: list[Path] = []
    for candidate, path in zip(candidates, prompt_paths, strict=True):
        prompt_text = build_score_prompt(prompt_body, candidate, contexts[candidate.key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt_text + "\n", encoding="utf-8")
        written_paths.append(path)
    return written_paths


def ranking_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    status_priority = {"eligible": 0, "rejected": 1, "parse_failed": 2}
    final_score = row.get("final_score")
    sort_score = -(int(final_score) if isinstance(final_score, int) else -1)
    return (
        status_priority.get(str(row.get("rank_status")), 99),
        sort_score,
        int(row["paper_id"]),
        int(row["task_idx"]),
    )


def flatten_row_for_csv(row: dict[str, Any], global_rank: int) -> dict[str, Any]:
    scores = row.get("scores") or {}
    return {
        "global_rank": global_rank,
        "rank_status": row.get("rank_status"),
        "paper_id": row.get("paper_id"),
        "task_idx": row.get("task_idx"),
        "paper_title": row.get("paper_title"),
        "task_brief": row.get("task_brief"),
        "input_dataset": row.get("input_dataset"),
        "primary_bucket": row.get("primary_bucket"),
        "method_family": row.get("method_family"),
        "target_family": row.get("target_family"),
        "hard_reject": row.get("hard_reject"),
        "hard_reject_reasons": "|".join(row.get("hard_reject_reasons") or []),
        "final_score": row.get("final_score"),
        "feasibility": scores.get("feasibility"),
        "specificity": scores.get("specificity"),
        "evaluability": scores.get("evaluability"),
        "practicality": scores.get("practicality"),
        "novelty": scores.get("novelty"),
        "raw_response_path": row.get("raw_response_path"),
        "parse_status": row.get("parse_status"),
    }


def aggregate_scores(output_dir: Path) -> dict[str, Any]:
    paper_score_files = sorted(
        path for path in output_dir.glob("*_scores.json")
        if path.is_file()
    )
    rows: list[dict[str, Any]] = []
    for path in paper_score_files:
        payload = read_json(path)
        rows.extend(payload.get("rows", []))

    ranked_rows = sorted(rows, key=ranking_sort_key)
    ranked_rows_with_rank: list[dict[str, Any]] = []
    for index, row in enumerate(ranked_rows, start=1):
        ranked_rows_with_rank.append({"global_rank": index, **row})

    csv_rows = [flatten_row_for_csv(row, row["global_rank"]) for row in ranked_rows_with_rank]
    task_scores_path = output_dir / "task_scores.jsonl"
    ranking_jsonl_path = output_dir / "task_ranking.jsonl"
    ranking_csv_path = output_dir / "task_ranking.csv"
    write_jsonl(task_scores_path, rows)
    write_jsonl(ranking_jsonl_path, ranked_rows_with_rank)
    write_csv(ranking_csv_path, csv_rows)

    return {
        "paper_file_count": len(paper_score_files),
        "task_count": len(rows),
        "eligible_count": sum(1 for row in rows if row.get("rank_status") == "eligible"),
        "rejected_count": sum(1 for row in rows if row.get("rank_status") == "rejected"),
        "parse_failed_count": sum(1 for row in rows if row.get("rank_status") == "parse_failed"),
        "task_scores_path": str(task_scores_path),
        "task_ranking_jsonl": str(ranking_jsonl_path),
        "task_ranking_csv": str(ranking_csv_path),
    }


def score_paper(args: argparse.Namespace) -> dict[str, Any]:
    tasks_path = resolve_tasks_path(DEFAULT_INPUT_ROOT, args.paper_id, args.tasks_path)
    candidates = discover_task_candidates(tasks_path)
    if not args.overwrite:
        existing_paths = existing_output_paths(args.output_dir, candidates[0].paper_id, len(candidates), args.output_prompt_only)
        if existing_paths:
            return {
                "paper_id": candidates[0].paper_id,
                "paper_title": candidates[0].paper_title,
                "tasks_path": str(tasks_path),
                "output_prompt_only": args.output_prompt_only,
                "skipped": True,
                "existing_paths": [str(path) for path in existing_paths],
            }

    prompt_body = generate_tasks.extract_prompt_body(PROMPT_PATH)
    contexts = build_candidate_contexts(candidates)
    if args.output_prompt_only:
        prompt_paths = write_prompt_outputs(args.output_dir, prompt_body, candidates, contexts, args.overwrite)
        return {
            "paper_id": candidates[0].paper_id,
            "paper_title": candidates[0].paper_title,
            "tasks_path": str(tasks_path),
            "prompt_paths": [str(path) for path in prompt_paths],
            "output_prompt_only": True,
        }

    llm_config = generate_tasks.load_llm_config(generate_tasks.CONFIG_PATH, args.model_key)
    client = generate_tasks.build_client(llm_config)
    rows: list[dict[str, Any]] = []
    response_index_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        record, index_row = score_single_candidate(
            client=client,
            llm_config=llm_config,
            prompt_body=prompt_body,
            candidate=candidate,
            derived_context=contexts[candidate.key],
            output_dir=args.output_dir,
            max_output_tokens=args.max_output_tokens,
        )
        rows.append(record)
        response_index_rows.append(index_row)

    paper_id = candidates[0].paper_id
    scores_path = paper_scores_output_path(args.output_dir, paper_id)
    response_index_path = response_index_output_path(args.output_dir, paper_id)
    write_json(
        scores_path,
        {
            "paper_id": paper_id,
            "paper_title": candidates[0].paper_title,
            "tasks_path": relative_path(tasks_path),
            "task_count": len(rows),
            "rows": rows,
        },
    )
    write_json(
        response_index_path,
        {
            "paper_id": paper_id,
            "paper_title": candidates[0].paper_title,
            "rows": response_index_rows,
        },
    )

    summary = {
        "paper_id": paper_id,
        "paper_title": candidates[0].paper_title,
        "tasks_path": str(tasks_path),
        "task_count": len(rows),
        "eligible_count": sum(1 for row in rows if row.get("rank_status") == "eligible"),
        "rejected_count": sum(1 for row in rows if row.get("rank_status") == "rejected"),
        "parse_failed_count": sum(1 for row in rows if row.get("rank_status") == "parse_failed"),
        "scores_path": str(scores_path),
        "response_index_path": str(response_index_path),
    }
    log(
        f"paper={paper_id} scoring finished eligible={summary['eligible_count']} "
        f"rejected={summary['rejected_count']} parse_failed={summary['parse_failed_count']}"
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.aggregate_only:
        summary = aggregate_scores(args.output_dir)
    else:
        summary = score_paper(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
