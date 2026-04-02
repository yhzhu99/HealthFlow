from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator


DEFAULT_MODEL_KEY = "openai/gpt-5.4"
DEFAULT_TASK_COUNT = 2
TASKS_PER_API_CALL = 1
DEFAULT_MAX_OUTPUT_TOKENS = 65536
DEFAULT_REASONING_EFFORT = "medium"
TASK_TYPE = "report_generation"
PROMPT_DATASET_METADATA_PLACEHOLDER = "{{DATASET_METADATA_BLOCK}}"
PROMPT_TASK_ASSIGNMENT_PLACEHOLDER = "{{TASK_ASSIGNMENT_BLOCK}}"
FIXED_REPORT_REQUIREMENTS = (
    "State the objective and paper-inspired hypothesis.",
    "Describe the input data used and preprocessing steps.",
    "Explain the method or analysis design.",
    "Report quantitative results.",
    "Provide figure and/or table evidence.",
    "State the final conclusion.",
)


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "config.toml").exists() and (parent / "data").exists():
            return parent
    raise FileNotFoundError("Could not locate project root")


PROJECT_ROOT = find_project_root()
DATASET_ROOT = PROJECT_ROOT / "data" / "ehrflowbench"
MARKDOWN_ROOT = DATASET_ROOT / "raw" / "papers" / "markdowns"
PROMPT_PATH = DATASET_ROOT / "scripts" / "prompt_ehrflowbench.md"
CONFIG_PATH = PROJECT_ROOT / "config.toml"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "processed" / "papers" / "generated_tasks"


@dataclass(frozen=True)
class DatasetPromptConfig:
    key: str
    display_name: str
    parquet_path: Path
    split_metadata_path: Path
    required_inputs: tuple[str, ...]
    value_reference_path: Path | None = None


DATASET_PROMPT_CONFIGS = (
    DatasetPromptConfig(
        key="tjh",
        display_name="TJH",
        parquet_path=DATASET_ROOT / "processed" / "tjh" / "tjh_formatted_ehr.parquet",
        split_metadata_path=DATASET_ROOT / "processed" / "tjh" / "split_metadata.json",
        required_inputs=(
            "data/ehrflowbench/processed/tjh/tjh_formatted_ehr.parquet",
            "data/ehrflowbench/processed/tjh/split_metadata.json",
        ),
    ),
    DatasetPromptConfig(
        key="mimic_iv_demo",
        display_name="MIMIC-IV-demo",
        parquet_path=DATASET_ROOT / "processed" / "mimic_iv_demo" / "mimic_iv_demo_formatted_ehr.parquet",
        split_metadata_path=DATASET_ROOT / "processed" / "mimic_iv_demo" / "split_metadata.json",
        required_inputs=(
            "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet",
            "data/ehrflowbench/processed/mimic_iv_demo/split_metadata.json",
            "data/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_value_reference.md",
        ),
        value_reference_path=DATASET_ROOT / "processed" / "mimic_iv_demo" / "mimic_iv_demo_value_reference.md",
    ),
)


class LLMGeneratedTask(BaseModel):
    task_brief: str
    focus_areas: list[str]
    task: str
    deliverables: list[str]


class LLMGeneratedTaskBundle(BaseModel):
    task: LLMGeneratedTask


class GeneratedTask(BaseModel):
    task_brief: str
    task_type: str
    focus_areas: list[str]
    task: str
    required_inputs: list[str]
    deliverables: list[str]
    report_requirements: list[str]

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, value: str) -> str:
        if value != TASK_TYPE:
            raise ValueError(f"task_type must be {TASK_TYPE}")
        return value


class GeneratedTaskBundle(BaseModel):
    tasks: list[GeneratedTask] = Field(min_length=DEFAULT_TASK_COUNT, max_length=DEFAULT_TASK_COUNT)


@dataclass(frozen=True)
class LLMConfig:
    api_key_env: str
    base_url: str
    model_name: str
    reasoning_effort: str
    input_cost_per_million_tokens: float | None
    output_cost_per_million_tokens: float | None


@dataclass(frozen=True)
class PaperPaths:
    paper_id: int
    paper_dir: Path
    pdf_path: Path
    markdown_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EhrFlowBench tasks via the OpenAI-compatible Responses API.")
    parser.add_argument("--paper-id", type=int, help="Numeric paper id prefix from the markdown folder name, for example 1.")
    parser.add_argument("--paper-dir", type=str, help="Explicit paper directory under raw/papers/markdowns.")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY, help="LLM key under config.toml.")
    parser.add_argument(
        "--task-count",
        type=int,
        default=DEFAULT_TASK_COUNT,
        help="Fixed at 2 tasks per paper: task 1 uses TJH and task 2 uses MIMIC-IV-demo.",
    )
    parser.add_argument("--output-prompt-only", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_llm_config(config_path: Path, model_key: str) -> LLMConfig:
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    llm_section = payload.get("llm", {})
    if model_key not in llm_section:
        raise KeyError(f"Model key {model_key!r} was not found in {config_path}")
    model_payload = llm_section[model_key]
    return LLMConfig(
        api_key_env=model_payload["api_key_env"],
        base_url=model_payload["base_url"],
        model_name=model_payload["model_name"],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        input_cost_per_million_tokens=model_payload.get("input_cost_per_million_tokens"),
        output_cost_per_million_tokens=model_payload.get("output_cost_per_million_tokens"),
    )


def extract_prompt_body(prompt_path: Path) -> str:
    text = prompt_path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)\n```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_path_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required local metadata file does not exist: {path}")
    return path


def summarize_split_metadata(path: Path) -> dict[str, Any]:
    payload = load_json(ensure_path_exists(path))
    keys = (
        "dataset",
        "seed",
        "key_column",
        "label_column",
        "split_ratios",
        "key_count",
        "row_count",
        "label_distribution",
        "split_key_counts",
        "split_row_counts",
    )
    return {key: payload[key] for key in keys if key in payload}


def summarize_parquet(path: Path) -> dict[str, Any]:
    table = pq.read_table(ensure_path_exists(path))
    sample_rows = table.slice(0, 1).to_pylist()
    return {
        "num_rows": table.num_rows,
        "num_columns": table.num_columns,
        "schema": [(field.name, str(field.type)) for field in table.schema],
        "sample_rows": sample_rows,
    }


def extract_supported_targets(schema_fields: list[tuple[str, str]]) -> list[str]:
    preferred_targets = ("Outcome", "LOS", "Readmission")
    available = {name for name, _ in schema_fields}
    return [name for name in preferred_targets if name in available]


def build_dataset_metadata_block(dataset_configs: tuple[DatasetPromptConfig, ...]) -> str:
    lines = []

    for index, config in enumerate(dataset_configs, start=1):
        parquet_summary = summarize_parquet(config.parquet_path)
        split_summary = summarize_split_metadata(config.split_metadata_path)
        supported_targets = extract_supported_targets(parquet_summary["schema"])

        lines.append(f"### Dataset: {config.display_name}")
        lines.append(f"- Dataset key: `{config.key}`")
        lines.append(f"- Canonical parquet path: `{config.required_inputs[0]}`")
        lines.append(f"- Canonical split metadata path: `{config.required_inputs[1]}`")
        if config.value_reference_path is not None:
            lines.append(f"- Canonical value-reference path: `{config.required_inputs[2]}`")
        lines.append(
            f"- Table shape: {parquet_summary['num_rows']} rows x {parquet_summary['num_columns']} columns."
        )
        lines.append(f"- Key column: `{split_summary.get('key_column', 'unknown')}`")
        if split_summary.get("label_column"):
            lines.append(f"- Main label column: `{split_summary['label_column']}`")
        if supported_targets:
            lines.append("- Available label columns: " + ", ".join(f"`{value}`" for value in supported_targets))
        if split_summary.get("split_ratios"):
            lines.append(
                "- Split ratios: "
                + json.dumps(split_summary["split_ratios"], ensure_ascii=False, sort_keys=True)
            )
        if split_summary.get("split_key_counts"):
            lines.append(
                "- Split key counts: "
                + json.dumps(split_summary["split_key_counts"], ensure_ascii=False, sort_keys=True)
            )
        if split_summary.get("label_distribution"):
            lines.append(
                "- Label distribution summary: "
                + json.dumps(split_summary["label_distribution"], ensure_ascii=False, sort_keys=True)
            )
        lines.append("- Column schema:")
        for field_name, field_type in parquet_summary["schema"]:
            lines.append(f"  - `{field_name}`: `{field_type}`")
        lines.append("- Sample row:")
        lines.append("```json")
        lines.append(json.dumps(parquet_summary["sample_rows"], ensure_ascii=False, indent=2))
        lines.append("```")
        if config.value_reference_path is not None:
            value_reference = ensure_path_exists(config.value_reference_path).read_text(encoding="utf-8").strip()
            lines.append("- Value-reference content:")
            lines.append("```markdown")
            lines.append(value_reference)
            lines.append("```")
        lines.append("")

    return "\n".join(lines).strip()


def build_task_assignment_block(dataset_config: DatasetPromptConfig) -> str:
    lines = [
        f"The task must use `{dataset_config.display_name}` only.",
        "Do not ask this task to access, compare against, or mention any other dataset."
    ]
    return "\n".join(lines)


def inject_prompt_block(prompt: str, placeholder: str, block: str, title: str) -> str:
    if placeholder in prompt:
        return prompt.replace(placeholder, block)
    return f"{prompt.rstrip()}\n\n## {title}\n\n{block}\n"


def adapt_prompt(prompt_body: str, task_count: int, dataset_config: DatasetPromptConfig) -> str:
    if task_count != DEFAULT_TASK_COUNT:
        raise ValueError(
            f"EhrFlowBench task generation is fixed to {DEFAULT_TASK_COUNT} tasks "
            f"(one TJH task and one MIMIC-IV-demo task); received {task_count}"
        )

    prompt = inject_prompt_block(
        prompt_body,
        PROMPT_DATASET_METADATA_PLACEHOLDER,
        build_dataset_metadata_block((dataset_config,)),
        "Injected Local EHR Metadata",
    )
    prompt = inject_prompt_block(
        prompt,
        PROMPT_TASK_ASSIGNMENT_PLACEHOLDER,
        build_task_assignment_block(dataset_config),
        "Injected Task Allocation",
    )
    return prompt


def paper_id_from_dir_name(name: str) -> int:
    match = re.match(r"(\d+)_", name)
    if not match:
        raise ValueError(f"Could not parse paper id from directory name: {name}")
    return int(match.group(1))


def discover_paper_dirs(markdown_root: Path) -> list[Path]:
    if not markdown_root.exists():
        raise FileNotFoundError(f"Markdown root does not exist: {markdown_root}")
    return sorted(path for path in markdown_root.iterdir() if path.is_dir())


def resolve_paper_paths(markdown_root: Path, paper_id: int | None, explicit_dir: str | None) -> PaperPaths:
    paper_dir: Path | None = None
    if explicit_dir:
        candidate = Path(explicit_dir)
        if not candidate.is_absolute():
            candidate = markdown_root / candidate
        paper_dir = candidate.resolve()
    elif paper_id is not None:
        for candidate in discover_paper_dirs(markdown_root):
            if paper_id_from_dir_name(candidate.name) == paper_id:
                paper_dir = candidate.resolve()
                break
    else:
        raise ValueError("Either --paper-id or --paper-dir must be provided")

    if paper_dir is None or not paper_dir.exists():
        raise FileNotFoundError("Could not resolve the requested paper directory")

    pdf_candidates = sorted(paper_dir.glob("*_origin.pdf"))
    if not pdf_candidates:
        pdf_candidates = sorted(paper_dir.glob("*.pdf"))
    if len(pdf_candidates) != 1:
        raise ValueError(f"Expected exactly one PDF in {paper_dir}, found {len(pdf_candidates)}")

    markdown_path = paper_dir / "full.md"
    return PaperPaths(
        paper_id=paper_id_from_dir_name(paper_dir.name),
        paper_dir=paper_dir,
        pdf_path=pdf_candidates[0],
        markdown_path=markdown_path if markdown_path.exists() else None,
    )


def build_client(config: LLMConfig) -> OpenAI:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Environment variable {config.api_key_env} is not set")
    return OpenAI(api_key=api_key, base_url=config.base_url)


def estimate_cost(usage: Any, config: LLMConfig) -> dict[str, float | None]:
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    input_cost = None
    output_cost = None
    total_cost = None
    if input_tokens is not None and config.input_cost_per_million_tokens is not None:
        input_cost = (input_tokens / 1_000_000) * config.input_cost_per_million_tokens
    if output_tokens is not None and config.output_cost_per_million_tokens is not None:
        output_cost = (output_tokens / 1_000_000) * config.output_cost_per_million_tokens
    if input_cost is not None or output_cost is not None:
        total_cost = (input_cost or 0.0) + (output_cost or 0.0)
    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


def model_to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: model_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [model_to_jsonable(item) for item in value]
    return value


def encode_pdf_as_base64(pdf_path: Path) -> str:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    return f"data:application/pdf;base64,{encoded}"


def build_generation_request_input(prompt_text: str, paper_paths: PaperPaths) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {
                    "type": "input_file",
                    "filename": paper_paths.pdf_path.name,
                    "file_data": encode_pdf_as_base64(paper_paths.pdf_path),
                },
            ],
        }
    ]


def normalize_string_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        cleaned.append(normalized)
        seen.add(normalized)
    return cleaned


def normalize_focus_areas(values: list[str]) -> list[str]:
    cleaned = normalize_string_list(values)
    if not 2 <= len(cleaned) <= 4:
        raise ValueError(f"focus_areas must contain 2-4 unique non-empty values, received {cleaned!r}")
    return cleaned


def normalize_deliverables(values: list[str]) -> list[str]:
    cleaned = normalize_string_list(values)
    if "report.md" not in cleaned:
        cleaned.insert(0, "report.md")
    return cleaned


def task_mentions_other_dataset(task: LLMGeneratedTask, assigned_dataset_key: str) -> bool:
    text = " ".join(
        [
            task.task_brief,
            " ".join(task.focus_areas),
            task.task,
            " ".join(task.deliverables),
        ]
    ).lower()
    forbidden_markers = {
        "tjh": ("mimic", "mimic-iv", "mimic iv", "mimic_iv_demo"),
        "mimic_iv_demo": ("tjh", "tongji"),
    }
    return any(marker in text for marker in forbidden_markers[assigned_dataset_key])


def enrich_generated_task(task: LLMGeneratedTask, dataset_config: DatasetPromptConfig) -> GeneratedTask:
    if task_mentions_other_dataset(task, dataset_config.key):
        raise ValueError(
            f"Task assigned to {dataset_config.display_name} mentions the other dataset; "
            "single-task single-dataset contract violated"
        )

    return GeneratedTask(
        task_brief=task.task_brief.strip(),
        task_type=TASK_TYPE,
        focus_areas=normalize_focus_areas(task.focus_areas),
        task=task.task.strip(),
        required_inputs=list(dataset_config.required_inputs),
        deliverables=normalize_deliverables(task.deliverables),
        report_requirements=list(FIXED_REPORT_REQUIREMENTS),
    )


def extract_generated_task(bundle: LLMGeneratedTaskBundle, dataset_config: DatasetPromptConfig) -> GeneratedTask:
    if len(bundle.tasks) != TASKS_PER_API_CALL:
        raise ValueError(
            f"Expected {TASKS_PER_API_CALL} task from the model for {dataset_config.display_name}, "
            f"received {len(bundle.tasks)}"
        )
    return enrich_generated_task(bundle.tasks[0], dataset_config)


def response_debug_payload(
    *,
    response: Any,
    llm_config: LLMConfig,
    paper_paths: PaperPaths,
    max_output_tokens: int,
    reasoning_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "response_id": getattr(response, "id", None),
        "model": llm_config.model_name,
        "paper_id": paper_paths.paper_id,
        "paper_dir": str(paper_paths.paper_dir),
        "pdf_path": str(paper_paths.pdf_path),
        "request": {
            "max_output_tokens": max_output_tokens,
            "reasoning": reasoning_config,
        },
        "status": getattr(response, "status", None),
        "error": model_to_jsonable(getattr(response, "error", None)),
        "incomplete_details": model_to_jsonable(getattr(response, "incomplete_details", None)),
        "usage": model_to_jsonable(getattr(response, "usage", None)),
        "output_text": getattr(response, "output_text", None),
        "output": model_to_jsonable(getattr(response, "output", None)),
    }


def log_response_debug(
    *,
    response: Any,
    llm_config: LLMConfig,
    paper_paths: PaperPaths,
    max_output_tokens: int,
    reasoning_config: dict[str, Any],
) -> None:
    payload = response_debug_payload(
        response=response,
        llm_config=llm_config,
        paper_paths=paper_paths,
        max_output_tokens=max_output_tokens,
        reasoning_config=reasoning_config,
    )
    print("response_debug_begin", file=sys.stderr)
    print(json.dumps(payload, indent=2, ensure_ascii=False), file=sys.stderr)
    print("response_debug_end", file=sys.stderr)


def describe_response_state(response: Any) -> str:
    status = getattr(response, "status", None) or "unknown"
    incomplete_details = model_to_jsonable(getattr(response, "incomplete_details", None))
    if isinstance(incomplete_details, dict) and incomplete_details.get("reason"):
        return f"status={status}, incomplete_reason={incomplete_details['reason']}"
    return f"status={status}"


def should_retry_without_reasoning_max_tokens(exc: Exception) -> bool:
    message = str(exc).lower()
    if "reasoning.max_tokens" not in message:
        return False

    unsupported_markers = (
        "unknown parameter",
        "unsupported",
        "not allowed",
        "not supported",
        "invalid",
    )
    return any(marker in message for marker in unsupported_markers)


def call_generation_api(
    *,
    client: OpenAI,
    llm_config: LLMConfig,
    prompt_text: str,
    paper_paths: PaperPaths,
    dataset_config: DatasetPromptConfig,
    max_output_tokens: int,
) -> tuple[GeneratedTask, dict[str, Any]]:
    request_input = build_generation_request_input(prompt_text, paper_paths)
    reasoning_config = {
        "effort": llm_config.reasoning_effort
    }

    response = client.responses.parse(
        model=llm_config.model_name,
        input=request_input,
        reasoning=reasoning_config,
        text_format=LLMGeneratedTaskBundle,
        max_output_tokens=max_output_tokens,
        store=False,
    )

    parsed = response.output_parsed
    if parsed is None:
        log_response_debug(
            response=response,
            llm_config=llm_config,
            paper_paths=paper_paths,
            max_output_tokens=max_output_tokens,
            reasoning_config=reasoning_config,
        )
        raise ValueError(f"The model response did not contain a parsed payload ({describe_response_state(response)})")

    result = extract_generated_task(parsed, dataset_config)

    metadata = {
        "response_id": response.id,
        "model": llm_config.model_name,
        "paper_id": paper_paths.paper_id,
        "paper_dir": str(paper_paths.paper_dir),
        "pdf_path": str(paper_paths.pdf_path),
        "markdown_path": str(paper_paths.markdown_path) if paper_paths.markdown_path else None,
        "usage": model_to_jsonable(response.usage),
        "estimated_cost": estimate_cost(response.usage, llm_config),
        "output_text": response.output_text,
        "uploaded_via": "responses.input_file.file_data",
        "task_dataset_assignment": dataset_config.key,
    }
    return result, metadata


def aggregate_numeric_dicts(payloads: list[dict[str, Any] | None]) -> dict[str, int | float | None] | None:
    keys = {
        key
        for payload in payloads
        if isinstance(payload, dict)
        for key in payload
    }
    if not keys:
        return None

    aggregated: dict[str, int | float | None] = {}
    saw_numeric = False
    for key in sorted(keys):
        numeric_values = [
            value
            for payload in payloads
            if isinstance(payload, dict)
            for value in [payload.get(key)]
            if isinstance(value, (int, float))
        ]
        if numeric_values:
            saw_numeric = True
            total = sum(numeric_values)
            aggregated[key] = int(total) if all(isinstance(value, int) for value in numeric_values) else total
        else:
            aggregated[key] = None
    return aggregated if saw_numeric else None


def combine_generation_metadata(
    llm_config: LLMConfig,
    paper_paths: PaperPaths,
    response_metadatas: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "response_ids": [metadata["response_id"] for metadata in response_metadatas],
        "model": llm_config.model_name,
        "paper_id": paper_paths.paper_id,
        "paper_dir": str(paper_paths.paper_dir),
        "pdf_path": str(paper_paths.pdf_path),
        "markdown_path": str(paper_paths.markdown_path) if paper_paths.markdown_path else None,
        "usage": aggregate_numeric_dicts([metadata.get("usage") for metadata in response_metadatas]),
        "estimated_cost": aggregate_numeric_dicts([metadata.get("estimated_cost") for metadata in response_metadatas]),
        "uploaded_via": "responses.input_file.file_data",
        "task_dataset_assignments": [config.key for config in DATASET_PROMPT_CONFIGS],
        "responses": response_metadatas,
    }


def prompt_output_path(output_dir: Path, paper_paths: PaperPaths, dataset_config: DatasetPromptConfig) -> Path:
    return output_dir / f"{paper_paths.paper_id}_{dataset_config.key}_prompt.md"


def write_prompt_outputs(
    output_dir: Path,
    paper_paths: PaperPaths,
    prompt_payloads: list[tuple[DatasetPromptConfig, str]],
    overwrite: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_paths = [
        prompt_output_path(output_dir, paper_paths, dataset_config)
        for dataset_config, _prompt_text in prompt_payloads
    ]
    if not overwrite and any(path.exists() for path in prompt_paths):
        raise FileExistsError(f"Prompt output already exists for paper {paper_paths.paper_id}; use --overwrite")

    written_paths: list[Path] = []
    for (dataset_config, prompt_text), prompt_path in zip(prompt_payloads, prompt_paths, strict=True):
        prompt_sections = [
            "# Prompt Input",
            "",
            f"- paper_id: {paper_paths.paper_id}",
            f"- dataset_key: `{dataset_config.key}`",
            f"- dataset_name: `{dataset_config.display_name}`",
            f"- paper_dir: `{paper_paths.paper_dir}`",
            f"- pdf_path: `{paper_paths.pdf_path}`",
            f"- markdown_path: `{paper_paths.markdown_path}`" if paper_paths.markdown_path else "- markdown_path: null",
            "",
            "## Prompt Text",
            "",
            prompt_text.rstrip(),
            "",
        ]
        prompt_path.write_text("\n".join(prompt_sections), encoding="utf-8")
        written_paths.append(prompt_path)
    return written_paths


def write_outputs(output_dir: Path, paper_paths: PaperPaths, result: GeneratedTaskBundle, metadata: dict[str, Any], overwrite: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = output_dir / f"{paper_paths.paper_id}_tasks.json"
    metadata_path = output_dir / f"{paper_paths.paper_id}_response.json"
    if not overwrite and (tasks_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"Output already exists for paper {paper_paths.paper_id}; use --overwrite")

    tasks_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(model_to_jsonable(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return tasks_path, metadata_path


def main() -> None:
    paper_paths = resolve_paper_paths(MARKDOWN_ROOT, ARGS.paper_id, ARGS.paper_dir)
    prompt_body = extract_prompt_body(PROMPT_PATH)
    prompt_payloads = [
        (dataset_config, adapt_prompt(prompt_body, ARGS.task_count, dataset_config))
        for dataset_config in DATASET_PROMPT_CONFIGS
    ]
    if ARGS.output_prompt_only:
        prompt_paths = write_prompt_outputs(ARGS.output_dir, paper_paths, prompt_payloads, ARGS.overwrite)
        print(json.dumps(
            {
                "paper_id": paper_paths.paper_id,
                "paper_dir": str(paper_paths.paper_dir),
                "pdf_path": str(paper_paths.pdf_path),
                "prompt_paths": [str(path) for path in prompt_paths],
                "output_prompt_only": True,
            },
            indent=2,
        ))
        return

    llm_config = load_llm_config(CONFIG_PATH, ARGS.model_key)
    client = build_client(llm_config)
    generated_tasks: list[GeneratedTask] = []
    response_metadatas: list[dict[str, Any]] = []
    for dataset_config, prompt_text in prompt_payloads:
        task, response_metadata = call_generation_api(
            client=client,
            llm_config=llm_config,
            prompt_text=prompt_text,
            paper_paths=paper_paths,
            dataset_config=dataset_config,
            max_output_tokens=ARGS.max_output_tokens,
        )
        generated_tasks.append(task)
        response_metadatas.append(response_metadata)

    result = GeneratedTaskBundle(tasks=generated_tasks)
    metadata = combine_generation_metadata(llm_config, paper_paths, response_metadatas)
    tasks_path, metadata_path = write_outputs(ARGS.output_dir, paper_paths, result, metadata, ARGS.overwrite)
    print(json.dumps(
        {
            "paper_id": paper_paths.paper_id,
            "paper_dir": str(paper_paths.paper_dir),
            "pdf_path": str(paper_paths.pdf_path),
            "tasks_path": str(tasks_path),
            "metadata_path": str(metadata_path),
            "task_count": len(result.tasks),
            "response_ids": metadata["response_ids"],
            "estimated_cost": metadata["estimated_cost"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    try:
        ARGS = parse_args()
        main()
    except ValidationError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
