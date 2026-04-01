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

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator


DEFAULT_MODEL_KEY = "openai/gpt-5.4"
DEFAULT_TASK_COUNT = 2
DEFAULT_MAX_OUTPUT_TOKENS = 6000


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
        if value != "report_generation":
            raise ValueError("task_type must be report_generation")
        return value


class GeneratedTaskBundle(BaseModel):
    tasks: list[GeneratedTask] = Field(min_length=DEFAULT_TASK_COUNT, max_length=DEFAULT_TASK_COUNT)


@dataclass(frozen=True)
class LLMConfig:
    api_key_env: str
    base_url: str
    model_name: str
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
    parser.add_argument("--task-count", type=int, default=DEFAULT_TASK_COUNT, help="Number of tasks to request per paper.")
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
        input_cost_per_million_tokens=model_payload.get("input_cost_per_million_tokens"),
        output_cost_per_million_tokens=model_payload.get("output_cost_per_million_tokens"),
    )


def extract_prompt_body(prompt_path: Path) -> str:
    text = prompt_path.read_text(encoding="utf-8")
    match = re.search(r"```text\n(.*?)\n```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def adapt_prompt(prompt_body: str, task_count: int) -> str:
    prompt = prompt_body
    prompt = re.sub(
        r"Your first step is to read the research paper markdown at:\n`[^`]+`",
        "Your first step is to read the uploaded research paper PDF provided in this API request. Use the uploaded PDF as the source paper for task generation.",
        prompt,
        count=1,
    )
    prompt = re.sub(r"generate `3-5`", f"generate `{task_count}`", prompt)
    prompt = re.sub(r"The `3-5` tasks", f"The `{task_count}` tasks", prompt)
    prompt = re.sub(
        r"Now read the paper at `[^`]+` and return the JSON object only\.",
        "Now read the uploaded PDF and return the JSON object only.",
        prompt,
        count=1,
    )
    prompt += (
        "\n\nAdditional API-run constraints:\n"
        f"- Generate exactly {task_count} tasks.\n"
        "- The uploaded PDF is the only paper file available to you in this request.\n"
        "- The generated tasks must stay grounded in targets and variables that are directly available in the processed EHR tables.\n"
        "- Do not assume diagnosis-category labels, symptom-dialogue annotations, medications, code systems, or event-token schemas exist unless they are plainly available in the processed inputs.\n"
        "- When the paper is about diagnosis dialogue or other unavailable supervision, rewrite the task around stable targets that are commonly visible in the processed tables, such as `Outcome`, `LOS`, `Readmission`, or directly derived descriptive quantities.\n"
        "- Prefer lightweight tabular or longitudinal analyses over sequence-generation or token-level setups unless the processed schema clearly supports them.\n"
        "- Return valid JSON that matches the required schema exactly.\n"
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
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def call_generation_api(
    *,
    client: OpenAI,
    llm_config: LLMConfig,
    prompt_text: str,
    paper_paths: PaperPaths,
    max_output_tokens: int,
) -> tuple[GeneratedTaskBundle, dict[str, Any]]:
    response = client.responses.parse(
        model=llm_config.model_name,
        input=[
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
        ],
        text_format=GeneratedTaskBundle,
        max_output_tokens=max_output_tokens,
        store=False,
    )

    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("The model response did not contain a parsed payload")

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
    }
    return parsed, metadata


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
    llm_config = load_llm_config(CONFIG_PATH, ARGS.model_key)
    paper_paths = resolve_paper_paths(MARKDOWN_ROOT, ARGS.paper_id, ARGS.paper_dir)
    prompt_body = extract_prompt_body(PROMPT_PATH)
    prompt_text = adapt_prompt(prompt_body, ARGS.task_count)
    client = build_client(llm_config)
    result, metadata = call_generation_api(
        client=client,
        llm_config=llm_config,
        prompt_text=prompt_text,
        paper_paths=paper_paths,
        max_output_tokens=ARGS.max_output_tokens,
    )
    tasks_path, metadata_path = write_outputs(ARGS.output_dir, paper_paths, result, metadata, ARGS.overwrite)
    print(json.dumps(
        {
            "paper_id": paper_paths.paper_id,
            "paper_dir": str(paper_paths.paper_dir),
            "pdf_path": str(paper_paths.pdf_path),
            "tasks_path": str(tasks_path),
            "metadata_path": str(metadata_path),
            "task_count": len(result.tasks),
            "response_id": metadata["response_id"],
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
