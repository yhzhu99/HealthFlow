from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
_REPORT_EXTENSIONS = {".md", ".txt", ".pdf", ".rst", ".doc", ".docx"}
_CODE_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".r",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
}
_DATA_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".feather", ".xlsx", ".xls", ".arrow"}
_TEXT_EXTENSIONS = _REPORT_EXTENSIONS | _CODE_EXTENSIONS | _DATA_EXTENSIONS | {".yaml", ".yml", ".toml"}
_EXCLUDED_RUNTIME_FILES = {
    "report.md",
    "task_state.json",
    "memory_context.json",
    "evaluation.json",
    "cost_analysis.json",
    "run_manifest.json",
    "run_result.json",
    "full_history.json",
}
_COMMENT_PREFIXES = ("#", "//", "--", ";", "/*", "*")
_WHITESPACE_RE = re.compile(r"\s+")


def generate_task_report(task_workspace: Path) -> Path:
    run_manifest = _read_json(task_workspace / "run_manifest.json")
    run_result = _read_json(task_workspace / "run_result.json")
    full_history = _read_json(task_workspace / "full_history.json")
    evaluation = _read_json(task_workspace / "evaluation.json")
    cost_analysis = _read_json(task_workspace / "cost_analysis.json")

    prompt_relative_path = _relative_runtime_path(run_result.get("prompt_path"), task_workspace)
    runtime_paths = _runtime_artifact_paths(task_workspace, run_result.get("backend"), prompt_relative_path)
    deliverables = _collect_deliverables(task_workspace, runtime_paths)

    report_markdown = _render_report(
        task_workspace=task_workspace,
        run_manifest=run_manifest,
        run_result=run_result,
        full_history=full_history,
        evaluation=evaluation,
        cost_analysis=cost_analysis,
        deliverables=deliverables,
        runtime_paths=runtime_paths,
    )

    report_path = task_workspace / "report.md"
    temp_path = report_path.with_suffix(".md.tmp")
    temp_path.write_text(report_markdown, encoding="utf-8")
    temp_path.replace(report_path)
    return report_path


def _render_report(
    task_workspace: Path,
    run_manifest: dict[str, Any],
    run_result: dict[str, Any],
    full_history: dict[str, Any],
    evaluation: dict[str, Any],
    cost_analysis: dict[str, Any],
    deliverables: list[dict[str, Any]],
    runtime_paths: list[str],
) -> str:
    attempts = full_history.get("attempts", [])
    final_status = str(run_result.get("evaluation_status") or evaluation.get("status") or "unknown")
    final_score = run_result.get("evaluation_score")
    task_id = str(run_manifest.get("task_id") or task_workspace.name)
    summary = _sanitize_text(str(run_result.get("final_summary") or "No final summary was recorded."), task_workspace)
    user_request = _sanitize_text(str(run_manifest.get("user_request") or full_history.get("user_request") or ""), task_workspace)
    feedback = _sanitize_text(str(evaluation.get("feedback") or "No evaluator feedback was recorded."), task_workspace)
    reasoning = _sanitize_text(str(evaluation.get("reasoning") or "No evaluator reasoning was recorded."), task_workspace)
    llm_role_models = run_result.get("llm_role_models") or run_manifest.get("llm_role_models") or {}
    execution_time = run_result.get("execution_time")

    lines = [
        "# HealthFlow Report",
        "",
        f"- Task ID: `{task_id}`",
        f"- Outcome: `{final_status}`",
        f"- Success: `{bool(run_result.get('success', False))}`",
        f"- Evaluation Score: `{_format_scalar(final_score)}`",
        f"- Attempts: `{len(attempts)}`",
        f"- Backend: `{_format_scalar(run_result.get('backend') or run_manifest.get('backend'))}`",
        f"- Executor Model: `{_format_scalar(run_result.get('executor_model') or run_manifest.get('executor_model'))}`",
        f"- Executor Provider: `{_format_scalar(run_result.get('executor_provider') or run_manifest.get('executor_provider'))}`",
        f"- Planner Model: `{_format_scalar(run_result.get('reasoning_model') or run_manifest.get('reasoning_model'))}`",
        f"- Role Models: `{_format_role_models(llm_role_models)}`",
        f"- Memory Write Policy: `{_format_scalar(run_result.get('memory_write_policy') or run_manifest.get('memory_write_policy'))}`",
        f"- Execution Time: `{_format_duration(execution_time)}`",
        "",
        "## Abstract",
        _build_abstract(
            user_request=user_request,
            final_status=final_status,
            final_score=final_score,
            attempt_count=len(attempts),
            execution_time=execution_time,
            feedback=feedback,
            reasoning=reasoning,
            deliverables=deliverables,
        ),
        "",
        "## Original Task",
        user_request or "No original task was recorded.",
        "",
        "## Final Outcome",
        f"- Final Summary: {summary}",
        f"- Evaluator Feedback: {feedback}",
        f"- Evaluator Reasoning: {reasoning}",
    ]

    repair_instructions = evaluation.get("repair_instructions") or []
    if repair_instructions:
        lines.append(f"- Repair Instructions: {_render_inline_list(repair_instructions, task_workspace)}")

    violated_constraints = evaluation.get("violated_constraints") or []
    if violated_constraints:
        lines.append(f"- Violated Constraints: {_render_inline_list(violated_constraints, task_workspace)}")

    repair_hypotheses = evaluation.get("repair_hypotheses") or []
    if repair_hypotheses:
        lines.append(f"- Repair Hypotheses: {_render_inline_list(repair_hypotheses, task_workspace)}")

    memory_worthy_insights = evaluation.get("memory_worthy_insights") or []
    if memory_worthy_insights:
        lines.append(f"- Memory-Worthy Insights: {_render_inline_list(memory_worthy_insights, task_workspace)}")

    lines.extend(["", "## Attempt Trajectory"])
    if attempts:
        lines.extend(_render_attempts(attempts, task_workspace))
    else:
        lines.append("No attempt history was recorded.")

    lines.extend(["", "## Produced Deliverables"])
    if deliverables:
        lines.extend(_render_deliverables(deliverables))
    else:
        lines.append("No non-runtime deliverables were found in the task workspace.")

    lines.extend(
        [
            "",
            "## Evaluation and Analysis",
            f"- Final Evaluation Status: `{final_status}`",
            f"- Final Evaluation Score: `{_format_scalar(final_score)}`",
            f"- Attempt Count: `{len(attempts)}`",
            f"- Total Estimated Cost (USD): `{_format_scalar(cost_analysis.get('run_total', {}).get('total_estimated_cost_usd'))}`",
            f"- LLM Estimated Cost (USD): `{_format_scalar(cost_analysis.get('run_total', {}).get('llm_estimated_cost_usd'))}`",
            f"- Executor Estimated Cost (USD): `{_format_scalar(cost_analysis.get('run_total', {}).get('executor_estimated_cost_usd'))}`",
        ]
    )
    lines.extend(_render_usage_analysis(run_result.get("usage_summary") or {}))

    lines.extend(["", "## Runtime Artifacts"])
    lines.extend(_render_runtime_artifacts(task_workspace, runtime_paths))

    lines.extend(["", "## Appendix: Full Artifact Inventory"])
    if deliverables:
        lines.extend(_render_inventory(deliverables))
    else:
        lines.append("No non-runtime artifacts were available for inventory.")

    return "\n".join(lines).strip() + "\n"


def _build_abstract(
    user_request: str,
    final_status: str,
    final_score: Any,
    attempt_count: int,
    execution_time: Any,
    feedback: str,
    reasoning: str,
    deliverables: list[dict[str, Any]],
) -> str:
    notable_paths = ", ".join(f"`{item['path']}`" for item in deliverables[:3]) or "no non-runtime deliverables"
    score_text = _format_scalar(final_score)
    duration_text = _format_duration(execution_time)
    return (
        f"HealthFlow executed the request \"{_truncate_text(user_request, 180)}\" over {attempt_count} attempt(s) "
        f"and finished with status `{final_status}` (score `{score_text}`) in `{duration_text}`. "
        f"Notable deliverables include {notable_paths}. "
        f"Evaluator feedback: {_truncate_text(feedback, 220)} "
        f"Reasoning: {_truncate_text(reasoning, 220)}"
    )


def _render_attempts(attempts: list[dict[str, Any]], task_workspace: Path) -> list[str]:
    lines: list[str] = []
    for attempt in attempts:
        attempt_id = attempt.get("attempt", "?")
        plan = attempt.get("plan", {})
        execution = attempt.get("execution", {})
        evaluation = attempt.get("evaluation", {})
        artifacts = attempt.get("artifacts", {}).get("workspace_paths", [])
        lines.extend(
            [
                f"### Attempt {attempt_id}",
                f"- Objective: {_sanitize_text(str(plan.get('objective') or 'No objective recorded.'), task_workspace)}",
                f"- Execution: success=`{bool(execution.get('success', False))}`, return_code=`{_format_scalar(execution.get('return_code'))}`, duration=`{_format_duration(execution.get('duration_seconds'))}`, timed_out=`{bool(execution.get('timed_out', False))}`",
                f"- Evaluation: status=`{_format_scalar(evaluation.get('status'))}`, score=`{_format_scalar(evaluation.get('score'))}`, retry_recommended=`{bool(evaluation.get('retry_recommended', False))}`",
                f"- Feedback: {_sanitize_text(str(evaluation.get('feedback') or 'No evaluator feedback recorded.'), task_workspace)}",
                f"- Workspace Artifacts Observed: `{len(artifacts)}`",
            ]
        )
        recommended_steps = plan.get("recommended_steps") or []
        if recommended_steps:
            lines.append(f"- Planned Steps: {_render_inline_list(recommended_steps[:4], task_workspace)}")
        artifact_preview = [item for item in artifacts if not _is_runtime_path(item)]
        if artifact_preview:
            lines.append(
                "- Deliverable Preview: "
                + ", ".join(f"`{_sanitize_text(str(item), task_workspace)}`" for item in artifact_preview[:5])
            )
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _render_deliverables(deliverables: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = [f"Found `{len(deliverables)}` non-runtime deliverable(s) in the workspace.", ""]
    grouped = {
        "reports/docs": [item for item in deliverables if item["category"] == "reports/docs"],
        "images": [item for item in deliverables if item["category"] == "images"],
        "code/notebooks": [item for item in deliverables if item["category"] == "code/notebooks"],
        "tables/data": [item for item in deliverables if item["category"] == "tables/data"],
        "other outputs": [item for item in deliverables if item["category"] == "other outputs"],
    }
    image_embed_count = 0
    for category, items in grouped.items():
        if not items:
            continue
        lines.extend([f"### {category.title()}", ""])
        for item in items:
            lines.append(f"- {_artifact_link(item['path'])}: {item['descriptor']}")
            if item["category"] == "images" and image_embed_count < 5:
                lines.append(f"![{Path(item['path']).name}]({item['path']})")
                image_embed_count += 1
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _render_usage_analysis(usage_summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for component in ("planning", "execution", "evaluation", "reflection"):
        component_summary = usage_summary.get(component) or {}
        if not component_summary:
            continue
        models = component_summary.get("models") or []
        lines.append(
            f"- {component.title()} Usage: calls=`{_format_scalar(component_summary.get('calls'))}`, "
            f"estimated_cost_usd=`{_format_scalar(component_summary.get('estimated_cost_usd'))}`, "
            f"models=`{', '.join(models) if models else 'N/A'}`"
        )
    return lines


def _render_runtime_artifacts(task_workspace: Path, runtime_paths: list[str]) -> list[str]:
    lines: list[str] = []
    for relative_path in runtime_paths:
        artifact_path = task_workspace / relative_path
        if artifact_path.exists():
            lines.append(f"- {_artifact_link(relative_path)}")
        else:
            lines.append(f"- `{relative_path}` (not generated)")
    return lines


def _render_inventory(deliverables: list[dict[str, Any]]) -> list[str]:
    return [
        f"- {_artifact_link(item['path'])} | category=`{item['category']}` | size=`{_format_bytes(item['size_bytes'])}` | {item['descriptor']}"
        for item in deliverables
    ]


def _collect_deliverables(task_workspace: Path, runtime_paths: list[str]) -> list[dict[str, Any]]:
    runtime_set = set(runtime_paths)
    deliverables: list[dict[str, Any]] = []
    for path in sorted(task_workspace.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(task_workspace).as_posix()
        if relative_path in runtime_set or _is_runtime_path(relative_path):
            continue
        deliverables.append(
            {
                "path": relative_path,
                "category": _categorize_artifact(path),
                "descriptor": _artifact_descriptor(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return deliverables


def _runtime_artifact_paths(task_workspace: Path, backend: Any, prompt_relative_path: str | None) -> list[str]:
    backend_name = str(backend or "executor")
    latest_task_list = _latest_versioned_file(task_workspace, "task_list_v*.md")
    runtime_paths = [
        f"{backend_name}_execution.log",
        latest_task_list or "task_list_v*.md",
        "full_history.json",
        "memory_context.json",
        "evaluation.json",
        "cost_analysis.json",
        "run_manifest.json",
        "run_result.json",
    ]
    if prompt_relative_path:
        runtime_paths.append(prompt_relative_path)
    return runtime_paths


def _latest_versioned_file(task_workspace: Path, pattern: str) -> str | None:
    candidates = list(task_workspace.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=_version_sort_key).relative_to(task_workspace).as_posix()


def _version_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_v(\d+)", path.name)
    version = int(match.group(1)) if match else -1
    return (version, path.name)


def _relative_runtime_path(raw_path: Any, task_workspace: Path) -> str | None:
    if not raw_path:
        return None
    try:
        return Path(str(raw_path)).relative_to(task_workspace).as_posix()
    except ValueError:
        return None


def _artifact_descriptor(path: Path) -> str:
    if path.suffix.lower() == ".md":
        heading = _first_markdown_heading(path)
        if heading:
            return heading

    if path.suffix.lower() == ".ipynb":
        notebook_descriptor = _notebook_descriptor(path)
        if notebook_descriptor:
            return notebook_descriptor

    if path.suffix.lower() == ".py":
        python_descriptor = _python_descriptor(path)
        if python_descriptor:
            return python_descriptor

    text_descriptor = _text_descriptor(path)
    if text_descriptor:
        return text_descriptor

    return _generic_descriptor(path)


def _first_markdown_heading(path: Path) -> str | None:
    for line in _iter_text_lines(path, max_lines=40):
        stripped = line.strip()
        if stripped.startswith("#"):
            return _normalize_descriptor(stripped.lstrip("#").strip())
    return None


def _notebook_descriptor(path: Path) -> str | None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for cell in notebook.get("cells", [])[:10]:
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue
        if cell.get("cell_type") == "markdown":
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    return _normalize_descriptor(stripped.lstrip("#").strip())
                if stripped:
                    return _normalize_descriptor(stripped)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(_COMMENT_PREFIXES):
                return _normalize_descriptor(_strip_comment_prefix(stripped))
            if stripped:
                return _normalize_descriptor(stripped)
    return None


def _python_descriptor(path: Path) -> str | None:
    lines = list(_iter_text_lines(path, max_lines=40))
    docstring = _leading_docstring(lines)
    if docstring:
        return docstring
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            return _normalize_descriptor(_strip_comment_prefix(stripped))
        if stripped and not stripped.startswith(("import ", "from ", "def ", "class ")):
            return _normalize_descriptor(stripped)
    return None


def _leading_docstring(lines: list[str]) -> str | None:
    content = "\n".join(lines)
    match = re.search(r'^[\s\r\n]*("""|\'\'\')(.+?)(\1)', content, flags=re.DOTALL)
    if not match:
        return None
    first_line = next((line.strip() for line in match.group(2).splitlines() if line.strip()), "")
    return _normalize_descriptor(first_line) if first_line else None


def _text_descriptor(path: Path) -> str | None:
    if path.suffix.lower() not in _TEXT_EXTENSIONS:
        return None
    for line in _iter_text_lines(path, max_lines=30):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(_COMMENT_PREFIXES):
            return _normalize_descriptor(_strip_comment_prefix(stripped))
        if path.suffix.lower() == ".md" and stripped.startswith("#"):
            return _normalize_descriptor(stripped.lstrip("#").strip())
        return _normalize_descriptor(stripped)
    return None


def _iter_text_lines(path: Path, max_lines: int) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return text.splitlines()[:max_lines]


def _strip_comment_prefix(text: str) -> str:
    for prefix in _COMMENT_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):].strip(" */")
    return text.strip()


def _normalize_descriptor(text: str) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", text).strip("`*_ ")
    return _truncate_text(cleaned, 140)


def _generic_descriptor(path: Path) -> str:
    category = _categorize_artifact(path)
    fallback = {
        "reports/docs": "Document artifact",
        "images": "Image artifact",
        "code/notebooks": "Code or notebook artifact",
        "tables/data": "Data artifact",
        "other outputs": "Output artifact",
    }
    return fallback[category]


def _categorize_artifact(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "images"
    if suffix in _CODE_EXTENSIONS:
        return "code/notebooks"
    if suffix in _DATA_EXTENSIONS:
        return "tables/data"
    if suffix in _REPORT_EXTENSIONS:
        return "reports/docs"
    return "other outputs"


def _artifact_link(relative_path: str) -> str:
    label = Path(relative_path).name
    return f"[{label}]({relative_path})"


def _render_inline_list(items: list[Any], task_workspace: Path) -> str:
    return "; ".join(_sanitize_text(str(item), task_workspace) for item in items)


def _format_duration(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}s"
    return "N/A"


def _format_scalar(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_role_models(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "N/A"
    return ", ".join(f"{role}={model}" for role, model in sorted(value.items()))


def _format_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.1f}{units[unit_index]}"


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _sanitize_text(text: str, task_workspace: Path) -> str:
    workspace_str = str(task_workspace)
    sanitized = text.replace(workspace_str, ".")
    parts = re.split(r"(\s+)", sanitized)
    for index, part in enumerate(parts):
        stripped = part.strip("()[]{}<>,.;:'\"")
        if stripped.startswith("/") and not stripped.startswith("//"):
            replacement = "."
            if stripped.startswith(workspace_str):
                suffix = stripped[len(workspace_str):]
                replacement = f".{suffix}" if suffix else "."
            else:
                name = Path(stripped).name
                replacement = f"./{name}" if name else "."
            parts[index] = part.replace(stripped, replacement)
    return "".join(parts)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_runtime_path(relative_path: str) -> bool:
    path = Path(relative_path)
    if any(part.startswith(".") for part in path.parts):
        return True
    if path.name in _EXCLUDED_RUNTIME_FILES:
        return True
    if path.name.startswith("memory_context_v") and path.suffix == ".json":
        return True
    if path.name.startswith("task_list_v") and path.suffix == ".md":
        return True
    if path.name.endswith("_execution.log"):
        return True
    return False
