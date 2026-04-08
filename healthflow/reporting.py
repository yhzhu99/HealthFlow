from __future__ import annotations

import csv
import json
import os
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
_EXCLUDED_RUNTIME_FILES = {"report.md"}
_COMMENT_PREFIXES = ("#", "//", "--", ";", "/*", "*")
_WHITESPACE_RE = re.compile(r"\s+")
_PATH_LIKE_RE = re.compile(r"(?<!\w)(?:\./|\.\./|/)[^\s`]+")


def generate_task_report(
    task_workspace: Path,
    *,
    runtime_dir: Path | None = None,
    report_path: Path | None = None,
) -> Path:
    runtime_dir = runtime_dir or (task_workspace / "runtime")
    sandbox_dir = task_workspace / "sandbox"
    index = _read_json(runtime_dir / "index.json")
    run_summary = _read_json(runtime_dir / "run" / "summary.json")
    trajectory = _read_json(runtime_dir / "run" / "trajectory.json")
    evaluation = _read_json(runtime_dir / "run" / "final_evaluation.json")
    cost_analysis = _read_json(runtime_dir / "run" / "costs.json")

    resolved_report_path = report_path or (runtime_dir / "report.md")
    deliverables = _collect_deliverables(sandbox_dir, resolved_report_path)
    report_markdown = _render_report(
        task_workspace=task_workspace,
        index=index,
        run_summary=run_summary,
        trajectory=trajectory,
        evaluation=evaluation,
        cost_analysis=cost_analysis,
        deliverables=deliverables,
    )

    report_path = resolved_report_path
    temp_path = report_path.with_suffix(".md.tmp")
    temp_path.write_text(report_markdown, encoding="utf-8")
    temp_path.replace(report_path)
    return report_path


def _render_report(
    task_workspace: Path,
    index: dict[str, Any],
    run_summary: dict[str, Any],
    trajectory: dict[str, Any],
    evaluation: dict[str, Any],
    cost_analysis: dict[str, Any],
    deliverables: list[dict[str, Any]],
) -> str:
    attempts = trajectory.get("attempts") or []
    latest_attempt = attempts[-1] if attempts else {}
    final_status = str(run_summary.get("evaluation_status") or evaluation.get("status") or "unknown")
    final_score = run_summary.get("evaluation_score")
    success = bool(run_summary.get("success", False))
    task_id = str(index.get("task_id") or trajectory.get("task_id") or task_workspace.name)
    execution_time = index.get("execution_time")
    user_request = _sanitize_text(str(index.get("user_request") or trajectory.get("user_request") or ""), task_workspace)
    summary = _sanitize_text(str(run_summary.get("final_summary") or "No final summary was recorded."), task_workspace)
    feedback = _sanitize_text(str(evaluation.get("feedback") or "No evaluator feedback was recorded."), task_workspace)
    reasoning = _sanitize_text(str(evaluation.get("reasoning") or "No evaluator reasoning was recorded."), task_workspace)
    answer = _sanitize_text(
        str(run_summary.get("answer") or latest_attempt.get("artifacts", {}).get("generated_answer") or ""),
        task_workspace,
    )
    data_profile = trajectory.get("data_profile") or {}
    risk_findings = trajectory.get("risk_findings") or []
    plan = latest_attempt.get("plan") or {}
    paper_title = _build_paper_title(plan.get("objective"), user_request)
    primary_result = _primary_result_context(deliverables, answer, task_workspace)

    lines: list[str] = ["# HealthFlow Report", "", f"## {paper_title}"]
    lines.extend(
        _section(
            "Abstract",
            [
                _build_abstract(
                    user_request=user_request,
                    data_profile=data_profile,
                    success=success,
                    final_status=final_status,
                    execution_time=execution_time,
                    primary_finding=primary_result.get("finding"),
                    primary_artifact=primary_result.get("artifact_path"),
                )
            ],
        )
    )
    lines.extend(
        _section(
            "Problem",
            _build_problem_section(
                user_request=user_request,
                plan=plan,
                data_profile=data_profile,
                success=success,
                final_status=final_status,
                attempt_count=len(attempts),
            ),
        )
    )
    lines.extend(
        _section(
            "HealthFlow Analysis",
            _build_analysis_section(
                task_workspace=task_workspace,
                plan=plan,
                risk_findings=risk_findings,
                evaluation=evaluation,
                trajectory=trajectory,
            ),
        )
    )
    lines.extend(
        _section(
            "Execution",
            _build_execution_section(
                task_workspace=task_workspace,
                plan=plan,
                latest_attempt=latest_attempt,
                execution_time=execution_time,
                feedback=feedback,
                reasoning=reasoning,
            ),
        )
    )
    lines.extend(
        _section(
            "Results",
            _build_results_section(
                task_workspace=task_workspace,
                deliverables=deliverables,
                run_summary=run_summary,
                evaluation=evaluation,
                success=success,
                final_status=final_status,
                primary_result=primary_result,
            ),
        )
    )
    lines.extend(
        _section(
            "Conclusion",
            _build_conclusion_section(
                task_workspace=task_workspace,
                success=success,
                final_status=final_status,
                summary=summary,
                feedback=feedback,
                primary_finding=str(primary_result.get("finding") or ""),
                evaluation=evaluation,
            ),
        )
    )
    lines.extend(
        _section(
            "Appendix: Reproducibility and Audit",
            _build_appendix_section(
                task_workspace=task_workspace,
                task_id=task_id,
                final_status=final_status,
                final_score=final_score,
                success=success,
                execution_time=execution_time,
                index=index,
                run_summary=run_summary,
                trajectory=trajectory,
                cost_analysis=cost_analysis,
                deliverables=deliverables,
            ),
        )
    )
    return "\n".join(_trim_blank_edges(lines)).strip() + "\n"


def _section(title: str, body_lines: list[str]) -> list[str]:
    return ["", f"## {title}", "", *_trim_blank_edges(body_lines)]


def _build_paper_title(objective: Any, user_request: str) -> str:
    text = str(objective or user_request or "Analytical Task Report").strip()
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    replacements = (
        ("analyze ", "Analysis of "),
        ("summarize ", "Summary of "),
        ("compare ", "Comparison of "),
        ("evaluate ", "Evaluation of "),
        ("build ", "Study of "),
        ("create ", "Analysis of "),
    )
    lower_text = text.lower()
    for prefix, replacement in replacements:
        if lower_text.startswith(prefix):
            text = replacement + text[len(prefix) :]
            break
    text = re.sub(r"\s+and\s+(?:create|generate|save|write|produce)\b.+$", "", text, flags=re.IGNORECASE)
    text = text[:1].upper() + text[1:] if text else "Analytical Task Report"
    return _truncate_text(text, 120)


def _build_abstract(
    *,
    user_request: str,
    data_profile: dict[str, Any],
    success: bool,
    final_status: str,
    execution_time: Any,
    primary_finding: str | None,
    primary_artifact: str | None,
) -> str:
    objective = user_request or "an analytical request recorded in the runtime metadata"
    dataset_clause = _dataset_scope_sentence(data_profile)
    outcome_clause = (
        f"The run completed successfully in `{_format_duration(execution_time)}`."
        if success
        else f"The run ended with status `{final_status}` in `{_format_duration(execution_time)}`."
    )
    finding_clause = primary_finding or "The produced artifacts preserve the main analytical evidence for inspection."
    artifact_clause = (
        f"The primary quantitative evidence is available in {_artifact_link(primary_artifact)}."
        if primary_artifact
        else "The quantitative evidence is summarized in the generated workspace artifacts."
    )
    return " ".join(
        sentence
        for sentence in [
            f"This report addresses the request to {objective.rstrip('.')}.".replace("..", "."),
            dataset_clause,
            outcome_clause,
            finding_clause,
            artifact_clause,
        ]
        if sentence
    )


def _build_problem_section(
    *,
    user_request: str,
    plan: dict[str, Any],
    data_profile: dict[str, Any],
    success: bool,
    final_status: str,
    attempt_count: int,
) -> list[str]:
    objective = str(plan.get("objective") or user_request or "No objective was recorded.").strip()
    lines = [
        f"The analytical objective was to {objective.rstrip('.')}.",
        (
            f"HealthFlow resolved the task in {attempt_count} attempt(s) with final status `{final_status}`."
            if not success
            else f"HealthFlow resolved the task in {attempt_count} attempt(s) and produced a complete analytical package."
        ),
    ]
    overview_rows = _data_overview_rows(data_profile)
    if overview_rows:
        lines.extend(["", "### Study Inputs", "", *_render_table(["Field", "Value"], overview_rows)])
    return lines


def _build_analysis_section(
    *,
    task_workspace: Path,
    plan: dict[str, Any],
    risk_findings: list[dict[str, Any]],
    evaluation: dict[str, Any],
    trajectory: dict[str, Any],
) -> list[str]:
    assumptions = [_sanitize_text(str(item), task_workspace) for item in (plan.get("assumptions_to_check") or []) if str(item).strip()]
    insights = [_sanitize_text(str(item), task_workspace) for item in (evaluation.get("memory_worthy_insights") or []) if str(item).strip()]
    lines: list[str] = []
    if assumptions:
        assumption_text = "; ".join(assumptions[:4])
        lines.append(f"HealthFlow first grounded the task by checking the following assumptions: {assumption_text}.")
    else:
        lines.append("HealthFlow grounded the task by checking the recorded inputs, runtime context, and requested analytical objective.")
    risk_rows = _risk_overview_rows(risk_findings)
    if risk_rows:
        lines.extend(["", "### Data Risks and Safeguards", "", *_render_table(["Severity", "Risk", "Why It Mattered"], risk_rows)])
    elif trajectory.get("data_profile", {}).get("domain_focus") == "ehr":
        lines.append(
            "Because the request involved EHR data, the analysis path emphasized schema verification and task-relevant safeguards before final reporting."
        )
    if insights:
        lines.extend(["", "### Key Analytical Insights", ""])
        lines.extend(f"- {item}" for item in insights)
    return lines


def _build_execution_section(
    *,
    task_workspace: Path,
    plan: dict[str, Any],
    latest_attempt: dict[str, Any],
    execution_time: Any,
    feedback: str,
    reasoning: str,
) -> list[str]:
    execution = latest_attempt.get("execution") or {}
    lines: list[str] = []
    duration = execution.get("duration_seconds")
    if duration is not None:
        lines.append(
            f"The execution layer completed the main attempt in `{_format_duration(duration)}` within a total end-to-end runtime of `{_format_duration(execution_time)}`."
        )
    else:
        lines.append(f"The execution layer completed the recorded workflow in `{_format_duration(execution_time)}`.")
    method_summary = _best_execution_summary(reasoning, feedback, task_workspace)
    if method_summary:
        lines.append(method_summary)
    steps = [_sanitize_text(str(step), task_workspace) for step in (plan.get("recommended_steps") or []) if str(step).strip()]
    if steps:
        lines.extend(["", "### Workflow", ""])
        lines.extend(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    avoidances = [_sanitize_text(str(item), task_workspace) for item in (plan.get("avoidances") or []) if str(item).strip()]
    if avoidances:
        lines.extend(["", "### Safeguards", ""])
        lines.extend(f"- {item}" for item in avoidances)
    return lines


def _build_results_section(
    *,
    task_workspace: Path,
    deliverables: list[dict[str, Any]],
    run_summary: dict[str, Any],
    evaluation: dict[str, Any],
    success: bool,
    final_status: str,
    primary_result: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    answer = _sanitize_text(str(run_summary.get("answer") or ""), task_workspace)
    summary = _sanitize_text(str(run_summary.get("final_summary") or ""), task_workspace)
    feedback = _sanitize_text(str(evaluation.get("feedback") or ""), task_workspace)
    intro = primary_result.get("finding") or _best_results_narrative(answer, summary, feedback)
    if intro:
        lines.append(intro)
    elif success:
        lines.append("The run produced user-facing analytical artifacts, but no concise narrative summary was available in the runtime metadata.")
    else:
        lines.append(f"The run ended with status `{final_status}`. The current artifacts capture partial evidence, but the evaluator did not consider the task fully satisfied.")

    structured_blocks = _structured_result_blocks(deliverables, task_workspace)
    if structured_blocks:
        lines.extend(["", "### Quantitative Findings"])
        for block in structured_blocks:
            lines.extend(["", f"#### {_artifact_link(block['path'])}", ""])
            lines.extend(block["lines"])

    document_blocks = _document_result_blocks(deliverables, task_workspace)
    if document_blocks:
        lines.extend(["", "### Narrative Findings"])
        for block in document_blocks:
            lines.extend(["", f"#### {_artifact_link(block['path'])}", "", block["excerpt"]])

    figure_lines = _figure_blocks(deliverables)
    if figure_lines:
        lines.extend(["", "### Figures", "", *figure_lines])

    artifact_rows = [
        [_artifact_link(item["path"]), item["category"], item["descriptor"]]
        for item in deliverables
    ]
    if artifact_rows:
        lines.extend(["", "### Supporting Outputs", "", *_render_table(["Artifact", "Type", "Role"], artifact_rows)])
    else:
        lines.extend(["", "No non-runtime deliverables were found in the task workspace."])
    return lines


def _build_conclusion_section(
    *,
    task_workspace: Path,
    success: bool,
    final_status: str,
    summary: str,
    feedback: str,
    primary_finding: str,
    evaluation: dict[str, Any],
) -> list[str]:
    repair_instructions = [_sanitize_text(str(item), task_workspace) for item in (evaluation.get("repair_instructions") or []) if str(item).strip()]
    lines: list[str] = []
    if success:
        closing = primary_finding or summary or feedback
        lines.append(
            " ".join(
                sentence
                for sentence in [
                    "HealthFlow produced a readable, inspectable analytical package for the requested task.",
                    closing,
                ]
                if sentence
            )
        )
    else:
        lines.append(
            " ".join(
                sentence
                for sentence in [
                    f"The run did not achieve a final success state and ended with status `{final_status}`.",
                    feedback or summary,
                ]
                if sentence
            )
        )
    if repair_instructions:
        lines.append(f"The main follow-up action is: {repair_instructions[0]}.")
    return lines


def _build_appendix_section(
    *,
    task_workspace: Path,
    task_id: str,
    final_status: str,
    final_score: Any,
    success: bool,
    execution_time: Any,
    index: dict[str, Any],
    run_summary: dict[str, Any],
    trajectory: dict[str, Any],
    cost_analysis: dict[str, Any],
    deliverables: list[dict[str, Any]],
) -> list[str]:
    attempts = trajectory.get("attempts") or []
    latest_attempt = attempts[-1] if attempts else {}
    snapshot_rows = [
        ["Task ID", f"`{task_id}`"],
        ["Outcome", f"`{final_status}`"],
        ["Success", f"`{success}`"],
        ["Evaluation Score", f"`{_format_scalar(final_score)}`"],
        ["Attempts", f"`{len(attempts)}`"],
        ["Execution Time", f"`{_format_duration(execution_time)}`"],
    ]
    environment_rows = [
        ["Backend", f"`{_format_scalar(index.get('backend'))}`"],
        ["Executor Model", f"`{_format_scalar((index.get('models') or {}).get('executor'))}`"],
        ["Planner Model", f"`{_format_scalar((index.get('models') or {}).get('planner'))}`"],
        ["Memory Write Policy", f"`{_format_scalar(index.get('memory_write_policy'))}`"],
    ]
    execution_environment = index.get("execution_environment") or {}
    if execution_environment:
        environment_rows.extend(
            [
                ["Python Version", f"`{_format_scalar(execution_environment.get('python_version'))}`"],
                ["Package Manager", f"`{_format_scalar(execution_environment.get('package_manager'))}`"],
                ["Run Prefix", f"`{_format_scalar(execution_environment.get('run_prefix'))}`"],
            ]
        )
    cost_rows = _cost_rows(cost_analysis)
    audit_rows = _audit_rows(run_summary, latest_attempt)
    output_rows = [[_artifact_link(item["path"]), item["descriptor"]] for item in deliverables]

    lines: list[str] = ["### Task Snapshot", "", *_render_table(["Field", "Value"], snapshot_rows)]
    lines.extend(["", "### Reproducibility Notes", "", *_render_table(["Field", "Value"], environment_rows)])
    if cost_rows:
        lines.extend(["", "### Runtime Cost Summary", "", *_render_table(["Field", "Value"], cost_rows)])
    if audit_rows:
        lines.extend(["", "### Audit Trail", "", *_render_table(["Artifact", "Purpose"], audit_rows)])
    if output_rows:
        lines.extend(["", "### Generated Outputs", "", *_render_table(["Artifact", "Description"], output_rows)])
    return lines


def _primary_result_context(
    deliverables: list[dict[str, Any]],
    answer: str,
    task_workspace: Path,
) -> dict[str, Any]:
    primary_artifact = _select_primary_result_artifact(deliverables)
    rows = _structured_rows_for_artifact(primary_artifact, task_workspace) if primary_artifact else []
    document_excerpt = _first_document_excerpt(deliverables, task_workspace)
    finding = (
        _interpretive_sentence(answer)
        or _interpretive_sentence(document_excerpt or "")
        or _table_summary_sentence(rows)
    )
    return {
        "finding": finding,
        "artifact_path": primary_artifact["path"] if primary_artifact else None,
        "rows": rows,
        "document_excerpt": document_excerpt,
    }


def _dataset_scope_sentence(data_profile: dict[str, Any]) -> str:
    if not data_profile:
        return ""
    row_count = data_profile.get("row_count")
    schemas = data_profile.get("schemas") or []
    file_names = [Path(str(schema.get("file_name"))).name for schema in schemas if schema.get("file_name")]
    files_text = ", ".join(file_names)
    clauses: list[str] = []
    if files_text:
        clauses.append(f"Input data included `{files_text}`")
    if row_count:
        clauses.append(f"with `{row_count}` recorded rows")
    patient_ids = data_profile.get("patient_id_columns") or []
    if patient_ids:
        clauses.append(f"and patient identifiers in `{', '.join(patient_ids)}`")
    if not clauses:
        return ""
    return _sentence_from_clause(", ".join(clauses))


def _data_overview_rows(data_profile: dict[str, Any]) -> list[list[str]]:
    if not data_profile:
        return []
    schemas = data_profile.get("schemas") or []
    file_names = [Path(str(schema.get("file_name"))).name for schema in schemas if schema.get("file_name")]
    rows: list[list[str]] = []
    if file_names:
        rows.append(["Input Datasets", ", ".join(f"`{name}`" for name in file_names)])
    if data_profile.get("domain_focus"):
        rows.append(["Domain Focus", str(data_profile.get("domain_focus"))])
    if data_profile.get("task_family"):
        rows.append(["Task Family", str(data_profile.get("task_family"))])
    if data_profile.get("row_count") not in (None, ""):
        rows.append(["Row Count", f"`{data_profile.get('row_count')}`"])
    patient_ids = data_profile.get("patient_id_columns") or []
    if patient_ids:
        rows.append(["Patient Identifier Columns", ", ".join(f"`{col}`" for col in patient_ids)])
    targets = data_profile.get("target_columns") or []
    if targets:
        rows.append(["Target Columns", ", ".join(f"`{col}`" for col in targets)])
    time_columns = data_profile.get("time_columns") or []
    if time_columns:
        rows.append(["Time Columns", ", ".join(f"`{col}`" for col in time_columns)])
    return rows


def _risk_overview_rows(risk_findings: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for finding in risk_findings:
        rows.append(
            [
                str(finding.get("severity") or "info").title(),
                str(finding.get("category") or "unspecified").replace("_", " "),
                str(finding.get("message") or "").strip(),
            ]
        )
    return rows


def _best_execution_summary(reasoning: str, feedback: str, task_workspace: Path) -> str:
    text = reasoning or feedback
    cleaned = _sanitize_text(text, task_workspace)
    sentence = _interpretive_sentence(cleaned)
    if sentence:
        return sentence
    plain = _plain_text_excerpt(cleaned, max_chars=320)
    return plain if plain else ""


def _best_results_narrative(answer: str, summary: str, feedback: str) -> str:
    for candidate in (answer, summary, feedback):
        sentence = _interpretive_sentence(candidate)
        if sentence:
            return sentence
    for candidate in (answer, summary, feedback):
        excerpt = _plain_text_excerpt(candidate, max_chars=360)
        if excerpt:
            return excerpt
    return ""


def _structured_result_blocks(deliverables: list[dict[str, Any]], task_workspace: Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for item in deliverables:
        if item["category"] != "tables/data":
            continue
        rows = _structured_rows_for_artifact(item, task_workspace)
        if not rows:
            continue
        blocks.append({"path": item["path"], "lines": _render_table(["Metric", "Value"], rows)})
    return blocks


def _document_result_blocks(deliverables: list[dict[str, Any]], task_workspace: Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for item in deliverables:
        if item["category"] != "reports/docs":
            continue
        excerpt = _document_excerpt(item["source_path"], task_workspace)
        if not excerpt:
            continue
        blocks.append({"path": item["path"], "excerpt": excerpt})
    return blocks


def _figure_blocks(deliverables: list[dict[str, Any]]) -> list[str]:
    figures = [item for item in deliverables if item["category"] == "images"]
    lines: list[str] = []
    for index, item in enumerate(figures, start=1):
        caption = _figure_caption(item)
        lines.append(f"**Figure {index}.** {caption}")
        lines.append(f"![{Path(item['path']).name}]({item['path']})")
        lines.append("")
    return _trim_blank_edges(lines)


def _figure_caption(item: dict[str, Any]) -> str:
    descriptor = item.get("descriptor") or ""
    if descriptor and descriptor != "Image artifact":
        return descriptor.rstrip(".")
    stem = Path(item["path"]).stem.replace("_", " ")
    return stem[:1].upper() + stem[1:]


def _select_primary_result_artifact(deliverables: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [item for item in deliverables if item["category"] == "tables/data"]
    if not candidates:
        return None
    return max(candidates, key=_result_artifact_score)


def _result_artifact_score(item: dict[str, Any]) -> tuple[int, int]:
    name = Path(item["path"]).name.lower()
    score = 0
    for token, weight in (
        ("metric", 8),
        ("metrics", 8),
        ("stat", 7),
        ("stats", 7),
        ("result", 6),
        ("results", 6),
        ("summary", 4),
        ("performance", 5),
        ("evaluation", 5),
    ):
        if token in name:
            score += weight
    if name.endswith(".json"):
        score += 3
    if name.endswith(".csv") or name.endswith(".tsv"):
        score += 2
    return (score, -int(item.get("size_bytes") or 0))


def _structured_rows_for_artifact(item: dict[str, Any] | None, task_workspace: Path) -> list[list[str]]:
    if not item:
        return []
    path = item["source_path"]
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _json_rows(path, task_workspace)
    if suffix in {".csv", ".tsv"}:
        return _tabular_rows(path, task_workspace)
    return []


def _json_rows(path: Path, task_workspace: Path) -> list[list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    flattened = _flatten_json_scalars(payload)
    if not flattened or len(flattened) > 24:
        return []
    return [[_humanize_metric_key(key), _format_display_value(value, task_workspace)] for key, value in flattened]


def _flatten_json_scalars(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_json_scalars(nested_value, nested_prefix))
        return rows
    if isinstance(value, list):
        if not value:
            return [(prefix, "[]")] if prefix else []
        if all(_is_scalar(item) for item in value) and len(value) <= 8:
            return [(prefix, "; ".join(_format_scalar(item) for item in value))]
        return []
    if _is_scalar(value):
        return [(prefix, value)] if prefix else []
    return []


def _tabular_rows(path: Path, task_workspace: Path) -> list[list[str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with open(path, "r", encoding="utf-8") as handle:
            reader = list(csv.reader(handle, delimiter=delimiter))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    if len(reader) < 2:
        return []
    header = reader[0]
    body = reader[1:]
    if not header or len(header) > 6 or len(body) > 12:
        return []
    if not all(len(row) == len(header) for row in body):
        return []
    if "metric" in {column.strip().lower() for column in header[:2]}:
        return [[_sanitize_text(str(row[0]), task_workspace), _sanitize_text(str(row[1]), task_workspace)] for row in body if len(row) >= 2]
    return []


def _first_document_excerpt(deliverables: list[dict[str, Any]], task_workspace: Path) -> str | None:
    for item in deliverables:
        if item["category"] != "reports/docs":
            continue
        excerpt = _document_excerpt(item["source_path"], task_workspace)
        if excerpt:
            return excerpt
    return None


def _document_excerpt(path: Path, task_workspace: Path) -> str | None:
    if path.suffix.lower() not in {".md", ".txt", ".rst"}:
        return None
    lines = _iter_text_lines(path, max_lines=120)
    snippets: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("==="):
            continue
        if stripped.isupper() and len(stripped) < 48:
            continue
        lowered = stripped.lower()
        if lowered.startswith("analysis date:"):
            continue
        if lowered.startswith("generated artifacts:"):
            break
        if lowered.startswith("summary statistics:"):
            continue
        if lowered.startswith("gender coding:"):
            continue
        if re.match(r"^\d+\.\s+\S", stripped):
            continue
        cleaned = _sanitize_text(stripped, task_workspace)
        snippets.append(cleaned)
        if len(snippets) >= 4:
            break
    if not snippets:
        return None
    excerpt = " ".join(snippets)
    return _plain_text_excerpt(excerpt, max_chars=360)


def _interpretive_sentence(text: str) -> str | None:
    if not text:
        return None
    sentences = _split_sentences(_plain_text_excerpt(text, max_chars=500))
    weighted_keywords = {
        "show": 3,
        "shows": 3,
        "indicate": 3,
        "indicates": 3,
        "suggest": 3,
        "suggests": 3,
        "demonstrate": 3,
        "demonstrates": 3,
        "predominance": 4,
        "association": 4,
        "difference": 3,
        "improve": 3,
        "improves": 3,
        "increase": 2,
        "decrease": 2,
        "ratio": 2,
        "distribution": 1,
        "performance": 2,
    }
    best_sentence = ""
    best_score = -1
    for sentence in sentences:
        lowered = sentence.lower()
        score = sum(weight for keyword, weight in weighted_keywords.items() if keyword in lowered)
        if score > best_score:
            best_sentence = sentence
            best_score = score
    if best_score > 0:
        return best_sentence
    return sentences[0] if sentences else None


def _table_summary_sentence(rows: list[list[str]]) -> str | None:
    if not rows:
        return None
    summary_pairs = [f"{label} = {value}" for label, value in rows[:4]]
    if not summary_pairs:
        return None
    return f"Key quantitative findings included {', '.join(summary_pairs)}."


def _sentence_from_clause(clause: str) -> str:
    cleaned = clause.strip().rstrip(".")
    return f"{cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned}."


def _plain_text_excerpt(text: str, max_chars: int) -> str:
    if not text:
        return ""
    lines: list[str] = []
    skip_list_mode = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            skip_list_mode = False
            continue
        lowered = stripped.lower()
        if "generated artifacts" in lowered or "key safeguards" in lowered:
            skip_list_mode = True
            continue
        if stripped.startswith("#"):
            continue
        if skip_list_mode and (stripped.startswith("- ") or re.match(r"^\d+\.\s+", stripped)):
            continue
        if skip_list_mode:
            skip_list_mode = False
        if stripped.startswith("- "):
            stripped = stripped[2:]
        if re.match(r"^\d+\.\s+", stripped) and "`" in stripped:
            continue
        stripped = stripped.replace("**", "").replace("__", "").replace("`", "")
        if stripped.lower().startswith("here are the key findings"):
            continue
        if stripped[-1] not in ".!?:":
            stripped = f"{stripped}."
        lines.append(stripped)
    combined = " ".join(lines)
    combined = re.sub(r"\bHere are the key findings:\s*", "", combined, flags=re.IGNORECASE)
    combined = _WHITESPACE_RE.sub(" ", combined).strip()
    return _truncate_text(_normalize_report_voice(combined), max_chars)


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = list(row) + [""] * (len(headers) - len(row))
        table.append("| " + " | ".join(_escape_table_cell(cell) for cell in padded[: len(headers)]) + " |")
    return table


def _escape_table_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _cost_rows(cost_analysis: dict[str, Any]) -> list[list[str]]:
    run_total = cost_analysis.get("run_total") or {}
    rows: list[list[str]] = []
    for label, key in (
        ("Total Estimated Cost (USD)", "total_estimated_cost_usd"),
        ("LLM Estimated Cost (USD)", "llm_estimated_cost_usd"),
        ("Executor Estimated Cost (USD)", "executor_estimated_cost_usd"),
    ):
        value = run_total.get(key)
        if value is not None:
            rows.append([label, f"`{_format_scalar(value)}`"])
    return rows


def _audit_rows(run_summary: dict[str, Any], latest_attempt: dict[str, Any]) -> list[list[str]]:
    paths = run_summary.get("paths") or {}
    default_paths = {
        "trajectory": "run/trajectory.json",
        "final_evaluation": "run/final_evaluation.json",
        "costs": "run/costs.json",
        "task_state": "run/task_state.json",
    }
    audit_rows: list[list[str]] = []
    for key, purpose in (
        ("trajectory", "Complete run trajectory"),
        ("final_evaluation", "Evaluator verdict and reasoning"),
        ("costs", "Runtime cost summary"),
        ("task_state", "Detected data profile and risk state"),
        ("direct_response", "Direct-response payload when no execution attempt was needed"),
    ):
        relative_path = _runtime_relative_path(paths.get(key) or default_paths.get(key))
        if relative_path:
            audit_rows.append([_artifact_link(relative_path), purpose])
    latest_paths = latest_attempt.get("paths") or {}
    planner_plan = _runtime_relative_path(((latest_paths.get("planner") or {}).get("plan")) if latest_paths else None)
    executor_log = _runtime_relative_path(((latest_paths.get("executor") or {}).get("combined_log")) if latest_paths else None)
    if planner_plan:
        audit_rows.append([_artifact_link(planner_plan), "Planner output for the latest attempt"])
    if executor_log:
        audit_rows.append([_artifact_link(executor_log), "Executor log for the latest attempt"])
    return audit_rows


def _collect_deliverables(sandbox_dir: Path, report_path: Path) -> list[dict[str, Any]]:
    deliverables: list[dict[str, Any]] = []
    if not sandbox_dir.exists():
        return deliverables
    for path in sorted(sandbox_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(sandbox_dir).as_posix()
        if _is_runtime_path(relative_path):
            continue
        report_relative_path = os.path.relpath(path, report_path.parent).replace("\\", "/")
        deliverables.append(
            {
                "path": report_relative_path,
                "sandbox_relative_path": relative_path,
                "source_path": path,
                "category": _categorize_artifact(path),
                "descriptor": _artifact_descriptor(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return deliverables


def _artifact_descriptor(path: Path) -> str:
    if path.suffix.lower() == ".md":
        heading = _first_markdown_heading(path)
        if heading:
            return heading
    if path.suffix.lower() == ".json":
        json_descriptor = _json_descriptor(path)
        if json_descriptor:
            return json_descriptor
    if path.suffix.lower() in {".csv", ".tsv"}:
        csv_descriptor = _csv_descriptor(path)
        if csv_descriptor:
            return csv_descriptor
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
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
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
        if stripped.startswith("==="):
            continue
        if stripped.lower().startswith(("summary statistics:", "generated artifacts:", "analysis date:")):
            continue
        if path.suffix.lower() == ".json" and stripped in {"{", "}", "[", "]"}:
            continue
        if stripped.startswith(_COMMENT_PREFIXES):
            return _normalize_descriptor(_strip_comment_prefix(stripped))
        if path.suffix.lower() == ".md" and stripped.startswith("#"):
            return _normalize_descriptor(stripped.lstrip("#").strip())
        return _normalize_descriptor(stripped)
    return None


def _json_descriptor(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and payload:
        keys = ", ".join(str(key) for key in list(payload.keys())[:4])
        return _normalize_descriptor(f"JSON fields: {keys}")
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        keys = ", ".join(str(key) for key in list(payload[0].keys())[:4])
        return _normalize_descriptor(f"JSON table fields: {keys}")
    return None


def _csv_descriptor(path: Path) -> str | None:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, [])
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration):
        return None
    if not header:
        return None
    return _normalize_descriptor(f"Columns: {', '.join(header[:6])}")


def _iter_text_lines(path: Path, max_lines: int) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return text.splitlines()[:max_lines]


def _strip_comment_prefix(text: str) -> str:
    for prefix in _COMMENT_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :].strip(" */")
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


def _humanize_metric_key(key: str) -> str:
    parts = [part for part in key.split(".") if part]
    if not parts:
        return "Metric"
    rendered = []
    for index, part in enumerate(parts):
        if index > 0 and part.isdigit():
            rendered.append(part)
            continue
        rendered.append(part.replace("_", " ").strip().title())
    return " / ".join(rendered)


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


def _format_display_value(value: Any, task_workspace: Path) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return _sanitize_text(str(value), task_workspace)


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
                suffix = stripped[len(workspace_str) :]
                replacement = f".{suffix}" if suffix else "."
            else:
                name = Path(stripped).name
                replacement = f"./{name}" if name else "."
            parts[index] = part.replace(stripped, replacement)
    return "".join(parts)


def _normalize_report_voice(text: str) -> str:
    normalized = text
    replacements = (
        (r"\bI've\b", "HealthFlow has"),
        (r"\bI have\b", "HealthFlow has"),
        (r"\bI'm\b", "HealthFlow is"),
        (r"\bI'll\b", "HealthFlow will"),
        (r"\bI\b", "HealthFlow"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def _runtime_relative_path(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value)
    if text.startswith("runtime/"):
        return text[len("runtime/") :]
    return text


def _trim_blank_edges(lines: list[str]) -> list[str]:
    trimmed = list(lines)
    while trimmed and not trimmed[0].strip():
        trimmed.pop(0)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return trimmed


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_runtime_path(relative_path: str) -> bool:
    path = Path(relative_path)
    if any(part.startswith(".") for part in path.parts):
        return True
    if path.name in _EXCLUDED_RUNTIME_FILES:
        return True
    if path.parts and path.parts[0] == ".healthflow_pi_agent":
        return True
    return False


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))
