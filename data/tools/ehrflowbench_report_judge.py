"""Reference-based LLM judge for EHRFlowBench ``report_generation`` submissions.

Unlike ``medagentboard_llm_eval`` which compares per-artifact reference/submission
summaries, EHRFlowBench scores the generated report itself against the released
reference report. Both reports are attached to the judge request as PDFs, so a
markdown-only submission or reference is rendered to PDF first.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DEFAULT_JUDGE_LLM = "openai/gpt-5.4"
# Rubric anchor: "3 (Acceptable): a competent, complete submission that meets
# baseline expectations."
DEFAULT_PASS_THRESHOLD = 3.0
DIMENSION_NAMES = (
    "method_soundness",
    "presentation_quality",
    "artifact_generation",
    "overall_score",
)
SCORE_MIN = 1
SCORE_MAX = 5
SUBMISSION_REPORT_NAMES = ("report.pdf", "report.md")
PAGE_MARGIN = 0.65 * inch


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
    return submission_root / "ehrflowbench.eval.json"


def guess_media_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".md", ".txt"}:
        return "text"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        return "image"
    if suffix == ".parquet":
        return "parquet"
    return "binary"


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

    return JudgeConfig(
        llm_key=str(candidate_key),
        api_key=str(api_key),
        base_url=str(raw["base_url"]) if raw.get("base_url") is not None else None,
        model_name=str(raw.get("model_name") or candidate_key),
    )


def load_manifest_outputs(
    benchmark_root: Path,
    row: dict[str, Any],
) -> tuple[dict[str, Any], list[RequiredOutput]]:
    manifest_path = benchmark_root / str(row["reference_answer"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "required_outputs" in manifest:
        required = [
            RequiredOutput(
                file_name=str(item["file_name"]),
                reference_path=str(item["reference_path"]),
                media_type=str(item.get("media_type") or guess_media_type(str(item["file_name"]))),
            )
            for item in manifest["required_outputs"]
        ]
        return manifest, required

    primary_outputs = manifest.get("primary_outputs", manifest.get("deliverables", []))
    required = [
        RequiredOutput(
            file_name=Path(str(item)).name,
            reference_path=str(item),
            media_type=guess_media_type(str(item)),
        )
        for item in primary_outputs
    ]
    if not required:
        raise ValueError(f"answer manifest does not declare any outputs: {manifest_path}")
    return manifest, required


def find_report_reference_path(required_outputs: list[RequiredOutput], *, qid: Any) -> str:
    for item in required_outputs:
        if Path(item.file_name).stem == "report":
            return item.reference_path
    raise FileNotFoundError(f"qid {qid}: the answer manifest does not declare a report output")


def find_submission_report(submission_root: Path) -> Path | None:
    for name in SUBMISSION_REPORT_NAMES:
        candidate = submission_root / name
        if candidate.exists():
            return candidate
    return None


def _resolve_font_names() -> tuple[str, str, str]:
    """Prefer a bundled CJK-capable CID font, fall back to the built-in Latin fonts."""
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    except Exception:  # noqa: BLE001 - font support varies by reportlab build
        return "Helvetica", "Helvetica-Bold", "Courier"
    return "STSong-Light", "STSong-Light", "Courier"


def _build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    body_font, heading_font, code_font = _resolve_font_names()
    return {
        "body": ParagraphStyle(
            "EhrBody", parent=sample["BodyText"], fontName=body_font, fontSize=10.5, leading=14, spaceAfter=6
        ),
        "h1": ParagraphStyle(
            "EhrH1", parent=sample["Heading1"], fontName=heading_font, fontSize=18, leading=22, spaceAfter=8
        ),
        "h2": ParagraphStyle(
            "EhrH2", parent=sample["Heading2"], fontName=heading_font, fontSize=14, leading=18, spaceAfter=6
        ),
        "h3": ParagraphStyle(
            "EhrH3", parent=sample["Heading3"], fontName=heading_font, fontSize=12, leading=15, spaceAfter=4
        ),
        "list": ParagraphStyle(
            "EhrList",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=10.5,
            leading=14,
            leftIndent=12,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "EhrCode",
            parent=sample["BodyText"],
            fontName=code_font,
            fontSize=8.5,
            leading=10,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=4,
            spaceAfter=6,
        ),
        "table": ParagraphStyle(
            "EhrTable", parent=sample["BodyText"], fontName=body_font, fontSize=8.5, leading=10
        ),
    }


def _inline_text(text: str) -> str:
    value = escape(text)
    value = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", value)
    value = re.sub(r"\*\*\*([^*]+)\*\*\*", r"<b><i>\1</i></b>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    value = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', value)
    return value


def _is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _build_table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    width = (letter[0] - 2 * PAGE_MARGIN) / max(len(rows[0]), 1)
    table = Table(
        [[Paragraph(_inline_text(cell), styles["table"]) for cell in row] for row in rows],
        colWidths=[width] * len(rows[0]),
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _build_image(markdown_path: Path, target: str) -> Any:
    path = (markdown_path.parent / target).resolve()
    if not path.exists():
        return Paragraph(f"[missing image: {escape(target)}]", _build_styles()["body"])
    max_width = letter[0] - 2 * PAGE_MARGIN
    return Image(str(path), width=max_width, height=max_width * 0.6, kind="proportional")


def markdown_to_flowables(markdown_path: Path) -> list[Any]:
    styles = _build_styles()
    lines = markdown_path.read_text(encoding="utf-8", errors="replace").splitlines()
    flowables: list[Any] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            flowables.append(Preformatted("\n".join(code_lines), styles["code"]))
            continue

        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            flowables.append(_build_image(markdown_path, image_match.group(2)))
            flowables.append(Spacer(1, 0.12 * inch))
            index += 1
            continue

        if "|" in stripped and index + 1 < len(lines) and _is_table_separator(lines[index + 1]):
            rows = [_table_row(lines[index]), _table_row(lines[index + 2])]
            index += 3
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(_table_row(lines[index]))
                index += 1
            flowables.append(_build_table(rows, styles))
            flowables.append(Spacer(1, 0.12 * inch))
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            style = styles["h1"] if level <= 1 else styles["h2"] if level == 2 else styles["h3"]
            flowables.append(Paragraph(_inline_text(heading.group(2)), style))
            index += 1
            continue

        bullet = re.match(r"^([-*]|\d+\.)\s+(.*)$", stripped)
        if bullet:
            while index < len(lines):
                match = re.match(r"^([-*]|\d+\.)\s+(.*)$", lines[index].strip())
                if not match:
                    break
                flowables.append(Paragraph(f"• {_inline_text(match.group(2))}", styles["list"]))
                index += 1
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or candidate.startswith("```"):
                break
            if re.match(r"^(#{1,6})\s+", candidate) or re.match(r"^([-*]|\d+\.)\s+", candidate):
                break
            if re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", candidate):
                break
            paragraph.append(candidate)
            index += 1
        flowables.append(Paragraph(_inline_text(" ".join(paragraph)), styles["body"]))

    return flowables


def render_markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> Path:
    markdown_path = markdown_path.resolve()
    pdf_path = pdf_path.resolve()
    if not markdown_path.exists():
        raise FileNotFoundError(f"Missing markdown report: {markdown_path}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
    )
    document.build(markdown_to_flowables(markdown_path))
    return pdf_path


def load_prompt(prompt_root: Path, task_type: str) -> str:
    prompt_path = prompt_root / f"{task_type}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"missing prompt template: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def resolve_report_pdf(source: Path, *, generated_root: Path, label: str) -> Path:
    """Return a PDF for ``source``, rendering markdown when that is all we have."""
    if source.suffix.lower() == ".pdf":
        if not source.exists():
            raise FileNotFoundError(f"missing {label} report: {source}")
        return source

    sibling_pdf = source.with_suffix(".pdf")
    if sibling_pdf.exists():
        return sibling_pdf
    if not source.exists():
        raise FileNotFoundError(f"missing {label} report: {source}")
    return render_markdown_to_pdf(source, generated_root / f"{label}.pdf")


def encode_pdf_attachment(path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "file",
        "file": {
            "filename": path.name,
            "file_data": f"data:application/pdf;base64,{encoded}",
        },
    }


def normalize_judge_payload(
    payload: dict[str, Any],
    *,
    qid: Any,
    task_type: str,
    dataset: str,
    pass_threshold: float | None,
    judge_model: str,
) -> dict[str, Any]:
    def dimension_score(field_name: str) -> int:
        raw_value = payload.get(field_name, SCORE_MIN)
        if isinstance(raw_value, dict):
            raw_value = raw_value.get("score", SCORE_MIN)
        try:
            score = int(round(float(raw_value)))
        except (TypeError, ValueError):
            return SCORE_MIN
        if math.isnan(score):
            return SCORE_MIN
        return max(SCORE_MIN, min(SCORE_MAX, score))

    dimensions = {field_name: dimension_score(field_name) for field_name in DIMENSION_NAMES}
    score = float(dimensions["overall_score"])

    passed_value = payload.get("passed")
    if isinstance(passed_value, bool):
        passed = passed_value
    elif passed_value is None:
        passed = pass_threshold is not None and score >= float(pass_threshold)
    else:
        passed = str(passed_value).strip().lower() in {"1", "true", "yes", "pass"}

    file_level_notes = payload.get("file_level_notes", [])
    if not isinstance(file_level_notes, list):
        file_level_notes = []

    return {
        "qid": qid,
        "task_type": task_type,
        "dataset": dataset,
        "status": "scored",
        **dimensions,
        "dimensions": dimensions,
        "score": score,
        "passed": passed,
        "correct": passed,
        "summary": str(payload.get("summary", "")).strip(),
        "reason": str(payload.get("reason", "")).strip(),
        "file_level_notes": file_level_notes,
        "judge_model": judge_model,
        "raw_judge_payload": payload,
    }


def failed_result(
    *,
    qid: Any,
    task_type: str,
    dataset: str,
    reason: str,
    judge_model: str,
) -> dict[str, Any]:
    dimensions = {field_name: 0 for field_name in DIMENSION_NAMES}
    return {
        "qid": qid,
        "task_type": task_type,
        "dataset": dataset,
        "status": "failed",
        **dimensions,
        "dimensions": dimensions,
        "score": 0.0,
        "passed": False,
        "correct": False,
        "summary": "",
        "reason": reason,
        "file_level_notes": [],
        "judge_model": judge_model,
    }


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


def call_judge(
    client: OpenAI,
    judge_config: JudgeConfig,
    prompt_text: str,
    *,
    reference_pdf: Path,
    submission_pdf: Path,
    qid: Any,
    task_type: str,
    dataset: str,
    pass_threshold: float | None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Two PDFs are attached.\n"
                f"- {reference_pdf.name} is the reference report (role: reference).\n"
                f"- {submission_pdf.name} is the generated submission (role: submission).\n"
                "Score the submission against the reference report using the rubric."
            ),
        },
        {"type": "text", "text": f"Reference report: {reference_pdf.name}"},
        encode_pdf_attachment(reference_pdf),
        {"type": "text", "text": f"Submission report: {submission_pdf.name}"},
        encode_pdf_attachment(submission_pdf),
    ]
    completion = client.chat.completions.create(
        model=judge_config.model_name,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": content},
        ],
        temperature=0,
    )
    message = completion.choices[0].message
    raw = message.content or ""
    if isinstance(raw, list):
        raw = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in raw
        )
    payload = extract_json_object(str(raw))
    return normalize_judge_payload(
        payload,
        qid=qid,
        task_type=task_type,
        dataset=dataset,
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
    pass_threshold: float | None,
    qids: set[int] | None = None,
) -> dict[str, Any]:
    benchmark_rows = load_jsonl(benchmark_file)
    benchmark_root = benchmark_file.parent
    judge_config = resolve_judge_config(config_path, judge_llm)
    client = OpenAI(api_key=judge_config.api_key, base_url=judge_config.base_url)
    generated_root = submission_root / "generated_pdfs"

    results: list[dict[str, Any]] = []
    for row in benchmark_rows:
        qid = int(row["qid"])
        if qids and qid not in qids:
            continue
        task_type = str(row["task_type"])
        dataset = str(row["dataset"])
        try:
            _, required_outputs = load_manifest_outputs(benchmark_root, row)
            prompt_text = load_prompt(prompt_root, task_type)
            reference_pdf = resolve_report_pdf(
                benchmark_root / find_report_reference_path(required_outputs, qid=qid),
                generated_root=generated_root / "reference",
                label=f"reference-{qid}",
            )
            submission_report = find_submission_report(submission_root / str(qid))
            if submission_report is None:
                raise FileNotFoundError(
                    "missing generated report. Expected "
                    f"{(submission_root / str(qid) / 'report.pdf')} or "
                    f"{(submission_root / str(qid) / 'report.md')}"
                )
            submission_pdf = resolve_report_pdf(
                submission_report,
                generated_root=generated_root / "submission",
                label=f"submission-{qid}",
            )
            results.append(
                call_judge(
                    client,
                    judge_config,
                    prompt_text,
                    reference_pdf=reference_pdf,
                    submission_pdf=submission_pdf,
                    qid=qid,
                    task_type=task_type,
                    dataset=dataset,
                    pass_threshold=pass_threshold,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad task must not abort the batch
            results.append(
                failed_result(
                    qid=qid,
                    task_type=task_type,
                    dataset=dataset,
                    reason=f"{type(exc).__name__}: {exc}",
                    judge_model=judge_config.model_name,
                )
            )

    scored = [item for item in results if item["status"] == "scored"]
    dimension_averages = {
        f"{field_name}_average": (
            sum(float(item[field_name]) for item in scored) / len(scored) if scored else None
        )
        for field_name in DIMENSION_NAMES
    }
    return {
        "benchmark_file": str(benchmark_file),
        "submission_root": str(submission_root),
        "config_file": str(config_path),
        "judge_llm": judge_config.llm_key,
        "judge_model": judge_config.model_name,
        "score_scale": {
            "min": SCORE_MIN,
            "max": SCORE_MAX,
            "pass_threshold": pass_threshold,
            "dimensions": list(DIMENSION_NAMES),
        },
        "total_questions": len(results),
        "scored_questions": len(scored),
        "failed_questions": len(results) - len(scored),
        "passed_questions": sum(int(item["passed"]) for item in results),
        "average_score": dimension_averages["overall_score_average"],
        "dimension_averages": dimension_averages,
        "results": results,
    }


def run_cli(
    *,
    default_benchmark_file: str | None = None,
    default_prompt_root: str | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EHRFlowBench submissions with a reference-based LLM judge."
    )
    parser.add_argument(
        "--submission-root",
        type=Path,
        required=True,
        help="Directory laid out as <root>/<qid>/<files>.",
    )
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
        help=f"Pass threshold on the {SCORE_MIN}-{SCORE_MAX} overall_score scale.",
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
        help="Optional output json path. Defaults to <submission-root>/ehrflowbench.eval.json.",
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
                "scored_questions": payload["scored_questions"],
                "passed_questions": payload["passed_questions"],
                "average_score": payload["average_score"],
                "output_file": str(output_file),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
