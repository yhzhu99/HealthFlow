"""EHRFlowBench PDF/artifact evaluation using the published 1–5 rubric."""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import random
import subprocess

import httpx

SCORES = ("method_soundness", "presentation_quality", "artifact_generation", "overall_score")
PROMPT = """Compare the submitted EHR analysis report PDF and artifacts with the
human reference and task requirements. Alternative methods are acceptable when
methodologically sound and well justified. Assign integer scores from 1 to 5:
- method_soundness: valid data handling, methods, evaluation and interpretation;
- presentation_quality: clear, complete, readable report, tables and figures;
- artifact_generation: correct, usable outputs consistent with the report;
- overall_score: a separate holistic assessment of task completion and validity.
Use these anchors: 1 fundamentally flawed; 2 major errors/incomplete; 3 adequate
with material limitations; 4 sound and complete with minor issues; 5 exemplary.
An attractive presentation does not compensate for invalid analysis. Overall
score is an independent judgment, not the arithmetic mean of dimension scores.
Return one JSON object containing these four scores and a 'reason' string.
"""


def bootstrap(scores: list[float], seed: int = 42) -> dict:
    rng = random.Random(seed)
    means = [sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(100)]
    mean = sum(means) / len(means)
    return {"n": len(scores), "sample_mean": sum(scores) / len(scores),
            "bootstrap_mean": mean,
            "bootstrap_std": (sum((v - mean) ** 2 for v in means) / len(means)) ** 0.5}


def select_attempt(attempts: list[dict], threshold: float = 0.8) -> dict | None:
    for attempt in sorted(attempts, key=lambda item: item["attempt"]):
        verdict = attempt.get("evaluation", {})
        if attempt["attempt"] <= 3 and verdict.get("status") == "success" and verdict.get("score", 0) >= threshold:
            return attempt
    return next((a for a in attempts if a["attempt"] == 3), None)


def submission_directory(root: Path, threshold: float) -> tuple[Path | None, int | None]:
    manifest = root / "attempts.json"
    trajectory = root / "runtime/run/trajectory.json"
    if manifest.is_file():
        selected = select_attempt(json.loads(manifest.read_text())["attempts"], threshold)
        return (root / selected["artifact_dir"], selected["attempt"]) if selected else (None, None)
    if trajectory.is_file():
        attempts = json.loads(trajectory.read_text())["attempts"]
        selected = select_attempt(attempts, threshold)
        if selected is None:
            return None, None
        if selected["attempt"] != max(a["attempt"] for a in attempts):
            raise ValueError("Earlier attempt files were overwritten; supply separate snapshots in attempts.json")
        return root / "sandbox", selected["attempt"]
    return root, None  # Single-attempt or caller-selected submission.


def render_pdf(markdown: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["pandoc", str(markdown.resolve()), "--from=markdown+tex_math_single_backslash",
         "--standalone", "--pdf-engine=xelatex", "--resource-path", str(markdown.parent.resolve()),
         "-V", "geometry:margin=1in", "-o", str(output.resolve())],
        cwd=markdown.parent, capture_output=True, text=True, timeout=180)
    if result.returncode or "Could not fetch resource" in result.stderr:
        raise RuntimeError(f"PDF rendering failed for {markdown}: {result.stderr[-2000:]}")


def file_part(path: Path) -> dict:
    if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
        return {"inlineData": {"mimeType": mimetypes.guess_type(path.name)[0],
                               "data": base64.b64encode(path.read_bytes()).decode()}}
    if path.suffix.lower() == ".parquet":
        import pandas as pd
        return {"text": pd.read_parquet(path).to_csv(index=False)}
    return {"text": path.read_text(encoding="utf-8")}


def judge(parts: list[dict], model: str) -> tuple[dict, dict]:
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=180,
        json={"systemInstruction": {"parts": [{"text": PROMPT}]},
              "contents": [{"role": "user", "parts": parts}],
              "generationConfig": {"responseMimeType": "application/json"}})
    response.raise_for_status()
    raw = response.json()
    candidate = raw["candidates"][0]
    if candidate.get("finishReason") != "STOP":
        raise ValueError("Incomplete judge response")
    text = "".join(p.get("text", "") for p in candidate["content"]["parts"] if not p.get("thought"))
    payload = json.loads(text)
    if any(type(payload.get(k)) is not int or not 1 <= payload[k] <= 5 for k in SCORES):
        raise ValueError("Judge must return four integer scores in [1, 5]")
    return {**{k: payload[k] for k in SCORES}, "reason": payload["reason"]}, raw


def evaluate_task(row: dict, benchmark_root: Path, submissions: Path, work: Path,
                  model: str, threshold: float = 0.8, dry_run: bool = False) -> dict:
    manifest = json.loads((benchmark_root / row["reference_answer"]).read_text())
    outputs = manifest["required_outputs"]
    folder, attempt = submission_directory(submissions / str(row["qid"]), threshold)
    result = {"qid": row["qid"], "dataset": row["dataset"], "attempt": attempt}
    for item in outputs:
        reference = benchmark_root / item["reference_path"]
        if not reference.is_file() or reference.stat().st_size == 0:
            raise FileNotFoundError(f"Missing reference: {reference}")
    missing = [item["file_name"] for item in outputs if folder is None
               or not (folder / item["file_name"]).is_file() or (folder / item["file_name"]).stat().st_size == 0]
    if missing:
        return {**result, **{k: 0 for k in SCORES}, "missing_outputs": missing}
    parts = [{"text": row["task"]}]
    for item in outputs:
        for label, path in (("Reference", benchmark_root / item["reference_path"]),
                            ("Submission", folder / item["file_name"])):
            if item["file_name"] == "report.md":
                pdf = work / str(row["qid"]) / label / "report.pdf"
                render_pdf(path, pdf)
                path = pdf
            parts.extend([{"text": f"{label}: {item['file_name']}"}, file_part(path)])
    if dry_run:
        return {**result, "preflight": "ok"}
    verdict, raw = judge(parts, model)
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{row['qid']}.judge.json").write_text(json.dumps(raw, indent=2))
    return {**result, **verdict}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-file", type=Path, default=Path(__file__).resolve().parents[1] / "processed/test.jsonl")
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--judge-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--qid", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.benchmark_file.read_text().splitlines() if line.strip()]
    if args.qid:
        rows = [row for row in rows if row["qid"] in args.qid]
    if not rows:
        parser.error("No tasks selected")
    results = []
    output = {"judge_model": args.judge_model, "bootstrap_seed": args.bootstrap_seed,
              "benchmark_file": str(args.benchmark_file), "dry_run": args.dry_run,
              "partial_subset": bool(args.qid), "results": results, "complete": False}
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    work = args.output_file.parent / (args.output_file.stem + ".artifacts")
    for row in rows:
        results.append(evaluate_task(row, args.benchmark_file.parent, args.submission_root, work,
                                    args.judge_model, args.success_threshold, args.dry_run))
        args.output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    if not args.dry_run:
        output["overall"] = bootstrap([r["overall_score"] for r in results], args.bootstrap_seed)
        output["by_dataset"] = {dataset: {score: bootstrap([r[score] for r in results if r["dataset"] == dataset], args.bootstrap_seed)
                                         for score in SCORES} for dataset in sorted({r["dataset"] for r in results})}
    output["complete"] = True
    args.output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_file": str(args.output_file), "tasks": len(results), "overall": output.get("overall")}, indent=2))


if __name__ == "__main__":
    main()
