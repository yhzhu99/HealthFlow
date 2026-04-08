from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .runtime_artifacts import TaskRuntimePaths
from .session import TaskSessionState, TaskTurnRecord

SHOWCASE_TASK_ID = "showcase-ehr-mortality"
SHOWCASE_SEED_VERSION = 2
SHOWCASE_TITLE = "ICU Mortality Risk Modeling"
SHOWCASE_FILE_NAME = "icu_mortality_cohort.csv"
SHOWCASE_PROMPT = (
    "Use the attached ICU EHR cohort to produce a compact mortality risk diagnostic packet. "
    "Inspect the schema first, build a patient-level baseline model, generate calibration and "
    "ranking diagnostics, save the figures and tables into the workspace, and finish with a "
    "short clinical summary that references the artifacts."
)
SHOWCASE_DESCRIPTION = (
    "Reference ICU EHR workflow with inline figures, workspace deliverables, and a final report."
)
SHOWCASE_OBJECTIVE = "Build an in-hospital mortality risk packet from the ICU cohort."
SHOWCASE_RECOMMENDED_STEPS = [
    "Inspect the cohort schema and confirm outcome prevalence.",
    "Train a reproducible patient-level baseline model.",
    "Render ROC, calibration, and risk distribution diagnostics.",
    "Summarize high-risk patient patterns and export the deliverables.",
]
SHOWCASE_AVOIDANCES = [
    "Do not leak the mortality label into engineered features.",
    "Keep generated artifacts concise and presentation-ready.",
]
SHOWCASE_SUCCESS_SIGNALS = [
    "Inline figures render directly in the main panel.",
    "Workspace contains tables, notes, figures, and a final report.",
]
SHOWCASE_PRIORITY_PATHS = [
    "runtime/report.md",
    "sandbox/figures/roc_curve.png",
    "sandbox/figures/calibration.png",
    "sandbox/figures/risk_distribution.png",
    "sandbox/metrics.json",
    "sandbox/tables/predictions.csv",
    "sandbox/tables/feature_importance.csv",
    "sandbox/tables/cohort_profile.json",
    "sandbox/notes/patient_vignettes.md",
    "sandbox/reports/final_report.md",
]

_SHOWCASE_ROWS = [
    {
        "subject_id": "1001",
        "mortality": 0,
        "label": 0,
        "age": 64,
        "sex": "F",
        "heart_rate": 82,
        "systolic_bp": 126,
        "respiratory_rate": 18,
        "temperature_c": 36.8,
        "glucose": 118,
        "lactate": 1.4,
        "spo2": 97,
        "creatinine": 0.9,
        "los_days": 3.2,
    },
    {
        "subject_id": "1002",
        "mortality": 1,
        "label": 1,
        "age": 79,
        "sex": "M",
        "heart_rate": 109,
        "systolic_bp": 94,
        "respiratory_rate": 24,
        "temperature_c": 37.9,
        "glucose": 176,
        "lactate": 2.6,
        "spo2": 92,
        "creatinine": 1.5,
        "los_days": 4.9,
    },
    {
        "subject_id": "1003",
        "mortality": 0,
        "label": 0,
        "age": 58,
        "sex": "F",
        "heart_rate": 76,
        "systolic_bp": 132,
        "respiratory_rate": 17,
        "temperature_c": 36.6,
        "glucose": 109,
        "lactate": 1.2,
        "spo2": 98,
        "creatinine": 0.8,
        "los_days": 3.3,
    },
    {
        "subject_id": "1004",
        "mortality": 1,
        "label": 1,
        "age": 83,
        "sex": "M",
        "heart_rate": 118,
        "systolic_bp": 88,
        "respiratory_rate": 27,
        "temperature_c": 38.1,
        "glucose": 201,
        "lactate": 3.1,
        "spo2": 90,
        "creatinine": 1.8,
        "los_days": 5.8,
    },
    {
        "subject_id": "1005",
        "mortality": 0,
        "label": 0,
        "age": 47,
        "sex": "F",
        "heart_rate": 74,
        "systolic_bp": 138,
        "respiratory_rate": 16,
        "temperature_c": 36.7,
        "glucose": 97,
        "lactate": 1.1,
        "spo2": 99,
        "creatinine": 0.7,
        "los_days": 2.1,
    },
    {
        "subject_id": "1006",
        "mortality": 0,
        "label": 0,
        "age": 69,
        "sex": "M",
        "heart_rate": 88,
        "systolic_bp": 122,
        "respiratory_rate": 19,
        "temperature_c": 37.0,
        "glucose": 136,
        "lactate": 1.5,
        "spo2": 96,
        "creatinine": 1.0,
        "los_days": 4.0,
    },
    {
        "subject_id": "1007",
        "mortality": 1,
        "label": 1,
        "age": 76,
        "sex": "F",
        "heart_rate": 121,
        "systolic_bp": 92,
        "respiratory_rate": 28,
        "temperature_c": 38.4,
        "glucose": 214,
        "lactate": 3.4,
        "spo2": 89,
        "creatinine": 1.7,
        "los_days": 6.1,
    },
    {
        "subject_id": "1008",
        "mortality": 0,
        "label": 0,
        "age": 55,
        "sex": "M",
        "heart_rate": 79,
        "systolic_bp": 129,
        "respiratory_rate": 17,
        "temperature_c": 36.5,
        "glucose": 111,
        "lactate": 1.3,
        "spo2": 98,
        "creatinine": 0.9,
        "los_days": 2.4,
    },
    {
        "subject_id": "1009",
        "mortality": 0,
        "label": 0,
        "age": 62,
        "sex": "F",
        "heart_rate": 85,
        "systolic_bp": 124,
        "respiratory_rate": 18,
        "temperature_c": 36.9,
        "glucose": 126,
        "lactate": 1.6,
        "spo2": 97,
        "creatinine": 1.0,
        "los_days": 3.8,
    },
    {
        "subject_id": "1010",
        "mortality": 1,
        "label": 1,
        "age": 81,
        "sex": "M",
        "heart_rate": 116,
        "systolic_bp": 90,
        "respiratory_rate": 26,
        "temperature_c": 38.0,
        "glucose": 188,
        "lactate": 2.9,
        "spo2": 90,
        "creatinine": 1.6,
        "los_days": 5.6,
    },
    {
        "subject_id": "1011",
        "mortality": 0,
        "label": 0,
        "age": 51,
        "sex": "F",
        "heart_rate": 72,
        "systolic_bp": 136,
        "respiratory_rate": 15,
        "temperature_c": 36.4,
        "glucose": 102,
        "lactate": 1.0,
        "spo2": 99,
        "creatinine": 0.8,
        "los_days": 2.8,
    },
    {
        "subject_id": "1012",
        "mortality": 1,
        "label": 1,
        "age": 74,
        "sex": "M",
        "heart_rate": 113,
        "systolic_bp": 96,
        "respiratory_rate": 25,
        "temperature_c": 37.8,
        "glucose": 192,
        "lactate": 2.8,
        "spo2": 91,
        "creatinine": 1.5,
        "los_days": 6.0,
    },
    {
        "subject_id": "1013",
        "mortality": 0,
        "label": 0,
        "age": 67,
        "sex": "F",
        "heart_rate": 90,
        "systolic_bp": 119,
        "respiratory_rate": 20,
        "temperature_c": 37.1,
        "glucose": 145,
        "lactate": 1.7,
        "spo2": 95,
        "creatinine": 1.1,
        "los_days": 3.9,
    },
    {
        "subject_id": "1014",
        "mortality": 0,
        "label": 0,
        "age": 59,
        "sex": "M",
        "heart_rate": 77,
        "systolic_bp": 130,
        "respiratory_rate": 16,
        "temperature_c": 36.6,
        "glucose": 107,
        "lactate": 1.2,
        "spo2": 98,
        "creatinine": 0.8,
        "los_days": 2.6,
    },
    {
        "subject_id": "1015",
        "mortality": 1,
        "label": 1,
        "age": 86,
        "sex": "F",
        "heart_rate": 124,
        "systolic_bp": 87,
        "respiratory_rate": 29,
        "temperature_c": 38.3,
        "glucose": 226,
        "lactate": 3.5,
        "spo2": 88,
        "creatinine": 1.9,
        "los_days": 6.4,
    },
    {
        "subject_id": "1016",
        "mortality": 0,
        "label": 0,
        "age": 63,
        "sex": "M",
        "heart_rate": 83,
        "systolic_bp": 127,
        "respiratory_rate": 18,
        "temperature_c": 36.8,
        "glucose": 121,
        "lactate": 1.4,
        "spo2": 97,
        "creatinine": 0.9,
        "los_days": 3.1,
    },
    {
        "subject_id": "1017",
        "mortality": 1,
        "label": 1,
        "age": 78,
        "sex": "F",
        "heart_rate": 119,
        "systolic_bp": 91,
        "respiratory_rate": 27,
        "temperature_c": 38.2,
        "glucose": 205,
        "lactate": 3.2,
        "spo2": 89,
        "creatinine": 1.7,
        "los_days": 6.2,
    },
    {
        "subject_id": "1018",
        "mortality": 0,
        "label": 0,
        "age": 53,
        "sex": "M",
        "heart_rate": 75,
        "systolic_bp": 134,
        "respiratory_rate": 16,
        "temperature_c": 36.5,
        "glucose": 99,
        "lactate": 1.1,
        "spo2": 99,
        "creatinine": 0.8,
        "los_days": 2.9,
    },
    {
        "subject_id": "1019",
        "mortality": 1,
        "label": 1,
        "age": 72,
        "sex": "F",
        "heart_rate": 111,
        "systolic_bp": 98,
        "respiratory_rate": 24,
        "temperature_c": 37.7,
        "glucose": 184,
        "lactate": 2.7,
        "spo2": 92,
        "creatinine": 1.4,
        "los_days": 5.1,
    },
    {
        "subject_id": "1020",
        "mortality": 0,
        "label": 0,
        "age": 60,
        "sex": "M",
        "heart_rate": 81,
        "systolic_bp": 128,
        "respiratory_rate": 17,
        "temperature_c": 36.7,
        "glucose": 116,
        "lactate": 1.3,
        "spo2": 97,
        "creatinine": 0.9,
        "los_days": 3.0,
    },
    {
        "subject_id": "1021",
        "mortality": 1,
        "label": 1,
        "age": 84,
        "sex": "M",
        "heart_rate": 122,
        "systolic_bp": 86,
        "respiratory_rate": 30,
        "temperature_c": 38.5,
        "glucose": 232,
        "lactate": 3.8,
        "spo2": 87,
        "creatinine": 2.0,
        "los_days": 6.8,
    },
    {
        "subject_id": "1022",
        "mortality": 0,
        "label": 0,
        "age": 57,
        "sex": "F",
        "heart_rate": 78,
        "systolic_bp": 131,
        "respiratory_rate": 17,
        "temperature_c": 36.6,
        "glucose": 108,
        "lactate": 1.2,
        "spo2": 98,
        "creatinine": 0.8,
        "los_days": 2.7,
    },
    {
        "subject_id": "1023",
        "mortality": 1,
        "label": 1,
        "age": 77,
        "sex": "M",
        "heart_rate": 114,
        "systolic_bp": 93,
        "respiratory_rate": 25,
        "temperature_c": 37.9,
        "glucose": 196,
        "lactate": 3.0,
        "spo2": 91,
        "creatinine": 1.6,
        "los_days": 5.7,
    },
    {
        "subject_id": "1024",
        "mortality": 0,
        "label": 0,
        "age": 61,
        "sex": "F",
        "heart_rate": 84,
        "systolic_bp": 123,
        "respiratory_rate": 18,
        "temperature_c": 36.9,
        "glucose": 124,
        "lactate": 1.5,
        "spo2": 96,
        "creatinine": 1.0,
        "los_days": 3.5,
    },
]


def ensure_showcase_task(workspace_dir: Path, *, reset: bool = False) -> Path:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if reset:
        clear_task_workspace(workspace_dir)
    task_root = workspace_dir / SHOWCASE_TASK_ID
    if _showcase_seed_is_current(task_root):
        return task_root
    if task_root.exists():
        shutil.rmtree(task_root)
    _seed_showcase_task(task_root)
    return task_root


def clear_task_workspace(workspace_dir: Path) -> None:
    if not workspace_dir.exists():
        return
    for child in workspace_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)


def showcase_message_lines() -> list[str]:
    return [
        f"**Objective**\n\n{SHOWCASE_OBJECTIVE}",
        "**Recommended steps**\n\n"
        + "\n".join(f"{index}. {item}" for index, item in enumerate(SHOWCASE_RECOMMENDED_STEPS, start=1)),
        "**Stage progress**\n\n- Memory retrieval completed.\n- Planner locked the clinical objective.\n- Executor generated figures, tables, and the report.\n- Evaluator cleared the packet for clinical review.",
        "**Workspace outputs**\n\n- `figures/roc_curve.png`\n- `figures/calibration.png`\n- `figures/risk_distribution.png`\n- `tables/predictions.csv`\n- `reports/final_report.md`",
    ]


def _showcase_seed_is_current(task_root: Path) -> bool:
    session_path = task_root / "runtime" / "session.json"
    manifest_path = task_root / "runtime" / "showcase_seed.json"
    if not session_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        int(manifest.get("version") or 0) == SHOWCASE_SEED_VERSION
        and str(manifest.get("title") or "") == SHOWCASE_TITLE
        and str(manifest.get("file_name") or "") == SHOWCASE_FILE_NAME
    )


def _seed_showcase_task(task_root: Path) -> None:
    paths = TaskRuntimePaths.build(task_root)
    paths.ensure_base_dirs()
    turn_paths = TaskRuntimePaths.build(
        task_root,
        sandbox_dir=paths.sandbox_dir,
        runtime_dir=task_root / "runtime" / "turns" / "turn_001",
    )
    turn_paths.ensure_base_dirs()

    now = datetime.now(timezone.utc).replace(microsecond=0)
    created_at = (now - timedelta(minutes=18)).isoformat().replace("+00:00", "Z")
    updated_at = (now - timedelta(minutes=2)).isoformat().replace("+00:00", "Z")

    rows = _cohort_with_predictions()
    metrics = _metrics_payload(rows)
    feature_importance = _feature_importance_rows()
    profile = _cohort_profile(rows)
    answer = _final_answer(metrics, profile)
    summary = _summary_text(metrics)
    report_content = _final_report(profile, metrics, feature_importance)
    vignettes_content = _patient_vignettes(rows)

    _write_csv(paths.sandbox_dir / SHOWCASE_FILE_NAME, rows, fieldnames=_csv_fieldnames(rows))
    _write_json(paths.sandbox_dir / "metrics.json", metrics)
    _write_csv(paths.sandbox_dir / "tables" / "predictions.csv", rows, fieldnames=_csv_fieldnames(rows))
    _write_csv(
        paths.sandbox_dir / "tables" / "feature_importance.csv",
        feature_importance,
        fieldnames=["feature", "importance", "clinical_note"],
    )
    _write_json(paths.sandbox_dir / "tables" / "cohort_profile.json", profile)
    _write_text(paths.sandbox_dir / "notes" / "patient_vignettes.md", vignettes_content)
    _write_text(paths.sandbox_dir / "reports" / "final_report.md", report_content)
    _write_text(paths.report_path, report_content)
    _write_text(turn_paths.report_path, report_content)

    _write_chart_roc(paths.sandbox_dir / "figures" / "roc_curve.png")
    _write_chart_calibration(paths.sandbox_dir / "figures" / "calibration.png")
    _write_chart_distribution(paths.sandbox_dir / "figures" / "risk_distribution.png", rows)

    trajectory = _trajectory_payload(answer)
    run_summary = {
        "success": True,
        "cancelled": False,
        "evaluation_status": "success",
        "final_summary": summary,
        "answer": answer,
    }
    events = _runtime_events(created_at)
    runtime_index = {
        "schema_version": "2.0",
        "task_id": SHOWCASE_TASK_ID,
        "user_request": SHOWCASE_PROMPT,
        "mode": "workflow_run",
        "success": True,
        "cancelled": False,
        "paths": {
            "task_root": str(task_root),
            "sandbox": str(paths.sandbox_dir),
            "runtime": str(paths.runtime_dir),
            "summary": str(paths.run_summary_path),
            "trajectory": str(paths.run_trajectory_path),
            "report": str(paths.report_path),
        },
        "workflow_recommendations": ["predictive_modeling", "ehr_tabular_case"],
        "available_project_cli_tools": ["python", "pytest"],
    }

    _write_json(paths.run_trajectory_path, trajectory)
    _write_json(paths.run_summary_path, run_summary)
    _write_json(paths.index_path, runtime_index)
    _write_json(turn_paths.run_trajectory_path, trajectory)
    _write_json(turn_paths.run_summary_path, run_summary)
    _write_json(turn_paths.index_path, runtime_index)
    _write_text(paths.events_path, "\n".join(json.dumps(item) for item in events) + "\n")
    _write_text(turn_paths.events_path, "\n".join(json.dumps(item) for item in events) + "\n")

    turn_record = TaskTurnRecord(
        turn_number=1,
        user_message=SHOWCASE_PROMPT,
        answer=answer,
        status="success",
        runtime_dir=str(turn_paths.runtime_dir),
        report_path=str(paths.report_path),
        evaluation_feedback=summary,
        uploaded_files=[
            {
                "original_name": SHOWCASE_FILE_NAME,
                "upload_path": f"uploads/turn_001/{SHOWCASE_FILE_NAME}",
                "sandbox_path": f"sandbox/{SHOWCASE_FILE_NAME}",
            }
        ],
        created_at_utc=updated_at,
    )
    state = TaskSessionState(
        task_id=SHOWCASE_TASK_ID,
        task_root=str(task_root),
        created_at_utc=created_at,
        updated_at_utc=updated_at,
        original_goal=SHOWCASE_PROMPT,
        display_title=SHOWCASE_TITLE,
        turn_count=1,
        latest_turn_number=1,
        latest_turn_status="success",
    )
    upload_path = task_root / "uploads" / "turn_001" / SHOWCASE_FILE_NAME
    _write_csv(upload_path, rows, fieldnames=_csv_fieldnames(rows))
    _write_json(task_root / "runtime" / "session.json", state.to_dict())
    _write_json(
        task_root / "runtime" / "showcase_seed.json",
        {
            "version": SHOWCASE_SEED_VERSION,
            "title": SHOWCASE_TITLE,
            "file_name": SHOWCASE_FILE_NAME,
        },
    )
    _write_text(task_root / "runtime" / "history.jsonl", json.dumps(turn_record.to_dict()) + "\n")


def _cohort_with_predictions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _SHOWCASE_ROWS:
        risk = _predicted_risk(item)
        enriched = dict(item)
        enriched["predicted_risk"] = round(risk, 3)
        enriched["predicted_label"] = int(risk >= 0.5)
        enriched["risk_bucket"] = _risk_bucket(risk)
        rows.append(enriched)
    return rows


def _predicted_risk(item: dict[str, Any]) -> float:
    logit = (
        -2.8
        + 0.055 * (float(item["age"]) - 60)
        + 0.025 * (float(item["heart_rate"]) - 85)
        - 0.03 * (float(item["systolic_bp"]) - 118)
        + 0.07 * (float(item["respiratory_rate"]) - 18)
        + 0.018 * (float(item["glucose"]) - 115)
        + 0.85 * (float(item["lactate"]) - 1.2)
        - 0.12 * (float(item["spo2"]) - 95)
        + 0.35 * (float(item["creatinine"]) - 0.9)
    )
    return 1.0 / (1.0 + math.exp(-logit))


def _risk_bucket(value: float) -> str:
    if value >= 0.75:
        return "critical"
    if value >= 0.45:
        return "elevated"
    return "low"


def _metrics_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prevalence = sum(int(item["mortality"]) for item in rows) / max(len(rows), 1)
    return {
        "task": "icu_mortality_prediction",
        "cohort_size": len(rows),
        "positive_cases": sum(int(item["mortality"]) for item in rows),
        "outcome_prevalence": round(prevalence, 3),
        "auroc": 0.892,
        "auprc": 0.801,
        "brier_score": 0.123,
        "sensitivity_at_0_5": 0.889,
        "specificity_at_0_5": 0.867,
        "top_risk_pattern": "Older patients with hypotension, tachycardia, and elevated lactate clustered in the highest-risk bucket.",
    }


def _feature_importance_rows() -> list[dict[str, Any]]:
    return [
        {"feature": "lactate", "importance": 0.28, "clinical_note": "Perfusion failure was the strongest nonlinear driver."},
        {"feature": "systolic_bp", "importance": 0.21, "clinical_note": "Persistent hypotension sharply increased predicted risk."},
        {"feature": "age", "importance": 0.17, "clinical_note": "Advanced age amplified every hemodynamic signal."},
        {"feature": "glucose", "importance": 0.13, "clinical_note": "Stress hyperglycemia separated several high-risk patients."},
        {"feature": "respiratory_rate", "importance": 0.12, "clinical_note": "Tachypnea remained important after adjustment."},
        {"feature": "spo2", "importance": 0.09, "clinical_note": "Low oxygen saturation pushed borderline cases upward."},
    ]


def _cohort_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    female_count = sum(1 for item in rows if item["sex"] == "F")
    high_risk = [item for item in rows if item["risk_bucket"] == "critical"]
    return {
        "cohort_size": len(rows),
        "female_count": female_count,
        "male_count": len(rows) - female_count,
        "median_age": 68,
        "median_los_days": 4.0,
        "high_risk_count": len(high_risk),
        "critical_subject_ids": [item["subject_id"] for item in high_risk],
    }


def _final_answer(metrics: dict[str, Any], profile: dict[str, Any]) -> str:
    return (
        f"Built a compact ICU mortality packet for {profile['cohort_size']} admissions. "
        f"The baseline model reached AUROC {metrics['auroc']:.3f} with good rank ordering, and the "
        "highest-risk patients consistently combined older age, low systolic blood pressure, high lactate, "
        "and tachycardia. The main panel now includes ROC, calibration, and risk distribution figures, and "
        "the workspace contains predictions, feature importance, cohort profile, vignettes, and the final report."
    )


def _summary_text(metrics: dict[str, Any]) -> str:
    return (
        f"Clinical example completed successfully. ROC and calibration diagnostics were generated, the evaluator "
        f"accepted the packet, and the final AUROC was {metrics['auroc']:.3f}."
    )


def _final_report(
    profile: dict[str, Any],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
) -> str:
    top_features = "\n".join(
        f"- **{row['feature']}** ({row['importance']:.2f}): {row['clinical_note']}" for row in feature_importance[:4]
    )
    return (
        "# ICU Mortality Risk Report\n\n"
        "## Executive Summary\n\n"
        f"- Cohort size: {profile['cohort_size']} admissions\n"
        f"- Positive outcome count: {metrics['positive_cases']}\n"
        f"- AUROC: {metrics['auroc']:.3f}\n"
        f"- AUPRC: {metrics['auprc']:.3f}\n"
        f"- Brier score: {metrics['brier_score']:.3f}\n\n"
        "## Clinical Takeaways\n\n"
        f"{metrics['top_risk_pattern']}\n\n"
        "## Leading Predictors\n\n"
        f"{top_features}\n\n"
        "## Deliverables\n\n"
        "- `metrics.json`\n"
        "- `tables/predictions.csv`\n"
        "- `tables/feature_importance.csv`\n"
        "- `tables/cohort_profile.json`\n"
        "- `notes/patient_vignettes.md`\n"
        "- `figures/roc_curve.png`\n"
        "- `figures/calibration.png`\n"
        "- `figures/risk_distribution.png`\n"
    )


def _patient_vignettes(rows: list[dict[str, Any]]) -> str:
    high_risk = sorted(rows, key=lambda item: float(item["predicted_risk"]), reverse=True)[:3]
    blocks = []
    for item in high_risk:
        blocks.append(
            f"## Subject {item['subject_id']}\n\n"
            f"- Risk bucket: {item['risk_bucket']}\n"
            f"- Predicted risk: {float(item['predicted_risk']):.3f}\n"
            f"- Snapshot: age {item['age']}, SBP {item['systolic_bp']}, HR {item['heart_rate']}, "
            f"RR {item['respiratory_rate']}, lactate {item['lactate']}, glucose {item['glucose']}\n"
        )
    return "# Patient Vignettes\n\n" + "\n".join(blocks)


def _trajectory_payload(answer: str) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "task_id": SHOWCASE_TASK_ID,
        "user_request": SHOWCASE_PROMPT,
        "backend": "showcase",
        "executor_model": "showcase-fixture",
        "planner_model": "showcase-fixture",
        "llm_role_models": {
            "planner": "showcase-fixture",
            "evaluator": "showcase-fixture",
            "reflector": "showcase-fixture",
        },
        "workflow_recommendations": ["predictive_modeling", "risk_stratification"],
        "attempts": [
            {
                "attempt": 1,
                "memory": {"retrieval": {"selected": [], "skipped": True}},
                "plan": {
                    "objective": SHOWCASE_OBJECTIVE,
                    "assumptions_to_check": ["Outcome label is patient-level and binary."],
                    "recommended_steps": list(SHOWCASE_RECOMMENDED_STEPS),
                    "recommended_workflows": ["tabular_modeling", "ehr_quality_checks"],
                    "avoidances": list(SHOWCASE_AVOIDANCES),
                    "success_signals": list(SHOWCASE_SUCCESS_SIGNALS),
                },
                "execution": {
                    "success": True,
                    "return_code": 0,
                    "duration_seconds": 8.4,
                    "timed_out": False,
                    "cancelled": False,
                    "backend": "showcase",
                },
                "artifacts": {
                    "sandbox_paths": [
                        SHOWCASE_FILE_NAME,
                        "metrics.json",
                        "tables/predictions.csv",
                        "tables/feature_importance.csv",
                        "tables/cohort_profile.json",
                        "notes/patient_vignettes.md",
                        "reports/final_report.md",
                        "figures/roc_curve.png",
                        "figures/calibration.png",
                        "figures/risk_distribution.png",
                    ],
                    "generated_answer": answer,
                },
                "gate": {
                    "execution_ok": True,
                    "evaluation_threshold_ok": True,
                    "verdict_success": True,
                    "retry_recommended": False,
                },
                "evaluation": {
                    "status": "success",
                    "score": 0.94,
                    "failure_type": None,
                    "feedback": "The workspace is coherent, clinically plausible, and presentation-ready.",
                    "reasoning": "Key outputs are present and align with the plan.",
                    "retry_recommended": False,
                },
            }
        ],
        "reflection": {
            "model_name": "showcase-fixture",
            "estimated_cost_usd": 0.0,
        },
    }


def _runtime_events(created_at: str) -> list[dict[str, Any]]:
    timestamp = created_at
    return [
        {
            "timestamp_utc": timestamp,
            "task_id": SHOWCASE_TASK_ID,
            "attempt": 1,
            "stage": "memory",
            "event": "retrieval_completed",
            "status": "completed",
            "metadata": {"selected_count": 0, "skipped": True},
        },
        {
            "timestamp_utc": timestamp,
            "task_id": SHOWCASE_TASK_ID,
            "attempt": 1,
            "stage": "planner",
            "event": "plan_generated",
            "status": "completed",
            "metadata": {"objective": SHOWCASE_OBJECTIVE},
        },
        {
            "timestamp_utc": timestamp,
            "task_id": SHOWCASE_TASK_ID,
            "attempt": 1,
            "stage": "executor",
            "event": "execution_completed",
            "status": "completed",
            "metadata": {"artifact_count": 9},
        },
        {
            "timestamp_utc": timestamp,
            "task_id": SHOWCASE_TASK_ID,
            "attempt": 1,
            "stage": "evaluator",
            "event": "evaluation_completed",
            "status": "success",
            "metadata": {"score": 0.94},
        },
    ]


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_chart_roc(path: Path) -> None:
    import matplotlib.pyplot as plt

    fpr = [0.0, 0.03, 0.08, 0.15, 0.25, 1.0]
    tpr = [0.0, 0.46, 0.71, 0.82, 0.92, 1.0]
    figure, axis = plt.subplots(figsize=(5.8, 4.0), dpi=160)
    figure.patch.set_facecolor("#f7fbfd")
    axis.set_facecolor("#ffffff")
    axis.plot(fpr, tpr, color="#155a8a", linewidth=2.8, label="HealthFlow baseline")
    axis.plot([0, 1], [0, 1], color="#94a3b8", linewidth=1.2, linestyle="--", label="Chance")
    axis.fill_between(fpr, tpr, alpha=0.08, color="#155a8a")
    axis.set_title("ROC Curve", fontsize=12, fontweight="bold")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(alpha=0.18, linewidth=0.8)
    _save_figure(figure, path)


def _write_chart_calibration(path: Path) -> None:
    import matplotlib.pyplot as plt

    predicted = [0.08, 0.18, 0.32, 0.51, 0.67, 0.83]
    observed = [0.05, 0.17, 0.28, 0.56, 0.7, 0.87]
    figure, axis = plt.subplots(figsize=(5.8, 4.0), dpi=160)
    figure.patch.set_facecolor("#f7fbfd")
    axis.set_facecolor("#ffffff")
    axis.plot(predicted, observed, marker="o", color="#0f4a74", linewidth=2.5)
    axis.plot([0, 1], [0, 1], color="#94a3b8", linewidth=1.2, linestyle="--")
    axis.set_title("Calibration", fontsize=12, fontweight="bold")
    axis.set_xlabel("Predicted risk")
    axis.set_ylabel("Observed mortality")
    axis.grid(alpha=0.18, linewidth=0.8)
    _save_figure(figure, path)


def _write_chart_distribution(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    positive = [float(item["predicted_risk"]) for item in rows if int(item["mortality"]) == 1]
    negative = [float(item["predicted_risk"]) for item in rows if int(item["mortality"]) == 0]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    figure, axis = plt.subplots(figsize=(5.8, 4.0), dpi=160)
    figure.patch.set_facecolor("#f7fbfd")
    axis.set_facecolor("#ffffff")
    axis.hist([negative, positive], bins=bins, stacked=True, color=["#94a3b8", "#155a8a"], label=["Survived", "Died"])
    axis.set_title("Risk Distribution", fontsize=12, fontweight="bold")
    axis.set_xlabel("Predicted risk")
    axis.set_ylabel("Patients")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18, linewidth=0.8, axis="y")
    _save_figure(figure, path)


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, format="png", bbox_inches="tight")
    figure.clf()
    import matplotlib.pyplot as plt

    plt.close(figure)
