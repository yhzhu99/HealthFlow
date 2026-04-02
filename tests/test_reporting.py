import json
import tempfile
import unittest
from pathlib import Path

from healthflow.reporting import generate_task_report


class ReportingTests(unittest.TestCase):
    def test_report_uses_relative_links_and_renders_paper_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "task"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_runtime_files(workspace)
            sandbox = workspace / "sandbox"
            (sandbox / "analysis.md").write_text("# Analysis Summary\n\nOutcome details.\n", encoding="utf-8")
            (sandbox / "plots").mkdir()
            (sandbox / "plots" / "roc.png").write_bytes(b"png")
            (sandbox / "code").mkdir()
            (sandbox / "code" / "train.py").write_text('"""Train the model."""\nprint("ok")\n', encoding="utf-8")
            (sandbox / "data").mkdir()
            (sandbox / "data" / "metrics.json").write_text('{"auroc": 0.81}\n', encoding="utf-8")

            report_path = generate_task_report(workspace)
            report = report_path.read_text(encoding="utf-8")

            self.assertIn("## Abstract", report)
            self.assertIn("## Problem", report)
            self.assertIn("## HealthFlow Analysis", report)
            self.assertIn("## Execution", report)
            self.assertIn("## Results", report)
            self.assertIn("## Conclusion", report)
            self.assertIn("## Appendix: Reproducibility and Audit", report)
            self.assertIn("[analysis.md](../sandbox/analysis.md)", report)
            self.assertIn("[roc.png](../sandbox/plots/roc.png)", report)
            self.assertIn("![roc.png](../sandbox/plots/roc.png)", report)
            self.assertIn("[train.py](../sandbox/code/train.py)", report)
            self.assertIn("| Metric | Value |", report)
            self.assertIn("| Auroc | 0.8100 |", report)
            self.assertNotIn(str(workspace), report)

    def test_report_embeds_all_user_facing_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "task"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_runtime_files(workspace)
            sandbox = workspace / "sandbox"
            (sandbox / "figures").mkdir()
            for index in range(1, 7):
                (sandbox / "figures" / f"fig{index}.png").write_bytes(b"png")

            report = generate_task_report(workspace).read_text(encoding="utf-8")

            self.assertEqual(report.count("!["), 6)
            self.assertIn("![fig6.png](../sandbox/figures/fig6.png)", report)

    def test_report_extracts_descriptors_from_markdown_and_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "task"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_runtime_files(workspace)
            sandbox = workspace / "sandbox"
            (sandbox / "notes.md").write_text("# Executive Summary\n\nBody.\n", encoding="utf-8")
            (sandbox / "scripts").mkdir()
            (sandbox / "scripts" / "train.py").write_text('"""Train the risk model."""\nimport json\n', encoding="utf-8")
            (sandbox / "scripts" / "cohort.sql").write_text("-- Derive study cohort\nselect 1;\n", encoding="utf-8")

            report = generate_task_report(workspace).read_text(encoding="utf-8")

            self.assertIn("Executive Summary", report)
            self.assertIn("Train the risk model.", report)
            self.assertIn("Derive study cohort", report)

    def test_report_excludes_runtime_noise_but_keeps_user_deliverables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "task"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_runtime_files(workspace)
            sandbox = workspace / "sandbox"
            runtime = workspace / "runtime"
            (sandbox / "final_report.md").write_text("# User Report\n\nTask output.\n", encoding="utf-8")
            (sandbox / ".healthflow_pi_agent").mkdir()
            (sandbox / ".healthflow_pi_agent" / "models.json").write_text('{"providers": {}}', encoding="utf-8")
            (runtime / "attempts" / "attempt_002" / "planner").mkdir(parents=True, exist_ok=True)
            (runtime / "attempts" / "attempt_002" / "planner" / "plan.md").write_text("# Retry Plan\n", encoding="utf-8")

            report = generate_task_report(workspace).read_text(encoding="utf-8")

            self.assertIn("[final_report.md](../sandbox/final_report.md)", report)
            self.assertIn("[trajectory.json](run/trajectory.json)", report)
            self.assertNotIn(".healthflow_pi_agent", report)
            self.assertIn("### Generated Outputs", report)

    def test_report_handles_direct_response_without_attempts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "task"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_runtime_files(workspace)
            runtime = workspace / "runtime"

            (runtime / "run" / "trajectory.json").write_text(
                json.dumps(
                    {
                        "task_id": "task-demo",
                        "user_request": "Say hello and explain that the request used a direct response path.",
                        "response_mode": "direct",
                        "direct_response_reason": "The prompt was lightweight and did not require a full execution attempt.",
                        "attempts": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (runtime / "run" / "direct_response.json").write_text(
                json.dumps(
                    {
                        "mode": "direct",
                        "answer": "HealthFlow returned a built-in response.",
                        "reason": "The prompt was lightweight.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (runtime / "run" / "summary.json").write_text(
                json.dumps(
                    {
                        "success": True,
                        "evaluation_status": "success",
                        "evaluation_score": 1.0,
                        "attempt_count": 0,
                        "final_summary": "Returned a built-in direct response.",
                        "answer": "HealthFlow returned a built-in response.",
                        "available_project_cli_tools": [],
                        "workflow_recommendations": [],
                        "paths": {
                            "direct_response": "run/direct_response.json",
                            "trajectory": "run/trajectory.json",
                            "costs": "run/costs.json",
                            "final_evaluation": "run/final_evaluation.json",
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (runtime / "run" / "final_evaluation.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "score": 1.0,
                        "feedback": "Returned a built-in direct response.",
                        "reasoning": "No execution attempt was needed for this lightweight prompt.",
                        "repair_instructions": [],
                        "violated_constraints": [],
                        "repair_hypotheses": [],
                        "memory_worthy_insights": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            report = generate_task_report(workspace).read_text(encoding="utf-8")

            self.assertIn("HealthFlow returned a built-in response.", report)
            self.assertIn("[direct_response.json](run/direct_response.json)", report)
            self.assertIn("## Results", report)

    def test_report_uses_task_relevant_ehr_language_without_privacy_claim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "task"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_runtime_files(workspace)
            runtime = workspace / "runtime"
            (runtime / "run" / "trajectory.json").write_text(
                json.dumps(
                    {
                        "task_id": "task-demo",
                        "user_request": "Convert the uploaded EHR workbook to CSV.",
                        "data_profile": {"domain_focus": "ehr", "task_family": "format_conversion"},
                        "risk_findings": [],
                        "attempts": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            report = generate_task_report(workspace).read_text(encoding="utf-8")

            self.assertIn("task-relevant safeguards", report)
            self.assertNotIn("privacy-preserving", report)

    def _write_runtime_files(self, workspace: Path) -> None:
        sandbox = workspace / "sandbox"
        runtime = workspace / "runtime"
        (runtime / "run").mkdir(parents=True, exist_ok=True)
        (runtime / "attempts" / "attempt_001" / "planner").mkdir(parents=True, exist_ok=True)
        (runtime / "attempts" / "attempt_001" / "executor").mkdir(parents=True, exist_ok=True)
        (runtime / "attempts" / "attempt_001" / "evaluator").mkdir(parents=True, exist_ok=True)
        sandbox.mkdir(parents=True, exist_ok=True)

        (runtime / "events.jsonl").write_text("", encoding="utf-8")
        (runtime / "attempts" / "attempt_001" / "planner" / "plan.md").write_text("# Execution Plan\n", encoding="utf-8")
        (runtime / "attempts" / "attempt_001" / "executor" / "combined.log").write_text("STDOUT: done\n", encoding="utf-8")

        (runtime / "run" / "trajectory.json").write_text(
            json.dumps(
                {
                    "task_id": "task-demo",
                    "user_request": "Analyze ./cohort.csv and summarize the outcome.",
                    "attempts": [
                        {
                            "attempt": 1,
                            "plan": {
                                "objective": "Analyze the cohort.",
                                "recommended_steps": ["Inspect the input.", "Write the output."],
                            },
                            "execution": {
                                "success": True,
                                "return_code": 0,
                                "duration_seconds": 0.42,
                                "timed_out": False,
                            },
                            "evaluation": {
                                "status": "success",
                                "score": 0.91,
                                "retry_recommended": False,
                                "feedback": "Artifacts were easy to inspect.",
                            },
                            "artifacts": {
                                "sandbox_paths": [
                                    "analysis.md",
                                    "plots/roc.png",
                                    "code/train.py",
                                    "data/metrics.json",
                                ]
                            },
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (runtime / "run" / "final_evaluation.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "score": 0.91,
                    "feedback": "Artifacts were easy to inspect.",
                    "reasoning": "The requested output files are present.",
                    "repair_instructions": [],
                    "violated_constraints": [],
                    "repair_hypotheses": [],
                    "memory_worthy_insights": ["Explicit artifacts simplified review."],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (runtime / "run" / "costs.json").write_text(
            json.dumps(
                {
                    "run_total": {
                        "total_estimated_cost_usd": 0.25,
                        "llm_estimated_cost_usd": 0.05,
                        "executor_estimated_cost_usd": 0.20,
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (runtime / "run" / "summary.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "evaluation_status": "success",
                    "evaluation_score": 0.91,
                    "attempt_count": 1,
                    "final_summary": "Task completed successfully.",
                    "answer": "The analysis determined that the cohort-level result was summarized in the generated artifacts, including an AUROC of 0.81.",
                    "available_project_cli_tools": [],
                    "workflow_recommendations": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (runtime / "index.json").write_text(
            json.dumps(
                {
                    "task_id": "task-demo",
                    "user_request": "Analyze ./cohort.csv and summarize the outcome.",
                    "backend": "opencode",
                    "executor_provider": "demo-provider",
                    "models": {
                        "planner": "demo-planner",
                        "executor": "demo-executor",
                        "evaluator": "demo-judge",
                        "reflector": "demo-reflector",
                    },
                    "runtime_llm_keys": {
                        "planner": "demo-planner-key",
                        "evaluator": "demo-judge-key",
                        "reflector": "demo-reflector-key",
                        "executor": "demo-executor-key",
                    },
                    "memory_write_policy": "append",
                    "execution_time": 12.34,
                    "usage_summary": {
                        "planning": {"calls": 1, "estimated_cost_usd": 0.01, "models": ["demo-planner"]},
                        "execution": {"calls": 1, "estimated_cost_usd": 0.20, "models": ["demo-executor"]},
                        "evaluation": {"calls": 1, "estimated_cost_usd": 0.04, "models": ["demo-judge"]},
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
