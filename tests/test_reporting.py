import json
import tempfile
import unittest
from pathlib import Path

from healthflow.reporting import generate_task_report


class ReportingTests(unittest.TestCase):
    def test_report_uses_relative_links_and_avoids_absolute_workspace_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "task"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_runtime_files(workspace)
            (workspace / "analysis.md").write_text("# Analysis Summary\n\nOutcome details.\n", encoding="utf-8")
            (workspace / "plots").mkdir()
            (workspace / "plots" / "roc.png").write_bytes(b"png")
            (workspace / "code").mkdir()
            (workspace / "code" / "train.py").write_text('"""Train the model."""\nprint("ok")\n', encoding="utf-8")
            (workspace / "data").mkdir()
            (workspace / "data" / "metrics.json").write_text('{"auroc": 0.81}\n', encoding="utf-8")

            report_path = generate_task_report(workspace)
            report = report_path.read_text(encoding="utf-8")

            self.assertIn("[analysis.md](analysis.md)", report)
            self.assertIn("[roc.png](plots/roc.png)", report)
            self.assertIn("![roc.png](plots/roc.png)", report)
            self.assertIn("[train.py](code/train.py)", report)
            self.assertNotIn(str(workspace), report)

    def test_report_embeds_only_first_five_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "task"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_runtime_files(workspace)
            (workspace / "figures").mkdir()
            for index in range(1, 7):
                (workspace / "figures" / f"fig{index}.png").write_bytes(b"png")

            report = generate_task_report(workspace).read_text(encoding="utf-8")

            self.assertEqual(report.count("!["), 5)
            self.assertIn("[fig6.png](figures/fig6.png)", report)
            self.assertNotIn("![fig6.png](figures/fig6.png)", report)

    def test_report_extracts_descriptors_from_markdown_and_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "task"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_runtime_files(workspace)
            (workspace / "notes.md").write_text("# Executive Summary\n\nBody.\n", encoding="utf-8")
            (workspace / "scripts").mkdir()
            (workspace / "scripts" / "train.py").write_text('"""Train the risk model."""\nimport json\n', encoding="utf-8")
            (workspace / "scripts" / "cohort.sql").write_text("-- Derive study cohort\nselect 1;\n", encoding="utf-8")

            report = generate_task_report(workspace).read_text(encoding="utf-8")

            self.assertIn("Executive Summary", report)
            self.assertIn("Train the risk model.", report)
            self.assertIn("Derive study cohort", report)

    def test_report_excludes_runtime_noise_but_keeps_user_deliverables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "task"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_runtime_files(workspace)
            (workspace / "task_list_v2.md").write_text("# Retry Plan\n", encoding="utf-8")
            (workspace / "memory_context_v1.json").write_text('{"task_family": "general_analysis"}', encoding="utf-8")
            (workspace / "final_report.md").write_text("# User Report\n\nTask output.\n", encoding="utf-8")
            (workspace / ".healthflow_pi_agent").mkdir()
            (workspace / ".healthflow_pi_agent" / "models.json").write_text('{"providers": {}}', encoding="utf-8")

            report = generate_task_report(workspace).read_text(encoding="utf-8")

            self.assertIn("[final_report.md](final_report.md)", report)
            self.assertIn("[task_list_v2.md](task_list_v2.md)", report)
            self.assertNotIn("memory_context_v1.json", report)
            self.assertNotIn(".healthflow_pi_agent", report)
            self.assertIn("Found `1` non-runtime deliverable(s)", report)

    def _write_runtime_files(self, workspace: Path) -> None:
        (workspace / "opencode_execution.log").write_text("STDOUT: done\n", encoding="utf-8")
        (workspace / "task_list_v1.md").write_text("# Execution Plan\n", encoding="utf-8")
        (workspace / "full_history.json").write_text(
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
                                "workspace_paths": [
                                    "analysis.md",
                                    "plots/roc.png",
                                    "code/train.py",
                                    "data/metrics.json",
                                    "task_list_v1.md",
                                ]
                            },
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (workspace / "memory_context.json").write_text('{"task_family": "general_analysis"}', encoding="utf-8")
        (workspace / "evaluation.json").write_text(
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
        (workspace / "cost_analysis.json").write_text(
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
        (workspace / "run_manifest.json").write_text(
            json.dumps(
                {
                    "task_id": "task-demo",
                    "user_request": "Analyze ./cohort.csv and summarize the outcome.",
                    "backend": "opencode",
                    "executor_model": "demo-executor",
                    "executor_provider": "demo-provider",
                    "planner_model": "demo-planner",
                    "llm_role_models": {
                        "planner": "demo-planner",
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
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (workspace / "run_result.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "backend": "opencode",
                    "executor_model": "demo-executor",
                    "executor_provider": "demo-provider",
                    "planner_model": "demo-planner",
                    "llm_role_models": {
                        "planner": "demo-planner",
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
                    "evaluation_status": "success",
                    "evaluation_score": 0.91,
                    "execution_time": 12.34,
                    "final_summary": "Task completed successfully.",
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
