import json
import tempfile
import unittest
from pathlib import Path

from healthflow.core.contracts import EvaluationVerdict, ExecutionPlan
from healthflow.core.config import EvaluationConfig, ExecutorConfig, HealthFlowConfig
from healthflow.core.config import (
    EnvironmentConfig,
    LLMProviderConfig,
    LLMRoleConfig,
    LoggingConfig,
    MemoryConfig,
    SystemConfig,
)
from healthflow.core.config import default_executor_backends
from healthflow.execution.base import ExecutionResult
from healthflow.system import HealthFlowSystem


class _FakeMetaAgent:
    def __init__(self):
        self.last_usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        self.last_estimated_cost_usd = 0.0012
        self.last_model_name = "planner-model"

    async def generate_plan(self, **kwargs) -> ExecutionPlan:
        return ExecutionPlan(
            objective="Train a readmission prediction model on the uploaded cohort.",
            assumptions_to_check=["Confirm the uploaded file schema."],
            recommended_steps=["Inspect the data.", "Write artifacts.", "Summarize the result."],
            recommended_workflows=["Use reproducible Python scripts.", "Persist cohort and validation artifacts."],
            avoidances=["Do not skip the final answer."],
            success_signals=["The expected artifacts exist in the workspace."],
            executor_brief="Use a simple reproducible implementation path.",
        )


class _FakeEvaluator:
    def __init__(self):
        self.last_usage = {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
        self.last_estimated_cost_usd = 0.0006
        self.last_model_name = "judge-model"

    async def evaluate(self, **kwargs) -> EvaluationVerdict:
        return EvaluationVerdict(
            status="success",
            score=9.5,
            failure_type="none",
            feedback="Looks good.",
            repair_instructions=[],
            retry_recommended=False,
            memory_worthy_insights=["Explicit artifacts made the result easy to inspect."],
            reasoning="The task request was satisfied and the artifacts are present.",
        )


class _FakeReflector:
    def __init__(self):
        self.last_usage = {"input_tokens": 25, "output_tokens": 5, "total_tokens": 30}
        self.last_estimated_cost_usd = 0.0003
        self.last_model_name = "reflector-model"

    async def synthesize_experience(self, full_history, final_verdict: EvaluationVerdict):
        return []


class _LowScoreSuccessEvaluator(_FakeEvaluator):
    async def evaluate(self, **kwargs) -> EvaluationVerdict:
        return EvaluationVerdict(
            status="success",
            score=1.0,
            failure_type="none",
            feedback="The task completed successfully.",
            repair_instructions=[],
            retry_recommended=False,
            memory_worthy_insights=["Successful runs should not be rejected due to contradictory evaluator scores."],
            reasoning="The artifacts and final answer satisfy the task.",
        )


class _FakeExecutor:
    async def execute(self, context, working_dir: Path) -> ExecutionResult:
        (working_dir / "final_report.md").write_text(
            "# Task Summary\n# Data Profile\n# Method\n# Verification\n# Limitations\n"
            "# Cohort Definition\n# Split Evidence\n# Leakage Audit\n# Metrics Summary\n",
            encoding="utf-8",
        )
        (working_dir / "cohort_definition.json").write_text('{"name": "demo cohort"}', encoding="utf-8")
        (working_dir / "split_evidence.json").write_text('{"train": [1], "val": [], "test": []}', encoding="utf-8")
        (working_dir / "leakage_audit.md").write_text("# Leakage Audit\nNo leakage found.\n", encoding="utf-8")
        (working_dir / "metrics.json").write_text('{"auroc": 0.81}', encoding="utf-8")
        log_path = working_dir / "claude_code_execution.log"
        log_path.write_text("STDOUT: final answer: success\n", encoding="utf-8")
        return ExecutionResult(
            success=True,
            return_code=0,
            log="STDOUT: final answer: success\n",
            log_path=str(log_path),
            prompt_path=None,
            backend="claude_code",
            command=["claude", "--dangerously-skip-permissions", "--print", "prompt"],
            duration_seconds=0.01,
            timed_out=False,
            usage={
                "wall_time_seconds": 0.01,
                "timed_out": False,
                "input_tokens": 400,
                "output_tokens": 25,
                "total_tokens": 425,
                "tool_call_count": 1,
                "tool_time_seconds": 0.004,
                "model_time_seconds": 0.006,
                "estimated_cost_usd": 0.1234,
            },
            telemetry={
                "session_id": "ses_test",
                "models": ["openai/gpt-4.1"],
                "tool_names": ["read"],
                "step_reasons": {"stop": 1},
            },
        )


class SystemSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_task_writes_structured_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = HealthFlowConfig(
                active_llm_name="test-llm",
                active_executor_name="claude_code",
                llm_registry={
                    "test-llm": LLMProviderConfig(
                        api_key="key",
                        base_url="https://example.com/v1",
                        model_name="test-model",
                    ),
                },
                llm=LLMProviderConfig(
                    api_key="key",
                    base_url="https://example.com/v1",
                    model_name="test-model",
                ),
                llm_roles=LLMRoleConfig(),
                system=SystemConfig(max_attempts=1, workspace_dir=str(workspace_dir)),
                environment=EnvironmentConfig(),
                executor=ExecutorConfig(active_backend="claude_code", backends=default_executor_backends()),
                memory=MemoryConfig(write_policy="append"),
                evaluation=EvaluationConfig(success_threshold=8.0),
                logging=LoggingConfig(),
            )
            experience_path = workspace_root / "memory" / "experience.jsonl"
            system = HealthFlowSystem(config=config, experience_path=experience_path)
            system.meta_agent = _FakeMetaAgent()
            system.evaluator = _FakeEvaluator()
            system.reflector = _FakeReflector()
            system.executor = _FakeExecutor()

            result = await system.run_task(
                "Train a readmission prediction model on the uploaded cohort.",
                uploaded_files={"patients.csv": b"subject_id,outcome,event_time\n1,0,2020-01-01\n"},
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["backend"], "claude_code")
            self.assertEqual(result["executor_model"], "test-model")
            self.assertEqual(result["evaluation_status"], "success")
            self.assertTrue(Path(result["run_result_path"]).exists())
            self.assertTrue(Path(result["run_manifest_path"]).exists())
            self.assertTrue(Path(result["memory_context_path"]).exists())
            self.assertTrue(Path(result["evaluation_path"]).exists())
            self.assertTrue(Path(result["cost_analysis_path"]).exists())
            self.assertEqual(Path(result["workspace_path"]).parent, workspace_dir)
            self.assertTrue(experience_path.parent.exists())
            self.assertTrue((Path(result["workspace_path"]) / "task_state.json").exists())
            self.assertEqual(result["cost_summary"]["llm_estimated_cost_usd"], 0.0021)
            self.assertEqual(result["cost_summary"]["executor_estimated_cost_usd"], 0.1234)
            self.assertEqual(result["cost_summary"]["total_estimated_cost_usd"], 0.1255)
            self.assertEqual(result["usage_summary"]["execution"]["session_ids"], ["ses_test"])
            self.assertEqual(result["usage_summary"]["execution"]["tool_names"], ["read"])
            self.assertEqual(result["execution_environment"]["package_manager"], "uv")
            self.assertTrue(result["workflow_recommendations"])

            memory_context = json.loads(Path(result["memory_context_path"]).read_text(encoding="utf-8"))
            self.assertEqual(memory_context["task_family"], "predictive_modeling")
            self.assertEqual(memory_context["domain_focus"], "ehr")

            run_result = json.loads(Path(result["run_result_path"]).read_text(encoding="utf-8"))
            self.assertEqual(run_result["backend"], "claude_code")
            self.assertEqual(run_result["executor_model"], "test-model")
            self.assertEqual(run_result["memory_write_policy"], "append")
            self.assertEqual(run_result["execution_environment"]["run_prefix"], "uv run")
            self.assertEqual(run_result["cost_summary"]["executor_estimated_cost_usd"], 0.1234)
            self.assertEqual(run_result["evaluation_status"], "success")

            run_manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["execution_environment"]["python_version"], "3.12")
            self.assertTrue(run_manifest["workflow_recommendations"])

            cost_analysis = json.loads(Path(result["cost_analysis_path"]).read_text(encoding="utf-8"))
            self.assertEqual(cost_analysis["attempts"][0]["execution"]["estimated_cost_usd"], 0.1234)
            self.assertEqual(cost_analysis["attempts"][0]["attempt_total_estimated_cost_usd"], 0.1252)
            self.assertEqual(cost_analysis["run_total"]["reflection"]["estimated_cost_usd"], 0.0003)
            self.assertEqual(cost_analysis["run_total"]["total_estimated_cost_usd"], 0.1255)

    async def test_run_task_normalizes_contradictory_success_verdicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = HealthFlowConfig(
                active_llm_name="test-llm",
                active_executor_name="claude_code",
                llm_registry={
                    "test-llm": LLMProviderConfig(
                        api_key="key",
                        base_url="https://example.com/v1",
                        model_name="test-model",
                    ),
                },
                llm=LLMProviderConfig(
                    api_key="key",
                    base_url="https://example.com/v1",
                    model_name="test-model",
                ),
                llm_roles=LLMRoleConfig(),
                system=SystemConfig(max_attempts=1, workspace_dir=str(workspace_dir)),
                environment=EnvironmentConfig(),
                executor=ExecutorConfig(active_backend="claude_code", backends=default_executor_backends()),
                memory=MemoryConfig(write_policy="append"),
                evaluation=EvaluationConfig(success_threshold=8.0),
                logging=LoggingConfig(),
            )
            system = HealthFlowSystem(config=config, experience_path=workspace_root / "memory" / "experience.jsonl")
            system.meta_agent = _FakeMetaAgent()
            system.evaluator = _LowScoreSuccessEvaluator()
            system.reflector = _FakeReflector()
            system.executor = _FakeExecutor()

            result = await system.run_task("Create final_report.md with executor smoke ok.")

            self.assertTrue(result["success"])
            self.assertEqual(result["evaluation_status"], "success")
            self.assertEqual(result["evaluation_score"], 8.0)


if __name__ == "__main__":
    unittest.main()
