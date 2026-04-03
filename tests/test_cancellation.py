import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from healthflow.core.config import EvaluationConfig, ExecutorConfig, HealthFlowConfig
from healthflow.core.config import (
    BackendCLIConfig,
    EnvironmentConfig,
    LLMProviderConfig,
    LoggingConfig,
    MemoryConfig,
    SystemConfig,
)
from healthflow.core.contracts import ExecutionPlan
from healthflow.execution.base import ExecutionCancelledError, ExecutionContext, ExecutionResult
from healthflow.execution.cli_adapters import CLISubprocessExecutor
from healthflow.system import HealthFlowSystem


class _FakeMetaAgent:
    def __init__(self):
        self.last_usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        self.last_estimated_cost_usd = 0.001
        self.last_model_name = "planner-model"

    async def generate_plan(self, **kwargs) -> ExecutionPlan:
        return ExecutionPlan(
            objective="Inspect the workspace.",
            assumptions_to_check=["Confirm files in the workspace."],
            recommended_steps=["Inspect the workspace."],
            recommended_workflows=["Use reproducible commands."],
            avoidances=["Do not skip artifact logging."],
            success_signals=["The workspace contains inspectable artifacts."],
        )


class _UnexpectedCallEvaluator:
    async def evaluate(self, **kwargs):
        raise AssertionError("Evaluator should not run for cancelled tasks.")


class _UnexpectedCallReflector:
    async def synthesize_experience(self, full_history, final_verdict):
        raise AssertionError("Reflector should not run for cancelled tasks.")


class _CancellingExecutor:
    async def execute(self, context, working_dir: Path) -> ExecutionResult:
        log_path = working_dir / "cancelled_execution.log"
        log_path.write_text("STDERR: Process cancelled by user.\nSTDOUT: partial output\n", encoding="utf-8")
        prompt_path = working_dir / "cancelled_prompt.md"
        prompt_path.write_text(context.render_prompt(), encoding="utf-8")
        raise ExecutionCancelledError(
            ExecutionResult(
                success=False,
                return_code=-2,
                log=log_path.read_text(encoding="utf-8"),
                log_path=str(log_path),
                prompt_path=str(prompt_path),
                backend="cancel-demo",
                command=["cancel-demo"],
                backend_version="1.0.0",
                executor_metadata={"binary": "cancel-demo"},
                duration_seconds=0.02,
                timed_out=False,
                usage={"wall_time_seconds": 0.02, "estimated_cost_usd": 0.0},
                telemetry={},
                cancelled=True,
                cancel_reason="Execution cancelled by user.",
            )
        )


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_subprocess_executor_writes_partial_log_on_cancellation(self):
        executor = CLISubprocessExecutor(
            "python_runner",
            BackendCLIConfig(
                binary=sys.executable,
                args=[
                    "-c",
                    "import sys, time; sys.stdout.write('hello\\n'); sys.stdout.flush(); time.sleep(30)",
                ],
                prompt_mode="append",
                timeout_seconds=60,
                version_args=[],
            ),
        )
        context = ExecutionContext(
            user_request="Say hello.",
            plan=ExecutionPlan(
                objective="Say hello.",
                assumptions_to_check=["No assumptions."],
                recommended_steps=["Print hello."],
                recommended_workflows=["Use the Python executable."],
                avoidances=["Do not skip stdout."],
                success_signals=["stdout contains hello."],
            ),
            execution_environment=EnvironmentConfig(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir)
            task = asyncio.create_task(executor.execute(context, working_dir))
            await asyncio.sleep(1.0)
            task.cancel()

            with self.assertRaises(ExecutionCancelledError) as ctx:
                await task

            result = ctx.exception.result
            self.assertTrue(result.cancelled)
            self.assertTrue(Path(result.log_path).exists())
            log_text = Path(result.log_path).read_text(encoding="utf-8")
            self.assertIn("Process cancelled by user.", log_text)

    async def test_healthflow_run_task_returns_cancelled_result_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = HealthFlowConfig(
                planner_llm_name="test-llm",
                evaluator_llm_name="test-llm",
                reflector_llm_name="test-llm",
                executor_llm_name="test-llm",
                active_executor_name="claude_code",
                llm_registry={
                    "test-llm": LLMProviderConfig(
                        api_key="key",
                        base_url="https://example.com/v1",
                        model_name="test-model",
                    ),
                },
                system=SystemConfig(max_attempts=1, workspace_dir=str(workspace_dir)),
                environment=EnvironmentConfig(),
                executor=ExecutorConfig(active_backend="claude_code"),
                memory=MemoryConfig(write_policy="append"),
                evaluation=EvaluationConfig(success_threshold=0.8),
                logging=LoggingConfig(),
            )
            experience_path = workspace_root / "memory" / "experience.jsonl"
            system = HealthFlowSystem(config=config, experience_path=experience_path)
            system.meta_agent = _FakeMetaAgent()
            system.evaluator = _UnexpectedCallEvaluator()
            system.reflector = _UnexpectedCallReflector()
            system.executor = _CancellingExecutor()

            result = await system.run_task("Run a long task.")

            self.assertFalse(result["success"])
            self.assertTrue(result["cancelled"])
            self.assertEqual(result["evaluation_status"], "cancelled")
            self.assertTrue(Path(result["runtime_index_path"]).exists())
            self.assertTrue(Path(result["run_summary_path"]).exists())
            self.assertTrue(Path(result["final_evaluation_path"]).exists())
            runtime_index_text = Path(result["runtime_index_path"]).read_text(encoding="utf-8")
            self.assertIn('"cancelled": true', runtime_index_text)
            self.assertIn('"cancel_reason": "Execution cancelled by user."', runtime_index_text)
            self.assertIn('"available_project_cli_tools": [', runtime_index_text)
            self.assertTrue(Path(result["final_evaluation_path"]).exists())
            self.assertTrue(Path(result["last_executor_log_path"]).exists())
            self.assertTrue(result["available_project_cli_tools"])
            self.assertNotIn("oneehr", " ".join(result["available_project_cli_tools"]).lower())
            self.assertIn("tooluniverse", " ".join(result["available_project_cli_tools"]).lower())

    async def test_healthflow_cancelled_ehr_run_persists_oneehr_tool_contracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = HealthFlowConfig(
                planner_llm_name="test-llm",
                evaluator_llm_name="test-llm",
                reflector_llm_name="test-llm",
                executor_llm_name="test-llm",
                active_executor_name="claude_code",
                llm_registry={
                    "test-llm": LLMProviderConfig(
                        api_key="key",
                        base_url="https://example.com/v1",
                        model_name="test-model",
                    ),
                },
                system=SystemConfig(max_attempts=1, workspace_dir=str(workspace_dir)),
                environment=EnvironmentConfig(),
                executor=ExecutorConfig(active_backend="claude_code"),
                memory=MemoryConfig(write_policy="append"),
                evaluation=EvaluationConfig(success_threshold=0.8),
                logging=LoggingConfig(),
            )
            system = HealthFlowSystem(config=config, experience_path=workspace_root / "memory" / "experience.jsonl")
            system.meta_agent = _FakeMetaAgent()
            system.evaluator = _UnexpectedCallEvaluator()
            system.reflector = _UnexpectedCallReflector()
            system.executor = _CancellingExecutor()

            result = await system.run_task(
                "Train an EHR mortality prediction model from the uploaded cohort.",
                uploaded_files={
                    "patients.csv": b"subject_id,mortality,label,event_time\n1,0,0,2020-01-01\n2,1,1,2020-01-02\n"
                },
            )

            self.assertTrue(result["cancelled"])
            self.assertTrue(result["available_project_cli_tools"])
            self.assertIn("oneehr", " ".join(result["available_project_cli_tools"]).lower())
            runtime_index_text = Path(result["runtime_index_path"]).read_text(encoding="utf-8").lower()
            self.assertIn("oneehr", runtime_index_text)


if __name__ == "__main__":
    unittest.main()
