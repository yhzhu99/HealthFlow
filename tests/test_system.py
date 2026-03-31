import json
import tempfile
import unittest
from pathlib import Path

from healthflow.core.config import EHRConfig, EvaluationConfig, ExecutorConfig, HealthFlowConfig
from healthflow.core.config import LLMProviderConfig, LLMRoleConfig, LoggingConfig, MemoryConfig, SystemConfig
from healthflow.core.config import VerificationConfig, default_executor_backends
from healthflow.execution.base import ExecutionResult
from healthflow.system import HealthFlowSystem


class _FakeMetaAgent:
    async def generate_plan(self, **kwargs) -> str:
        return "# Plan\n\n- Inspect data\n- Save required artifacts\n"


class _FakeEvaluator:
    async def evaluate(self, **kwargs) -> dict:
        return {"score": 9.5, "feedback": "Looks good.", "reasoning": "Artifacts satisfy the contract."}


class _FakeReflector:
    async def synthesize_experience(self, full_history, verified: bool):
        return []


class _FakeExecutor:
    async def execute(self, context, working_dir: Path, prompt_file_name: str) -> ExecutionResult:
        (working_dir / prompt_file_name).write_text(context.render_prompt(), encoding="utf-8")
        (working_dir / "final_report.md").write_text(
            "# Task Summary\n# Data Profile\n# Method\n# Verification\n# Limitations\n"
            "# Cohort Definition\n# Split Evidence\n# Leakage Audit\n# Metrics Summary\n",
            encoding="utf-8",
        )
        (working_dir / "cohort_definition.json").write_text('{"name": "demo cohort"}', encoding="utf-8")
        (working_dir / "split_evidence.json").write_text('{"train": [1], "val": [], "test": []}', encoding="utf-8")
        (working_dir / "leakage_audit.md").write_text("# Leakage Audit\nNo leakage found.\n", encoding="utf-8")
        (working_dir / "metrics.json").write_text('{"auroc": 0.81}', encoding="utf-8")
        log_path = working_dir / "healthflow_agent_execution.log"
        log_path.write_text("STDOUT: final answer: success\n", encoding="utf-8")
        return ExecutionResult(
            success=True,
            return_code=0,
            log="STDOUT: final answer: success\n",
            log_path=str(log_path),
            prompt_path=str(working_dir / prompt_file_name),
            backend="healthflow_agent",
            command=["healthflow-agent", "-p", "prompt"],
            duration_seconds=0.01,
            timed_out=False,
            usage={"wall_time_seconds": 0.01, "timed_out": False},
        )


class SystemSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_task_writes_structured_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir) / "workspace"
            config = HealthFlowConfig(
                active_llm_name="test-llm",
                active_executor_name="healthflow_agent",
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
                executor=ExecutorConfig(active_backend="healthflow_agent", backends=default_executor_backends()),
                memory=MemoryConfig(mode="frozen_train"),
                ehr=EHRConfig(),
                verification=VerificationConfig(),
                evaluation=EvaluationConfig(success_threshold=8.0),
                logging=LoggingConfig(),
            )
            experience_path = workspace_dir / "experience.jsonl"
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
            self.assertEqual(result["backend"], "healthflow_agent")
            self.assertTrue(result["verification_passed"])
            self.assertTrue(Path(result["run_result_path"]).exists())
            self.assertTrue(Path(result["run_manifest_path"]).exists())
            self.assertTrue(Path(result["memory_context_path"]).exists())
            self.assertTrue(Path(result["verification_path"]).exists())

            run_result = json.loads(Path(result["run_result_path"]).read_text(encoding="utf-8"))
            self.assertEqual(run_result["backend"], "healthflow_agent")
            self.assertEqual(run_result["memory_mode"], "frozen_train")


if __name__ == "__main__":
    unittest.main()
