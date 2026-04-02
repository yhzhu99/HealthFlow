import json
import tempfile
import unittest
from pathlib import Path

from healthflow.core.contracts import EvaluationVerdict, ExecutionPlan
from healthflow.core.config import EvaluationConfig, ExecutorConfig, HealthFlowConfig
from healthflow.core.config import (
    EnvironmentConfig,
    LLMProviderConfig,
    LoggingConfig,
    MemoryConfig,
    SystemConfig,
)
from healthflow.core.config import default_executor_backends
from healthflow.core.direct_responses import DirectResponse
from healthflow.execution.base import ExecutionResult
from healthflow.experience.experience_models import Experience, MemoryKind, SourceOutcome
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
        )


class _FakeEvaluator:
    def __init__(self):
        self.last_usage = {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
        self.last_estimated_cost_usd = 0.0006
        self.last_model_name = "judge-model"

    async def evaluate(self, **kwargs) -> EvaluationVerdict:
        return EvaluationVerdict(
            status="success",
            score=0.95,
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


class _MemoryWritingReflector(_FakeReflector):
    async def synthesize_experience(self, full_history, final_verdict: EvaluationVerdict):
        return [
            Experience(
                kind=MemoryKind.WORKFLOW,
                category="repository_summary",
                content="Save a concise repository map when the task asks for structure or orientation.",
                source_task_id=full_history["task_id"],
                task_family=full_history.get("data_profile", {}).get("task_family", "general_analysis"),
                dataset_signature=full_history.get("data_profile", {}).get("dataset_signature", "unknown"),
                stage="reflection",
                backend=full_history.get("backend", "unknown"),
                source_outcome=SourceOutcome.SUCCESS,
                applicability_scope="workflow_generic",
            )
        ]


class _LowScoreSuccessEvaluator(_FakeEvaluator):
    async def evaluate(self, **kwargs) -> EvaluationVerdict:
        return EvaluationVerdict(
            status="success",
            score=0.1,
            failure_type="none",
            feedback="The task completed successfully.",
            repair_instructions=[],
            retry_recommended=False,
            memory_worthy_insights=["Successful runs should not be rejected due to contradictory evaluator scores."],
            reasoning="The artifacts and final answer satisfy the task.",
        )


class _FailedEvaluator(_FakeEvaluator):
    async def evaluate(self, **kwargs) -> EvaluationVerdict:
        return EvaluationVerdict(
            status="failed",
            score=0.2,
            failure_type="analysis_incomplete",
            feedback="The task did not satisfy the requested deliverable.",
            repair_instructions=["Produce the missing deliverable."],
            retry_recommended=False,
            memory_worthy_insights=["Failure reports should still preserve the audit trail."],
            reasoning="The produced artifacts are incomplete.",
        )


class _GeneratedAnswerCompletenessEvaluator(_FakeEvaluator):
    async def evaluate(self, **kwargs) -> EvaluationVerdict:
        generated_answer = kwargs["generated_answer"]
        if "Current time" in generated_answer and "Hong Kong weather today" in generated_answer:
            return EvaluationVerdict(
                status="success",
                score=0.95,
                failure_type="none",
                feedback="The generated answer includes both requested fields.",
                repair_instructions=[],
                retry_recommended=False,
                memory_worthy_insights=["Generated answers should preserve every requested field from the final stop block."],
                reasoning="Both the time and weather are present in the final extracted answer.",
            )
        return EvaluationVerdict(
            status="needs_retry",
            score=0.2,
            failure_type="output_assembly.omitted_requested_field",
            feedback="The generated answer omitted one of the requested fields.",
            repair_instructions=["Preserve both the current time and Hong Kong weather in the extracted answer."],
            retry_recommended=True,
            memory_worthy_insights=["Multi-item prompts need answer extraction that keeps all requested fields together."],
            reasoning="The extracted answer is incomplete.",
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


class _MultiPartStopStepExecutor:
    async def execute(self, context, working_dir: Path) -> ExecutionResult:
        log_text = "\n".join(
            [
                "EVENT: step_start #1",
                "STDOUT: I'll inspect the workspace to locate the captured time and weather data, then provide a corrected answer with both elements.",
                "EVENT: step_finish #1 reason=tool-calls cost=$0.000642 tokens(in=911, out=116, total=12291, cache_read=11264, cache_write=0)",
                "EVENT: step_start #2",
                "STDOUT: Perfect! I can see from the execution log that the script ran successfully and captured both the time and weather data. The log shows:",
                "STDOUT: ",
                "STDOUT: 1. **Current time:** 2026-04-01 23:58:25",
                "STDOUT: 2. **Hong Kong weather:** Overcast, 23.2°C (feels like 26.9°C), 85% humidity, no precipitation, light wind from southeast (143°)",
                "STDOUT: ",
                "STDOUT: Now I'll provide the corrected final answer with both elements explicitly included:",
                "STDOUT: ",
                "STDOUT: 🕐 **Current time:** 2026-04-01 23:58:25",
                "STDOUT: ",
                "STDOUT: 🌤️ **Hong Kong weather today:** Overcast, 23.2°C (feels like 26.9°C), 85% humidity, no precipitation, light wind from southeast (143°)",
                "EVENT: step_finish #2 reason=stop cost=$0.000916 tokens(in=1653, out=169, total=14558, cache_read=12736, cache_write=0)",
            ]
        )
        log_path = working_dir / "opencode_execution.log"
        log_path.write_text(log_text, encoding="utf-8")
        return ExecutionResult(
            success=True,
            return_code=0,
            log=log_text,
            log_path=str(log_path),
            prompt_path=None,
            backend="opencode",
            command=["opencode", "run"],
            duration_seconds=0.01,
            timed_out=False,
            usage={
                "wall_time_seconds": 0.01,
                "timed_out": False,
                "estimated_cost_usd": 0.0009,
            },
            telemetry={},
        )


class _UnexpectedCallMetaAgent:
    async def generate_plan(self, **kwargs) -> ExecutionPlan:
        raise AssertionError("MetaAgent should not be called for direct-response prompts.")


class _UnexpectedCallEvaluator:
    async def evaluate(self, **kwargs) -> EvaluationVerdict:
        raise AssertionError("Evaluator should not be called for direct-response prompts.")


class _UnexpectedCallReflector:
    async def synthesize_experience(self, full_history, final_verdict: EvaluationVerdict):
        raise AssertionError("Reflector should not be called for direct-response prompts.")


class _UnexpectedCallExecutor:
    async def execute(self, context, working_dir: Path) -> ExecutionResult:
        raise AssertionError("Executor should not be called for direct-response prompts.")


class _StubDirectResponseRouter:
    def __init__(self, responses: dict[str, DirectResponse]):
        self.responses = responses
        self.calls: list[tuple[str, bool]] = []

    async def maybe_build_direct_response(self, user_request: str, has_uploaded_files: bool = False) -> DirectResponse | None:
        self.calls.append((user_request, has_uploaded_files))
        return self.responses.get(user_request)


def _build_test_config(workspace_dir: Path, *, active_executor_name: str = "claude_code") -> HealthFlowConfig:
    llm_registry = {
        "test-llm": LLMProviderConfig(
            api_key="key",
            base_url="https://example.com/v1",
            model_name="test-model",
        ),
    }
    return HealthFlowConfig(
        planner_llm_name="test-llm",
        evaluator_llm_name="test-llm",
        reflector_llm_name="test-llm",
        executor_llm_name="test-llm",
        active_executor_name=active_executor_name,
        llm_registry=llm_registry,
        system=SystemConfig(max_attempts=1, workspace_dir=str(workspace_dir)),
        environment=EnvironmentConfig(),
        executor=ExecutorConfig(active_backend=active_executor_name, backends=default_executor_backends()),
        memory=MemoryConfig(write_policy="append"),
        evaluation=EvaluationConfig(success_threshold=0.8),
        logging=LoggingConfig(),
    )


class SystemSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_task_writes_structured_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = _build_test_config(workspace_dir)
            experience_path = workspace_root / "memory" / "experience.jsonl"
            system = HealthFlowSystem(config=config, experience_path=experience_path)
            system.meta_agent = _FakeMetaAgent()
            system.evaluator = _FakeEvaluator()
            system.reflector = _FakeReflector()
            system.executor = _FakeExecutor()

            result = await system.run_task(
                "Train a readmission prediction model on the uploaded cohort.",
                uploaded_files={"patients.csv": b"subject_id,outcome,event_time\n1,0,2020-01-01\n"},
                report_requested=True,
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["backend"], "claude_code")
            self.assertEqual(result["executor_model"], "test-model")
            self.assertEqual(result["evaluation_status"], "success")
            self.assertTrue(result["report_requested"])
            self.assertTrue(result["report_generated"])
            self.assertTrue(Path(result["report_path"]).exists())
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
            self.assertTrue(result["available_project_cli_tools"])
            self.assertIn("oneehr", " ".join(result["available_project_cli_tools"]).lower())

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
            self.assertTrue(run_result["report_requested"])
            self.assertTrue(run_result["report_generated"])
            self.assertEqual(Path(run_result["report_path"]).name, "report.md")
            self.assertTrue(run_result["available_project_cli_tools"])
            self.assertIn("oneehr", " ".join(run_result["available_project_cli_tools"]).lower())

            run_manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            self.assertTrue(run_manifest["report_requested"])
            self.assertTrue(run_manifest["report_generated"])
            self.assertEqual(Path(run_manifest["report_path"]).name, "report.md")
            self.assertEqual(run_manifest["execution_environment"]["python_version"], "3.12")
            self.assertTrue(run_manifest["workflow_recommendations"])
            self.assertTrue(run_manifest["available_project_cli_tools"])
            self.assertIn("oneehr", " ".join(run_manifest["available_project_cli_tools"]).lower())

            cost_analysis = json.loads(Path(result["cost_analysis_path"]).read_text(encoding="utf-8"))
            self.assertEqual(cost_analysis["attempts"][0]["execution"]["estimated_cost_usd"], 0.1234)
            self.assertEqual(cost_analysis["attempts"][0]["attempt_total_estimated_cost_usd"], 0.1252)
            self.assertEqual(cost_analysis["run_total"]["reflection"]["estimated_cost_usd"], 0.0003)
            self.assertEqual(cost_analysis["run_total"]["total_estimated_cost_usd"], 0.1255)

            report_text = Path(result["report_path"]).read_text(encoding="utf-8")
            self.assertIn("# HealthFlow Report", report_text)
            self.assertIn("[final_report.md](final_report.md)", report_text)
            self.assertNotIn(str(Path(result["workspace_path"])), report_text)

    async def test_run_task_normalizes_contradictory_success_verdicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = _build_test_config(workspace_dir)
            system = HealthFlowSystem(config=config, experience_path=workspace_root / "memory" / "experience.jsonl")
            system.meta_agent = _FakeMetaAgent()
            system.evaluator = _LowScoreSuccessEvaluator()
            system.reflector = _FakeReflector()
            system.executor = _FakeExecutor()

            result = await system.run_task("Create final_report.md with executor smoke ok.")

            self.assertFalse(result["success"])
            self.assertEqual(result["evaluation_status"], "success")
            self.assertEqual(result["evaluation_score"], 0.1)

    async def test_run_task_generates_report_for_failed_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = _build_test_config(workspace_dir)
            system = HealthFlowSystem(config=config, experience_path=workspace_root / "memory" / "experience.jsonl")
            system.meta_agent = _FakeMetaAgent()
            system.evaluator = _FailedEvaluator()
            system.reflector = _FakeReflector()
            system.executor = _FakeExecutor()

            result = await system.run_task("Create the requested deliverable.", report_requested=True)

            self.assertFalse(result["success"])
            self.assertEqual(result["evaluation_status"], "failed")
            self.assertTrue(result["report_requested"])
            self.assertTrue(result["report_generated"])
            self.assertTrue(Path(result["report_path"]).exists())
            report_text = Path(result["report_path"]).read_text(encoding="utf-8")
            self.assertIn("`failed`", report_text)
            self.assertIn("The task did not satisfy the requested deliverable.", report_text)

    async def test_run_task_writes_memory_when_reflection_produces_experience(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = _build_test_config(workspace_dir)
            experience_path = workspace_root / "memory" / "experience.jsonl"
            system = HealthFlowSystem(config=config, experience_path=experience_path)
            system.meta_agent = _FakeMetaAgent()
            system.evaluator = _FakeEvaluator()
            system.reflector = _MemoryWritingReflector()
            system.executor = _FakeExecutor()

            result = await system.run_task("Summarize the repository structure briefly.")

            self.assertTrue(result["success"])
            self.assertTrue(experience_path.exists())
            saved_lines = [line for line in experience_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(saved_lines), 1)
            saved_memory = json.loads(saved_lines[0])
            self.assertEqual(saved_memory["category"], "repository_summary")
            self.assertEqual(saved_memory["kind"], "workflow")

    async def test_run_task_recovers_multi_item_answer_from_stop_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = _build_test_config(workspace_dir)
            system = HealthFlowSystem(config=config, experience_path=workspace_root / "memory" / "experience.jsonl")
            system.meta_agent = _FakeMetaAgent()
            system.evaluator = _GeneratedAnswerCompletenessEvaluator()
            system.reflector = _FakeReflector()
            system.executor = _MultiPartStopStepExecutor()

            result = await system.run_task("please tell me the time now and how's the weather in hk today")

            self.assertTrue(result["success"])
            self.assertEqual(result["evaluation_status"], "success")
            self.assertIn("Current time", result["answer"])
            self.assertIn("Hong Kong weather today", result["answer"])

    async def test_run_task_uses_direct_response_path_for_lightweight_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = _build_test_config(workspace_dir)
            experience_path = workspace_root / "memory" / "experience.jsonl"
            system = HealthFlowSystem(config=config, experience_path=experience_path)
            system.meta_agent = _UnexpectedCallMetaAgent()
            system.evaluator = _UnexpectedCallEvaluator()
            system.reflector = _UnexpectedCallReflector()
            system.executor = _UnexpectedCallExecutor()
            system.direct_response_router = _StubDirectResponseRouter(
                {
                    "hi, who are you?": DirectResponse(
                        mode="direct_response",
                        category="identity",
                        answer="Hi! I'm HealthFlow, an AI assistant for this workspace.",
                        reason="Classified as a lightweight identity prompt.",
                    )
                }
            )

            result = await system.run_task("hi, who are you?")

            self.assertTrue(result["success"])
            self.assertEqual(result["response_mode"], "direct_response")
            self.assertEqual(result["evaluation_status"], "success")
            self.assertEqual(result["evaluation_score"], 1.0)
            self.assertIn("HealthFlow", result["answer"])
            self.assertEqual(result["available_project_cli_tools"], [])
            self.assertTrue(Path(result["evaluation_path"]).exists())
            self.assertTrue(Path(result["memory_context_path"]).exists())
            history = json.loads((Path(result["workspace_path"]) / "full_history.json").read_text(encoding="utf-8"))
            self.assertEqual(history["attempts"], [])
            self.assertEqual(history["response_mode"], "direct_response")
            self.assertEqual(history["available_project_cli_tools"], [])
            run_result = json.loads(Path(result["run_result_path"]).read_text(encoding="utf-8"))
            self.assertEqual(run_result["available_project_cli_tools"], [])
            run_manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["available_project_cli_tools"], [])

    async def test_run_task_uses_direct_response_path_for_name_question(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            config = _build_test_config(workspace_dir)
            experience_path = workspace_root / "memory" / "experience.jsonl"
            system = HealthFlowSystem(config=config, experience_path=experience_path)
            system.meta_agent = _UnexpectedCallMetaAgent()
            system.evaluator = _UnexpectedCallEvaluator()
            system.reflector = _UnexpectedCallReflector()
            system.executor = _UnexpectedCallExecutor()
            system.direct_response_router = _StubDirectResponseRouter(
                {
                    "what's your name?": DirectResponse(
                        mode="direct_response",
                        category="identity",
                        answer="I'm HealthFlow.",
                        reason="Classified as a lightweight identity prompt.",
                    )
                }
            )

            result = await system.run_task("what's your name?")

            self.assertTrue(result["success"])
            self.assertEqual(result["response_mode"], "direct_response")
            self.assertIn("HealthFlow", result["answer"])
            self.assertNotIn("MetaAgent", result["answer"])
            self.assertEqual(result["available_project_cli_tools"], [])
            history = json.loads((Path(result["workspace_path"]) / "full_history.json").read_text(encoding="utf-8"))
            self.assertEqual(history["direct_response_category"], "identity")
            self.assertEqual(history["response_mode"], "direct_response")
            self.assertEqual(history["available_project_cli_tools"], [])


class SystemAnswerExtractionTests(unittest.TestCase):
    def _build_system(self, workspace_root: Path, workspace_dir: Path) -> HealthFlowSystem:
        config = _build_test_config(workspace_dir)
        return HealthFlowSystem(config=config, experience_path=workspace_root / "memory" / "experience.jsonl")

    def test_extract_answer_prefers_substantive_reply_over_process_narration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            system = self._build_system(workspace_root, workspace_dir)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            task_workspace = workspace_dir / "task"
            task_workspace.mkdir(parents=True, exist_ok=True)
            execution_log = "\n".join(
                [
                    "STDOUT: I'll start by inspecting the workspace to understand the current environment.",
                    "STDOUT: Let me check what's in these files to understand the context better.",
                    "STDOUT: ",
                    "STDOUT: I am HealthFlow, an AI assistant that can help with analysis, coding, and structured task execution in this workspace.",
                    "STDOUT: ",
                    "STDOUT: How can I assist you today?",
                ]
            )

            answer = system._extract_answer_from_workspace(task_workspace, execution_log, "hi, who are you?")

            self.assertIn("I am HealthFlow", answer)
            self.assertIn("How can I assist you today?", answer)
            self.assertNotEqual(answer, "How can I assist you today?")

    def test_extract_answer_keeps_multi_item_stop_step_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            system = self._build_system(workspace_root, workspace_dir)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            task_workspace = workspace_dir / "task"
            task_workspace.mkdir(parents=True, exist_ok=True)
            execution_log = "\n".join(
                [
                    "EVENT: step_start #1",
                    "STDOUT: I'll inspect the workspace to locate the captured time and weather data, then provide a corrected answer with both elements.",
                    "EVENT: step_finish #1 reason=tool-calls cost=$0.000642 tokens(in=911, out=116, total=12291, cache_read=11264, cache_write=0)",
                    "EVENT: step_start #2",
                    "STDOUT: Perfect! I can see from the execution log that the script ran successfully and captured both the time and weather data. The log shows:",
                    "STDOUT: ",
                    "STDOUT: 1. **Current time:** 2026-04-01 23:58:25",
                    "STDOUT: 2. **Hong Kong weather:** Overcast, 23.2°C (feels like 26.9°C), 85% humidity, no precipitation, light wind from southeast (143°)",
                    "STDOUT: ",
                    "STDOUT: Now I'll provide the corrected final answer with both elements explicitly included:",
                    "STDOUT: ",
                    "STDOUT: 🕐 **Current time:** 2026-04-01 23:58:25",
                    "STDOUT: ",
                    "STDOUT: 🌤️ **Hong Kong weather today:** Overcast, 23.2°C (feels like 26.9°C), 85% humidity, no precipitation, light wind from southeast (143°)",
                    "EVENT: step_finish #2 reason=stop cost=$0.000916 tokens(in=1653, out=169, total=14558, cache_read=12736, cache_write=0)",
                ]
            )

            answer = system._extract_answer_from_workspace(
                task_workspace,
                execution_log,
                "please tell me the time now and how's the weather in hk today",
            )

            self.assertTrue(answer.startswith("🕐 **Current time:**"))
            self.assertIn("Hong Kong weather today", answer)
            self.assertNotIn("Now I'll provide", answer)
            self.assertNotIn("1. **Current time:**", answer)

    def test_extract_answer_supports_inline_final_answer_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_dir = workspace_root / "tasks"
            system = self._build_system(workspace_root, workspace_dir)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            task_workspace = workspace_dir / "task"
            task_workspace.mkdir(parents=True, exist_ok=True)
            execution_log = "\n".join(
                [
                    "EVENT: step_start #1",
                    "STDOUT: final answer: executor smoke ok",
                    "EVENT: step_finish #1 reason=stop cost=$0.000100 tokens(in=10, out=3, total=13, cache_read=0, cache_write=0)",
                ]
            )

            answer = system._extract_answer_from_workspace(
                task_workspace,
                execution_log,
                "create final_report.md with executor smoke ok",
            )

            self.assertEqual(answer, "executor smoke ok")


if __name__ == "__main__":
    unittest.main()
