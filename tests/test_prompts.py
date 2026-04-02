import json
import unittest

from healthflow.agents.meta_agent import MetaAgent
from healthflow.experience.experience_models import Experience, MemoryKind, SourceOutcome
from healthflow.prompts.templates import render_prompt


class _StubPlannerProvider:
    model_name = "planner-test-model"


class PromptRenderingTests(unittest.TestCase):
    def test_meta_agent_prompt_omits_empty_optional_sections(self):
        agent = MetaAgent(_StubPlannerProvider())

        rendered = agent._build_user_prompt(
            user_request="Build a cohort table.",
            safeguard_experiences=[],
            workflow_experiences=[],
            dataset_experiences=[],
            execution_experiences=[],
            execution_environment=["Preferred Python version: 3.12"],
            workflow_recommendations=[],
            previous_feedback=None,
        )

        self.assertIn("User request:", rendered)
        self.assertIn("Execution environment:", rendered)
        self.assertNotIn("Workflow recommendations:", rendered)
        self.assertNotIn("EHR safeguards:", rendered)
        self.assertNotIn("Workflow memories:", rendered)
        self.assertNotIn("Dataset memories:", rendered)
        self.assertNotIn("Execution memories:", rendered)
        self.assertNotIn("Feedback from Previous Failed Attempt:", rendered)
        self.assertNotIn("No EHR safeguard memory was retrieved.", rendered)
        self.assertNotIn("No workflow memory was retrieved.", rendered)

    def test_meta_agent_prompt_renders_optional_sections_when_present(self):
        agent = MetaAgent(_StubPlannerProvider())
        safeguard = Experience(
            kind=MemoryKind.SAFEGUARD,
            category="cohort_boundary",
            content="Validate inclusion criteria before writing the final cohort.",
            source_task_id="task-1",
            task_family="cohorting",
            dataset_signature="demo",
            stage="reflection",
            backend="codex",
            source_outcome=SourceOutcome.SUCCESS,
            applicability_scope="domain_ehr",
        )

        rendered = agent._build_user_prompt(
            user_request="Build a cohort table.",
            safeguard_experiences=[safeguard],
            workflow_experiences=[],
            dataset_experiences=[],
            execution_experiences=[],
            execution_environment=["Preferred Python version: 3.12"],
            workflow_recommendations=["Prefer `uv run` for repo-local scripts."],
            previous_feedback="The prior attempt never wrote the requested artifact.",
        )

        self.assertIn("Workflow recommendations:", rendered)
        self.assertIn("EHR safeguards:", rendered)
        self.assertIn("Feedback from Previous Failed Attempt:", rendered)
        self.assertIn("Validate inclusion criteria", rendered)
        self.assertIn("prior attempt never wrote the requested artifact", rendered.lower())

    def test_evaluator_prompt_renders_literal_json_block_and_runtime_json_values(self):
        generated_answer = json.dumps(
            {
                "status": "ok",
                "details": {"note": "literal braces should not crash"},
            },
            indent=2,
        )

        rendered = render_prompt(
            "evaluator_user",
            user_request="Summarize the task.",
            plan_markdown="1. Inspect\n2. Answer",
            execution_log='STDOUT: {"step": "done"}',
            generated_answer=generated_answer,
            workspace_artifacts="- final_report.md",
        )

        self.assertIn('"status": "<success|needs_retry|failed>"', rendered)
        self.assertIn('"violated_constraints": ["<constraint or contract that was violated>"]', rendered)
        self.assertIn('"details": {', rendered)
        self.assertIn('STDOUT: {"step": "done"}', rendered)

    def test_reflector_prompt_renders_embedded_history_json(self):
        task_history = json.dumps({"attempts": [{"execution": {"log": '{"ok": true}'}}]}, indent=2)

        rendered = render_prompt("reflector_user", task_history=task_history)

        self.assertIn('"experiences": [', rendered)
        self.assertIn('"memory_updates": [', rendered)
        self.assertIn('\\"ok\\": true', rendered)


if __name__ == "__main__":
    unittest.main()
