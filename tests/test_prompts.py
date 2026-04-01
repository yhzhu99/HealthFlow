import json
import unittest

from healthflow.prompts.templates import render_prompt


class PromptRenderingTests(unittest.TestCase):
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
