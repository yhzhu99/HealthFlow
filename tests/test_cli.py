import unittest
from unittest.mock import patch

from typer.testing import CliRunner

import run_healthflow


class _FakeSystem:
    async def run_task(self, task, live=None, spinner=None, report_requested=False):
        return {
            "success": True,
            "answer": "Hi! I'm HealthFlow.",
            "final_summary": "Task completed successfully.",
            "execution_time": 0.12,
            "report_generated": False,
        }


class CliOutputTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_run_defaults_to_concise_output(self):
        with patch.object(run_healthflow, "_initialize_system", return_value=_FakeSystem()):
            result = self.runner.invoke(
                run_healthflow.app,
                ["run", "hi, who are you?"],
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hi! I'm HealthFlow.", result.output)
        self.assertIn("Status: success", result.output)
        self.assertNotIn("ANSWER:", result.output)
        self.assertNotIn("Usage Summary", result.output)
        self.assertNotIn("Starting HealthFlow Task", result.output)

    def test_interactive_defaults_to_chat_style_output(self):
        with patch.object(run_healthflow, "_initialize_system", return_value=_FakeSystem()):
            result = self.runner.invoke(
                run_healthflow.app,
                ["interactive"],
                input="hi, who are you?\nexit\n",
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Hi! I'm HealthFlow.", result.output)
        self.assertIn("0.12s", result.output)
        self.assertNotIn("ANSWER:", result.output)
        self.assertNotIn("Usage Summary", result.output)
        self.assertNotIn("Starting HealthFlow Task", result.output)
        self.assertNotIn("Status: success", result.output)


if __name__ == "__main__":
    unittest.main()
