import unittest
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

import run_healthflow


class _FakeSystem:
    def __init__(self):
        self.created_sessions = []
        self.turn_calls = []

    def create_task_session(self):
        task_id = f"task-{len(self.created_sessions) + 1}"
        self.created_sessions.append(task_id)
        return SimpleNamespace(task_id=task_id)

    async def run_task_turn(self, task_id, task, live=None, spinner=None, report_requested=False, progress_callback=None):
        self.turn_calls.append((task_id, task))
        return {
            "success": True,
            "answer": "Hi! I'm HealthFlow.",
            "final_summary": "Task completed successfully.",
            "execution_time": 0.12,
            "report_generated": False,
            "task_id": task_id,
        }

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

    def test_interactive_reuses_task_until_new(self):
        system = _FakeSystem()
        with patch.object(run_healthflow, "_initialize_system", return_value=system):
            result = self.runner.invoke(
                run_healthflow.app,
                ["interactive"],
                input="first task\nfollow up\n/new\nfresh task\nexit\n",
            )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(system.turn_calls[0][0], system.turn_calls[1][0])
        self.assertNotEqual(system.turn_calls[1][0], system.turn_calls[2][0])


if __name__ == "__main__":
    unittest.main()
