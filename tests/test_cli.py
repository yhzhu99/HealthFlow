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

    async def run_task_turn(
        self,
        task_id,
        task,
        live=None,
        spinner=None,
        report_requested=False,
        progress_callback=None,
        uploaded_files=None,
    ):
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

    def test_web_command_invokes_launcher(self):
        system = _FakeSystem()
        captured = {}

        def _fake_launch(system_factory, *, server_name, server_port, share):
            captured["server_name"] = server_name
            captured["server_port"] = server_port
            captured["share"] = share
            captured["task_id"] = system_factory().create_task_session().task_id

        with patch.object(run_healthflow, "_initialize_system", return_value=system):
            with patch.object(run_healthflow, "launch_web_app", side_effect=_fake_launch):
                result = self.runner.invoke(
                    run_healthflow.app,
                    ["web", "--server-port", "7861", "--share"],
                )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(captured["server_name"], "127.0.0.1")
        self.assertEqual(captured["server_port"], 7861)
        self.assertTrue(captured["share"])
        self.assertEqual(captured["task_id"], "task-1")


if __name__ == "__main__":
    unittest.main()
