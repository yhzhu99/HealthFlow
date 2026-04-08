import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from healthflow.session import TaskTurnRecord
from healthflow.web_app import (
    WebTaskSessionStore,
    _artifact_files,
    _restore_main_history,
    _restore_trace_history,
    _result_answer_text,
    _task_info,
)


class _FakeSystem:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.created_task_ids: list[str | None] = []
        self.history_by_task_id: dict[str, list[TaskTurnRecord]] = {}
        self.state_by_task_id: dict[str, SimpleNamespace] = {}

    def create_task_session(self, task_id: str | None = None):
        resolved_task_id = task_id or f"task-{len(self.state_by_task_id) + 1}"
        self.created_task_ids.append(task_id)
        state = self.state_by_task_id.get(resolved_task_id)
        if state is None:
            task_root = self.workspace_dir / resolved_task_id
            (task_root / "sandbox").mkdir(parents=True, exist_ok=True)
            (task_root / "runtime").mkdir(parents=True, exist_ok=True)
            state = SimpleNamespace(
                task_id=resolved_task_id,
                task_root=str(task_root),
                turn_count=0,
                latest_turn_status=None,
            )
            self.state_by_task_id[resolved_task_id] = state
        return state

    def load_task_session(self, task_id: str):
        return self.state_by_task_id[task_id]

    def load_task_history(self, task_id: str):
        return list(self.history_by_task_id.get(task_id, []))


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_dir = Path(self.temp_dir.name)
        self.system = _FakeSystem(self.workspace_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_session_store_reuses_cached_client_for_same_task(self):
        factory_calls = 0

        def _factory():
            nonlocal factory_calls
            factory_calls += 1
            return self.system

        store = WebTaskSessionStore(_factory)
        first_client = store.new_client()
        restored_client = store.get_client(first_client.task_id)

        self.assertIs(restored_client, first_client)
        self.assertEqual(factory_calls, 1)

    def test_restore_main_history_replays_prior_turns_with_status(self):
        task_id = "task-restore"
        state = self.system.create_task_session(task_id)
        state.turn_count = 1
        state.latest_turn_status = "failed"
        self.system.history_by_task_id[task_id] = [
            TaskTurnRecord(
                turn_number=1,
                user_message="Analyze this table",
                answer="The run produced a draft answer.",
                status="failed",
                runtime_dir=str(self.workspace_dir / task_id / "runtime" / "turns" / "turn_001"),
                evaluation_feedback="The CSV parser crashed near line 42.",
            )
        ]

        client = WebTaskSessionStore(lambda: self.system).get_client(task_id)
        main_history = _restore_main_history(client)
        trace_history = _restore_trace_history(client, restored=True)

        self.assertEqual(main_history[0]["role"], "assistant")
        self.assertIn("Follow-ups stay on this task", main_history[0]["content"])
        self.assertEqual(main_history[1]["content"], "Analyze this table")
        self.assertIn("Status: failed", main_history[2]["content"])
        self.assertIn("line 42", main_history[2]["content"])
        self.assertIn("Reopened task session", trace_history[0]["content"])
        self.assertIn("Turn 1", trace_history[1]["content"])

    def test_task_info_and_artifacts_reflect_latest_state(self):
        task_id = "task-artifacts"
        state = self.system.create_task_session(task_id)
        state.turn_count = 2
        state.latest_turn_status = "success"
        task_root = Path(state.task_root)
        sandbox_file = task_root / "sandbox" / "result.csv"
        sandbox_file.write_text("a,b\n1,2\n", encoding="utf-8")
        report_path = task_root / "runtime" / "report.md"
        report_path.write_text("# Report\n", encoding="utf-8")

        client = WebTaskSessionStore(lambda: self.system).get_client(task_id)
        info = _task_info(client)
        artifacts = _artifact_files(client)

        self.assertIn("Mode:** Web UI", info)
        self.assertIn("Completed turns:", info)
        self.assertIn("Latest status:", info)
        self.assertIn(str(sandbox_file), artifacts)
        self.assertIn(str(report_path), artifacts)

    def test_result_answer_text_includes_explicit_status(self):
        success_text = _result_answer_text(
            {
                "success": True,
                "answer": "All checks passed.",
                "execution_time": 1.23,
            }
        )
        failed_text = _result_answer_text(
            {
                "success": False,
                "answer": "",
                "final_summary": "The executor exited with code 1.",
                "execution_time": 2.5,
            }
        )
        cancelled_text = _result_answer_text(
            {
                "success": False,
                "cancelled": True,
                "answer": "Partial output",
                "final_summary": "Task cancelled before completion.",
                "execution_time": 0.5,
            }
        )

        self.assertIn("Status: success", success_text)
        self.assertIn("Status: failed", failed_text)
        self.assertIn("exited with code 1", failed_text)
        self.assertIn("Status: cancelled", cancelled_text)


if __name__ == "__main__":
    unittest.main()
