import json
import tempfile
import unittest
from pathlib import Path

from healthflow.system import HealthFlowSystem


class TaskSessionListingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_task(self, task_id: str, *, original_goal: str, updated_at_utc: str, turn_count: int, latest_turn_status: str):
        runtime_dir = self.workspace_dir / task_id / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        session_payload = {
            "task_id": task_id,
            "task_root": str(self.workspace_dir / task_id),
            "created_at_utc": "2026-04-08T00:00:00Z",
            "updated_at_utc": updated_at_utc,
            "original_goal": original_goal,
            "turn_count": turn_count,
            "latest_turn_number": turn_count,
            "latest_turn_status": latest_turn_status,
        }
        (runtime_dir / "session.json").write_text(json.dumps(session_payload), encoding="utf-8")
        (runtime_dir / "history.jsonl").write_text("", encoding="utf-8")

    def test_list_task_sessions_returns_recent_tasks_sorted_by_update(self):
        self._write_task(
            "task-old",
            original_goal="Analyze old cohort",
            updated_at_utc="2026-04-08T01:00:00Z",
            turn_count=1,
            latest_turn_status="failed",
        )
        self._write_task(
            "task-new",
            original_goal="Analyze new cohort",
            updated_at_utc="2026-04-08T03:00:00Z",
            turn_count=2,
            latest_turn_status="success",
        )

        system = object.__new__(HealthFlowSystem)
        system.workspace_dir = self.workspace_dir

        summaries = HealthFlowSystem.list_task_sessions(system, limit=10)

        self.assertEqual([item.task_id for item in summaries], ["task-new", "task-old"])
        self.assertEqual(summaries[0].title, "Analyze new cohort")
        self.assertEqual(summaries[0].turn_count, 2)
        self.assertEqual(summaries[0].latest_turn_status, "success")


if __name__ == "__main__":
    unittest.main()
