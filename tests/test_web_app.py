import inspect
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

from healthflow.session import HealthFlowProgressEvent, TaskSessionSummary, TaskTurnRecord
from healthflow.web_app import (
    WebTaskSessionStore,
    _artifact_preview_outputs,
    _build_task_choices,
    _build_task_header,
    _default_selected_file,
    _restore_main_history,
    _restore_trace_history,
    _result_answer_text,
    _stream_task_turn,
    _visible_recent_tasks,
)


class _FakeGradio:
    @staticmethod
    def update(**kwargs):
        return kwargs


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
                original_goal="",
                display_title="",
                updated_at_utc="2026-04-08T00:00:00Z",
            )
            self.state_by_task_id[resolved_task_id] = state
        return state

    def load_task_session(self, task_id: str):
        return self.state_by_task_id[task_id]

    def load_task_history(self, task_id: str):
        return list(self.history_by_task_id.get(task_id, []))

    def list_task_sessions(self, limit: int = 20):
        summaries = []
        for task_id, state in self.state_by_task_id.items():
            summaries.append(
                TaskSessionSummary(
                    task_id=task_id,
                    title=state.display_title or state.original_goal or "Untitled task",
                    updated_at_utc=state.updated_at_utc,
                    turn_count=state.turn_count,
                    latest_turn_status=state.latest_turn_status,
                )
            )
        return summaries[:limit]

    def rename_task_session(self, task_id: str, display_title: str):
        state = self.state_by_task_id[task_id]
        state.display_title = display_title.strip()
        state.updated_at_utc = "2026-04-08T05:00:00Z"
        return state

    def delete_task_session(self, task_id: str):
        self.state_by_task_id.pop(task_id, None)
        self.history_by_task_id.pop(task_id, None)


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

    def test_session_store_lists_recent_tasks(self):
        first = self.system.create_task_session("task-a")
        first.original_goal = "Analyze the uploaded vitals cohort"
        second = self.system.create_task_session("task-b")
        second.original_goal = "Summarize model drift findings"
        second.turn_count = 2
        second.latest_turn_status = "success"

        store = WebTaskSessionStore(lambda: self.system)
        summaries = store.list_recent_tasks(limit=10)
        choices = _build_task_choices(summaries)

        self.assertEqual(len(summaries), 2)
        self.assertIn("Analyze the uploaded vitals cohort", choices[0][0])
        self.assertIn("Summarize model drift findings", choices[1][0])

    def test_visible_recent_tasks_keeps_untitled_drafts_accessible(self):
        draft = TaskSessionSummary(task_id="task-draft", title="Untitled task", updated_at_utc="2026-04-08T03:00:00Z")
        previous = TaskSessionSummary(
            task_id="task-real",
            title="Analyze the uploaded vitals cohort",
            updated_at_utc="2026-04-08T01:00:00Z",
            turn_count=2,
            latest_turn_status="success",
        )

        visible = _visible_recent_tasks([draft, previous], current_task_id="task-real")

        self.assertEqual([item.task_id for item in visible], ["task-draft", "task-real"])

    def test_visible_recent_tasks_inserts_current_task_when_missing_from_listing(self):
        previous = TaskSessionSummary(
            task_id="task-real",
            title="Analyze the uploaded vitals cohort",
            updated_at_utc="2026-04-08T01:00:00Z",
            turn_count=2,
            latest_turn_status="success",
        )

        visible = _visible_recent_tasks(
            [previous],
            current_task_id="task-draft",
            current_title="",
            current_updated_at="2026-04-08T04:00:00Z",
            current_turn_count=0,
        )

        self.assertEqual([item.task_id for item in visible], ["task-draft", "task-real"])

    def test_restore_main_history_replays_prior_turns(self):
        task_id = "task-restore"
        state = self.system.create_task_session(task_id)
        state.turn_count = 1
        state.latest_turn_status = "failed"
        state.original_goal = "Analyze this table"
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
        self.assertIn("fresh workspace", main_history[0]["content"])
        self.assertEqual(main_history[1]["content"], "Analyze this table")
        self.assertIn("Run status: failed", main_history[2]["content"])
        self.assertIn("line 42", main_history[2]["content"])
        self.assertIn("Previous execution details", trace_history[0]["content"])
        self.assertIn("Turn 1", trace_history[1]["content"])

    def test_task_header_is_user_facing(self):
        task_id = "task-header"
        state = self.system.create_task_session(task_id)
        state.turn_count = 2
        state.latest_turn_status = "success"
        state.original_goal = "Review the current asthma cohort analysis"
        state.updated_at_utc = "2026-04-08T01:00:00Z"
        client = WebTaskSessionStore(lambda: self.system).get_client(task_id)
        summaries = self.system.list_task_sessions()

        header = _build_task_header(client, summaries)

        self.assertIn("Review the current asthma cohort analysis", header)
        self.assertIn("Last run: **success**", header)
        self.assertNotIn(task_id, header)
        self.assertNotIn("workspace/tasks", header)

    def test_session_store_lists_custom_display_title(self):
        task = self.system.create_task_session("task-custom")
        task.original_goal = "Analyze the uploaded vitals cohort"
        task.display_title = "Pinned history title"

        store = WebTaskSessionStore(lambda: self.system)
        summaries = store.list_recent_tasks(limit=10)

        self.assertEqual(summaries[0].title, "Pinned history title")

    def test_result_answer_text_only_adds_status_for_failures(self):
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

        self.assertEqual(success_text, "All checks passed.")
        self.assertIn("Run status: failed", failed_text)
        self.assertIn("exited with code 1", failed_text)

    def test_default_selected_file_prefers_report(self):
        catalog = [
            {"origin": "generated", "source_path": "/tmp/sandbox/results.csv"},
            {"origin": "report", "source_path": "/tmp/runtime/report.md"},
        ]

        selected = _default_selected_file(catalog)

        self.assertEqual(selected, "/tmp/runtime/report.md")

    def test_artifact_preview_outputs_omit_verbose_metadata(self):
        task_root = self.workspace_dir / "task-preview"
        markdown_path = task_root / "runtime" / "report.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text("# Report\n\nSummary.\n", encoding="utf-8")

        preview_header, preview_markdown, *_ = _artifact_preview_outputs(
            str(markdown_path),
            task_root=task_root,
            gr=_FakeGradio,
        )

        self.assertIn("report.md", preview_header["value"])
        self.assertNotIn("Type:", preview_header["value"])
        self.assertNotIn("Source:", preview_header["value"])
        self.assertNotIn("Updated:", preview_header["value"])
        self.assertNotIn("Size:", preview_header["value"])
        self.assertNotIn("Summary:", preview_header["value"])
        self.assertEqual(preview_markdown["value"], "# Report\n\nSummary.\n")

    def test_stream_task_turn_returns_sync_iterator(self):
        class _FakeStreamingClient:
            task_id = "task-stream"

            async def run_turn(
                self,
                task,
                *,
                live=None,
                spinner=None,
                report_requested=False,
                progress_callback=None,
                uploaded_files=None,
            ):
                if progress_callback is not None:
                    progress_callback(
                        HealthFlowProgressEvent(
                            kind="status",
                            stage="executor",
                            status="running",
                            message="Working",
                        )
                    )
                await __import__("asyncio").sleep(0.02)
                return {"success": True, "answer": "Done"}

        iterator = _stream_task_turn(_FakeStreamingClient(), "run", report_requested=False)

        self.assertFalse(inspect.isasyncgen(iterator))
        kind, payload = next(iterator)
        self.assertEqual(kind, "progress")
        self.assertEqual(payload.stage, "executor")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            iterator.close()
            time.sleep(0.05)

        self.assertFalse(any("aclose" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
