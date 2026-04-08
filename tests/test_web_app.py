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
    _apply_progress_to_overview,
    _artifact_preview_outputs,
    _branding_header_html,
    _build_history_entries,
    _build_task_choices,
    _build_task_header,
    _composer_attachment_preview_update,
    _demo_case_uploads,
    _draft_browser_task_id,
    _draft_task_title,
    _default_selected_file,
    _empty_run_overview,
    _history_list_html,
    _main_progress_messages,
    _resolved_recent_task_id,
    _restore_main_history,
    _restore_trace_history,
    _result_answer_text,
    _run_overview_from_task_root,
    _run_overview_html,
    _starter_card_html,
    _stream_task_turn,
    _task_id_after_deletion,
    _task_title_text,
    _turn_task_id_for_submission,
    _user_message_content,
    _visible_recent_tasks,
    _workspace_tree_html,
    _workspace_tree_rows,
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

    def test_build_history_entries_formats_status_turns_and_time(self):
        summaries = [
            TaskSessionSummary(
                task_id="task-a",
                title="Analyze the uploaded vitals cohort",
                updated_at_utc="2026-04-08T04:59:00Z",
                turn_count=2,
                latest_turn_status="success",
            )
        ]

        entries = _build_history_entries(summaries)

        self.assertEqual(entries[0]["task_id"], "task-a")
        self.assertEqual(entries[0]["status"], "success")
        self.assertEqual(entries[0]["turns_text"], "2 turns")
        self.assertTrue(entries[0]["updated_text"])

    def test_visible_recent_tasks_hides_inactive_untitled_drafts(self):
        draft = TaskSessionSummary(task_id="task-draft", title="Untitled task", updated_at_utc="2026-04-08T03:00:00Z")
        previous = TaskSessionSummary(
            task_id="task-real",
            title="Analyze the uploaded vitals cohort",
            updated_at_utc="2026-04-08T01:00:00Z",
            turn_count=2,
            latest_turn_status="success",
        )

        visible = _visible_recent_tasks([draft, previous], current_task_id="task-real")

        self.assertEqual([item.task_id for item in visible], ["task-real"])

    def test_visible_recent_tasks_keeps_current_untitled_draft_accessible(self):
        draft = TaskSessionSummary(task_id="task-draft", title="Untitled task", updated_at_utc="2026-04-08T03:00:00Z")
        previous = TaskSessionSummary(
            task_id="task-real",
            title="Analyze the uploaded vitals cohort",
            updated_at_utc="2026-04-08T01:00:00Z",
            turn_count=2,
            latest_turn_status="success",
        )

        visible = _visible_recent_tasks([draft, previous], current_task_id="task-draft")

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

    def test_resolved_recent_task_id_falls_back_to_most_recent_entry(self):
        recent_tasks = [
            TaskSessionSummary(task_id="task-new", title="Newest", updated_at_utc="2026-04-08T05:00:00Z"),
            TaskSessionSummary(task_id="task-old", title="Older", updated_at_utc="2026-04-08T04:00:00Z"),
        ]

        self.assertEqual(_resolved_recent_task_id("task-old", recent_tasks), "task-old")
        self.assertEqual(_resolved_recent_task_id("missing-task", recent_tasks), "task-new")
        self.assertEqual(_resolved_recent_task_id(None, recent_tasks), "task-new")
        self.assertIsNone(_resolved_recent_task_id("missing-task", []))

    def test_task_id_after_deletion_only_switches_when_active_task_is_deleted(self):
        remaining_tasks = [
            TaskSessionSummary(task_id="task-b", title="Task B", updated_at_utc="2026-04-08T04:00:00Z"),
            TaskSessionSummary(task_id="task-c", title="Task C", updated_at_utc="2026-04-08T03:00:00Z"),
        ]

        self.assertEqual(_task_id_after_deletion("task-a", "task-z", remaining_tasks), "task-a")
        self.assertEqual(_task_id_after_deletion("task-a", "task-a", remaining_tasks), "task-b")
        self.assertIsNone(_task_id_after_deletion("task-a", "task-a", []))

    def test_renaming_inactive_sidebar_task_preserves_current_title_context(self):
        active = self.system.create_task_session("task-active")
        active.original_goal = "Current task"
        active.turn_count = 1
        active.latest_turn_status = "success"
        active.updated_at_utc = "2026-04-08T04:00:00Z"

        other = self.system.create_task_session("task-other")
        other.original_goal = "Other task"
        other.updated_at_utc = "2026-04-08T05:00:00Z"

        store = WebTaskSessionStore(lambda: self.system)
        active_client = store.get_client("task-active")
        store.rename_task("task-other", "Renamed sidebar task")

        visible = _visible_recent_tasks(
            store.list_recent_tasks(limit=10),
            current_task_id=active_client.task_id,
            current_title=active_client.display_title or active_client.original_goal,
            current_updated_at=active_client.updated_at_utc,
            current_turn_count=active_client.turn_count,
            current_status=active_client.latest_turn_status,
        )

        self.assertEqual(_task_title_text(active_client, visible), "Current task")
        self.assertEqual(next(item.title for item in visible if item.task_id == "task-other"), "Renamed sidebar task")

    def test_renaming_active_sidebar_task_updates_current_title_context(self):
        active = self.system.create_task_session("task-active")
        active.original_goal = "Current task"
        active.turn_count = 1
        active.latest_turn_status = "success"
        active.updated_at_utc = "2026-04-08T04:00:00Z"

        store = WebTaskSessionStore(lambda: self.system)
        active_client = store.get_client("task-active")
        store.rename_task("task-active", "Pinned current title")
        active_client.refresh_state()

        visible = _visible_recent_tasks(
            store.list_recent_tasks(limit=10),
            current_task_id=active_client.task_id,
            current_title=active_client.display_title or active_client.original_goal,
            current_updated_at=active_client.updated_at_utc,
            current_turn_count=active_client.turn_count,
            current_status=active_client.latest_turn_status,
        )

        self.assertEqual(_task_title_text(active_client, visible), "Pinned current title")

    def test_deleting_inactive_sidebar_task_keeps_current_task_visible(self):
        active = self.system.create_task_session("task-active")
        active.original_goal = "Current task"
        active.turn_count = 1
        active.latest_turn_status = "success"
        active.updated_at_utc = "2026-04-08T04:00:00Z"

        other = self.system.create_task_session("task-other")
        other.original_goal = "Other task"
        other.updated_at_utc = "2026-04-08T05:00:00Z"

        store = WebTaskSessionStore(lambda: self.system)
        active_client = store.get_client("task-active")
        store.delete_task("task-other")

        visible = _visible_recent_tasks(
            store.list_recent_tasks(limit=10),
            current_task_id=active_client.task_id,
            current_title=active_client.display_title or active_client.original_goal,
            current_updated_at=active_client.updated_at_utc,
            current_turn_count=active_client.turn_count,
            current_status=active_client.latest_turn_status,
        )

        self.assertEqual([item.task_id for item in visible], ["task-active"])

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

        self.assertEqual(main_history[0]["role"], "user")
        self.assertEqual(main_history[0]["content"], "Analyze this table")
        self.assertIn("Run status: failed", main_history[1]["content"])
        self.assertIn("line 42", main_history[1]["content"])
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

        self.assertEqual(header, "## Review the current asthma cohort analysis")
        self.assertNotIn(task_id, header)
        self.assertNotIn("workspace/tasks", header)

    def test_user_message_content_lists_uploaded_file_names(self):
        content = _user_message_content(
            "Please inspect the uploaded files and continue the current task.",
            ["cohort.csv", "report draft.md"],
        )

        self.assertIn("hf-chat-attachment", content)
        self.assertIn("cohort.csv", content)
        self.assertIn("report draft.md", content)
        self.assertNotIn("Please inspect the uploaded files", content)

    def test_branding_header_html_is_user_facing(self):
        header_html = _branding_header_html()

        self.assertIn("HealthFlow", header_html)
        self.assertNotIn("Continue a task", header_html)
        self.assertNotIn("Workspace-first AI", header_html)
        self.assertNotIn("Mode:", header_html)
        self.assertNotIn("task_id", header_html)

    def test_starter_card_html_surfaces_demo_case_details(self):
        html = _starter_card_html()

        self.assertIn("EHR Predictive Modeling Demo", html)
        self.assertIn("ehr_predictive_demo.csv", html)
        self.assertIn("roc_curve.png", html)
        self.assertIn("calibration.png", html)

    def test_demo_case_uploads_reads_builtin_demo_file(self):
        uploads = _demo_case_uploads()

        self.assertIn("ehr_predictive_demo.csv", uploads)
        self.assertIn(b"subject_id,mortality,label", uploads["ehr_predictive_demo.csv"])

    def test_composer_attachment_preview_update_lists_selected_file_names(self):
        prompt_input = {
            "text": "",
            "files": [
                {"path": "/tmp/uploads/abc123.txt", "orig_name": "notes.txt"},
                {"path": "/tmp/uploads/x.csv", "name": "labs.csv"},
            ],
        }

        update = _composer_attachment_preview_update(prompt_input, gr=_FakeGradio)

        self.assertTrue(update["visible"])
        self.assertIn("notes.txt", update["value"])
        self.assertIn("labs.csv", update["value"])
        self.assertIn("hf-composer-attachment", update["value"])

    def test_turn_task_id_for_submission_keeps_draft_submissions_new(self):
        draft_browser_task_id = _draft_browser_task_id()

        self.assertIsNone(_turn_task_id_for_submission(None, draft_browser_task_id))
        self.assertIsNone(_turn_task_id_for_submission("task-existing", draft_browser_task_id))
        self.assertEqual(_turn_task_id_for_submission("task-existing", "task-existing"), "task-existing")
        self.assertEqual(_turn_task_id_for_submission(None, "task-existing"), "task-existing")

    def test_draft_task_title_prefers_message_and_falls_back_to_uploads(self):
        self.assertEqual(_draft_task_title("Review the uploaded cohort", ["cohort.csv"]), "Review the uploaded cohort")
        self.assertEqual(_draft_task_title("", ["cohort.csv"]), "Review cohort.csv")
        self.assertEqual(_draft_task_title("", ["cohort.csv", "labs.csv"]), "Review uploaded files")
        self.assertEqual(_draft_task_title("", []), "")

    def test_workspace_tree_rows_group_nested_files(self):
        catalog = [
            {
                "origin": "uploaded",
                "display_name": "cohort.csv",
                "source_path": "/tmp/task/sandbox/cohort.csv",
                "task_relative_path": "sandbox/cohort.csv",
            },
            {
                "origin": "generated",
                "display_name": "summary.json",
                "source_path": "/tmp/task/sandbox/reports/summary.json",
                "task_relative_path": "sandbox/reports/summary.json",
            },
        ]

        rows = _workspace_tree_rows(catalog)

        self.assertEqual(rows[0]["kind"], "file")
        self.assertEqual(rows[0]["label"], "cohort.csv")
        self.assertEqual(rows[1]["kind"], "folder")
        self.assertEqual(rows[1]["label"], "reports")
        self.assertEqual(rows[2]["kind"], "file")
        self.assertEqual(rows[2]["label"], "summary.json")

    def test_history_list_html_renders_inline_actions_and_active_state(self):
        recent_tasks = [
            TaskSessionSummary(task_id="task-a", title="Alpha", updated_at_utc="2026-04-08T05:00:00Z"),
            TaskSessionSummary(task_id="task-b", title="Beta", updated_at_utc="2026-04-08T04:00:00Z"),
        ]

        html = _history_list_html(recent_tasks, "task-a")

        self.assertIn('data-task-id="task-a"', html)
        self.assertIn('data-history-action="rename"', html)
        self.assertIn('data-history-action="delete"', html)
        self.assertIn("hf-history-card is-active", html)

    def test_workspace_tree_html_includes_report_and_selected_file(self):
        catalog = [
            {
                "origin": "report",
                "display_name": "report.md",
                "source_path": "/tmp/task/runtime/report.md",
                "task_relative_path": "runtime/report.md",
            },
            {
                "origin": "generated",
                "display_name": "summary.json",
                "source_path": "/tmp/task/sandbox/reports/summary.json",
                "task_relative_path": "sandbox/reports/summary.json",
            },
        ]

        html = _workspace_tree_html(catalog, "/tmp/task/sandbox/reports/summary.json")

        self.assertIn("report.md", html)
        self.assertIn("hf-tree-folder", html)
        self.assertIn("hf-tree-file", html)
        self.assertIn("is-active", html)

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

    def test_apply_progress_to_overview_tracks_planner_objective_and_steps(self):
        overview = _apply_progress_to_overview(
            _empty_run_overview(),
            HealthFlowProgressEvent(
                kind="stage_finished",
                stage="planner",
                status="completed",
                attempt=2,
                message="Build the predictive modeling plan.",
                metadata={
                    "objective": "Predict in-hospital mortality.",
                    "recommended_steps": ["Inspect schema.", "Train baseline.", "Render ROC."],
                    "avoidances": ["Do not leak labels."],
                },
            ),
        )

        self.assertEqual(overview["attempt"], 2)
        self.assertEqual(overview["objective"], "Predict in-hospital mortality.")
        self.assertEqual(overview["recommended_steps"][1], "Train baseline.")
        self.assertEqual(overview["avoidances"], ["Do not leak labels."])
        self.assertEqual(overview["stage_status"]["planner"], "done")
        self.assertIn("Attempt 2", _run_overview_html(overview))

    def test_main_progress_messages_inline_generated_images(self):
        task_root = self.workspace_dir / "task-inline"
        image_path = task_root / "sandbox" / "figures" / "roc_curve.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"png")

        messages = _main_progress_messages(
            HealthFlowProgressEvent(
                kind="artifact_delta",
                stage="executor",
                status="completed",
                metadata={"artifacts": ["figures/roc_curve.png"]},
            ),
            task_root=task_root,
            seen_image_paths=set(),
        )

        self.assertEqual(messages[0]["metadata"]["title"], "Executor")
        self.assertEqual(messages[1]["content"], (str(image_path), "roc_curve.png"))

    def test_run_overview_from_task_root_restores_latest_plan_summary(self):
        task_root = self.workspace_dir / "task-overview"
        run_dir = task_root / "runtime" / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trajectory.json").write_text(
            inspect.cleandoc(
                """
                {
                  "task_id": "task-overview",
                  "user_request": "Predict mortality.",
                  "attempts": [
                    {
                      "attempt": 1,
                      "memory": {"retrieval": {}},
                      "plan": {
                        "objective": "Predict mortality.",
                        "recommended_steps": ["Inspect schema.", "Train baseline."]
                      },
                      "execution": {"success": true, "cancelled": false},
                      "evaluation": {"status": "success", "feedback": "Looks good."},
                      "gate": {"retry_recommended": false}
                    }
                  ]
                }
                """
            ),
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(
            inspect.cleandoc(
                """
                {
                  "success": true,
                  "final_summary": "Completed successfully.",
                  "evaluation_status": "success"
                }
                """
            ),
            encoding="utf-8",
        )

        overview = _run_overview_from_task_root(task_root)

        self.assertEqual(overview["mode"], "completed")
        self.assertEqual(overview["objective"], "Predict mortality.")
        self.assertEqual(overview["recommended_steps"], ["Inspect schema.", "Train baseline."])
        self.assertEqual(overview["stage_status"]["reflection"], "skipped")

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
