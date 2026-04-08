from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from .session import HealthFlowProgressEvent
from .session_client import TaskSessionClient

_MAIN_ASSISTANT_TEXT = (
    "This is HealthFlow Web. Describe the task, upload files if needed, and keep refining the same "
    "task here. Follow-ups stay on this task until you click New Task."
)
_TRACE_ASSISTANT_TEXT = (
    "Planner, executor, evaluator, and artifact updates will appear here. Refreshing the page keeps "
    "the same task session."
)


class WebTaskSessionStore:
    def __init__(self, system_factory: Callable[[], Any]):
        self._system_factory = system_factory
        self._clients: dict[str, TaskSessionClient] = {}

    def get_client(self, task_id: str | None = None) -> TaskSessionClient:
        normalized_task_id = str(task_id).strip() if task_id else None
        if normalized_task_id and normalized_task_id in self._clients:
            return self._clients[normalized_task_id]

        client = TaskSessionClient(self._system_factory(), task_id=normalized_task_id)
        self._clients[client.task_id] = client
        return client

    def new_client(self) -> TaskSessionClient:
        client = TaskSessionClient(self._system_factory())
        self._clients[client.task_id] = client
        return client


def _status_label(status: str | None) -> str:
    normalized = (status or "ready").strip().lower()
    if normalized in {"success", "completed"}:
        return "success"
    if normalized in {"cancelled", "canceled"}:
        return "cancelled"
    if normalized in {"failed", "failure", "error"}:
        return "failed"
    if normalized in {"needs_retry", "retry"}:
        return "needs_retry"
    return normalized or "ready"


def _task_info(client: TaskSessionClient) -> str:
    task_root = Path(client.task_root) if client.task_root else None
    lines = [
        f"**Mode:** Web UI",
        f"**Active task:** `{client.task_id}`",
        "Follow-up messages keep working on this same task until you click **New Task**.",
    ]
    if client.turn_count:
        lines.append(f"**Completed turns:** `{client.turn_count}`")
    if client.latest_turn_status:
        lines.append(f"**Latest status:** `{_status_label(client.latest_turn_status)}`")
    if task_root is not None:
        lines.append(f"**Workspace:** `{task_root}`")
        lines.append(f"**Sandbox:** `{task_root / 'sandbox'}`")
    return "\n\n".join(lines)


def _artifact_files(client: TaskSessionClient, result: dict[str, Any] | None = None) -> list[str]:
    task_root = Path(client.task_root) if client.task_root else None
    if task_root is None:
        return []

    sandbox_dir = task_root / "sandbox"
    files = [str(path) for path in sorted(sandbox_dir.rglob("*")) if path.is_file()]
    runtime_report = task_root / "runtime" / "report.md"
    if runtime_report.exists():
        files.append(str(runtime_report))
    if result and result.get("report_path"):
        files.append(str(result["report_path"]))
    return list(dict.fromkeys(files))[:100]


def _restore_main_history(client: TaskSessionClient) -> list[dict[str, str]]:
    history = client.load_history()
    main_history: list[dict[str, str]] = [{"role": "assistant", "content": _MAIN_ASSISTANT_TEXT}]
    for record in history:
        main_history.append({"role": "user", "content": record.user_message})
        main_history.append({"role": "assistant", "content": _history_answer_text(record)})
    return main_history


def _restore_trace_history(client: TaskSessionClient, *, restored: bool) -> list[dict[str, str]]:
    history = client.load_history()
    if not history:
        if restored:
            message = (
                f"Reopened task session `{client.task_id}`. New planner, executor, evaluator, and "
                "artifact updates will appear here."
            )
        else:
            message = _TRACE_ASSISTANT_TEXT
        return [{"role": "assistant", "content": message}]

    trace_history = [
        {
            "role": "assistant",
            "content": (
                f"Reopened task session `{client.task_id}` from saved runtime history. Prior turns are "
                "listed below; new progress updates will stream here."
            ),
        }
    ]
    for record in history[-5:]:
        status = _status_label(record.status)
        lines = [f"**Turn {record.turn_number}** · `{status}`"]
        if record.evaluation_feedback:
            lines.append(record.evaluation_feedback.strip())
        if record.report_path:
            lines.append(f"Report: `{record.report_path}`")
        trace_history.append({"role": "assistant", "content": "\n\n".join(lines)})
    return trace_history


def _history_answer_text(record: Any) -> str:
    answer = str(getattr(record, "answer", "") or "").strip() or "No answer available."
    status = _status_label(getattr(record, "status", None))
    lines = [answer, f"_Status: {status}_"]
    feedback = str(getattr(record, "evaluation_feedback", "") or "").strip()
    if feedback and status != "success":
        lines.append(feedback)
    return "\n\n".join(lines)


def _result_answer_text(result: dict[str, Any]) -> str:
    answer = str(result.get("answer") or "").strip() or "No answer available."
    status = "cancelled" if result.get("cancelled") else ("success" if result.get("success") else "failed")
    lines = [answer, f"_Status: {status} · {result.get('execution_time', 0):.2f}s_"]
    summary = str(result.get("final_summary") or "").strip()
    if summary and status != "success":
        lines.append(summary)
    return "\n\n".join(lines)


def _session_view(
    client: TaskSessionClient,
    *,
    restored: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str, str, list[str]]:
    return (
        _restore_main_history(client),
        _restore_trace_history(client, restored=restored),
        client.task_id,
        _task_info(client),
        _artifact_files(client),
    )


def launch_web_app(
    system_factory: Callable[[], Any],
    *,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is not installed. Install the web extra first, for example: uv sync --extra web"
        ) from None

    session_store = WebTaskSessionStore(system_factory)

    def _format_progress_event(event: HealthFlowProgressEvent) -> str:
        if event.kind == "log_chunk":
            body = (event.message or "").strip()
            body = body[-3000:] if len(body) > 3000 else body
            if body:
                return f"### Executor Log\n\n```text\n{body}\n```"
        if event.kind == "artifact_delta":
            artifacts = event.metadata.get("artifacts") if isinstance(event.metadata, dict) else None
            if artifacts:
                artifact_lines = "\n".join(f"- `{item}`" for item in artifacts[:15])
                return f"### Artifacts Updated\n\n{artifact_lines}"
        message = (event.message or f"{event.stage}: {event.status}").strip()
        return f"**{event.stage.title()}** · {message}"

    def _submit_display_text(text: str, upload_names: list[str]) -> str:
        cleaned = text.strip()
        if cleaned and upload_names:
            return f"{cleaned}\n\nUploaded files: {', '.join(upload_names)}"
        if upload_names:
            return f"Uploaded files: {', '.join(upload_names)}"
        return cleaned

    async def _run_turn(
        prompt_input: dict[str, Any] | None,
        main_history: list[dict[str, str]] | None,
        trace_history: list[dict[str, str]] | None,
        task_id: str | None,
        report_requested: bool,
    ):
        client = session_store.get_client(task_id)
        main_history = list(main_history or _restore_main_history(client))
        trace_history = list(trace_history or _restore_trace_history(client, restored=bool(task_id)))
        prompt_input = prompt_input or {}
        text = str(prompt_input.get("text") or "")
        files = list(prompt_input.get("files") or [])

        upload_payloads: dict[str, bytes] = {}
        upload_names: list[str] = []
        for raw_file in files:
            file_path = Path(getattr(raw_file, "path", raw_file))
            if not file_path.exists():
                continue
            upload_payloads[file_path.name] = file_path.read_bytes()
            upload_names.append(file_path.name)

        if not text.strip() and not upload_payloads:
            yield main_history, trace_history, client.task_id, _task_info(client), _artifact_files(client)
            return

        user_display = _submit_display_text(text, upload_names)
        user_message = text.strip() or "Please inspect the uploaded files and continue the current task."
        main_history.append({"role": "user", "content": user_display})
        trace_history.append({"role": "assistant", "content": f"Starting turn on `{client.task_id}`."})
        yield main_history, trace_history, client.task_id, _task_info(client), _artifact_files(client)

        queue: asyncio.Queue[HealthFlowProgressEvent] = asyncio.Queue()

        def _progress_callback(event: HealthFlowProgressEvent) -> None:
            queue.put_nowait(event)

        task = asyncio.create_task(
            client.run_turn(
                user_message,
                report_requested=report_requested,
                progress_callback=_progress_callback,
                uploaded_files=upload_payloads or None,
            )
        )

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                trace_history.append({"role": "assistant", "content": _format_progress_event(event)})
                yield main_history, trace_history, client.task_id, _task_info(client), _artifact_files(client)
            except asyncio.TimeoutError:
                if task.done():
                    break

        while not queue.empty():
            event = queue.get_nowait()
            trace_history.append({"role": "assistant", "content": _format_progress_event(event)})

        result = await task
        summary = str(result.get("final_summary") or "").strip()
        main_history.append({"role": "assistant", "content": _result_answer_text(result)})
        if summary:
            trace_history.append({"role": "assistant", "content": f"**Run summary**\n\n{summary}"})
        yield main_history, trace_history, client.task_id, _task_info(client), _artifact_files(client, result)

    def _new_task(
        _task_id: str | None,
    ):
        client = session_store.new_client()
        main_history, trace_history, task_id, task_info, artifact_files = _session_view(client, restored=False)
        trace_history[0]["content"] = f"Started a fresh task session: `{client.task_id}`."
        return main_history, trace_history, task_id, task_info, artifact_files

    def _load_session(task_id: str | None):
        client = session_store.get_client(task_id)
        return _session_view(client, restored=bool(task_id))

    with gr.Blocks(title="HealthFlow Web") as demo:
        browser_task_id = gr.BrowserState(None, storage_key="healthflow-active-task")
        gr.Markdown(
            """
            # HealthFlow Web

            Browser mode for HealthFlow. Work in one task session, upload files as you go, and keep
            giving follow-up feedback in the same workspace until you click **New Task**.
            """
        )
        with gr.Row():
            task_info = gr.Markdown()
            report_requested = gr.Checkbox(label="Generate report.md for each turn", value=False)
            new_task_button = gr.Button("New Task", variant="primary")
        with gr.Row():
            with gr.Column(scale=3):
                main_chatbot = gr.Chatbot(
                    label="HealthFlow",
                    type="messages",
                    value=[{"role": "assistant", "content": _MAIN_ASSISTANT_TEXT}],
                    height=680,
                    show_copy_button=True,
                )
            with gr.Column(scale=2):
                trace_chatbot = gr.Chatbot(
                    label="Execution Trace",
                    type="messages",
                    value=[{"role": "assistant", "content": _TRACE_ASSISTANT_TEXT}],
                    height=680,
                    show_copy_button=True,
                )
        prompt_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="Describe the task or provide follow-up feedback. Upload files if needed.",
            show_label=False,
        )
        artifact_files = gr.Files(label="Current task artifacts", interactive=False)

        demo.load(
            _load_session,
            [browser_task_id],
            [main_chatbot, trace_chatbot, browser_task_id, task_info, artifact_files],
            queue=False,
            show_progress="hidden",
        )

        prompt_input.submit(
            _run_turn,
            [prompt_input, main_chatbot, trace_chatbot, browser_task_id, report_requested],
            [main_chatbot, trace_chatbot, browser_task_id, task_info, artifact_files],
        ).then(lambda: gr.MultimodalTextbox(value=None), None, [prompt_input])

        new_task_button.click(
            _new_task,
            [browser_task_id],
            [main_chatbot, trace_chatbot, browser_task_id, task_info, artifact_files],
            queue=False,
        )

    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, share=share)
