from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifacts import collect_task_artifacts, read_structured_preview
from .session import HealthFlowProgressEvent, TaskSessionSummary
from .session_client import TaskSessionClient

_MAIN_ASSISTANT_TEXT = (
    "Describe your task, upload files if needed, and keep iterating here. Use New Task when you want "
    "to start over."
)
_TRACE_ASSISTANT_TEXT = "Advanced execution details for this task will appear here."
_EMPTY_FILES_TEXT = "Generated files and uploads will appear here."
_EMPTY_PREVIEW_TEXT = "Select a file to preview it."
_FILTER_DEFINITIONS = (
    ("All", "all"),
    ("Uploaded", "uploaded"),
    ("Generated", "generated"),
    ("Reports", "reports"),
    ("Data", "data"),
    ("Images", "images"),
    ("Code", "code"),
    ("Other", "other"),
)
_CATEGORY_LABELS = {
    "reports/docs": "Document",
    "images": "Image",
    "code/notebooks": "Code",
    "tables/data": "Data",
    "other outputs": "Other",
}
_ORIGIN_LABELS = {
    "uploaded": "Uploaded",
    "generated": "Generated",
    "report": "Report",
}


class WebTaskSessionStore:
    def __init__(self, system_factory: Callable[[], Any]):
        self._system_factory = system_factory
        self._clients: dict[str, TaskSessionClient] = {}
        self._listing_system: Any | None = None

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

    def list_recent_tasks(self, limit: int = 20) -> list[TaskSessionSummary]:
        if self._listing_system is None:
            self._listing_system = self._system_factory()
        if hasattr(self._listing_system, "list_task_sessions"):
            return list(self._listing_system.list_task_sessions(limit=limit))
        return []


def _status_label(status: str | None) -> str:
    normalized = (status or "ready").strip().lower()
    if normalized in {"success", "completed"}:
        return "success"
    if normalized in {"cancelled", "canceled"}:
        return "cancelled"
    if normalized in {"failed", "failure", "error"}:
        return "failed"
    if normalized in {"needs_retry", "retry"}:
        return "needs retry"
    return normalized or "ready"


def _truncate_text(text: str, max_chars: int = 72) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return "New task"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _relative_time_text(value: str | None) -> str:
    timestamp = _parse_utc_timestamp(value)
    if timestamp is None:
        return "recently"
    delta = datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)
    total_seconds = max(int(delta.total_seconds()), 0)
    if total_seconds < 60:
        return "just now"
    if total_seconds < 3600:
        minutes = total_seconds // 60
        return f"{minutes}m ago"
    if total_seconds < 86400:
        hours = total_seconds // 3600
        return f"{hours}h ago"
    days = total_seconds // 86400
    return f"{days}d ago"


def _task_title(client: TaskSessionClient, recent_tasks: Sequence[TaskSessionSummary]) -> str:
    for item in recent_tasks:
        if item.task_id == client.task_id:
            return _truncate_text(item.title)
    return _truncate_text(client.original_goal or "")


def _build_task_choices(recent_tasks: Sequence[TaskSessionSummary]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for item in recent_tasks:
        title = _truncate_text(item.title, max_chars=44)
        status = _status_label(item.latest_turn_status)
        turns = f"{item.turn_count} turn" if item.turn_count == 1 else f"{item.turn_count} turns"
        label = f"{title} · {status} · {turns} · {_relative_time_text(item.updated_at_utc)}"
        choices.append((label, item.task_id))
    return choices


def _visible_recent_tasks(
    recent_tasks: Sequence[TaskSessionSummary],
    *,
    current_task_id: str,
) -> list[TaskSessionSummary]:
    visible_tasks = [
        item
        for item in recent_tasks
        if item.task_id == current_task_id or item.turn_count > 0 or item.title.strip().lower() not in {"", "new task"}
    ]
    return visible_tasks or [item for item in recent_tasks if item.task_id == current_task_id]


def _build_task_header(client: TaskSessionClient, recent_tasks: Sequence[TaskSessionSummary]) -> str:
    title = _task_title(client, recent_tasks)
    if client.turn_count <= 0:
        return (
            f"## {title}\n\n"
            "Describe the task or upload files to begin. Files you share will appear in the Files panel."
        )

    status = _status_label(client.latest_turn_status)
    updated_text = _relative_time_text(client.updated_at_utc)
    turn_text = "turn" if client.turn_count == 1 else "turns"
    return (
        f"## {title}\n\n"
        f"Last run: **{status}**. This task has **{client.turn_count} {turn_text}** and was updated **{updated_text}**."
    )


def _result_answer_text(result: dict[str, Any]) -> str:
    answer = str(result.get("answer") or "").strip() or "No answer available."
    if result.get("success"):
        return answer

    status = "cancelled" if result.get("cancelled") else "failed"
    summary = str(result.get("final_summary") or "").strip()
    lines = [answer, f"_Run status: {status}_"]
    if summary:
        lines.append(summary)
    return "\n\n".join(lines)


def _history_answer_text(record: Any) -> str:
    answer = str(getattr(record, "answer", "") or "").strip() or "No answer available."
    status = _status_label(getattr(record, "status", None))
    if status == "success":
        return answer
    feedback = str(getattr(record, "evaluation_feedback", "") or "").strip()
    lines = [answer, f"_Run status: {status}_"]
    if feedback:
        lines.append(feedback)
    return "\n\n".join(lines)


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
        message = (
            "Previous execution details for this task are available here."
            if restored
            else _TRACE_ASSISTANT_TEXT
        )
        return [{"role": "assistant", "content": message}]

    trace_history = [
        {
            "role": "assistant",
            "content": "Previous execution details for this task are listed below. New progress updates will appear here.",
        }
    ]
    for record in history[-5:]:
        status = _status_label(record.status)
        lines = [f"**Turn {record.turn_number}** · `{status}`"]
        if record.evaluation_feedback:
            lines.append(record.evaluation_feedback.strip())
        trace_history.append({"role": "assistant", "content": "\n\n".join(lines)})
    return trace_history


def _collect_artifact_catalog(client: TaskSessionClient) -> list[dict[str, Any]]:
    task_root = Path(client.task_root) if client.task_root else None
    if task_root is None or not task_root.exists():
        return []
    return collect_task_artifacts(task_root, client.load_history())


def _filter_catalog(catalog: Sequence[dict[str, Any]], filter_value: str | None) -> list[dict[str, Any]]:
    normalized = (filter_value or "all").strip().lower()
    if normalized == "all":
        return list(catalog)
    if normalized == "uploaded":
        return [item for item in catalog if item.get("origin") == "uploaded"]
    if normalized == "generated":
        return [item for item in catalog if item.get("origin") == "generated"]
    if normalized == "reports":
        return [item for item in catalog if item.get("origin") == "report" or item.get("category") == "reports/docs"]
    if normalized == "data":
        return [item for item in catalog if item.get("category") == "tables/data"]
    if normalized == "images":
        return [item for item in catalog if item.get("category") == "images"]
    if normalized == "code":
        return [item for item in catalog if item.get("category") == "code/notebooks"]
    if normalized == "other":
        return [item for item in catalog if item.get("category") == "other outputs"]
    return list(catalog)


def _artifact_filter_choices(catalog: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for label, value in _FILTER_DEFINITIONS:
        count = len(_filter_catalog(catalog, value))
        choices.append((f"{label} ({count})", value))
    return choices


def _artifact_selection_label(item: dict[str, Any]) -> str:
    display_name = str(item.get("display_name") or "artifact")
    origin_label = _ORIGIN_LABELS.get(str(item.get("origin") or ""), "File")
    category_label = _CATEGORY_LABELS.get(str(item.get("category") or ""), "File")
    return f"{display_name} · {origin_label} · {category_label}"


def _default_selected_artifact(
    filtered_catalog: Sequence[dict[str, Any]],
    preferred_artifact: str | None = None,
) -> str | None:
    if preferred_artifact and any(item.get("task_relative_path") == preferred_artifact for item in filtered_catalog):
        return preferred_artifact
    if not filtered_catalog:
        return None
    for origin in ("report", "generated", "uploaded"):
        for item in filtered_catalog:
            if item.get("origin") == origin:
                return str(item.get("task_relative_path"))
    return str(filtered_catalog[0].get("task_relative_path"))


def _artifact_selector_update(
    catalog: Sequence[dict[str, Any]],
    filter_value: str,
    preferred_artifact: str | None,
    *,
    gr: Any,
) -> tuple[Any, str | None]:
    filtered_catalog = _filter_catalog(catalog, filter_value)
    selected_path = _default_selected_artifact(filtered_catalog, preferred_artifact=preferred_artifact)
    choices = [(_artifact_selection_label(item), str(item.get("task_relative_path"))) for item in filtered_catalog]
    return gr.update(choices=choices, value=selected_path), selected_path


def _human_size(size_bytes: Any) -> str:
    try:
        size = float(size_bytes)
    except (TypeError, ValueError):
        return "Unknown size"
    units = ["B", "KB", "MB", "GB"]
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.1f} {units[unit_index]}"


def _read_text_preview(path: Path, *, max_chars: int = 50000) -> tuple[str | None, bool]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, False
    if len(content) <= max_chars:
        return content, False
    trimmed = content[: max_chars - 25].rstrip()
    return f"{trimmed}\n\n[Preview truncated]", True


def _preview_meta_text(item: dict[str, Any], *, truncated: bool = False) -> str:
    display_name = str(item.get("display_name") or "artifact")
    descriptor = str(item.get("descriptor") or "").strip()
    origin_label = _ORIGIN_LABELS.get(str(item.get("origin") or ""), "File")
    category_label = _CATEGORY_LABELS.get(str(item.get("category") or ""), "File")
    updated_text = _relative_time_text(str(item.get("updated_at") or ""))
    meta_lines = [
        f"### {display_name}",
        f"**Type:** {category_label}  \n**Source:** {origin_label}  \n**Updated:** {updated_text}  \n**Size:** {_human_size(item.get('size_bytes'))}",
    ]
    if descriptor:
        meta_lines.append(f"**Summary:** {descriptor}")
    if truncated:
        meta_lines.append("_Preview truncated to keep the UI responsive._")
    return "\n\n".join(meta_lines)


def _artifact_preview_outputs(
    catalog: Sequence[dict[str, Any]],
    selected_artifact: str | None,
    *,
    gr: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    if not catalog:
        return (
            gr.update(value="### Preview", visible=True),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value=_EMPTY_FILES_TEXT, visible=True),
            gr.update(value=None, visible=False),
        )

    selected_item = next((item for item in catalog if item.get("task_relative_path") == selected_artifact), None)
    if selected_item is None:
        return (
            gr.update(value="### Preview", visible=True),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value=_EMPTY_PREVIEW_TEXT, visible=True),
            gr.update(value=None, visible=False),
        )

    path = Path(str(selected_item.get("source_path")))
    download_update = gr.update(value=str(path), visible=True)
    preview_kind = str(selected_item.get("preview_kind") or "download")

    if preview_kind == "markdown":
        content, truncated = _read_text_preview(path)
        if content is None:
            return (
                gr.update(value=_preview_meta_text(selected_item), visible=True),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(value="", language=None, visible=False),
                gr.update(value="Preview not available for this file type. Download to inspect it locally.", visible=True),
                download_update,
            )
        return (
            gr.update(value=_preview_meta_text(selected_item, truncated=truncated), visible=True),
            gr.update(value=content, visible=True),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value="", visible=False),
            download_update,
        )

    if preview_kind == "image":
        return (
            gr.update(value=_preview_meta_text(selected_item), visible=True),
            gr.update(value="", visible=False),
            gr.update(value=str(path), visible=True),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value="", visible=False),
            download_update,
        )

    if preview_kind == "table":
        structured_preview = read_structured_preview(path)
        if structured_preview is not None:
            return (
                gr.update(
                    value=_preview_meta_text(selected_item, truncated=bool(structured_preview.get("truncated"))),
                    visible=True,
                ),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(
                    value=structured_preview.get("rows") or [],
                    headers=structured_preview.get("headers") or [],
                    visible=True,
                ),
                gr.update(value="", language=None, visible=False),
                gr.update(value="", visible=False),
                download_update,
            )

    if preview_kind == "code":
        content, truncated = _read_text_preview(path)
        if content is not None:
            return (
                gr.update(value=_preview_meta_text(selected_item, truncated=truncated), visible=True),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(value=content, language=selected_item.get("preview_language"), visible=True),
                gr.update(value="", visible=False),
                download_update,
            )

    return (
        gr.update(value=_preview_meta_text(selected_item), visible=True),
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        gr.update(value="", language=None, visible=False),
        gr.update(value="Preview not available for this file type. Download to inspect it locally.", visible=True),
        download_update,
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

    def _compose_outputs(
        client: TaskSessionClient,
        *,
        main_history: list[dict[str, str]],
        trace_history: list[dict[str, str]],
        artifact_filter: str | None,
        preferred_artifact: str | None,
    ) -> tuple[Any, list[dict[str, str]], list[dict[str, str]], str, str, list[dict[str, Any]], Any, Any, str | None, Any, Any, Any, Any, Any, Any, Any]:
        recent_tasks = session_store.list_recent_tasks(limit=24)
        visible_recent_tasks = _visible_recent_tasks(recent_tasks, current_task_id=client.task_id)
        catalog = _collect_artifact_catalog(client)
        normalized_filter = artifact_filter if artifact_filter in {value for _, value in _FILTER_DEFINITIONS} else "all"
        recent_tasks_update = gr.update(choices=_build_task_choices(visible_recent_tasks), value=client.task_id)
        artifact_filter_update = gr.update(choices=_artifact_filter_choices(catalog), value=normalized_filter)
        artifact_selector_update, selected_artifact = _artifact_selector_update(
            catalog,
            normalized_filter,
            preferred_artifact,
            gr=gr,
        )
        preview_outputs = _artifact_preview_outputs(catalog, selected_artifact, gr=gr)
        return (
            recent_tasks_update,
            main_history,
            trace_history,
            client.task_id,
            _build_task_header(client, visible_recent_tasks),
            catalog,
            artifact_filter_update,
            artifact_selector_update,
            selected_artifact,
            *preview_outputs,
        )

    def _format_progress_event(event: HealthFlowProgressEvent) -> str:
        if event.kind == "log_chunk":
            body = (event.message or "").strip()
            body = body[-3000:] if len(body) > 3000 else body
            if body:
                return f"### Executor Log\n\n```text\n{body}\n```"
        if event.kind == "artifact_delta":
            artifacts = event.metadata.get("artifacts") if isinstance(event.metadata, dict) else None
            if artifacts:
                artifact_lines = "\n".join(f"- `{item}`" for item in artifacts[:10])
                return f"### Files Updated\n\n{artifact_lines}"
        message = (event.message or f"{event.stage}: {event.status}").strip()
        return f"**{event.stage.title()}** · {message}"

    def _submit_display_text(text: str, upload_count: int) -> str:
        cleaned = text.strip()
        if cleaned:
            return cleaned
        if upload_count == 1:
            return "Uploaded 1 file."
        if upload_count > 1:
            return f"Uploaded {upload_count} files."
        return ""

    def _load_session(task_id: str | None):
        client = session_store.get_client(task_id)
        return _compose_outputs(
            client,
            main_history=_restore_main_history(client),
            trace_history=_restore_trace_history(client, restored=bool(task_id)),
            artifact_filter="all",
            preferred_artifact=None,
        )

    def _switch_task(task_id: str | None):
        client = session_store.get_client(task_id)
        return _compose_outputs(
            client,
            main_history=_restore_main_history(client),
            trace_history=_restore_trace_history(client, restored=True),
            artifact_filter="all",
            preferred_artifact=None,
        )

    def _new_task():
        client = session_store.new_client()
        return _compose_outputs(
            client,
            main_history=_restore_main_history(client),
            trace_history=_restore_trace_history(client, restored=False),
            artifact_filter="all",
            preferred_artifact=None,
        )

    def _on_artifact_filter_change(
        artifact_filter: str,
        artifact_catalog: list[dict[str, Any]] | None,
        selected_artifact: str | None,
    ):
        catalog = list(artifact_catalog or [])
        artifact_selector_update, selected_value = _artifact_selector_update(
            catalog,
            artifact_filter or "all",
            selected_artifact,
            gr=gr,
        )
        preview_outputs = _artifact_preview_outputs(catalog, selected_value, gr=gr)
        return artifact_selector_update, selected_value, *preview_outputs

    def _on_artifact_selected(
        selected_artifact: str | None,
        artifact_catalog: list[dict[str, Any]] | None,
    ):
        catalog = list(artifact_catalog or [])
        preview_outputs = _artifact_preview_outputs(catalog, selected_artifact, gr=gr)
        return selected_artifact, *preview_outputs

    async def _run_turn(
        prompt_input: dict[str, Any] | None,
        main_history: list[dict[str, str]] | None,
        trace_history: list[dict[str, str]] | None,
        task_id: str | None,
        report_requested: bool,
        artifact_filter: str | None,
        selected_artifact: str | None,
    ):
        client = session_store.get_client(task_id)
        main_history = list(main_history or _restore_main_history(client))
        trace_history = list(trace_history or _restore_trace_history(client, restored=bool(task_id)))
        prompt_input = prompt_input or {}
        text = str(prompt_input.get("text") or "")
        files = list(prompt_input.get("files") or [])

        upload_payloads: dict[str, bytes] = {}
        for raw_file in files:
            file_path = Path(getattr(raw_file, "path", raw_file))
            if not file_path.exists():
                continue
            upload_payloads[file_path.name] = file_path.read_bytes()

        if not text.strip() and not upload_payloads:
            yield _compose_outputs(
                client,
                main_history=main_history,
                trace_history=trace_history,
                artifact_filter=artifact_filter,
                preferred_artifact=selected_artifact,
            )
            return

        user_display = _submit_display_text(text, len(upload_payloads))
        user_message = text.strip() or "Please inspect the uploaded files and continue the current task."
        main_history.append({"role": "user", "content": user_display})
        trace_history.append({"role": "assistant", "content": "Working on your latest request."})
        yield _compose_outputs(
            client,
            main_history=main_history,
            trace_history=trace_history,
            artifact_filter=artifact_filter,
            preferred_artifact=selected_artifact,
        )

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
                yield _compose_outputs(
                    client,
                    main_history=main_history,
                    trace_history=trace_history,
                    artifact_filter=artifact_filter,
                    preferred_artifact=selected_artifact,
                )
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

        preferred_artifact = None if (artifact_filter or "all") == "all" else selected_artifact
        yield _compose_outputs(
            client,
            main_history=main_history,
            trace_history=trace_history,
            artifact_filter=artifact_filter,
            preferred_artifact=preferred_artifact,
        )

    with gr.Blocks(title="HealthFlow Web") as demo:
        browser_task_id = gr.BrowserState(None, storage_key="healthflow-active-task")
        artifact_catalog_state = gr.State([])
        selected_artifact_state = gr.State(None)

        gr.Markdown(
            """
            # HealthFlow

            A browser workspace for task-driven analysis. Switch between recent tasks, keep the conversation
            going, and preview uploaded or generated files without leaving the page.
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                recent_tasks = gr.Radio(
                    label="Recent Tasks",
                    choices=[],
                    value=None,
                    interactive=True,
                )
                new_task_button = gr.Button("New Task", variant="primary")
            with gr.Column(scale=4):
                task_header = gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=3):
                        main_chatbot = gr.Chatbot(
                            label="Conversation",
                            type="messages",
                            value=[{"role": "assistant", "content": _MAIN_ASSISTANT_TEXT}],
                            height=720,
                            show_copy_button=True,
                        )
                        prompt_input = gr.MultimodalTextbox(
                            interactive=True,
                            file_count="multiple",
                            placeholder="Describe the task or provide follow-up feedback. Upload files if needed.",
                            show_label=False,
                        )
                        gr.Markdown(
                            "Uploads appear in the **Files** panel automatically. Use **New Task** when you want a fresh workspace."
                        )
                    with gr.Column(scale=2):
                        report_requested = gr.Checkbox(label="Generate report.md for each turn", value=False)
                        with gr.Tabs():
                            with gr.Tab("Files"):
                                artifact_filter = gr.Radio(
                                    label="Browse",
                                    choices=_artifact_filter_choices([]),
                                    value="all",
                                    interactive=True,
                                )
                                artifact_list = gr.Radio(
                                    label="Files",
                                    choices=[],
                                    value=None,
                                    interactive=True,
                                )
                                preview_meta = gr.Markdown(value="### Preview")
                                preview_markdown = gr.Markdown(visible=False)
                                preview_image = gr.Image(
                                    label="Image preview",
                                    visible=False,
                                    type="filepath",
                                    show_download_button=False,
                                )
                                preview_table = gr.Dataframe(
                                    label="Data preview",
                                    visible=False,
                                    interactive=False,
                                    show_copy_button=True,
                                    max_height=360,
                                )
                                preview_code = gr.Code(
                                    label="Code preview",
                                    visible=False,
                                    interactive=False,
                                    lines=18,
                                    max_lines=32,
                                )
                                preview_empty = gr.Markdown(value=_EMPTY_FILES_TEXT)
                                download_button = gr.DownloadButton("Download file", visible=False)
                            with gr.Tab("Advanced"):
                                trace_chatbot = gr.Chatbot(
                                    label="Execution Trace",
                                    type="messages",
                                    value=[{"role": "assistant", "content": _TRACE_ASSISTANT_TEXT}],
                                    height=620,
                                    show_copy_button=True,
                                )

        demo.load(
            _load_session,
            [browser_task_id],
            [
                recent_tasks,
                main_chatbot,
                trace_chatbot,
                browser_task_id,
                task_header,
                artifact_catalog_state,
                artifact_filter,
                artifact_list,
                selected_artifact_state,
                preview_meta,
                preview_markdown,
                preview_image,
                preview_table,
                preview_code,
                preview_empty,
                download_button,
            ],
            queue=False,
            show_progress="hidden",
        )

        recent_tasks.change(
            _switch_task,
            [recent_tasks],
            [
                recent_tasks,
                main_chatbot,
                trace_chatbot,
                browser_task_id,
                task_header,
                artifact_catalog_state,
                artifact_filter,
                artifact_list,
                selected_artifact_state,
                preview_meta,
                preview_markdown,
                preview_image,
                preview_table,
                preview_code,
                preview_empty,
                download_button,
            ],
            queue=False,
            show_progress="hidden",
        )

        new_task_button.click(
            _new_task,
            None,
            [
                recent_tasks,
                main_chatbot,
                trace_chatbot,
                browser_task_id,
                task_header,
                artifact_catalog_state,
                artifact_filter,
                artifact_list,
                selected_artifact_state,
                preview_meta,
                preview_markdown,
                preview_image,
                preview_table,
                preview_code,
                preview_empty,
                download_button,
            ],
            queue=False,
            show_progress="hidden",
        )

        artifact_filter.change(
            _on_artifact_filter_change,
            [artifact_filter, artifact_catalog_state, selected_artifact_state],
            [
                artifact_list,
                selected_artifact_state,
                preview_meta,
                preview_markdown,
                preview_image,
                preview_table,
                preview_code,
                preview_empty,
                download_button,
            ],
            queue=False,
            show_progress="hidden",
        )

        artifact_list.change(
            _on_artifact_selected,
            [artifact_list, artifact_catalog_state],
            [
                selected_artifact_state,
                preview_meta,
                preview_markdown,
                preview_image,
                preview_table,
                preview_code,
                preview_empty,
                download_button,
            ],
            queue=False,
            show_progress="hidden",
        )

        prompt_input.submit(
            _run_turn,
            [
                prompt_input,
                main_chatbot,
                trace_chatbot,
                browser_task_id,
                report_requested,
                artifact_filter,
                selected_artifact_state,
            ],
            [
                recent_tasks,
                main_chatbot,
                trace_chatbot,
                browser_task_id,
                task_header,
                artifact_catalog_state,
                artifact_filter,
                artifact_list,
                selected_artifact_state,
                preview_meta,
                preview_markdown,
                preview_image,
                preview_table,
                preview_code,
                preview_empty,
                download_button,
            ],
        ).then(lambda: gr.MultimodalTextbox(value=None), None, [prompt_input])

    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, share=share)
