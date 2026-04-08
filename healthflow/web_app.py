from __future__ import annotations

import asyncio
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifacts import artifact_preview_kind, artifact_preview_language, collect_task_artifacts, read_structured_preview
from .session import HealthFlowProgressEvent, TaskSessionSummary
from .session_client import TaskSessionClient

_MAIN_ASSISTANT_TEXT = (
    "Describe a task, upload files if needed, and keep iterating in the same workspace. Start a new task "
    "when you want a fresh workspace."
)
_TRACE_ASSISTANT_TEXT = "Advanced execution details for this task will appear here."
_EMPTY_WORKSPACE_TEXT = "No workspace files yet."
_EMPTY_PREVIEW_TEXT = "Select a file to preview it."
_BRANDING_DIR = Path(__file__).resolve().parent.parent / "assets" / "branding"
_WEB_APP_CSS = """
.gradio-container {
    max-width: 100% !important;
    padding: 0 1.25rem 1.25rem !important;
}

.hf-main {
    min-height: calc(100vh - 9rem);
    gap: 1rem;
}

.hf-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    margin: 0 0 1rem;
    padding: 1.25rem 1.5rem;
    border: 1px solid rgba(28, 168, 255, 0.14);
    border-radius: 28px;
    background: linear-gradient(135deg, #f4fbff 0%, #ecf8ff 54%, #ffffff 100%);
}

.hf-hero__brand {
    flex: 0 0 auto;
}

.hf-hero__brand svg {
    display: block;
    width: min(100%, 320px);
    height: auto;
}

.hf-hero__fallback {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #102033;
}

.hf-hero__copy {
    max-width: 38rem;
}

.hf-hero__eyebrow {
    margin: 0 0 0.35rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0b84db;
}

.hf-hero__summary {
    margin: 0;
    font-size: 1rem;
    line-height: 1.6;
    color: #415166;
}

@media (max-width: 900px) {
    .hf-hero {
        flex-direction: column;
        align-items: flex-start;
    }

    .hf-main {
        min-height: calc(100vh - 10rem);
    }
}
"""


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

    def has_task(self, task_id: str | None) -> bool:
        normalized_task_id = str(task_id).strip() if task_id else ""
        if not normalized_task_id:
            return False
        if self._listing_system is None:
            self._listing_system = self._system_factory()
        if hasattr(self._listing_system, "load_task_session"):
            try:
                self._listing_system.load_task_session(normalized_task_id)
            except FileNotFoundError:
                return False
            return True
        return any(item.task_id == normalized_task_id for item in self.list_recent_tasks(limit=0))

    def rename_task(self, task_id: str, title: str) -> TaskSessionClient:
        client = self.get_client(task_id)
        client.rename(title)
        return client

    def delete_task(self, task_id: str) -> None:
        if task_id in self._clients:
            self._clients[task_id].delete()
            self._clients.pop(task_id, None)
            return
        if not self.has_task(task_id):
            raise FileNotFoundError(f"Task session '{task_id}' does not exist.")
        client = self.get_client(task_id)
        client.delete()
        self._clients.pop(task_id, None)


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
        return "Untitled task"
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


def _task_title_text(client: TaskSessionClient, recent_tasks: Sequence[TaskSessionSummary]) -> str:
    for item in recent_tasks:
        if item.task_id == client.task_id:
            return str(item.title or "").strip() or "Untitled task"
    return client.display_title.strip() or client.original_goal.strip() or "Untitled task"


def _task_title(client: TaskSessionClient, recent_tasks: Sequence[TaskSessionSummary]) -> str:
    return _truncate_text(_task_title_text(client, recent_tasks))


def _load_branding_svg(filename: str) -> str:
    try:
        return (_BRANDING_DIR / filename).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _branding_header_html() -> str:
    logo_svg = _load_branding_svg("healthflow-logo.svg")
    if logo_svg:
        brand_markup = f'<div class="hf-hero__brand" aria-hidden="true">{logo_svg}</div>'
    else:
        brand_markup = '<p class="hf-hero__fallback">HealthFlow</p>'

    return (
        '<section class="hf-hero">'
        f"{brand_markup}"
        '<div class="hf-hero__copy">'
        '<p class="hf-hero__eyebrow">Workspace-first AI</p>'
        "<p class=\"hf-hero__summary\">Continue a task, switch across task history, and preview workspace files "
        "without leaving the page.</p>"
        "</div>"
        "</section>"
    )


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
    current_title: str = "",
    current_updated_at: str | None = None,
    current_turn_count: int = 0,
    current_status: str | None = None,
) -> list[TaskSessionSummary]:
    task_map = {item.task_id: item for item in recent_tasks}
    if current_task_id and current_task_id not in task_map:
        task_map[current_task_id] = TaskSessionSummary(
            task_id=current_task_id,
            title=current_title.strip() or "Untitled task",
            updated_at_utc=current_updated_at or "",
            turn_count=current_turn_count,
            latest_turn_status=current_status,
        )
    tasks = list(task_map.values())
    tasks.sort(key=lambda item: (item.updated_at_utc or "", item.created_at_utc or ""), reverse=True)
    return tasks


def _build_task_header(client: TaskSessionClient, recent_tasks: Sequence[TaskSessionSummary]) -> str:
    title = _task_title(client, recent_tasks)
    if client.turn_count <= 0:
        return (
            f"## {title}\n\n"
            "Describe the task or upload files to begin. Your workspace files will appear on the right."
        )

    status = _status_label(client.latest_turn_status)
    updated_text = _relative_time_text(client.updated_at_utc)
    turn_text = "turn" if client.turn_count == 1 else "turns"
    return (
        f"## {title}\n\n"
        f"Last run: **{status}**. This task has **{client.turn_count} {turn_text}** and was updated **{updated_text}**."
    )


def _history_notice_update(message: str | None, *, gr: Any) -> Any:
    text = str(message or "").strip()
    return gr.update(value=text, visible=bool(text))


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


def _read_text_preview(path: Path, *, max_chars: int = 50000) -> tuple[str | None, bool]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, False
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars].rstrip(), True


def _default_selected_file(
    catalog: Sequence[dict[str, Any]],
    preferred_file: str | None = None,
) -> str | None:
    if preferred_file and any(str(item.get("source_path")) == preferred_file for item in catalog):
        return preferred_file
    if not catalog:
        return None
    report_item = next((item for item in catalog if item.get("origin") == "report"), None)
    if report_item is not None:
        return str(report_item.get("source_path"))
    return str(catalog[0].get("source_path"))


def _task_root_path(task_id: str | None, session_store: WebTaskSessionStore) -> Path | None:
    if not task_id or not session_store.has_task(task_id):
        return None
    client = session_store.get_client(task_id)
    if not client.task_root:
        return None
    return Path(client.task_root)


def _task_relative_label(path: Path, task_root: Path | None) -> str:
    if task_root is None:
        return path.name
    try:
        return path.relative_to(task_root).as_posix()
    except ValueError:
        return path.name


def _preview_header_text(path: Path, *, task_root: Path | None) -> str:
    relative_label = _task_relative_label(path, task_root)
    if relative_label and relative_label != path.name:
        return f"### {path.name}\n\n`{relative_label}`"
    return f"### {path.name}"


def _workspace_tree_value(selected_file: str | None, *, task_root: Path | None) -> str | None:
    if not selected_file or task_root is None:
        return None
    sandbox_root = task_root / "sandbox"
    path = Path(selected_file)
    try:
        path.relative_to(sandbox_root)
    except ValueError:
        return None
    return str(path)


def _artifact_preview_outputs(
    selected_file: str | None,
    *,
    task_root: Path | None,
    gr: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    if not selected_file:
        return (
            gr.update(value="### Preview", visible=True),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value=_EMPTY_PREVIEW_TEXT, visible=True),
            gr.update(value=None, visible=False),
        )

    path = Path(selected_file)
    if not path.exists():
        return (
            gr.update(value="### Preview", visible=True),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value=_EMPTY_PREVIEW_TEXT, visible=True),
            gr.update(value=None, visible=False),
        )

    if path.is_dir():
        return (
            gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value="Choose a file in this folder to preview it.", visible=True),
            gr.update(value=None, visible=False),
        )

    download_update = gr.update(value=str(path), visible=True)
    preview_kind = artifact_preview_kind(path)

    if preview_kind == "markdown":
        content, _ = _read_text_preview(path)
        if content is None:
            return (
                gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(value="", language=None, visible=False),
                gr.update(value="Preview not available for this file type.", visible=True),
                download_update,
            )
        return (
            gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
            gr.update(value=content, visible=True),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(value="", language=None, visible=False),
            gr.update(value="", visible=False),
            download_update,
        )

    if preview_kind == "image":
        return (
            gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
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
                gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
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
        content, _ = _read_text_preview(path)
        if content is not None:
            return (
                gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(value=content, language=artifact_preview_language(path), visible=True),
                gr.update(value="", visible=False),
                download_update,
            )

    return (
        gr.update(value=_preview_header_text(path, task_root=task_root), visible=True),
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        gr.update(value="", language=None, visible=False),
        gr.update(value="Preview not available for this file type.", visible=True),
        download_update,
    )


def _stream_task_turn(
    client: TaskSessionClient,
    user_message: str,
    *,
    report_requested: bool,
    uploaded_files: dict[str, bytes] | None = None,
):
    stream_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    closed = threading.Event()

    def _progress_callback(event: HealthFlowProgressEvent) -> None:
        if not closed.is_set():
            stream_queue.put(("progress", event))

    def _runner() -> None:
        try:
            result = asyncio.run(
                client.run_turn(
                    user_message,
                    report_requested=report_requested,
                    progress_callback=_progress_callback,
                    uploaded_files=uploaded_files,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive fallback for unexpected runtime failures
            if not closed.is_set():
                stream_queue.put(("error", exc))
        else:
            if not closed.is_set():
                stream_queue.put(("result", result))
        finally:
            if not closed.is_set():
                stream_queue.put(("done", None))

    worker = threading.Thread(target=_runner, name=f"healthflow-web-turn-{client.task_id[:8]}", daemon=True)
    worker.start()

    finished = False
    try:
        while not finished or not stream_queue.empty():
            try:
                kind, payload = stream_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "done":
                finished = True
                continue
            yield kind, payload
    finally:
        closed.set()


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

    def _client_for_requested_task(task_id: str | None) -> TaskSessionClient:
        if task_id and session_store.has_task(task_id):
            return session_store.get_client(task_id)
        return session_store.new_client()

    def _compose_outputs(
        client: TaskSessionClient,
        *,
        main_history: list[dict[str, str]],
        trace_history: list[dict[str, str]],
        preferred_file: str | None,
        history_notice: str | None = None,
    ) -> tuple[Any, list[dict[str, str]], list[dict[str, str]], str, str, list[dict[str, Any]], str | None, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        recent_tasks = session_store.list_recent_tasks(limit=50)
        visible_recent_tasks = _visible_recent_tasks(
            recent_tasks,
            current_task_id=client.task_id,
            current_title=client.display_title or client.original_goal,
            current_updated_at=client.updated_at_utc,
            current_turn_count=client.turn_count,
            current_status=client.latest_turn_status,
        )
        catalog = _collect_artifact_catalog(client)
        selected_file = _default_selected_file(catalog, preferred_file=preferred_file)
        task_root = Path(client.task_root) if client.task_root else None
        recent_tasks_update = gr.update(choices=_build_task_choices(visible_recent_tasks), value=client.task_id)
        preview_outputs = _artifact_preview_outputs(selected_file, task_root=task_root, gr=gr)
        return (
            recent_tasks_update,
            main_history,
            trace_history,
            client.task_id,
            _build_task_header(client, visible_recent_tasks),
            catalog,
            selected_file,
            gr.update(value=_task_title_text(client, visible_recent_tasks)),
            gr.update(value=False),
            gr.update(interactive=False),
            _history_notice_update(history_notice, gr=gr),
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
        client = _client_for_requested_task(task_id)
        return _compose_outputs(
            client,
            main_history=_restore_main_history(client),
            trace_history=_restore_trace_history(client, restored=bool(task_id)),
            preferred_file=None,
        )

    def _switch_task(task_id: str | None):
        client = _client_for_requested_task(task_id)
        return _compose_outputs(
            client,
            main_history=_restore_main_history(client),
            trace_history=_restore_trace_history(client, restored=True),
            preferred_file=None,
        )

    def _new_task():
        client = session_store.new_client()
        return _compose_outputs(
            client,
            main_history=_restore_main_history(client),
            trace_history=_restore_trace_history(client, restored=False),
            preferred_file=None,
        )

    def _preview_outputs_for_task(selected_file: str | None, task_id: str | None):
        task_root = _task_root_path(task_id, session_store)
        return _artifact_preview_outputs(selected_file, task_root=task_root, gr=gr)

    def _on_workspace_file_selected(selected_file: str | None, task_id: str | None):
        return selected_file, *_preview_outputs_for_task(selected_file, task_id)

    def _open_report_file(report_path: str | None, task_id: str | None):
        return report_path, *_preview_outputs_for_task(report_path, task_id)

    def _toggle_delete_button(delete_confirmed: bool):
        return gr.update(interactive=bool(delete_confirmed))

    def _rename_task(
        task_title: str | None,
        main_history: list[dict[str, str]] | None,
        trace_history: list[dict[str, str]] | None,
        task_id: str | None,
        selected_file: str | None,
    ):
        client = _client_for_requested_task(task_id)
        main_history = list(main_history or _restore_main_history(client))
        trace_history = list(trace_history or _restore_trace_history(client, restored=bool(task_id)))
        session_store.rename_task(client.task_id, str(task_title or ""))
        client.refresh_state()
        notice = "Custom title cleared." if not str(task_title or "").strip() else "Title updated."
        return _compose_outputs(
            client,
            main_history=main_history,
            trace_history=trace_history,
            preferred_file=selected_file,
            history_notice=notice,
        )

    def _delete_task(
        delete_confirmed: bool,
        task_id: str | None,
    ):
        if not task_id:
            client = session_store.new_client()
            return _compose_outputs(
                client,
                main_history=_restore_main_history(client),
                trace_history=_restore_trace_history(client, restored=False),
                preferred_file=None,
            )
        if not delete_confirmed:
            client = _client_for_requested_task(task_id)
            return _compose_outputs(
                client,
                main_history=_restore_main_history(client),
                trace_history=_restore_trace_history(client, restored=bool(task_id)),
                preferred_file=None,
                history_notice="Confirm deletion before removing this task.",
            )

        session_store.delete_task(task_id)
        remaining_tasks = session_store.list_recent_tasks(limit=50)
        if remaining_tasks:
            client = session_store.get_client(remaining_tasks[0].task_id)
            return _compose_outputs(
                client,
                main_history=_restore_main_history(client),
                trace_history=_restore_trace_history(client, restored=True),
                preferred_file=None,
                history_notice="Task deleted.",
            )

        client = session_store.new_client()
        return _compose_outputs(
            client,
            main_history=_restore_main_history(client),
            trace_history=_restore_trace_history(client, restored=False),
            preferred_file=None,
            history_notice="Task deleted.",
        )

    def _run_turn(
        prompt_input: dict[str, Any] | None,
        main_history: list[dict[str, str]] | None,
        trace_history: list[dict[str, str]] | None,
        task_id: str | None,
        report_requested: bool,
        selected_file: str | None,
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
                preferred_file=selected_file,
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
            preferred_file=selected_file,
        )

        result: dict[str, Any] | None = None
        error: Exception | None = None
        for kind, payload in _stream_task_turn(
            client,
            user_message,
            report_requested=report_requested,
            uploaded_files=upload_payloads or None,
        ):
            if kind == "progress":
                event = payload
                trace_history.append({"role": "assistant", "content": _format_progress_event(event)})
                yield _compose_outputs(
                    client,
                    main_history=main_history,
                    trace_history=trace_history,
                    preferred_file=selected_file,
                )
                continue
            if kind == "error":
                error = payload
                break
            if kind == "result":
                result = payload

        if error is not None:
            result = {
                "success": False,
                "answer": "The web session failed before the run completed.",
                "final_summary": str(error),
            }

        result = result or {
            "success": False,
            "answer": "The run stopped before a result was returned.",
            "final_summary": "",
        }
        summary = str(result.get("final_summary") or "").strip()
        main_history.append({"role": "assistant", "content": _result_answer_text(result)})
        if summary:
            trace_history.append({"role": "assistant", "content": f"**Run summary**\n\n{summary}"})

        yield _compose_outputs(
            client,
            main_history=main_history,
            trace_history=trace_history,
            preferred_file=selected_file,
        )

    with gr.Blocks(title="HealthFlow Web", fill_width=True, fill_height=True, css=_WEB_APP_CSS) as demo:
        browser_task_id = gr.BrowserState(None, storage_key="healthflow-active-task")
        workspace_catalog_state = gr.State([])
        selected_file_state = gr.State(None)

        gr.HTML(_branding_header_html())

        with gr.Sidebar(label="History", open=True, width=360):
            recent_tasks = gr.Radio(
                label="Tasks",
                choices=[],
                value=None,
                interactive=True,
            )
            new_task_button = gr.Button("New Task", variant="primary")
            task_title_input = gr.Textbox(
                label="Task Title",
                placeholder="Leave blank to use an automatic title.",
                lines=1,
            )
            save_title_button = gr.Button("Save Title")
            delete_confirm = gr.Checkbox(
                label="I understand this permanently deletes this task.",
                value=False,
            )
            delete_task_button = gr.Button("Delete Task", variant="stop", interactive=False)
            history_notice = gr.Markdown(visible=False)

        with gr.Row(elem_classes=["hf-main"]):
            with gr.Column(scale=7, min_width=620):
                task_header = gr.Markdown()
                main_chatbot = gr.Chatbot(
                    label="Conversation",
                    type="messages",
                    value=[{"role": "assistant", "content": _MAIN_ASSISTANT_TEXT}],
                    height=820,
                    show_copy_button=True,
                )
                prompt_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_count="multiple",
                    placeholder="Describe the task or provide follow-up feedback. Upload files if needed.",
                    show_label=False,
                )
            with gr.Column(scale=5, min_width=420):
                with gr.Tabs():
                    with gr.Tab("Workspace"):
                        with gr.Row():
                            with gr.Column(scale=4, min_width=320):
                                gr.Markdown("### Workspace")

                                @gr.render(
                                    inputs=[browser_task_id, workspace_catalog_state, selected_file_state],
                                    queue=False,
                                    show_progress="hidden",
                                )
                                def _render_workspace_browser(
                                    current_task_id: str | None,
                                    workspace_catalog: list[dict[str, Any]] | None,
                                    current_selected_file: str | None,
                                ):
                                    if not current_task_id:
                                        gr.Markdown(_EMPTY_WORKSPACE_TEXT)
                                        return

                                    task_root = _task_root_path(current_task_id, session_store)
                                    if task_root is None:
                                        gr.Markdown(_EMPTY_WORKSPACE_TEXT)
                                        return

                                    catalog = list(workspace_catalog or [])
                                    report_path = next(
                                        (
                                            str(item.get("source_path"))
                                            for item in catalog
                                            if item.get("origin") == "report"
                                        ),
                                        None,
                                    )
                                    visible_files = [item for item in catalog if item.get("origin") != "report"]
                                    report_path_state = gr.State(report_path)

                                    if report_path:
                                        open_report_button = gr.Button(
                                            "report.md",
                                            variant="primary" if current_selected_file == report_path else "secondary",
                                        )
                                        open_report_button.click(
                                            _open_report_file,
                                            [report_path_state, browser_task_id],
                                            [
                                                selected_file_state,
                                                preview_header,
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

                                    sandbox_root = task_root / "sandbox"
                                    if visible_files:
                                        workspace_browser = gr.FileExplorer(
                                            root_dir=str(sandbox_root),
                                            file_count="single",
                                            label="Files",
                                            height=560,
                                            interactive=True,
                                            value=_workspace_tree_value(
                                                current_selected_file,
                                                task_root=task_root,
                                            ),
                                            key=("workspace-browser", current_task_id),
                                            preserved_by_key=[],
                                        )
                                        workspace_browser.change(
                                            _on_workspace_file_selected,
                                            [workspace_browser, browser_task_id],
                                            [
                                                selected_file_state,
                                                preview_header,
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
                                    elif not report_path:
                                        gr.Markdown(_EMPTY_WORKSPACE_TEXT)

                            with gr.Column(scale=6, min_width=420):
                                preview_header = gr.Markdown(value="### Preview")
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
                                    max_height=560,
                                )
                                preview_code = gr.Code(
                                    label="Code preview",
                                    visible=False,
                                    interactive=False,
                                    lines=24,
                                    max_lines=42,
                                )
                                preview_empty = gr.Markdown(value=_EMPTY_WORKSPACE_TEXT)
                                download_button = gr.DownloadButton("Download file", visible=False)
                    with gr.Tab("Advanced"):
                        report_requested = gr.Checkbox(label="Generate report.md for each turn", value=False)
                        trace_chatbot = gr.Chatbot(
                            label="Execution Trace",
                            type="messages",
                            value=[{"role": "assistant", "content": _TRACE_ASSISTANT_TEXT}],
                            height=780,
                            show_copy_button=True,
                        )

        app_outputs = [
            recent_tasks,
            main_chatbot,
            trace_chatbot,
            browser_task_id,
            task_header,
            workspace_catalog_state,
            selected_file_state,
            task_title_input,
            delete_confirm,
            delete_task_button,
            history_notice,
            preview_header,
            preview_markdown,
            preview_image,
            preview_table,
            preview_code,
            preview_empty,
            download_button,
        ]

        demo.load(
            _load_session,
            [browser_task_id],
            app_outputs,
            queue=False,
            show_progress="hidden",
        )

        recent_tasks.change(
            _switch_task,
            [recent_tasks],
            app_outputs,
            queue=False,
            show_progress="hidden",
        )

        new_task_button.click(
            _new_task,
            None,
            app_outputs,
            queue=False,
            show_progress="hidden",
        )

        save_title_button.click(
            _rename_task,
            [
                task_title_input,
                main_chatbot,
                trace_chatbot,
                browser_task_id,
                selected_file_state,
            ],
            app_outputs,
            queue=False,
            show_progress="hidden",
        )

        delete_confirm.change(
            _toggle_delete_button,
            [delete_confirm],
            [delete_task_button],
            queue=False,
            show_progress="hidden",
        )

        delete_task_button.click(
            _delete_task,
            [delete_confirm, browser_task_id],
            app_outputs,
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
                selected_file_state,
            ],
            app_outputs,
        ).then(lambda: gr.MultimodalTextbox(value=None), None, [prompt_input])

    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, share=share)
