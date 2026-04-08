from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from .session import HealthFlowProgressEvent
from .session_client import TaskSessionClient


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

    def _ensure_state(state: dict[str, Any] | None) -> dict[str, Any]:
        if isinstance(state, dict) and isinstance(state.get("client"), TaskSessionClient):
            return state
        client = TaskSessionClient(system_factory())
        return {"client": client}

    def _task_info(client: TaskSessionClient) -> str:
        task_root = Path(client.system.workspace_dir) / client.task_id if hasattr(client.system, "workspace_dir") else None
        lines = [
            f"**Active task:** `{client.task_id}`",
            "This chat stays on the same task session until you click **New Task**.",
        ]
        if task_root is not None:
            lines.append(f"**Workspace:** `{task_root}`")
            lines.append(f"**Sandbox:** `{task_root / 'sandbox'}`")
        return "\n\n".join(lines)

    def _artifact_files(client: TaskSessionClient, result: dict[str, Any] | None = None) -> list[str]:
        if not hasattr(client.system, "workspace_dir"):
            return []
        task_root = Path(client.system.workspace_dir) / client.task_id
        sandbox_dir = task_root / "sandbox"
        files = [str(path) for path in sorted(sandbox_dir.rglob("*")) if path.is_file()]
        if result and result.get("report_path"):
            files.append(str(result["report_path"]))
        return files[:100]

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
        state: dict[str, Any] | None,
        report_requested: bool,
    ):
        state = _ensure_state(state)
        client: TaskSessionClient = state["client"]
        main_history = list(main_history or [])
        trace_history = list(trace_history or [])
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
            yield main_history, trace_history, state, _task_info(client), _artifact_files(client)
            return

        user_display = _submit_display_text(text, upload_names)
        user_message = text.strip() or "Please inspect the uploaded files and continue the current task."
        main_history.append({"role": "user", "content": user_display})
        trace_history.append({"role": "assistant", "content": f"Starting turn on `{client.task_id}`."})
        yield main_history, trace_history, state, _task_info(client), _artifact_files(client)

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
                yield main_history, trace_history, state, _task_info(client), _artifact_files(client)
            except asyncio.TimeoutError:
                if task.done():
                    break

        while not queue.empty():
            event = queue.get_nowait()
            trace_history.append({"role": "assistant", "content": _format_progress_event(event)})

        result = await task
        answer = str(result.get("answer") or "No answer available.").strip()
        summary = str(result.get("final_summary") or "").strip()
        main_history.append({"role": "assistant", "content": answer})
        if summary:
            trace_history.append({"role": "assistant", "content": f"**Run summary**\n\n{summary}"})
        yield main_history, trace_history, state, _task_info(client), _artifact_files(client, result)

    def _new_task(
        state: dict[str, Any] | None,
    ):
        state = _ensure_state(state)
        client: TaskSessionClient = state["client"]
        client.new_task()
        main_history = [
            {
                "role": "assistant",
                "content": "Describe the task, upload any supporting files, and keep refining it here. Use this same session for follow-up feedback.",
            }
        ]
        trace_history = [
            {
                "role": "assistant",
                "content": f"Started a fresh task session: `{client.task_id}`.",
            }
        ]
        return main_history, trace_history, state, _task_info(client), []

    initial_state = _ensure_state(None)
    with gr.Blocks(title="HealthFlow Web") as demo:
        state = gr.State(initial_state)
        gr.Markdown(
            """
            # HealthFlow Web

            Work in one task session, upload files as you go, and keep giving expert feedback in the same workspace.
            """
        )
        with gr.Row():
            task_info = gr.Markdown(value=_task_info(initial_state["client"]))
            report_requested = gr.Checkbox(label="Generate report.md for each turn", value=False)
            new_task_button = gr.Button("New Task", variant="primary")
        with gr.Row():
            with gr.Column(scale=3):
                main_chatbot = gr.Chatbot(
                    label="HealthFlow",
                    type="messages",
                    value=[
                        {
                            "role": "assistant",
                            "content": "Describe the task, upload files if needed, and keep refining the same task here.",
                        }
                    ],
                    height=680,
                    show_copy_button=True,
                )
            with gr.Column(scale=2):
                trace_chatbot = gr.Chatbot(
                    label="Execution Trace",
                    type="messages",
                    value=[
                        {
                            "role": "assistant",
                            "content": "Planner, executor, evaluator, and artifact updates will appear here.",
                        }
                    ],
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

        prompt_input.submit(
            _run_turn,
            [prompt_input, main_chatbot, trace_chatbot, state, report_requested],
            [main_chatbot, trace_chatbot, state, task_info, artifact_files],
        ).then(lambda: gr.MultimodalTextbox(value=None), None, [prompt_input])

        new_task_button.click(
            _new_task,
            [state],
            [main_chatbot, trace_chatbot, state, task_info, artifact_files],
        )

    demo.queue()
    demo.launch(server_name=server_name, server_port=server_port, share=share)
