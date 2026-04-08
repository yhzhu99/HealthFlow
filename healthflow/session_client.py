from __future__ import annotations

from typing import Any


class TaskSessionClient:
    def __init__(self, system: Any):
        self.system = system
        self._state = None
        self._fallback_counter = 0
        self.new_task()

    def new_task(self) -> None:
        self._fallback_counter += 1
        if hasattr(self.system, "create_task_session"):
            self._state = self.system.create_task_session()
        else:
            self._state = {"task_id": f"session-{self._fallback_counter:03d}"}

    @property
    def task_id(self) -> str:
        if isinstance(self._state, dict):
            return str(self._state.get("task_id") or f"session-{self._fallback_counter:03d}")
        return getattr(self._state, "task_id", f"session-{self._fallback_counter:03d}")

    @property
    def session_label(self) -> str:
        return f"Task {self.task_id[:8]}"

    async def run_turn(
        self,
        task: str,
        *,
        live=None,
        spinner=None,
        report_requested: bool = False,
        progress_callback=None,
        uploaded_files: dict[str, bytes] | None = None,
    ) -> dict[str, Any]:
        if hasattr(self.system, "run_task_turn"):
            return await self.system.run_task_turn(
                self.task_id,
                task,
                live=live,
                spinner=spinner,
                uploaded_files=uploaded_files,
                report_requested=report_requested,
                progress_callback=progress_callback,
            )
        return await self.system.run_task(task, live, spinner, report_requested=report_requested, uploaded_files=uploaded_files)
