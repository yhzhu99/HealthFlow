from __future__ import annotations

from typing import Any


class TaskSessionClient:
    def __init__(self, system: Any, task_id: str | None = None):
        self.system = system
        self._state = None
        self._fallback_counter = 0
        self.bind_task(task_id)

    def bind_task(self, task_id: str | None) -> None:
        if task_id:
            if hasattr(self.system, "create_task_session"):
                self._state = self.system.create_task_session(task_id)
            else:
                self._state = {"task_id": task_id}
            return
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
    def task_root(self) -> str | None:
        if isinstance(self._state, dict):
            task_root = self._state.get("task_root")
        else:
            task_root = getattr(self._state, "task_root", None)
        return str(task_root) if task_root else None

    @property
    def turn_count(self) -> int:
        if isinstance(self._state, dict):
            return int(self._state.get("turn_count") or 0)
        return int(getattr(self._state, "turn_count", 0) or 0)

    @property
    def latest_turn_status(self) -> str | None:
        if isinstance(self._state, dict):
            status = self._state.get("latest_turn_status")
        else:
            status = getattr(self._state, "latest_turn_status", None)
        return str(status) if status else None

    @property
    def original_goal(self) -> str:
        if isinstance(self._state, dict):
            value = self._state.get("original_goal")
        else:
            value = getattr(self._state, "original_goal", "")
        return str(value or "")

    @property
    def updated_at_utc(self) -> str | None:
        if isinstance(self._state, dict):
            value = self._state.get("updated_at_utc")
        else:
            value = getattr(self._state, "updated_at_utc", None)
        return str(value) if value else None

    @property
    def session_label(self) -> str:
        return f"Task {self.task_id[:8]}"

    def load_history(self) -> list[Any]:
        if hasattr(self.system, "load_task_history"):
            return list(self.system.load_task_history(self.task_id))
        return []

    def refresh_state(self) -> None:
        if not hasattr(self.system, "load_task_session"):
            return
        try:
            self._state = self.system.load_task_session(self.task_id)
        except FileNotFoundError:
            return

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
            result = await self.system.run_task_turn(
                self.task_id,
                task,
                live=live,
                spinner=spinner,
                uploaded_files=uploaded_files,
                report_requested=report_requested,
                progress_callback=progress_callback,
            )
            self.refresh_state()
            return result
        return await self.system.run_task(task, live, spinner, report_requested=report_requested, uploaded_files=uploaded_files)
