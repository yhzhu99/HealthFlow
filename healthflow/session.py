from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HealthFlowProgressEvent:
    kind: str
    stage: str
    status: str
    turn_number: int | None = None
    attempt: int | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskTurnRecord:
    turn_number: int
    user_message: str
    answer: str
    status: str
    runtime_dir: str
    report_path: str | None = None
    evaluation_feedback: str | None = None
    uploaded_files: list[dict[str, str]] = field(default_factory=list)
    created_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskTurnRecord":
        return cls(
            turn_number=int(payload.get("turn_number", 0)),
            user_message=str(payload.get("user_message", "")),
            answer=str(payload.get("answer", "")),
            status=str(payload.get("status", "unknown")),
            runtime_dir=str(payload.get("runtime_dir", "")),
            report_path=payload.get("report_path"),
            evaluation_feedback=payload.get("evaluation_feedback"),
            uploaded_files=list(payload.get("uploaded_files") or []),
            created_at_utc=payload.get("created_at_utc"),
        )


@dataclass
class TaskSessionState:
    task_id: str
    task_root: str
    created_at_utc: str
    updated_at_utc: str
    original_goal: str
    turn_count: int = 0
    latest_turn_number: int = 0
    latest_turn_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSessionState":
        return cls(
            task_id=str(payload.get("task_id", "")),
            task_root=str(payload.get("task_root", "")),
            created_at_utc=str(payload.get("created_at_utc", "")),
            updated_at_utc=str(payload.get("updated_at_utc", "")),
            original_goal=str(payload.get("original_goal", "")),
            turn_count=int(payload.get("turn_count", 0)),
            latest_turn_number=int(payload.get("latest_turn_number", 0)),
            latest_turn_status=payload.get("latest_turn_status"),
        )


@dataclass
class SessionPromptContext:
    task_id: str
    turn_number: int
    original_goal: str
    latest_user_message: str
    previous_answer: str | None = None
    previous_feedback: str | None = None
    recent_turns: list[dict[str, str]] = field(default_factory=list)
    sandbox_artifacts: list[str] = field(default_factory=list)
    current_uploads: list[dict[str, str]] = field(default_factory=list)

    def planner_block(self) -> str:
        return self._render(include_feedback=True)

    def execution_block(self) -> str:
        return self._render(include_feedback=True)

    def evaluator_block(self) -> str:
        return self._render(include_feedback=True)

    def _render(self, *, include_feedback: bool) -> str:
        sections = [
            "Task session context:",
            "---",
            f"- Task session id: {self.task_id}",
            f"- Turn number: {self.turn_number}",
            f"- Original task goal: {self.original_goal.strip() or 'N/A'}",
            f"- Current user follow-up: {self.latest_user_message.strip() or 'N/A'}",
        ]
        if self.previous_answer:
            sections.append(f"- Previous assistant answer: {self.previous_answer.strip()}")
        if include_feedback and self.previous_feedback:
            sections.append(f"- Previous evaluator feedback: {self.previous_feedback.strip()}")
        if self.recent_turns:
            sections.extend(["- Recent conversation:"])
            for item in self.recent_turns:
                user = item.get("user", "").strip()
                assistant = item.get("assistant", "").strip()
                if user:
                    sections.append(f"  - User: {user}")
                if assistant:
                    sections.append(f"  - Assistant: {assistant}")
        if self.current_uploads:
            sections.extend(["- Current uploaded files:"])
            for item in self.current_uploads:
                original_name = item.get("original_name", "").strip()
                sandbox_path = item.get("sandbox_path", "").strip()
                if original_name or sandbox_path:
                    sections.append(f"  - {original_name or 'uploaded file'} -> {sandbox_path or 'N/A'}")
        if self.sandbox_artifacts:
            sections.extend(["- Current sandbox artifacts:"])
            for artifact in self.sandbox_artifacts[:20]:
                sections.append(f"  - {artifact}")
        sections.append("---")
        return "\n".join(sections)
