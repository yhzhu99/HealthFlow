from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SchemaSummary:
    file_name: str
    file_type: str
    columns: List[str] = field(default_factory=list)
    preview_rows: List[str] = field(default_factory=list)


@dataclass
class DataProfile:
    task_family: str
    dataset_signature: str
    modalities: List[str] = field(default_factory=list)
    schemas: List[SchemaSummary] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"- Task family: {self.task_family}",
            f"- Dataset signature: {self.dataset_signature}",
            f"- Modalities: {', '.join(self.modalities) if self.modalities else 'unknown'}",
        ]
        for schema in self.schemas:
            lines.append(f"- File `{schema.file_name}` ({schema.file_type}) columns: {', '.join(schema.columns) or 'n/a'}")
            if schema.preview_rows:
                lines.append(f"  - Preview: {' | '.join(schema.preview_rows)}")
        if self.notes:
            for note in self.notes:
                lines.append(f"- Note: {note}")
        return "\n".join(lines)


@dataclass
class RiskFinding:
    severity: str
    category: str
    message: str

    def to_bullet(self) -> str:
        return f"[{self.severity.upper()}] {self.category}: {self.message}"
