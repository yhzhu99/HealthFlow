from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Tuple

class ExperienceType(str, Enum):
    HEURISTIC = "heuristic"
    CODE_SNIPPET = "code_snippet"
    WORKFLOW_PATTERN = "workflow_pattern"
    WARNING = "warning"

class Experience(BaseModel):
    id: int = None
    type: ExperienceType
    category: str = Field(..., description="A category for the experience, e.g., 'data_cleaning', 'debugging'.")
    content: str = Field(..., description="The actual content of the experience.")
    source_task_id: str
    created_at: datetime = None

    @classmethod
    def from_db_row(cls, row: Tuple):
        return cls(
            id=row[0],
            type=ExperienceType(row[1]),
            category=row[2],
            content=row[3],
            source_task_id=row[4],
            created_at=datetime.fromisoformat(row[5])
        )