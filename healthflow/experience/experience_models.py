from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from uuid import UUID

class ExperienceType(str, Enum):
    """Enumeration for the types of experiences the system can learn."""
    HEURISTIC = "heuristic"
    CODE_SNIPPET = "code_snippet"
    WORKFLOW_PATTERN = "workflow_pattern"
    WARNING = "warning"

class Experience(BaseModel):
    """
    Pydantic model representing a single piece of learned knowledge.
    This structure is used for storing and retrieving experiences.
    """
    type: ExperienceType = Field(..., description="The type of the experience.")
    category: str = Field(..., description="A classification for the experience, e.g., 'medical_data_cleaning', 'hipaa_compliance', 'model_evaluation'.")
    content: str = Field(..., description="The actual content of the experience, e.g., a rule, a piece of code, or a warning message.")
    source_task_id: str = Field(..., description="The ID of the task from which this experience was synthesized.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of when the experience was created.")