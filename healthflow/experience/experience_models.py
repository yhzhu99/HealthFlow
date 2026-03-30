from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timezone
from typing import List, Optional

class ExperienceType(str, Enum):
    """Enumeration for the types of experiences the system can learn."""
    HEURISTIC = "heuristic"
    CODE_SNIPPET = "code_snippet"
    WORKFLOW_PATTERN = "workflow_pattern"
    WARNING = "warning"
    DATASET_PROFILE = "dataset_profile"
    VERIFIER_RULE = "verifier_rule"


class MemoryLayer(str, Enum):
    DATASET = "dataset"
    STRATEGY = "strategy"
    FAILURE = "failure"
    ARTIFACT = "artifact"


class ValidationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"

class Experience(BaseModel):
    """
    Pydantic model representing a single piece of learned knowledge.
    This structure is used for storing and retrieving experiences.
    """
    type: ExperienceType = Field(..., description="The type of the experience.")
    layer: MemoryLayer = Field(default=MemoryLayer.STRATEGY, description="Hierarchical memory layer.")
    category: str = Field(..., description="A classification for the experience, e.g., 'medical_data_cleaning', 'hipaa_compliance', 'model_evaluation'.")
    content: str = Field(..., description="The actual content of the experience, e.g., a rule, a piece of code, or a warning message.")
    source_task_id: str = Field(..., description="The ID of the task from which this experience was synthesized.")
    task_family: str = Field(default="general", description="Task family associated with the memory.")
    dataset_signature: str = Field(default="unknown", description="Stable summary of the dataset context.")
    stage: str = Field(default="reflection", description="Lifecycle stage that produced the memory.")
    backend: str = Field(default="unknown", description="Executor backend that produced the memory.")
    validation_status: ValidationStatus = Field(default=ValidationStatus.UNVERIFIED, description="Validation state for the memory item.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score assigned during reflection.")
    conflict_group: Optional[str] = Field(default=None, description="Conflict group identifier for contradictory memories.")
    tags: List[str] = Field(default_factory=list, description="Additional retrieval tags.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of when the experience was created.",
    )
