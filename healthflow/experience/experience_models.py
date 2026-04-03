from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
from typing import List, Optional

from pydantic import BaseModel, Field


class MemoryKind(str, Enum):
    SAFEGUARD = "safeguard"
    WORKFLOW = "workflow"
    DATASET = "dataset"


class SourceOutcome(str, Enum):
    SUCCESS = "success"
    RECOVERED = "recovered"
    FAILED = "failed"


class Experience(BaseModel):
    """A reusable memory item synthesized from prior task trajectories."""

    experience_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Stable identifier for this strategic memory item.",
    )
    kind: MemoryKind = Field(default=MemoryKind.WORKFLOW, description="EHR-adaptive memory class.")
    category: str = Field(..., description="Short category for routing and audit.")
    content: str = Field(..., description="Reusable memory content.")
    source_task_id: str = Field(..., description="Task identifier that produced the memory.")
    task_family: str = Field(default="general_analysis", description="Task family associated with the memory.")
    dataset_signature: str = Field(default="unknown", description="Stable summary of the profiled dataset context.")
    stage: str = Field(default="reflection", description="Lifecycle stage that produced the memory.")
    backend: str = Field(default="unknown", description="Executor backend that produced the memory.")
    source_outcome: SourceOutcome = Field(default=SourceOutcome.SUCCESS, description="Task outcome that produced the memory.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score assigned during synthesis.")
    conflict_slot: Optional[str] = Field(default=None, description="Domain-specific conflict slot identifier.")
    applicability_scope: str = Field(
        default="task_family",
        description="Where the memory applies, e.g. dataset_exact, task_family, workflow_generic, domain_ehr.",
    )
    risk_tags: List[str] = Field(default_factory=list, description="EHR risk-state tags associated with the memory.")
    schema_tags: List[str] = Field(default_factory=list, description="Schema/profile tags associated with the memory.")
    tags: List[str] = Field(default_factory=list, description="Additional retrieval tags.")
    supersedes: List[str] = Field(
        default_factory=list,
        description="Prior experience identifiers superseded by this memory item.",
    )
    provenance: dict = Field(default_factory=dict, description="Free-form provenance metadata.")
    times_retrieved: int = Field(
        default=0,
        ge=0,
        description="How many times this memory was selected into the planning context.",
    )
    validation_count: int = Field(
        default=0,
        ge=0,
        description="How many completed trajectories validated this memory as useful for future runs.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the memory item was created.",
    )
    last_validated_at: datetime | None = Field(
        default=None,
        description="Timestamp when this memory was most recently validated or retired.",
    )
    retired: bool = Field(
        default=False,
        description="Whether this memory has been retired from future retrieval.",
    )
    retired_reason: str | None = Field(
        default=None,
        description="Optional explanation for why the memory was retired.",
    )
    retired_at: datetime | None = Field(
        default=None,
        description="Timestamp when the memory was retired.",
    )


class MemoryScoreBreakdown(BaseModel):
    overlap_score: int = 0
    task_family_bonus: int = 0
    dataset_bonus: int = 0
    applicability_bonus: int = 0
    schema_bonus: int = 0
    risk_bonus: int = 0
    confidence_bonus: float = 0.0
    total_score: float = 0.0


class MemoryAuditEntry(BaseModel):
    experience_id: str
    source_task_id: str
    kind: MemoryKind
    source_outcome: SourceOutcome
    category: str
    content_preview: str
    conflict_slot: Optional[str] = None
    applicability_scope: str = "task_family"
    risk_tags: List[str] = Field(default_factory=list)
    schema_tags: List[str] = Field(default_factory=list)
    score: MemoryScoreBreakdown
    disposition: str
    rationale: str


class RetrievalContext(BaseModel):
    task_family: str = "general_analysis"
    domain_focus: str = "general"
    dataset_signature: str = "unknown"
    schema_tags: List[str] = Field(default_factory=list)
    risk_tags: List[str] = Field(default_factory=list)
    prior_failure_modes: List[str] = Field(default_factory=list)


class MemoryRetrievalAudit(BaseModel):
    query: str
    task_family: str
    domain_focus: str
    dataset_signature: str
    capacity: int = 0
    selection_policy: List[str] = Field(default_factory=list)
    selected: List[MemoryAuditEntry] = Field(default_factory=list)
    safeguard_overrides: List[MemoryAuditEntry] = Field(default_factory=list)
    suppressed_duplicates: List[MemoryAuditEntry] = Field(default_factory=list)
    suppressed_conflicts: List[MemoryAuditEntry] = Field(default_factory=list)
    suppressed: List[MemoryAuditEntry] = Field(default_factory=list)


class MemoryRetrievalResult(BaseModel):
    safeguard_experiences: List[Experience] = Field(default_factory=list)
    workflow_experiences: List[Experience] = Field(default_factory=list)
    dataset_experiences: List[Experience] = Field(default_factory=list)
    selected_experiences: List[Experience] = Field(default_factory=list)
    audit: MemoryRetrievalAudit


class MemoryUpdateAction(str, Enum):
    VALIDATE = "validate"
    RETIRE = "retire"


class MemoryUpdate(BaseModel):
    experience_id: str = Field(..., description="Identifier of the memory to update.")
    action: MemoryUpdateAction = Field(..., description="Lifecycle update to apply.")
    reason: str = Field(..., description="Concise justification grounded in the trajectory.")


class SynthesizedExperience(BaseModel):
    kind: MemoryKind = Field(default=MemoryKind.WORKFLOW, description="Memory class for the synthesized experience.")
    category: str = Field(..., description="Short category for routing and audit.")
    content: str = Field(..., description="Reusable memory content.")
    confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Confidence score assigned during synthesis.")
    conflict_slot: Optional[str] = Field(default=None, description="Domain-specific conflict slot identifier.")
    applicability_scope: str = Field(
        default="task_family",
        description="Where the memory applies, e.g. dataset_exact, task_family, workflow_generic, domain_ehr.",
    )
    risk_tags: List[str] = Field(default_factory=list, description="EHR risk-state tags associated with the memory.")
    schema_tags: List[str] = Field(default_factory=list, description="Schema/profile tags associated with the memory.")
    tags: List[str] = Field(default_factory=list, description="Additional retrieval tags.")
    supersedes: List[str] = Field(
        default_factory=list,
        description="Identifiers of prior memories that should be retired when this memory is stored.",
    )


class ReflectionSynthesisResult(BaseModel):
    experiences: List[SynthesizedExperience] = Field(default_factory=list)
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)
