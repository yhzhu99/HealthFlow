import tempfile
import unittest
from pathlib import Path

from healthflow.experience.experience_manager import ExperienceManager
from healthflow.experience.experience_models import Experience, MemoryKind, RetrievalContext, SourceOutcome


class MemoryManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_adaptive_retrieval_prioritizes_safeguards_dataset_anchor_and_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            experiences = [
                Experience(
                    kind=MemoryKind.SAFEGUARD,
                    category="split_policy",
                    content="Use patient-level splitting to prevent duplicate-patient leakage across folds.",
                    source_task_id="1",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.FAILED,
                    confidence=0.95,
                    conflict_slot="split_policy",
                    applicability_scope="domain_ehr",
                    risk_tags=["validation_strategy", "patient_linkage"],
                    schema_tags=["domain:ehr", "patient_id", "target_column"],
                ),
                Experience(
                    kind=MemoryKind.DATASET,
                    category="dataset_anchor",
                    content="Dataset abc123 includes subject_id, readmission labels, and event_time columns.",
                    source_task_id="2",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.SUCCESS,
                    confidence=0.8,
                    applicability_scope="dataset_exact",
                    schema_tags=["domain:ehr", "patient_id", "time_column", "target_column"],
                ),
                Experience(
                    kind=MemoryKind.WORKFLOW,
                    category="modeling_workflow",
                    content="Inspect the schema, audit splits, train the model, and save metrics plus split notes.",
                    source_task_id="3",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.SUCCESS,
                    confidence=0.9,
                    schema_tags=["domain:ehr", "patient_id", "target_column"],
                ),
                Experience(
                    kind=MemoryKind.EXECUTION,
                    category="artifact_pattern",
                    content="Write cohort_notes.md and metrics.json before drafting the final answer.",
                    source_task_id="4",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.SUCCESS,
                    confidence=0.7,
                    schema_tags=["domain:ehr", "target_column"],
                ),
            ]
            await manager.save_experiences(experiences)
            retrieval = await manager.retrieve_experiences(
                "build a readmission prediction model from the uploaded cohort",
                retrieval_context=RetrievalContext(
                    task_family="predictive_modeling",
                    domain_focus="ehr",
                    dataset_signature="abc123",
                    schema_tags=["domain:ehr", "patient_id", "time_column", "target_column"],
                    risk_tags=["target_leakage", "validation_strategy", "patient_linkage"],
                    prior_failure_modes=[],
                ),
            )

            retrieved = retrieval.selected_experiences
            self.assertEqual(retrieved[0].kind, MemoryKind.SAFEGUARD)
            self.assertTrue(any(item.kind == MemoryKind.DATASET for item in retrieved))
            self.assertTrue(any(item.kind == MemoryKind.WORKFLOW for item in retrieved))
            self.assertEqual(len(retrieval.safeguard_experiences), 1)
            self.assertEqual(len(retrieval.dataset_experiences), 1)
            self.assertGreaterEqual(retrieval.audit.capacity, 5)
            self.assertTrue(retrieval.audit.selection_policy)
            safeguard_index = next(index for index, item in enumerate(retrieved) if item.kind == MemoryKind.SAFEGUARD)
            workflow_index = next(index for index, item in enumerate(retrieved) if item.kind == MemoryKind.WORKFLOW)
            self.assertLess(safeguard_index, workflow_index)

    async def test_safeguard_memory_suppresses_conflicting_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            experiences = [
                Experience(
                    kind=MemoryKind.WORKFLOW,
                    category="split_policy",
                    content="Use visit-level random splits for faster validation.",
                    source_task_id="1",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.SUCCESS,
                    confidence=0.8,
                    conflict_slot="split_policy",
                    risk_tags=["validation_strategy"],
                ),
                Experience(
                    kind=MemoryKind.SAFEGUARD,
                    category="split_policy",
                    content="Visit-level random splits are unsafe; patient-level splitting is required.",
                    source_task_id="2",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.FAILED,
                    confidence=0.9,
                    conflict_slot="split_policy",
                    applicability_scope="domain_ehr",
                    risk_tags=["validation_strategy", "patient_linkage"],
                    schema_tags=["domain:ehr", "patient_id"],
                ),
            ]
            await manager.save_experiences(experiences)
            retrieval = await manager.retrieve_experiences(
                "train a predictive model with safe split strategy",
                retrieval_context=RetrievalContext(
                    task_family="predictive_modeling",
                    domain_focus="ehr",
                    dataset_signature="abc123",
                    schema_tags=["domain:ehr", "patient_id", "target_column"],
                    risk_tags=["validation_strategy", "patient_linkage"],
                    prior_failure_modes=["unsafe_split"],
                ),
            )

            retrieved = retrieval.selected_experiences
            self.assertEqual(len([item for item in retrieved if item.kind == MemoryKind.SAFEGUARD]), 1)
            self.assertEqual(len([item for item in retrieved if item.kind == MemoryKind.WORKFLOW]), 0)
            self.assertEqual(len(retrieval.safeguard_experiences), 1)
            self.assertEqual(len(retrieval.workflow_experiences), 0)
            self.assertEqual(len(retrieval.audit.safeguard_overrides), 1)
            self.assertEqual(
                retrieval.audit.safeguard_overrides[0].disposition,
                "suppressed_safeguard_override",
            )


if __name__ == "__main__":
    unittest.main()
