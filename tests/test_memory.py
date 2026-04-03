import tempfile
import unittest
from pathlib import Path

from healthflow.experience.experience_manager import ExperienceManager
from healthflow.experience.experience_models import (
    Experience,
    MemoryKind,
    MemoryUpdate,
    MemoryUpdateAction,
    RetrievalContext,
    SourceOutcome,
)


class MemoryManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_lane_retrieval_prioritizes_safeguard_dataset_and_workflows(self):
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
                    kind=MemoryKind.WORKFLOW,
                    category="reporting_workflow",
                    content="Summarize the split evidence and report the saved metrics artifact in the final answer.",
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
            self.assertEqual(retrieval.audit.capacity, 4)
            self.assertEqual(retrieved[0].kind, MemoryKind.SAFEGUARD)
            self.assertEqual(len(retrieval.safeguard_experiences), 1)
            self.assertEqual(len(retrieval.dataset_experiences), 1)
            self.assertEqual(len(retrieval.workflow_experiences), 2)
            self.assertEqual(len(retrieved), 4)
            self.assertGreaterEqual(retrieved[0].times_retrieved, 1)

    async def test_safeguard_memory_requires_current_actionable_risk_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            await manager.save_experiences(
                [
                    Experience(
                        kind=MemoryKind.SAFEGUARD,
                        category="identifier_handling",
                        content="Do not expose identifiers in modeling artifacts when identifier-policy risk is active.",
                        source_task_id="1",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.RECOVERED,
                        confidence=0.9,
                        conflict_slot="identifier_policy",
                        applicability_scope="domain_ehr",
                        risk_tags=["identifier_policy"],
                        schema_tags=["domain:ehr", "patient_id"],
                    ),
                    Experience(
                        kind=MemoryKind.WORKFLOW,
                        category="conversion_workflow",
                        content="Load the workbook and write a CSV with the same columns and values.",
                        source_task_id="2",
                        task_family="format_conversion",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.SUCCESS,
                        confidence=0.8,
                        applicability_scope="task_family",
                        schema_tags=["domain:ehr", "patient_id"],
                    ),
                ]
            )

            retrieval = await manager.retrieve_experiences(
                "convert the workbook to csv",
                retrieval_context=RetrievalContext(
                    task_family="format_conversion",
                    domain_focus="ehr",
                    dataset_signature="abc123",
                    schema_tags=["domain:ehr", "patient_id"],
                    risk_tags=[],
                    prior_failure_modes=[],
                ),
            )

            self.assertEqual(len(retrieval.safeguard_experiences), 0)
            self.assertEqual(len(retrieval.workflow_experiences), 1)
            self.assertTrue(
                any(
                    "did not match any actionable current task risk" in entry.rationale.lower()
                    for entry in retrieval.audit.suppressed
                )
            )

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
                    prior_failure_modes=[],
                ),
            )

            retrieved = retrieval.selected_experiences
            self.assertEqual(len([item for item in retrieved if item.kind == MemoryKind.SAFEGUARD]), 1)
            self.assertEqual(len([item for item in retrieved if item.kind == MemoryKind.WORKFLOW]), 0)
            self.assertEqual(len(retrieval.audit.safeguard_overrides), 1)
            self.assertEqual(
                retrieval.audit.safeguard_overrides[0].disposition,
                "suppressed_safeguard_override",
            )

    async def test_memory_updates_validate_and_retire(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            experience = Experience(
                kind=MemoryKind.WORKFLOW,
                category="general_workflow",
                content="Inspect the dataset and produce a concise answer.",
                source_task_id="1",
                task_family="general_analysis",
                dataset_signature="general123",
                source_outcome=SourceOutcome.SUCCESS,
                confidence=0.8,
            )
            await manager.save_experiences([experience])

            await manager.apply_memory_updates(
                [
                    MemoryUpdate(
                        experience_id=experience.experience_id,
                        action=MemoryUpdateAction.VALIDATE,
                        reason="This workflow stayed within scope and produced the requested deliverable.",
                    )
                ]
            )

            validated = (await manager._load_all_experiences())[0]
            self.assertEqual(validated.validation_count, 1)
            self.assertIsNotNone(validated.last_validated_at)

            await manager.apply_memory_updates(
                [
                    MemoryUpdate(
                        experience_id=experience.experience_id,
                        action=MemoryUpdateAction.RETIRE,
                        reason="A newer strategy replaced this workflow.",
                    )
                ]
            )

            retrieval = await manager.retrieve_experiences(
                "analyze the uploaded dataset",
                retrieval_context=RetrievalContext(
                    task_family="general_analysis",
                    domain_focus="general",
                    dataset_signature="general123",
                ),
            )

            self.assertEqual(retrieval.selected_experiences, [])

    async def test_new_same_slot_workflow_retires_older_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            older = Experience(
                kind=MemoryKind.WORKFLOW,
                category="csv_conversion",
                content="Convert the workbook to csv with a quick one-off script.",
                source_task_id="1",
                task_family="format_conversion",
                dataset_signature="dataset-a",
                source_outcome=SourceOutcome.SUCCESS,
                confidence=0.7,
                conflict_slot="conversion_path",
                applicability_scope="task_family",
            )
            newer = Experience(
                kind=MemoryKind.WORKFLOW,
                category="csv_conversion",
                content="Convert the workbook to csv while preserving the source column order and values.",
                source_task_id="2",
                task_family="format_conversion",
                dataset_signature="dataset-b",
                source_outcome=SourceOutcome.SUCCESS,
                confidence=0.9,
                conflict_slot="conversion_path",
                applicability_scope="task_family",
            )

            await manager.save_experiences([older])
            await manager.save_experiences([newer])

            all_experiences = await manager._load_all_experiences()
            older_after = next(item for item in all_experiences if item.source_task_id == "1")
            newer_after = next(item for item in all_experiences if item.source_task_id == "2")
            self.assertTrue(older_after.retired)
            self.assertFalse(newer_after.retired)


if __name__ == "__main__":
    unittest.main()
