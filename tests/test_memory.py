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
    async def test_adaptive_retrieval_can_return_multiple_memory_types(self):
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
                    applicability_scope="domain_ehr",
                    risk_tags=["validation_strategy", "patient_linkage"],
                    schema_tags=["domain:ehr", "patient_id", "target_column"],
                ),
                Experience(
                    kind=MemoryKind.SAFEGUARD,
                    category="target_leakage_guard",
                    content="Verify label timestamps and feature timestamps before training to avoid target leakage.",
                    source_task_id="2",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.RECOVERED,
                    confidence=0.9,
                    applicability_scope="domain_ehr",
                    risk_tags=["target_leakage"],
                    schema_tags=["domain:ehr", "time_column", "target_column"],
                ),
                Experience(
                    kind=MemoryKind.SAFEGUARD,
                    category="identifier_handling",
                    content="Do not expose identifiers in artifacts when identifier misuse risk is active.",
                    source_task_id="3",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.RECOVERED,
                    confidence=0.8,
                    applicability_scope="domain_ehr",
                    risk_tags=["identifier_misuse"],
                    schema_tags=["domain:ehr", "patient_id"],
                ),
                Experience(
                    kind=MemoryKind.DATASET_ANCHOR,
                    category="core_schema",
                    content="Dataset abc123 includes subject_id, readmission labels, and event_time columns.",
                    source_task_id="4",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.SUCCESS,
                    confidence=0.85,
                    applicability_scope="dataset_exact",
                    schema_tags=["domain:ehr", "patient_id", "time_column", "target_column"],
                ),
                Experience(
                    kind=MemoryKind.WORKFLOW,
                    category="model_training",
                    content="Inspect the schema, validate the split, train the model, and save metrics plus cohort notes.",
                    source_task_id="5",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.SUCCESS,
                    confidence=0.88,
                    applicability_scope="task_family",
                    schema_tags=["domain:ehr", "patient_id", "target_column"],
                ),
                Experience(
                    kind=MemoryKind.CODE_SNIPPET,
                    category="split_assertion",
                    content="assert set(train.subject_id).isdisjoint(set(test.subject_id))",
                    source_task_id="6",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    source_outcome=SourceOutcome.SUCCESS,
                    confidence=0.86,
                    applicability_scope="task_family",
                    schema_tags=["domain:ehr", "patient_id"],
                    tags=["python", "validation"],
                ),
            ]
            await manager.save_experiences(experiences)

            retrieval = await manager.retrieve_experiences(
                "build a readmission prediction model and verify the patient split before training",
                retrieval_context=RetrievalContext(
                    task_family="predictive_modeling",
                    domain_focus="ehr",
                    dataset_signature="abc123",
                    schema_tags=["domain:ehr", "patient_id", "time_column", "target_column"],
                    risk_tags=["target_leakage", "validation_strategy", "patient_linkage"],
                    prior_failure_modes=[],
                ),
            )

            self.assertEqual(retrieval.audit.capacity, 6)
            self.assertEqual(len(retrieval.safeguard_experiences), 2)
            self.assertEqual(len(retrieval.dataset_anchor_experiences), 1)
            self.assertEqual(len(retrieval.workflow_experiences), 1)
            self.assertEqual(len(retrieval.code_snippet_experiences), 1)
            self.assertEqual(len(retrieval.selected_experiences), 5)
            self.assertEqual(retrieval.selected_experiences[0].kind, MemoryKind.SAFEGUARD)
            self.assertTrue(
                any(
                    "did not match any actionable current task risk" in entry.rationale.lower()
                    for entry in retrieval.audit.suppressed
                )
            )

    async def test_exact_dataset_match_retrieves_single_dataset_anchor_per_competing_scope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            await manager.save_experiences(
                [
                    Experience(
                        kind=MemoryKind.DATASET_ANCHOR,
                        category="schema_summary",
                        content="Dataset abc123 contains subject_id and event_time.",
                        source_task_id="1",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.SUCCESS,
                        confidence=0.7,
                        applicability_scope="dataset_exact",
                    ),
                    Experience(
                        kind=MemoryKind.DATASET_ANCHOR,
                        category="schema_summary",
                        content="Dataset abc123 contains subject_id, event_time, and readmission labels.",
                        source_task_id="2",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.SUCCESS,
                        confidence=0.9,
                        applicability_scope="dataset_exact",
                    ),
                ]
            )

            retrieval = await manager.retrieve_experiences(
                "train on the uploaded cohort",
                retrieval_context=RetrievalContext(
                    task_family="predictive_modeling",
                    domain_focus="ehr",
                    dataset_signature="abc123",
                    schema_tags=["domain:ehr", "patient_id", "time_column", "target_column"],
                ),
            )

            self.assertEqual(len(retrieval.dataset_anchor_experiences), 1)
            self.assertEqual(retrieval.dataset_anchor_experiences[0].source_task_id, "2")
            self.assertEqual(len(retrieval.audit.suppressed_competitors), 1)

    async def test_two_workflows_with_same_category_and_scope_do_not_appear_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            await manager.save_experiences(
                [
                    Experience(
                        kind=MemoryKind.WORKFLOW,
                        category="csv_conversion",
                        content="Convert the workbook to csv with a quick one-off script.",
                        source_task_id="1",
                        task_family="format_conversion",
                        dataset_signature="dataset-a",
                        source_outcome=SourceOutcome.SUCCESS,
                        confidence=0.7,
                        applicability_scope="task_family",
                    ),
                    Experience(
                        kind=MemoryKind.WORKFLOW,
                        category="csv_conversion",
                        content="Convert the workbook to csv while preserving source column order and values.",
                        source_task_id="2",
                        task_family="format_conversion",
                        dataset_signature="dataset-b",
                        source_outcome=SourceOutcome.SUCCESS,
                        confidence=0.9,
                        applicability_scope="task_family",
                    ),
                ]
            )

            retrieval = await manager.retrieve_experiences(
                "convert the workbook to csv",
                retrieval_context=RetrievalContext(
                    task_family="format_conversion",
                    domain_focus="general",
                    dataset_signature="dataset-c",
                ),
            )

            self.assertEqual(len(retrieval.workflow_experiences), 1)
            self.assertEqual(retrieval.workflow_experiences[0].source_task_id, "2")
            self.assertEqual(len(retrieval.audit.suppressed_competitors), 1)

    async def test_two_safeguards_with_same_category_and_scope_do_not_appear_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            await manager.save_experiences(
                [
                    Experience(
                        kind=MemoryKind.SAFEGUARD,
                        category="split_policy",
                        content="Check group leakage before running validation.",
                        source_task_id="1",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.RECOVERED,
                        confidence=0.6,
                        applicability_scope="domain_ehr",
                        risk_tags=["validation_strategy"],
                    ),
                    Experience(
                        kind=MemoryKind.SAFEGUARD,
                        category="split_policy",
                        content="Require patient-level splitting before reporting any model metrics.",
                        source_task_id="2",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.FAILED,
                        confidence=0.95,
                        applicability_scope="domain_ehr",
                        risk_tags=["validation_strategy", "patient_linkage"],
                    ),
                ]
            )

            retrieval = await manager.retrieve_experiences(
                "train a predictive model with a safe split strategy",
                retrieval_context=RetrievalContext(
                    task_family="predictive_modeling",
                    domain_focus="ehr",
                    dataset_signature="abc123",
                    schema_tags=["domain:ehr", "patient_id", "target_column"],
                    risk_tags=["validation_strategy", "patient_linkage"],
                ),
            )

            self.assertEqual(len(retrieval.safeguard_experiences), 1)
            self.assertEqual(retrieval.safeguard_experiences[0].source_task_id, "2")
            self.assertEqual(len(retrieval.audit.suppressed_competitors), 1)

    async def test_safeguard_workflow_and_code_snippet_can_coexist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            await manager.save_experiences(
                [
                    Experience(
                        kind=MemoryKind.SAFEGUARD,
                        category="split_policy",
                        content="Use patient-level splitting before reporting any model metrics.",
                        source_task_id="1",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.FAILED,
                        confidence=0.95,
                        applicability_scope="domain_ehr",
                        risk_tags=["validation_strategy", "patient_linkage"],
                    ),
                    Experience(
                        kind=MemoryKind.WORKFLOW,
                        category="split_policy",
                        content="Audit the split, train the model, and save metrics plus a split evidence note.",
                        source_task_id="2",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.SUCCESS,
                        confidence=0.9,
                        applicability_scope="task_family",
                        schema_tags=["domain:ehr", "patient_id", "target_column"],
                    ),
                    Experience(
                        kind=MemoryKind.CODE_SNIPPET,
                        category="split_policy",
                        content="assert set(train.subject_id).isdisjoint(set(test.subject_id))",
                        source_task_id="3",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.SUCCESS,
                        confidence=0.85,
                        applicability_scope="task_family",
                        schema_tags=["domain:ehr", "patient_id"],
                    ),
                ]
            )

            retrieval = await manager.retrieve_experiences(
                "train a predictive model and verify the patient split",
                retrieval_context=RetrievalContext(
                    task_family="predictive_modeling",
                    domain_focus="ehr",
                    dataset_signature="abc123",
                    schema_tags=["domain:ehr", "patient_id", "target_column"],
                    risk_tags=["validation_strategy", "patient_linkage"],
                ),
            )

            self.assertEqual(len(retrieval.safeguard_experiences), 1)
            self.assertEqual(len(retrieval.workflow_experiences), 1)
            self.assertEqual(len(retrieval.code_snippet_experiences), 1)
            self.assertEqual(len(retrieval.audit.safeguard_overrides), 0)

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

    async def test_competing_memories_are_not_retired_automatically(self):
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
                applicability_scope="task_family",
            )
            newer = Experience(
                kind=MemoryKind.WORKFLOW,
                category="csv_conversion",
                content="Convert the workbook to csv while preserving source column order and values.",
                source_task_id="2",
                task_family="format_conversion",
                dataset_signature="dataset-b",
                source_outcome=SourceOutcome.SUCCESS,
                confidence=0.9,
                applicability_scope="task_family",
            )

            await manager.save_experiences([older])
            await manager.save_experiences([newer])

            all_experiences = await manager._load_all_experiences()
            older_after = next(item for item in all_experiences if item.source_task_id == "1")
            newer_after = next(item for item in all_experiences if item.source_task_id == "2")
            self.assertFalse(older_after.retired)
            self.assertFalse(newer_after.retired)

            retrieval = await manager.retrieve_experiences(
                "convert the workbook to csv",
                retrieval_context=RetrievalContext(
                    task_family="format_conversion",
                    domain_focus="general",
                    dataset_signature="dataset-c",
                ),
            )

            self.assertEqual(len(retrieval.workflow_experiences), 1)
            self.assertEqual(retrieval.workflow_experiences[0].source_task_id, "2")

    async def test_retrieval_count_varies_with_relevance_within_ranges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            await manager.save_experiences(
                [
                    Experience(
                        kind=MemoryKind.SAFEGUARD,
                        category="split_policy",
                        content="Use patient-level splitting before reporting metrics.",
                        source_task_id="1",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.FAILED,
                        confidence=0.95,
                        applicability_scope="domain_ehr",
                        risk_tags=["validation_strategy", "patient_linkage"],
                    ),
                    Experience(
                        kind=MemoryKind.DATASET_ANCHOR,
                        category="core_schema",
                        content="Dataset abc123 includes subject_id and readmission labels.",
                        source_task_id="2",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.SUCCESS,
                        confidence=0.8,
                        applicability_scope="dataset_exact",
                    ),
                    Experience(
                        kind=MemoryKind.WORKFLOW,
                        category="model_training",
                        content="Inspect the schema, audit the split, and save metrics.",
                        source_task_id="3",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.SUCCESS,
                        confidence=0.85,
                        applicability_scope="task_family",
                    ),
                    Experience(
                        kind=MemoryKind.CODE_SNIPPET,
                        category="split_assertion",
                        content="assert set(train.subject_id).isdisjoint(set(test.subject_id))",
                        source_task_id="4",
                        task_family="predictive_modeling",
                        dataset_signature="abc123",
                        source_outcome=SourceOutcome.SUCCESS,
                        confidence=0.84,
                        applicability_scope="task_family",
                    ),
                ]
            )

            broad_retrieval = await manager.retrieve_experiences(
                "build a predictive model and verify the patient split before training",
                retrieval_context=RetrievalContext(
                    task_family="predictive_modeling",
                    domain_focus="ehr",
                    dataset_signature="abc123",
                    schema_tags=["domain:ehr", "patient_id", "target_column"],
                    risk_tags=["validation_strategy", "patient_linkage"],
                ),
            )
            narrow_retrieval = await manager.retrieve_experiences(
                "summarize the uploaded dataset briefly",
                retrieval_context=RetrievalContext(
                    task_family="general_analysis",
                    domain_focus="general",
                    dataset_signature="unknown",
                ),
            )

            self.assertGreater(len(broad_retrieval.selected_experiences), len(narrow_retrieval.selected_experiences))
            self.assertLessEqual(len(broad_retrieval.selected_experiences), 6)
            self.assertLessEqual(len(narrow_retrieval.selected_experiences), 6)

    async def test_read_only_memory_never_persists_retrieval_or_updates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            writer = ExperienceManager(memory_path)
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
            await writer.save_experiences([experience])
            original_text = memory_path.read_text(encoding="utf-8")

            reader = ExperienceManager(memory_path, read_only=True)
            retrieval = await reader.retrieve_experiences(
                "analyze the uploaded dataset",
                retrieval_context=RetrievalContext(
                    task_family="general_analysis",
                    domain_focus="general",
                    dataset_signature="general123",
                ),
            )
            applied = await reader.apply_memory_updates(
                [
                    MemoryUpdate(
                        experience_id=experience.experience_id,
                        action=MemoryUpdateAction.VALIDATE,
                        reason="read-only validation should be ignored",
                    )
                ]
            )

            self.assertEqual(len(retrieval.selected_experiences), 1)
            self.assertEqual(applied, [])
            self.assertEqual(memory_path.read_text(encoding="utf-8"), original_text)


if __name__ == "__main__":
    unittest.main()
