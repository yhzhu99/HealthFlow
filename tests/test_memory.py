import tempfile
import unittest
from pathlib import Path

from healthflow.experience.experience_manager import ExperienceManager
from healthflow.experience.experience_models import Experience, ExperienceType, MemoryLayer, RetrievalContext, ValidationStatus


class MemoryManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_adaptive_retrieval_prioritizes_guardrails_dataset_anchor_and_verified_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            experiences = [
                Experience(
                    type=ExperienceType.WARNING,
                    layer=MemoryLayer.FAILURE,
                    category="predictive_modeling",
                    content="Patient-level split is mandatory; visit-level splits cause leakage.",
                    source_task_id="1",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.FAILED,
                    confidence=0.95,
                    conflict_group="split_policy",
                    applicability_scope="safety_global",
                    safety_critical=True,
                    verifier_supported=True,
                ),
                Experience(
                    type=ExperienceType.DATASET_PROFILE,
                    layer=MemoryLayer.DATASET,
                    category="dataset_profile",
                    content="Dataset abc123 contains patient_id, encounter_id, and readmission label.",
                    source_task_id="2",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.VERIFIED,
                    confidence=0.8,
                    applicability_scope="dataset_exact",
                ),
                Experience(
                    type=ExperienceType.HEURISTIC,
                    layer=MemoryLayer.STRATEGY,
                    category="predictive_modeling",
                    content="Use grouped validation and save metrics plus split evidence artifacts.",
                    source_task_id="3",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.VERIFIED,
                    confidence=0.9,
                ),
                Experience(
                    type=ExperienceType.WORKFLOW_PATTERN,
                    layer=MemoryLayer.ARTIFACT,
                    category="artifact",
                    content="Persist metrics.json and split_evidence.json for deterministic review.",
                    source_task_id="4",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.UNVERIFIED,
                    confidence=0.7,
                ),
            ]
            await manager.save_experiences(experiences)
            retrieval = await manager.retrieve_experiences(
                "build a readmission prediction model",
                retrieval_context=RetrievalContext(
                    task_family="predictive_modeling",
                    domain_focus="ehr",
                    dataset_signature="abc123",
                    risk_findings=["[HIGH] target_leakage", "[MEDIUM] validation_strategy"],
                    verification_targets=["split_evidence", "audit_evidence", "metrics_artifact"],
                ),
            )

            retrieved = retrieval.selected_experiences
            self.assertEqual(retrieved[0].layer, MemoryLayer.FAILURE)
            self.assertTrue(any(item.layer == MemoryLayer.DATASET for item in retrieved))
            self.assertTrue(any(item.layer == MemoryLayer.STRATEGY for item in retrieved))
            self.assertTrue(retrieval.audit.capacity >= 5)
            self.assertTrue(retrieval.audit.selection_policy)
            verified_positions = [
                index for index, entry in enumerate(retrieval.audit.selected) if entry.validation_status == ValidationStatus.VERIFIED
            ]
            unverified_positions = [
                index for index, entry in enumerate(retrieval.audit.selected) if entry.validation_status == ValidationStatus.UNVERIFIED
            ]
            self.assertTrue(verified_positions)
            self.assertTrue(unverified_positions)
            self.assertLess(min(verified_positions), min(unverified_positions))

    async def test_safety_critical_failure_suppresses_conflicting_positive_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            experiences = [
                Experience(
                    type=ExperienceType.HEURISTIC,
                    layer=MemoryLayer.STRATEGY,
                    category="predictive_modeling",
                    content="Use visit-level random splits for faster validation.",
                    source_task_id="1",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.VERIFIED,
                    confidence=0.8,
                    conflict_group="split_policy",
                ),
                Experience(
                    type=ExperienceType.VERIFIER_RULE,
                    layer=MemoryLayer.FAILURE,
                    category="predictive_modeling",
                    content="Visit-level random splits are unsafe; patient-level splitting is required.",
                    source_task_id="2",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.FAILED,
                    confidence=0.9,
                    conflict_group="split_policy",
                    applicability_scope="safety_global",
                    safety_critical=True,
                    verifier_supported=True,
                ),
            ]
            await manager.save_experiences(experiences)
            retrieval = await manager.retrieve_experiences(
                "train a predictive model with safe split strategy",
                retrieval_context=RetrievalContext(
                    task_family="predictive_modeling",
                    domain_focus="ehr",
                    dataset_signature="abc123",
                    risk_findings=["[MEDIUM] validation_strategy"],
                    verification_targets=["split_evidence"],
                ),
            )

            retrieved = retrieval.selected_experiences
            self.assertEqual(len([item for item in retrieved if item.layer == MemoryLayer.FAILURE]), 1)
            self.assertEqual(len([item for item in retrieved if item.layer == MemoryLayer.STRATEGY]), 0)
            self.assertEqual(len(retrieval.audit.safety_overrides), 1)
            self.assertEqual(retrieval.audit.safety_overrides[0].disposition, "suppressed_safety_override")


if __name__ == "__main__":
    unittest.main()
