import tempfile
import unittest
from pathlib import Path

from healthflow.experience.experience_manager import ExperienceManager
from healthflow.experience.experience_models import Experience, ExperienceType, MemoryLayer, ValidationStatus


class MemoryManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_uses_layer_budgets_and_conflict_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            experiences = [
                Experience(
                    type=ExperienceType.HEURISTIC,
                    layer=MemoryLayer.STRATEGY,
                    category="predictive_modeling",
                    content="Use patient-level splits for readmission prediction.",
                    source_task_id="1",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.VERIFIED,
                    confidence=0.9,
                ),
                Experience(
                    type=ExperienceType.WARNING,
                    layer=MemoryLayer.FAILURE,
                    category="predictive_modeling",
                    content="Do not use discharge-time features for prospective modeling.",
                    source_task_id="2",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.FAILED,
                    confidence=0.8,
                    conflict_group="leakage",
                    applicability_scope="safety_global",
                    safety_critical=True,
                    verifier_supported=True,
                ),
                Experience(
                    type=ExperienceType.WARNING,
                    layer=MemoryLayer.FAILURE,
                    category="predictive_modeling",
                    content="Conflicting warning that should be suppressed by conflict grouping.",
                    source_task_id="3",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.FAILED,
                    confidence=0.1,
                    conflict_group="leakage",
                ),
                Experience(
                    type=ExperienceType.DATASET_PROFILE,
                    layer=MemoryLayer.DATASET,
                    category="dataset_profile",
                    content="Columns include hadm_id, label, age.",
                    source_task_id="4",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.VERIFIED,
                    confidence=0.7,
                ),
                Experience(
                    type=ExperienceType.HEURISTIC,
                    layer=MemoryLayer.STRATEGY,
                    category="predictive_modeling",
                    content="Unverified heuristic that should only be used as fallback.",
                    source_task_id="5",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.UNVERIFIED,
                    confidence=0.95,
                ),
            ]
            await manager.save_experiences(experiences)
            retrieval = await manager.retrieve_experiences(
                "build a predictive model for readmission",
                task_family="predictive_modeling",
                dataset_signature="abc123",
                budgets={"strategy": 1, "failure": 1, "dataset": 1, "artifact": 0},
            )
            retrieved = retrieval.selected_experiences
            self.assertEqual(len(retrieved), 3)
            failure_memories = [item for item in retrieved if item.layer == MemoryLayer.FAILURE]
            self.assertEqual(len(failure_memories), 1)
            strategy_memories = [item for item in retrieved if item.layer == MemoryLayer.STRATEGY]
            self.assertEqual(strategy_memories[0].validation_status, ValidationStatus.VERIFIED)
            self.assertEqual(len(retrieval.audit.suppressed_conflicts), 1)
            self.assertTrue(retrieval.audit.applied_precedence_rules)
            self.assertTrue(any(entry.disposition == "selected" for entry in retrieval.audit.selected))

    async def test_safety_critical_failure_overrides_conflicting_strategy_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "experience.jsonl"
            manager = ExperienceManager(memory_path)
            experiences = [
                Experience(
                    type=ExperienceType.HEURISTIC,
                    layer=MemoryLayer.STRATEGY,
                    category="predictive_modeling",
                    content="Use random visit-level splits for model validation.",
                    source_task_id="1",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.VERIFIED,
                    confidence=0.8,
                    conflict_group="split_policy",
                    applicability_scope="task_family",
                ),
                Experience(
                    type=ExperienceType.VERIFIER_RULE,
                    layer=MemoryLayer.FAILURE,
                    category="predictive_modeling",
                    content="Patient-level splits are required; visit-level random splits cause leakage.",
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
                Experience(
                    type=ExperienceType.DATASET_PROFILE,
                    layer=MemoryLayer.DATASET,
                    category="dataset_profile",
                    content="Dataset uses patient_id and encounter_id with longitudinal visits.",
                    source_task_id="3",
                    task_family="predictive_modeling",
                    dataset_signature="abc123",
                    validation_status=ValidationStatus.VERIFIED,
                    confidence=0.7,
                    applicability_scope="dataset_exact",
                ),
            ]
            await manager.save_experiences(experiences)
            retrieval = await manager.retrieve_experiences(
                "train a predictive model with safe split strategy",
                task_family="predictive_modeling",
                dataset_signature="abc123",
                budgets={"strategy": 1, "failure": 1, "dataset": 1, "artifact": 0},
            )
            retrieved = retrieval.selected_experiences
            self.assertEqual(len([item for item in retrieved if item.layer == MemoryLayer.FAILURE]), 1)
            self.assertEqual(len([item for item in retrieved if item.layer == MemoryLayer.DATASET]), 1)
            self.assertEqual(len([item for item in retrieved if item.layer == MemoryLayer.STRATEGY]), 0)
            self.assertEqual(len(retrieval.audit.safety_overrides), 1)
            self.assertEqual(retrieval.audit.safety_overrides[0].disposition, "suppressed_safety_override")


if __name__ == "__main__":
    unittest.main()
