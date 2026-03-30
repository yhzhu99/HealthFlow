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
            self.assertTrue(any(entry.disposition == "selected" for entry in retrieval.audit.selected))


if __name__ == "__main__":
    unittest.main()
