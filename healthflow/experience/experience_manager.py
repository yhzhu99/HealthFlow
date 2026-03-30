import json
import aiofiles
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
from loguru import logger
import re

from .experience_models import Experience, MemoryLayer, ValidationStatus
from ..core.llm_provider import LLMProvider

class ExperienceManager:
    """
    Manages the persistent storage and retrieval of structured experiences
    using a JSONL file. This forms the system's long-term, evolving memory.
    """
    def __init__(self, experience_path: Path, llm_provider: LLMProvider = None):
        self.experience_path = experience_path
        self.llm_provider = llm_provider
        # Ensure the parent directory exists
        self.experience_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the file exists
        if not self.experience_path.exists():
            self.experience_path.touch()
        logger.info(f"ExperienceManager initialized. Knowledge base at: {self.experience_path}")

    async def save_experiences(self, experiences: List[Experience]):
        """
        Appends a list of new Experience objects to the experience.jsonl file.
        Each experience is saved as a single JSON line.
        """
        if not experiences:
            return

        async with aiofiles.open(self.experience_path, mode='a', encoding='utf-8') as f:
            for exp in experiences:
                await f.write(exp.model_dump_json() + '\n')

        logger.info(f"Saved {len(experiences)} new experiences to the knowledge base.")

    async def reset(self):
        """Truncate the memory file."""
        async with aiofiles.open(self.experience_path, mode='w', encoding='utf-8') as f:
            await f.write("")
        logger.info("Reset experience memory at {}", self.experience_path)

    async def retrieve_experiences(
        self,
        query: str,
        task_family: str = "general",
        dataset_signature: str = "unknown",
        budgets: Dict[str, int] | None = None,
    ) -> List[Experience]:
        """
        Retrieves the most relevant experiences using lightweight hybrid scoring.
        """
        if not self.experience_path.exists() or self.experience_path.stat().st_size == 0:
            return []

        all_experiences: List[Experience] = []
        async with aiofiles.open(self.experience_path, mode='r', encoding='utf-8') as f:
            async for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        all_experiences.append(Experience(**data))
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        logger.warning(f"Skipping corrupted line in experience.jsonl: {e}")

        if not all_experiences:
            return []

        budgets = budgets or {
            MemoryLayer.STRATEGY.value: 3,
            MemoryLayer.FAILURE.value: 2,
            MemoryLayer.DATASET.value: 1,
            MemoryLayer.ARTIFACT.value: 1,
        }
        query_words = set(re.findall(r"\b\w+\b", query.lower()))
        scored_experiences = []

        for exp in all_experiences:
            exp_words = set(
                re.findall(
                    r"\b\w+\b",
                    " ".join(
                        [
                            exp.content,
                            exp.category,
                            exp.task_family,
                            exp.dataset_signature,
                            " ".join(exp.tags),
                        ]
                    ).lower(),
                )
            )
            overlap_score = len(query_words.intersection(exp_words))
            task_family_bonus = 4 if exp.task_family == task_family else 0
            dataset_bonus = 3 if dataset_signature != "unknown" and exp.dataset_signature == dataset_signature else 0
            validation_bonus = {
                ValidationStatus.VERIFIED: 3,
                ValidationStatus.UNVERIFIED: 1,
                ValidationStatus.FAILED: 0,
            }[exp.validation_status]
            recency_bonus = 1 if exp.created_at else 0
            total_score = overlap_score + task_family_bonus + dataset_bonus + validation_bonus + recency_bonus
            if total_score > 0:
                scored_experiences.append((total_score, exp))

        scored_experiences.sort(key=lambda item: (item[0], item[1].confidence, item[1].created_at), reverse=True)

        filtered_by_conflict: List[Experience] = []
        seen_conflicts = set()
        for _, exp in scored_experiences:
            if exp.conflict_group and exp.conflict_group in seen_conflicts:
                continue
            filtered_by_conflict.append(exp)
            if exp.conflict_group:
                seen_conflicts.add(exp.conflict_group)

        grouped: Dict[str, List[Experience]] = defaultdict(list)
        for exp in filtered_by_conflict:
            grouped[exp.layer.value].append(exp)

        selected: List[Experience] = []
        for layer_name, layer_budget in budgets.items():
            selected.extend(grouped.get(layer_name, [])[:layer_budget])

        # Fill any remaining slots with the next best memories across layers.
        target_count = sum(budgets.values())
        if len(selected) < target_count:
            selected_ids = {id(exp) for exp in selected}
            for exp in filtered_by_conflict:
                if id(exp) not in selected_ids:
                    selected.append(exp)
                if len(selected) >= target_count:
                    break

        logger.debug(
            "Retrieved {} memories for task_family='{}' dataset_signature='{}'",
            len(selected),
            task_family,
            dataset_signature,
        )
        return selected
