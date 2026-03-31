import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import aiofiles
from loguru import logger

from ..core.llm_provider import LLMProvider
from .experience_models import Experience
from .experience_models import MemoryAuditEntry, MemoryLayer, MemoryRetrievalAudit, MemoryRetrievalResult
from .experience_models import MemoryScoreBreakdown, ValidationStatus


class ExperienceManager:
    """
    Manages the persistent storage and retrieval of structured experiences
    using a JSONL file. This forms the system's long-term, evolving memory.
    """

    def __init__(self, experience_path: Path, llm_provider: LLMProvider = None):
        self.experience_path = experience_path
        self.llm_provider = llm_provider
        self.experience_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.experience_path.exists():
            self.experience_path.touch()
        logger.info("ExperienceManager initialized. Knowledge base at: {}", self.experience_path)

    async def save_experiences(self, experiences: List[Experience]):
        """Append new experiences to the JSONL memory file."""
        if not experiences:
            return

        async with aiofiles.open(self.experience_path, mode="a", encoding="utf-8") as handle:
            for exp in experiences:
                await handle.write(exp.model_dump_json() + "\n")

        logger.info("Saved {} new experiences to the knowledge base.", len(experiences))

    async def reset(self):
        """Truncate the memory file."""
        async with aiofiles.open(self.experience_path, mode="w", encoding="utf-8") as handle:
            await handle.write("")
        logger.info("Reset experience memory at {}", self.experience_path)

    async def retrieve_experiences(
        self,
        query: str,
        task_family: str = "general",
        dataset_signature: str = "unknown",
        budgets: Dict[str, int] | None = None,
    ) -> MemoryRetrievalResult:
        """Retrieve relevant memories and return a structured audit trail."""
        budgets = budgets or {
            MemoryLayer.STRATEGY.value: 3,
            MemoryLayer.FAILURE.value: 2,
            MemoryLayer.DATASET.value: 1,
            MemoryLayer.ARTIFACT.value: 1,
        }
        audit = MemoryRetrievalAudit(
            query=query,
            task_family=task_family,
            dataset_signature=dataset_signature,
            budgets=budgets,
        )
        audit.applied_precedence_rules.extend(
            [
                "Verified positive memories outrank unverified positive memories within the same layer.",
                "Safety-critical failure memories override conflicting strategy memories when they share a conflict group.",
            ]
        )
        if not self.experience_path.exists() or self.experience_path.stat().st_size == 0:
            return MemoryRetrievalResult(selected_experiences=[], audit=audit)

        all_experiences: List[Experience] = []
        async with aiofiles.open(self.experience_path, mode="r", encoding="utf-8") as handle:
            async for line in handle:
                if not line.strip():
                    continue
                try:
                    all_experiences.append(Experience(**json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning("Skipping corrupted line in experience.jsonl: {}", exc)

        if not all_experiences:
            return MemoryRetrievalResult(selected_experiences=[], audit=audit)

        query_words = set(re.findall(r"\b\w+\b", query.lower()))
        scored_experiences: list[tuple[MemoryScoreBreakdown, Experience]] = []
        for exp in all_experiences:
            score = self._score_experience(exp, query_words, task_family, dataset_signature)
            if score.total_score > 0:
                scored_experiences.append((score, exp))

        scored_experiences.sort(
            key=lambda item: (item[0].total_score, item[1].confidence, item[1].created_at),
            reverse=True,
        )

        filtered_by_conflict: list[tuple[MemoryScoreBreakdown, Experience]] = []
        seen_conflicts: dict[str, Experience] = {}
        for score, exp in scored_experiences:
            if exp.conflict_group and exp.conflict_group in seen_conflicts:
                existing = seen_conflicts[exp.conflict_group]
                if self._allow_safety_override_pair(existing, exp):
                    filtered_by_conflict.append((score, exp))
                    continue
                audit.suppressed_conflicts.append(
                    self._make_audit_entry(
                        exp,
                        score,
                        disposition="suppressed_conflict",
                        rationale=f"Suppressed because conflict group '{exp.conflict_group}' already had a higher-ranked memory.",
                    )
                )
                continue
            filtered_by_conflict.append((score, exp))
            if exp.conflict_group:
                seen_conflicts[exp.conflict_group] = exp

        grouped: Dict[str, list[tuple[MemoryScoreBreakdown, Experience]]] = defaultdict(list)
        for score, exp in filtered_by_conflict:
            grouped[exp.layer.value].append((score, exp))

        selected: list[Experience] = []
        selected_ids: set[int] = set()
        blocked_conflict_groups: set[str] = set()

        failure_budget = budgets.get(MemoryLayer.FAILURE.value, 0)
        failure_candidates = grouped.get(MemoryLayer.FAILURE.value, [])
        selected_failures = self._select_for_layer(MemoryLayer.FAILURE.value, failure_candidates, failure_budget)
        selected_failure_ids = {id(exp) for _, exp, _ in selected_failures}
        for score, exp, rationale in selected_failures:
            selected.append(exp)
            selected_ids.add(id(exp))
            if exp.safety_critical and exp.conflict_group:
                blocked_conflict_groups.add(exp.conflict_group)
            audit.selected.append(
                self._make_audit_entry(
                    exp,
                    score,
                    disposition="selected",
                    rationale=rationale,
                )
            )
        for score, exp in failure_candidates:
            if id(exp) not in selected_failure_ids:
                audit.suppressed_by_budget.append(
                    self._make_audit_entry(
                        exp,
                        score,
                        disposition="suppressed_budget",
                        rationale="Not selected because the failure-memory budget was exhausted or a stronger safety-critical failure memory was available.",
                    )
                )

        for layer_name in [MemoryLayer.STRATEGY.value, MemoryLayer.DATASET.value, MemoryLayer.ARTIFACT.value]:
            layer_budget = budgets.get(layer_name, 0)
            layer_candidates = grouped.get(layer_name, [])
            allowed_candidates: list[tuple[MemoryScoreBreakdown, Experience]] = []
            for score, exp in layer_candidates:
                if exp.conflict_group and exp.conflict_group in blocked_conflict_groups:
                    audit.safety_overrides.append(
                        self._make_audit_entry(
                            exp,
                            score,
                            disposition="suppressed_safety_override",
                            rationale=f"Suppressed because a safety-critical failure memory already occupies conflict group '{exp.conflict_group}'.",
                        )
                    )
                    continue
                allowed_candidates.append((score, exp))

            selected_for_layer = self._select_for_layer(layer_name, allowed_candidates, layer_budget)
            selected_layer_ids = {id(exp) for _, exp, _ in selected_for_layer}
            for score, exp, rationale in selected_for_layer:
                selected.append(exp)
                selected_ids.add(id(exp))
                audit.selected.append(
                    self._make_audit_entry(
                        exp,
                        score,
                        disposition="selected",
                        rationale=rationale,
                    )
                )

            for score, exp in allowed_candidates:
                if id(exp) not in selected_layer_ids:
                    audit.suppressed_by_budget.append(
                        self._make_audit_entry(
                            exp,
                            score,
                            disposition="suppressed_budget",
                            rationale=f"Not selected because the '{layer_name}' layer budget was exhausted or a stronger verified memory was available.",
                        )
                    )

        target_count = sum(budgets.values())
        if len(selected) < target_count:
            for score, exp in filtered_by_conflict:
                if exp.layer == MemoryLayer.FAILURE or id(exp) in selected_ids:
                    continue
                if exp.conflict_group and exp.conflict_group in blocked_conflict_groups:
                    continue
                selected.append(exp)
                selected_ids.add(id(exp))
                audit.selected.append(
                    self._make_audit_entry(
                        exp,
                        score,
                        disposition="selected",
                        rationale="Selected as cross-layer fallback after the primary layer budgets left free capacity.",
                    )
                )
                if len(selected) >= target_count:
                    break

        logger.debug(
            "Retrieved {} memories for task_family='{}' dataset_signature='{}'",
            len(selected),
            task_family,
            dataset_signature,
        )
        return MemoryRetrievalResult(selected_experiences=selected, audit=audit)

    def _score_experience(
        self,
        exp: Experience,
        query_words: set[str],
        task_family: str,
        dataset_signature: str,
    ) -> MemoryScoreBreakdown:
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
        applicability_bonus = self._applicability_bonus(exp, task_family, dataset_signature)
        confidence_bonus = round(exp.confidence, 2)
        recency_bonus = 1 if exp.created_at else 0
        validation_bonus = self._validation_bonus(exp)
        verifier_bonus = 2 if exp.verifier_supported else 0
        safety_bonus = 5 if exp.safety_critical and exp.layer == MemoryLayer.FAILURE else (2 if exp.safety_critical else 0)
        total_score = (
            overlap_score
            + task_family_bonus
            + dataset_bonus
            + applicability_bonus
            + validation_bonus
            + verifier_bonus
            + safety_bonus
            + confidence_bonus
            + recency_bonus
        )
        return MemoryScoreBreakdown(
            overlap_score=overlap_score,
            task_family_bonus=task_family_bonus,
            dataset_bonus=dataset_bonus,
            applicability_bonus=applicability_bonus,
            validation_bonus=validation_bonus,
            verifier_bonus=verifier_bonus,
            safety_bonus=safety_bonus,
            confidence_bonus=confidence_bonus,
            recency_bonus=recency_bonus,
            total_score=round(total_score, 2),
        )

    def _applicability_bonus(self, exp: Experience, task_family: str, dataset_signature: str) -> int:
        if exp.applicability_scope == "dataset_exact":
            return 4 if dataset_signature != "unknown" and exp.dataset_signature == dataset_signature else -1
        if exp.applicability_scope == "safety_global":
            return 3
        if exp.applicability_scope == "task_family":
            return 2 if exp.task_family == task_family else 0
        if exp.applicability_scope == "workflow_generic":
            return 1
        return 0

    def _allow_safety_override_pair(self, existing: Experience, candidate: Experience) -> bool:
        return (
            existing.layer == MemoryLayer.FAILURE
            and existing.safety_critical
            and candidate.layer != MemoryLayer.FAILURE
        )

    def _validation_bonus(self, exp: Experience) -> int:
        if exp.layer == MemoryLayer.FAILURE:
            return {
                ValidationStatus.FAILED: 4,
                ValidationStatus.VERIFIED: 2,
                ValidationStatus.UNVERIFIED: 1,
            }[exp.validation_status]
        return {
            ValidationStatus.VERIFIED: 4,
            ValidationStatus.UNVERIFIED: 1,
            ValidationStatus.FAILED: -2,
        }[exp.validation_status]

    def _select_for_layer(
        self,
        layer_name: str,
        layer_candidates: list[tuple[MemoryScoreBreakdown, Experience]],
        layer_budget: int,
    ) -> list[tuple[MemoryScoreBreakdown, Experience, str]]:
        if layer_budget <= 0 or not layer_candidates:
            return []

        if layer_name == MemoryLayer.FAILURE.value:
            return [
                (
                    score,
                    exp,
                    "Selected within the failure budget to guide future error avoidance.",
                )
                for score, exp in layer_candidates[:layer_budget]
            ]

        verified_candidates = [
            (score, exp)
            for score, exp in layer_candidates
            if exp.validation_status == ValidationStatus.VERIFIED
        ]
        fallback_candidates = [
            (score, exp)
            for score, exp in layer_candidates
            if exp.validation_status == ValidationStatus.UNVERIFIED
        ]
        selected: list[tuple[MemoryScoreBreakdown, Experience, str]] = [
            (
                score,
                exp,
                f"Selected from verified {layer_name} memory.",
            )
            for score, exp in verified_candidates[:layer_budget]
        ]
        remaining_slots = layer_budget - len(selected)
        if remaining_slots > 0:
            selected.extend(
                (
                    score,
                    exp,
                    f"Selected as unverified fallback because the verified {layer_name} pool was smaller than the layer budget.",
                )
                for score, exp in fallback_candidates[:remaining_slots]
            )
        return selected

    def _make_audit_entry(
        self,
        exp: Experience,
        score: MemoryScoreBreakdown,
        disposition: str,
        rationale: str,
    ) -> MemoryAuditEntry:
        return MemoryAuditEntry(
            source_task_id=exp.source_task_id,
            layer=exp.layer,
            validation_status=exp.validation_status,
            category=exp.category,
            content_preview=exp.content[:200],
            conflict_group=exp.conflict_group,
            applicability_scope=exp.applicability_scope,
            safety_critical=exp.safety_critical,
            verifier_supported=exp.verifier_supported,
            score=score,
            disposition=disposition,
            rationale=rationale,
        )
