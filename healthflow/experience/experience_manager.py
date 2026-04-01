import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List

import aiofiles
from loguru import logger

from ..core.llm_provider import LLMProvider
from .experience_models import Experience
from .experience_models import MemoryAuditEntry, MemoryLayer, MemoryRetrievalAudit, MemoryRetrievalResult
from .experience_models import MemoryScoreBreakdown, RetrievalContext, ValidationStatus
from .experience_models import ExperiencePolarity


class ExperienceManager:
    """
    Manages persistent structured experiences and retrieves context-aware memory
    for future runs.
    """

    def __init__(self, experience_path: Path, llm_provider: LLMProvider = None):
        self.experience_path = experience_path
        self.llm_provider = llm_provider
        self.experience_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.experience_path.exists():
            self.experience_path.touch()
        logger.info("ExperienceManager initialized. Knowledge base at: {}", self.experience_path)

    async def save_experiences(self, experiences: List[Experience]):
        if not experiences:
            return

        async with aiofiles.open(self.experience_path, mode="a", encoding="utf-8") as handle:
            for exp in experiences:
                await handle.write(exp.model_dump_json() + "\n")

        logger.info("Saved {} new experiences to the knowledge base.", len(experiences))

    async def reset(self):
        async with aiofiles.open(self.experience_path, mode="w", encoding="utf-8") as handle:
            await handle.write("")
        logger.info("Reset experience memory at {}", self.experience_path)

    async def retrieve_experiences(
        self,
        query: str,
        retrieval_context: RetrievalContext | None = None,
    ) -> MemoryRetrievalResult:
        context = retrieval_context or RetrievalContext()
        capacity = self._target_retrieval_capacity(context)
        audit = MemoryRetrievalAudit(
            query=query,
            task_family=context.task_family,
            domain_focus=context.domain_focus,
            dataset_signature=context.dataset_signature,
            capacity=capacity,
        )
        audit.selection_policy.extend(
            [
                "Safety-critical and verifier-supported failure memories are selected first as guardrails.",
                "At most one dataset-specific anchor memory is selected when an exact dataset match exists.",
                "Verified strategy and artifact memories outrank unverified positive memories.",
                "Safety-critical failure memories suppress conflicting non-failure memories.",
            ]
        )
        if not self.experience_path.exists() or self.experience_path.stat().st_size == 0:
            return MemoryRetrievalResult(selected_experiences=[], audit=audit)

        all_experiences = await self._load_all_experiences()
        if not all_experiences:
            return MemoryRetrievalResult(selected_experiences=[], audit=audit)

        query_words = self._tokenize(query)
        context_words = self._context_words(context)
        scored_experiences: list[tuple[MemoryScoreBreakdown, Experience]] = []
        for exp in all_experiences:
            score = self._score_experience(exp, query_words, context_words, context)
            if score.total_score > 0:
                scored_experiences.append((score, exp))

        scored_experiences.sort(
            key=lambda item: (item[0].total_score, item[1].confidence, item[1].created_at),
            reverse=True,
        )
        filtered_by_conflict = self._filter_conflicts(scored_experiences, audit)
        grouped = self._group_by_layer(filtered_by_conflict)

        selected: list[Experience] = []
        selected_ids: set[int] = set()
        blocked_conflict_groups: set[str] = set()

        failure_candidates = grouped.get(MemoryLayer.FAILURE.value, [])
        failure_slots = min(capacity, self._priority_failure_capacity(context))
        selected_failures = self._select_priority_failures(failure_candidates, failure_slots)
        for score, exp, rationale in selected_failures:
            self._record_selection(audit, selected, selected_ids, exp, score, rationale)
            if exp.safety_critical and exp.conflict_group:
                blocked_conflict_groups.add(exp.conflict_group)

        remaining_capacity = max(0, capacity - len(selected))
        if remaining_capacity:
            dataset_anchor = self._select_dataset_anchor(
                grouped.get(MemoryLayer.DATASET.value, []),
                selected_ids,
                context,
            )
            if dataset_anchor:
                score, exp, rationale = dataset_anchor
                self._record_selection(audit, selected, selected_ids, exp, score, rationale)
                remaining_capacity -= 1

        if remaining_capacity:
            verified_positive = self._select_positive_memories(
                filtered_by_conflict,
                selected_ids,
                blocked_conflict_groups,
                remaining_capacity,
                require_verified=True,
                audit=audit,
            )
            for score, exp, rationale in verified_positive:
                self._record_selection(audit, selected, selected_ids, exp, score, rationale)
            remaining_capacity -= len(verified_positive)

        if remaining_capacity:
            fallback_positive = self._select_positive_memories(
                filtered_by_conflict,
                selected_ids,
                blocked_conflict_groups,
                remaining_capacity,
                require_verified=False,
                audit=audit,
            )
            for score, exp, rationale in fallback_positive:
                self._record_selection(audit, selected, selected_ids, exp, score, rationale)

        self._record_remaining_suppressions(filtered_by_conflict, selected_ids, blocked_conflict_groups, audit)

        logger.debug(
            "Retrieved {} memories for task_family='{}' dataset_signature='{}'",
            len(selected),
            context.task_family,
            context.dataset_signature,
        )
        avoidance = [exp for exp in selected if exp.layer == MemoryLayer.FAILURE or exp.polarity == ExperiencePolarity.AVOID]
        recommended = [exp for exp in selected if exp not in avoidance]
        ordered = [*avoidance, *recommended]
        return MemoryRetrievalResult(
            recommended_experiences=recommended,
            avoidance_experiences=avoidance,
            selected_experiences=ordered,
            audit=audit,
        )

    async def _load_all_experiences(self) -> list[Experience]:
        all_experiences: list[Experience] = []
        async with aiofiles.open(self.experience_path, mode="r", encoding="utf-8") as handle:
            async for line in handle:
                if not line.strip():
                    continue
                try:
                    exp = Experience(**json.loads(line))
                    if exp.layer == MemoryLayer.FAILURE and exp.polarity != ExperiencePolarity.AVOID:
                        exp.polarity = ExperiencePolarity.AVOID
                    elif exp.layer != MemoryLayer.FAILURE and exp.polarity == ExperiencePolarity.AVOID:
                        exp.polarity = ExperiencePolarity.RECOMMEND
                    all_experiences.append(exp)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning("Skipping corrupted line in experience.jsonl: {}", exc)
        return all_experiences

    def _filter_conflicts(
        self,
        scored_experiences: list[tuple[MemoryScoreBreakdown, Experience]],
        audit: MemoryRetrievalAudit,
    ) -> list[tuple[MemoryScoreBreakdown, Experience]]:
        filtered: list[tuple[MemoryScoreBreakdown, Experience]] = []
        seen_conflicts: dict[str, Experience] = {}
        for score, exp in scored_experiences:
            if exp.conflict_group and exp.conflict_group in seen_conflicts:
                existing = seen_conflicts[exp.conflict_group]
                if self._allow_safety_override_pair(existing, exp):
                    filtered.append((score, exp))
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
            filtered.append((score, exp))
            if exp.conflict_group:
                seen_conflicts[exp.conflict_group] = exp
        return filtered

    def _group_by_layer(
        self,
        filtered_by_conflict: list[tuple[MemoryScoreBreakdown, Experience]],
    ) -> dict[str, list[tuple[MemoryScoreBreakdown, Experience]]]:
        grouped: dict[str, list[tuple[MemoryScoreBreakdown, Experience]]] = defaultdict(list)
        for score, exp in filtered_by_conflict:
            grouped[exp.layer.value].append((score, exp))
        return grouped

    def _target_retrieval_capacity(self, context: RetrievalContext) -> int:
        capacity = 3
        if context.dataset_signature != "unknown":
            capacity += 1
        if context.domain_focus == "ehr":
            capacity += 1
        if context.risk_findings:
            capacity += 1
        if len(context.verification_targets) >= 3:
            capacity += 1
        return min(capacity, 6)

    def _priority_failure_capacity(self, context: RetrievalContext) -> int:
        if context.domain_focus == "ehr" or len(context.risk_findings) >= 2:
            return 2
        return 1

    def _select_priority_failures(
        self,
        candidates: list[tuple[MemoryScoreBreakdown, Experience]],
        limit: int,
    ) -> list[tuple[MemoryScoreBreakdown, Experience, str]]:
        prioritized = [
            (score, exp)
            for score, exp in candidates
            if exp.safety_critical or exp.verifier_supported or exp.applicability_scope == "safety_global"
        ]
        if not prioritized:
            prioritized = candidates[:1] if candidates and limit > 0 else []
        return [
            (
                score,
                exp,
                "Selected as a high-priority failure guardrail for the upcoming run.",
            )
            for score, exp in prioritized[:limit]
        ]

    def _select_dataset_anchor(
        self,
        candidates: list[tuple[MemoryScoreBreakdown, Experience]],
        selected_ids: set[int],
        context: RetrievalContext,
    ) -> tuple[MemoryScoreBreakdown, Experience, str] | None:
        if context.dataset_signature == "unknown":
            return None
        for score, exp in candidates:
            if id(exp) in selected_ids:
                continue
            if exp.dataset_signature == context.dataset_signature or exp.applicability_scope == "dataset_exact":
                return (
                    score,
                    exp,
                    "Selected as the dataset-specific anchor memory for this run.",
                )
        return None

    def _select_positive_memories(
        self,
        filtered_by_conflict: list[tuple[MemoryScoreBreakdown, Experience]],
        selected_ids: set[int],
        blocked_conflict_groups: set[str],
        limit: int,
        require_verified: bool,
        audit: MemoryRetrievalAudit,
    ) -> list[tuple[MemoryScoreBreakdown, Experience, str]]:
        selected: list[tuple[MemoryScoreBreakdown, Experience, str]] = []
        for score, exp in filtered_by_conflict:
            if exp.layer == MemoryLayer.FAILURE or id(exp) in selected_ids:
                continue
            if exp.conflict_group and exp.conflict_group in blocked_conflict_groups:
                if not any(entry.source_task_id == exp.source_task_id for entry in audit.safety_overrides):
                    audit.safety_overrides.append(
                        self._make_audit_entry(
                            exp,
                            score,
                            disposition="suppressed_safety_override",
                            rationale=f"Suppressed because a safety-critical failure memory already occupies conflict group '{exp.conflict_group}'.",
                        )
                    )
                continue
            if require_verified and exp.validation_status != ValidationStatus.VERIFIED:
                continue
            if not require_verified and exp.validation_status == ValidationStatus.VERIFIED:
                continue
            if exp.validation_status == ValidationStatus.FAILED:
                continue
            rationale = (
                f"Selected from verified {exp.layer.value} memory."
                if require_verified
                else f"Selected as unverified {exp.layer.value} fallback after higher-confidence memory was exhausted."
            )
            selected.append((score, exp, rationale))
            if len(selected) >= limit:
                break
        return selected

    def _record_remaining_suppressions(
        self,
        filtered_by_conflict: list[tuple[MemoryScoreBreakdown, Experience]],
        selected_ids: set[int],
        blocked_conflict_groups: set[str],
        audit: MemoryRetrievalAudit,
    ) -> None:
        for score, exp in filtered_by_conflict:
            if id(exp) in selected_ids:
                continue
            if exp.conflict_group and exp.conflict_group in blocked_conflict_groups:
                continue
            audit.suppressed.append(
                self._make_audit_entry(
                    exp,
                    score,
                    disposition="suppressed",
                    rationale="Not selected by the adaptive retrieval controller after higher-priority memories consumed the available capacity.",
                )
            )

    def _record_selection(
        self,
        audit: MemoryRetrievalAudit,
        selected: list[Experience],
        selected_ids: set[int],
        exp: Experience,
        score: MemoryScoreBreakdown,
        rationale: str,
    ) -> None:
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

    def _score_experience(
        self,
        exp: Experience,
        query_words: set[str],
        context_words: set[str],
        context: RetrievalContext,
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
        context_bonus = min(3, len(context_words.intersection(exp_words)))
        task_family_bonus = 4 if exp.task_family == context.task_family else 0
        dataset_bonus = 3 if context.dataset_signature != "unknown" and exp.dataset_signature == context.dataset_signature else 0
        applicability_bonus = self._applicability_bonus(exp, context.task_family, context.dataset_signature)
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
            + context_bonus
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
            context_bonus=context_bonus,
            validation_bonus=validation_bonus,
            verifier_bonus=verifier_bonus,
            safety_bonus=safety_bonus,
            confidence_bonus=confidence_bonus,
            recency_bonus=recency_bonus,
            total_score=round(total_score, 2),
        )

    def _context_words(self, context: RetrievalContext) -> set[str]:
        joined = " ".join(
            [
                context.task_family,
                context.domain_focus,
                " ".join(context.risk_findings),
                " ".join(context.verification_targets),
            ]
        )
        return self._tokenize(joined)

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

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
            (existing.layer == MemoryLayer.FAILURE and existing.safety_critical and candidate.layer != MemoryLayer.FAILURE)
            or (candidate.layer == MemoryLayer.FAILURE and candidate.safety_critical and existing.layer != MemoryLayer.FAILURE)
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
