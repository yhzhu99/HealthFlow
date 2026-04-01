import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import aiofiles
from loguru import logger

from ..core.llm_provider import LLMProvider
from .experience_models import Experience
from .experience_models import MemoryAuditEntry, MemoryKind, MemoryRetrievalAudit, MemoryRetrievalResult
from .experience_models import MemoryScoreBreakdown, MemoryUpdate, MemoryUpdateAction, RetrievalContext, SourceOutcome


ARTIFACT_HEAVY_FAMILIES = {
    "cohort_extraction",
    "predictive_modeling",
    "survival_analysis",
    "time_series_modeling",
    "report_generation",
}
EHR_MODELING_FAMILIES = {"predictive_modeling", "survival_analysis", "time_series_modeling"}
HIGH_PRIORITY_RISK_TAGS = {"temporal_leakage", "target_leakage", "patient_linkage", "cohort_definition"}


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

        existing_experiences = await self._load_all_experiences()
        existing_signatures = {self._dedupe_signature(exp) for exp in existing_experiences}
        appended_count = 0
        superseded_ids = {
            experience_id
            for exp in experiences
            for experience_id in exp.supersedes
        }

        if superseded_ids:
            retired_count = self._retire_experiences_by_id(
                existing_experiences,
                superseded_ids,
                reason="Superseded by a newer strategic memory synthesized from a later trajectory.",
            )
            if retired_count:
                logger.info("Retired {} superseded memories before saving new strategic experiences.", retired_count)

        for exp in experiences:
            signature = self._dedupe_signature(exp)
            if signature in existing_signatures:
                continue
            existing_experiences.append(exp)
            existing_signatures.add(signature)
            appended_count += 1

        await self._persist_experiences(existing_experiences)
        logger.info("Saved {} new experiences to the knowledge base.", appended_count)

    async def reset(self):
        await self._persist_experiences([])
        logger.info("Reset experience memory at {}", self.experience_path)

    async def apply_memory_updates(self, updates: List[MemoryUpdate]) -> list[str]:
        if not updates:
            return []

        all_experiences = await self._load_all_experiences()
        experiences_by_id = {exp.experience_id: exp for exp in all_experiences}
        applied_ids: list[str] = []
        now = datetime.now(timezone.utc)

        for update in updates:
            exp = experiences_by_id.get(update.experience_id)
            if exp is None:
                logger.debug("Skipping memory update for unknown experience_id='{}'", update.experience_id)
                continue
            exp.last_validated_at = now
            if update.action == MemoryUpdateAction.VALIDATE:
                exp.times_helped += 1
            elif update.action == MemoryUpdateAction.PENALIZE:
                exp.times_hurt += 1
            elif update.action == MemoryUpdateAction.RETIRE:
                exp.retired = True
                exp.retired_reason = update.reason
                exp.retired_at = now
            applied_ids.append(exp.experience_id)

        if applied_ids:
            await self._persist_experiences(all_experiences)
            logger.info("Applied {} memory lifecycle updates.", len(applied_ids))
        return applied_ids

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
                "Adaptive retrieval uses task family, dataset signature, schema tags, and EHR risk tags.",
                "Safeguard memories are prioritized for elevated-risk EHR tasks and suppress conflicting workflow or execution memories.",
                "At most one dataset anchor is selected for an exact dataset match.",
                "Workflow memories fill most remaining capacity, with execution memories limited to artifact-heavy tasks.",
            ]
        )
        if not self.experience_path.exists() or self.experience_path.stat().st_size == 0:
            return MemoryRetrievalResult(selected_experiences=[], audit=audit)

        all_experiences = await self._load_all_experiences()
        active_experiences = [exp for exp in all_experiences if not exp.retired]
        if not active_experiences:
            return MemoryRetrievalResult(selected_experiences=[], audit=audit)

        query_words = self._tokenize(query)
        failure_words = self._tokenize(" ".join(context.prior_failure_modes))
        scored_experiences: list[tuple[MemoryScoreBreakdown, Experience]] = []
        for exp in active_experiences:
            score = self._score_experience(exp, query_words, failure_words, context)
            if score.total_score > 0:
                scored_experiences.append((score, exp))

        scored_experiences.sort(
            key=lambda item: (item[0].total_score, item[1].confidence, item[1].created_at),
            reverse=True,
        )
        deduped = self._filter_duplicates(scored_experiences, audit)
        filtered_by_conflict = self._filter_same_kind_conflicts(deduped, audit)

        selected: list[Experience] = []
        selected_ids: set[int] = set()
        blocked_conflict_slots: set[str] = set()

        safeguard_limit = min(capacity, self._safeguard_capacity(context))
        selected_safeguards = self._select_kind_memories(
            filtered_by_conflict,
            kind=MemoryKind.SAFEGUARD,
            selected_ids=selected_ids,
            blocked_conflict_slots=blocked_conflict_slots,
            limit=safeguard_limit,
            audit=audit,
            rationale="Selected as an EHR safeguard for the current task and risk state.",
        )
        for score, exp, rationale in selected_safeguards:
            self._record_selection(audit, selected, selected_ids, exp, score, rationale)
            if exp.conflict_slot:
                blocked_conflict_slots.add(exp.conflict_slot)

        remaining_capacity = max(0, capacity - len(selected))
        if remaining_capacity:
            dataset_anchor = self._select_dataset_anchor(filtered_by_conflict, selected_ids, context)
            if dataset_anchor:
                score, exp, rationale = dataset_anchor
                self._record_selection(audit, selected, selected_ids, exp, score, rationale)
                remaining_capacity -= 1

        execution_limit = 1 if remaining_capacity and self._allow_execution_lane(context) else 0
        workflow_limit = max(0, remaining_capacity - execution_limit)
        if workflow_limit:
            selected_workflows = self._select_kind_memories(
                filtered_by_conflict,
                kind=MemoryKind.WORKFLOW,
                selected_ids=selected_ids,
                blocked_conflict_slots=blocked_conflict_slots,
                limit=workflow_limit,
                audit=audit,
                rationale="Selected as a reusable workflow for the current task family.",
            )
            for score, exp, rationale in selected_workflows:
                self._record_selection(audit, selected, selected_ids, exp, score, rationale)
            remaining_capacity = max(0, capacity - len(selected))

        if execution_limit and remaining_capacity:
            selected_execution = self._select_kind_memories(
                filtered_by_conflict,
                kind=MemoryKind.EXECUTION,
                selected_ids=selected_ids,
                blocked_conflict_slots=blocked_conflict_slots,
                limit=min(execution_limit, remaining_capacity),
                audit=audit,
                rationale="Selected as an execution pattern that can improve task completion.",
            )
            for score, exp, rationale in selected_execution:
                self._record_selection(audit, selected, selected_ids, exp, score, rationale)
            remaining_capacity = max(0, capacity - len(selected))

        if remaining_capacity:
            overflow = self._select_overflow_memories(
                filtered_by_conflict,
                selected_ids,
                blocked_conflict_slots,
                limit=remaining_capacity,
                audit=audit,
            )
            for score, exp, rationale in overflow:
                self._record_selection(audit, selected, selected_ids, exp, score, rationale)

        self._record_remaining_suppressions(filtered_by_conflict, selected_ids, blocked_conflict_slots, audit)
        if selected:
            for exp in selected:
                exp.times_retrieved += 1
            await self._persist_experiences(all_experiences)

        logger.debug(
            "Retrieved {} memories for task_family='{}' dataset_signature='{}'",
            len(selected),
            context.task_family,
            context.dataset_signature,
        )
        grouped = self._selected_by_kind(selected)
        return MemoryRetrievalResult(
            safeguard_experiences=grouped[MemoryKind.SAFEGUARD],
            workflow_experiences=grouped[MemoryKind.WORKFLOW],
            dataset_experiences=grouped[MemoryKind.DATASET],
            execution_experiences=grouped[MemoryKind.EXECUTION],
            selected_experiences=selected,
            audit=audit,
        )

    async def _load_all_experiences(self) -> list[Experience]:
        all_experiences: list[Experience] = []
        async with aiofiles.open(self.experience_path, mode="r", encoding="utf-8") as handle:
            async for line in handle:
                if not line.strip():
                    continue
                try:
                    all_experiences.append(Experience(**json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning("Skipping corrupted line in experience.jsonl: {}", exc)
        return all_experiences

    async def _persist_experiences(self, experiences: list[Experience]) -> None:
        async with aiofiles.open(self.experience_path, mode="w", encoding="utf-8") as handle:
            for exp in experiences:
                await handle.write(exp.model_dump_json() + "\n")

    def _filter_duplicates(
        self,
        scored_experiences: list[tuple[MemoryScoreBreakdown, Experience]],
        audit: MemoryRetrievalAudit,
    ) -> list[tuple[MemoryScoreBreakdown, Experience]]:
        filtered: list[tuple[MemoryScoreBreakdown, Experience]] = []
        seen_signatures: set[str] = set()
        for score, exp in scored_experiences:
            signature = self._dedupe_signature(exp)
            if signature in seen_signatures:
                audit.suppressed_duplicates.append(
                    self._make_audit_entry(
                        exp,
                        score,
                        disposition="suppressed_duplicate",
                        rationale="Suppressed because an equivalent memory had already been ranked higher.",
                    )
                )
                continue
            seen_signatures.add(signature)
            filtered.append((score, exp))
        return filtered

    def _filter_same_kind_conflicts(
        self,
        scored_experiences: list[tuple[MemoryScoreBreakdown, Experience]],
        audit: MemoryRetrievalAudit,
    ) -> list[tuple[MemoryScoreBreakdown, Experience]]:
        filtered: list[tuple[MemoryScoreBreakdown, Experience]] = []
        seen_conflicts: dict[tuple[str, str, str], Experience] = {}
        for score, exp in scored_experiences:
            if not exp.conflict_slot:
                filtered.append((score, exp))
                continue
            key = (exp.kind.value, exp.conflict_slot, exp.applicability_scope)
            if key in seen_conflicts:
                audit.suppressed_conflicts.append(
                    self._make_audit_entry(
                        exp,
                        score,
                        disposition="suppressed_conflict",
                        rationale=(
                            f"Suppressed because another {exp.kind.value} memory already occupied "
                            f"conflict slot '{exp.conflict_slot}' for scope '{exp.applicability_scope}'."
                        ),
                    )
                )
                continue
            seen_conflicts[key] = exp
            filtered.append((score, exp))
        return filtered

    def _target_retrieval_capacity(self, context: RetrievalContext) -> int:
        capacity = 4
        if context.domain_focus == "ehr":
            capacity += 1
        if self._elevated_risk(context):
            capacity += 1
        return min(capacity, 6)

    def _safeguard_capacity(self, context: RetrievalContext) -> int:
        if context.domain_focus == "ehr" and (
            context.task_family in EHR_MODELING_FAMILIES or self._elevated_risk(context)
        ):
            return 2
        return 1 if context.risk_tags else 0

    def _allow_execution_lane(self, context: RetrievalContext) -> bool:
        return context.task_family in ARTIFACT_HEAVY_FAMILIES

    def _select_dataset_anchor(
        self,
        candidates: list[tuple[MemoryScoreBreakdown, Experience]],
        selected_ids: set[int],
        context: RetrievalContext,
    ) -> tuple[MemoryScoreBreakdown, Experience, str] | None:
        if context.dataset_signature == "unknown":
            return None
        for score, exp in candidates:
            if id(exp) in selected_ids or exp.kind != MemoryKind.DATASET:
                continue
            if exp.dataset_signature == context.dataset_signature:
                return score, exp, "Selected as the exact dataset anchor for this run."
        return None

    def _select_kind_memories(
        self,
        candidates: list[tuple[MemoryScoreBreakdown, Experience]],
        kind: MemoryKind,
        selected_ids: set[int],
        blocked_conflict_slots: set[str],
        limit: int,
        audit: MemoryRetrievalAudit,
        rationale: str,
    ) -> list[tuple[MemoryScoreBreakdown, Experience, str]]:
        selected: list[tuple[MemoryScoreBreakdown, Experience, str]] = []
        for score, exp in candidates:
            if exp.kind != kind or id(exp) in selected_ids:
                continue
            if kind in {MemoryKind.WORKFLOW, MemoryKind.EXECUTION} and exp.conflict_slot in blocked_conflict_slots:
                if not any(entry.source_task_id == exp.source_task_id for entry in audit.safeguard_overrides):
                    audit.safeguard_overrides.append(
                        self._make_audit_entry(
                            exp,
                            score,
                            disposition="suppressed_safeguard_override",
                            rationale=(
                                f"Suppressed because a safeguard memory already occupied conflict slot "
                                f"'{exp.conflict_slot}'."
                            ),
                        )
                    )
                continue
            selected.append((score, exp, rationale))
            if len(selected) >= limit:
                break
        return selected

    def _select_overflow_memories(
        self,
        candidates: list[tuple[MemoryScoreBreakdown, Experience]],
        selected_ids: set[int],
        blocked_conflict_slots: set[str],
        limit: int,
        audit: MemoryRetrievalAudit,
    ) -> list[tuple[MemoryScoreBreakdown, Experience, str]]:
        selected: list[tuple[MemoryScoreBreakdown, Experience, str]] = []
        preferred_order = [MemoryKind.WORKFLOW, MemoryKind.EXECUTION, MemoryKind.SAFEGUARD]
        for kind in preferred_order:
            kind_selected = self._select_kind_memories(
                candidates,
                kind=kind,
                selected_ids=selected_ids | {id(exp) for _, exp, _ in selected},
                blocked_conflict_slots=blocked_conflict_slots,
                limit=max(0, limit - len(selected)),
                audit=audit,
                rationale="Selected as an overflow memory after the primary lane budgets were exhausted.",
            )
            selected.extend(kind_selected)
            if len(selected) >= limit:
                break
        return selected[:limit]

    def _record_remaining_suppressions(
        self,
        filtered_by_conflict: list[tuple[MemoryScoreBreakdown, Experience]],
        selected_ids: set[int],
        blocked_conflict_slots: set[str],
        audit: MemoryRetrievalAudit,
    ) -> None:
        for score, exp in filtered_by_conflict:
            if id(exp) in selected_ids:
                continue
            if exp.kind in {MemoryKind.WORKFLOW, MemoryKind.EXECUTION} and exp.conflict_slot in blocked_conflict_slots:
                continue
            audit.suppressed.append(
                self._make_audit_entry(
                    exp,
                    score,
                    disposition="suppressed",
                    rationale="Not selected after higher-priority EHR memory lanes consumed the available capacity.",
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
        failure_words: set[str],
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
                        exp.conflict_slot or "",
                        " ".join(exp.tags),
                        " ".join(exp.risk_tags),
                        " ".join(exp.schema_tags),
                    ]
                ).lower(),
            )
        )
        schema_overlap = len(set(exp.schema_tags).intersection(context.schema_tags))
        risk_overlap = len(set(exp.risk_tags).intersection(context.risk_tags))
        overlap_score = len(query_words.intersection(exp_words))
        task_family_bonus = 4 if exp.task_family == context.task_family else 0
        dataset_bonus = 3 if context.dataset_signature != "unknown" and exp.dataset_signature == context.dataset_signature else 0
        applicability_bonus = self._applicability_bonus(exp, context)
        schema_bonus = min(4, schema_overlap * 2)
        risk_bonus = min(4, risk_overlap * 2)
        failure_bonus = min(3, len(failure_words.intersection(exp_words)))
        kind_bonus = self._kind_bonus(exp, context)
        source_bonus = self._source_bonus(exp)
        confidence_bonus = round(exp.confidence, 2)
        utility_bonus = self._utility_bonus(exp)
        recency_bonus = 1 if exp.last_validated_at or exp.created_at else 0
        total_score = (
            overlap_score
            + task_family_bonus
            + dataset_bonus
            + applicability_bonus
            + schema_bonus
            + risk_bonus
            + failure_bonus
            + kind_bonus
            + source_bonus
            + confidence_bonus
            + utility_bonus
            + recency_bonus
        )
        return MemoryScoreBreakdown(
            overlap_score=overlap_score,
            task_family_bonus=task_family_bonus,
            dataset_bonus=dataset_bonus,
            applicability_bonus=applicability_bonus,
            schema_bonus=schema_bonus,
            risk_bonus=risk_bonus,
            failure_bonus=failure_bonus,
            kind_bonus=kind_bonus,
            source_bonus=source_bonus,
            confidence_bonus=confidence_bonus,
            utility_bonus=utility_bonus,
            recency_bonus=recency_bonus,
            total_score=round(total_score, 2),
        )

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def _applicability_bonus(self, exp: Experience, context: RetrievalContext) -> int:
        if exp.applicability_scope == "dataset_exact":
            return 4 if context.dataset_signature != "unknown" and exp.dataset_signature == context.dataset_signature else -1
        if exp.applicability_scope == "task_family":
            return 2 if exp.task_family == context.task_family else 0
        if exp.applicability_scope == "domain_ehr":
            return 2 if context.domain_focus == "ehr" else 0
        if exp.applicability_scope == "workflow_generic":
            return 1
        return 0

    def _kind_bonus(self, exp: Experience, context: RetrievalContext) -> int:
        if exp.kind == MemoryKind.SAFEGUARD:
            return 3 if self._elevated_risk(context) or context.task_family in EHR_MODELING_FAMILIES else 1
        if exp.kind == MemoryKind.WORKFLOW:
            return 2
        if exp.kind == MemoryKind.DATASET:
            return 2 if exp.dataset_signature == context.dataset_signature else 1
        if exp.kind == MemoryKind.EXECUTION:
            return 1 if self._allow_execution_lane(context) else 0
        return 0

    def _source_bonus(self, exp: Experience) -> int:
        if exp.kind == MemoryKind.SAFEGUARD:
            return {
                SourceOutcome.FAILED: 2,
                SourceOutcome.RECOVERED: 1,
                SourceOutcome.SUCCESS: 0,
            }[exp.source_outcome]
        return {
            SourceOutcome.SUCCESS: 2,
            SourceOutcome.RECOVERED: 2,
            SourceOutcome.FAILED: 0,
        }[exp.source_outcome]

    def _elevated_risk(self, context: RetrievalContext) -> bool:
        if len(context.risk_tags) >= 2:
            return True
        return any(tag in HIGH_PRIORITY_RISK_TAGS for tag in context.risk_tags)

    def _utility_bonus(self, exp: Experience) -> float:
        feedback_total = exp.times_helped + exp.times_hurt
        if feedback_total == 0:
            return 0.0
        normalized_utility = (exp.times_helped - exp.times_hurt) / feedback_total
        return round(max(min(normalized_utility * 2.5, 2.5), -2.5), 2)

    def _dedupe_signature(self, exp: Experience) -> str:
        normalized = "|".join(
            [
                exp.kind.value,
                exp.category.strip().lower(),
                re.sub(r"\s+", " ", exp.content.strip().lower()),
                exp.task_family.strip().lower(),
                exp.dataset_signature.strip().lower(),
                (exp.conflict_slot or "").strip().lower(),
                exp.applicability_scope.strip().lower(),
            ]
        )
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def _selected_by_kind(self, selected: list[Experience]) -> dict[MemoryKind, list[Experience]]:
        grouped: dict[MemoryKind, list[Experience]] = defaultdict(list)
        for exp in selected:
            grouped[exp.kind].append(exp)
        return grouped

    def _make_audit_entry(
        self,
        exp: Experience,
        score: MemoryScoreBreakdown,
        disposition: str,
        rationale: str,
    ) -> MemoryAuditEntry:
        return MemoryAuditEntry(
            experience_id=exp.experience_id,
            source_task_id=exp.source_task_id,
            kind=exp.kind,
            source_outcome=exp.source_outcome,
            category=exp.category,
            content_preview=exp.content[:200],
            conflict_slot=exp.conflict_slot,
            applicability_scope=exp.applicability_scope,
            risk_tags=exp.risk_tags,
            schema_tags=exp.schema_tags,
            score=score,
            disposition=disposition,
            rationale=rationale,
        )

    def _retire_experiences_by_id(
        self,
        experiences: list[Experience],
        experience_ids: set[str],
        reason: str,
    ) -> int:
        now = datetime.now(timezone.utc)
        retired_count = 0
        for exp in experiences:
            if exp.experience_id not in experience_ids or exp.retired:
                continue
            exp.retired = True
            exp.retired_reason = reason
            exp.retired_at = now
            exp.last_validated_at = now
            retired_count += 1
        return retired_count
