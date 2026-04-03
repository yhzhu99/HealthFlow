import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import aiofiles
from loguru import logger

from .experience_models import Experience
from .experience_models import MemoryAuditEntry, MemoryKind, MemoryRetrievalAudit, MemoryRetrievalResult
from .experience_models import MemoryScoreBreakdown, MemoryUpdate, MemoryUpdateAction, RetrievalContext


MAX_RETRIEVAL_CAPACITY = 4
SAFEGUARD_LIMIT = 1
DATASET_LIMIT = 1
WORKFLOW_LIMIT = 2


class ExperienceManager:
    """
    Manages persistent structured experiences and retrieves explicit, inspectable
    memory for future runs.
    """

    def __init__(self, experience_path: Path):
        self.experience_path = experience_path
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
        retired_count = 0
        superseded_ids = {
            experience_id
            for exp in experiences
            for experience_id in exp.supersedes
        }

        if superseded_ids:
            retired_count += self._retire_experiences_by_id(
                existing_experiences,
                superseded_ids,
                reason="Superseded by a newer strategic memory synthesized from a later trajectory.",
            )

        for exp in experiences:
            signature = self._dedupe_signature(exp)
            if signature in existing_signatures:
                continue
            if exp.kind in {MemoryKind.SAFEGUARD, MemoryKind.WORKFLOW} and exp.conflict_slot:
                retired_count += self._retire_conflict_slot_memories(existing_experiences, exp)
            existing_experiences.append(exp)
            existing_signatures.add(signature)
            appended_count += 1

        await self._persist_experiences(existing_experiences)
        if retired_count:
            logger.info("Retired {} superseded or replaced memories before saving new experiences.", retired_count)
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
                exp.validation_count += 1
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
        audit = MemoryRetrievalAudit(
            query=query,
            task_family=context.task_family,
            domain_focus=context.domain_focus,
            dataset_signature=context.dataset_signature,
            capacity=MAX_RETRIEVAL_CAPACITY,
        )
        audit.selection_policy.extend(
            [
                "Retrieval uses lexical overlap, task-family match, exact dataset match, applicability scope, schema overlap, risk overlap, and confidence.",
                "At most one safeguard is selected, and only when its risk tags match an actionable current task risk.",
                "At most one dataset memory is selected, and only as an exact dataset anchor.",
                "At most two workflow memories are selected after safeguard and dataset selection.",
            ]
        )
        if not self.experience_path.exists() or self.experience_path.stat().st_size == 0:
            return MemoryRetrievalResult(selected_experiences=[], audit=audit)

        all_experiences = await self._load_all_experiences()
        active_experiences = [exp for exp in all_experiences if not exp.retired]
        if not active_experiences:
            return MemoryRetrievalResult(selected_experiences=[], audit=audit)

        query_words = self._tokenize(query)
        scored_experiences: list[tuple[MemoryScoreBreakdown, Experience]] = []
        for exp in active_experiences:
            score = self._score_experience(exp, query_words, context)
            if score.total_score > 0:
                scored_experiences.append((score, exp))

        scored_experiences.sort(
            key=lambda item: (
                item[0].total_score,
                item[1].validation_count,
                item[1].confidence,
                item[1].created_at,
            ),
            reverse=True,
        )
        deduped = self._filter_duplicates(scored_experiences, audit)
        filtered_by_conflict = self._filter_same_kind_conflicts(deduped, audit)

        selected: list[Experience] = []
        selected_ids: set[int] = set()
        blocked_conflict_slots: set[str] = set()

        safeguard_candidate = self._select_safeguard(filtered_by_conflict, selected_ids, context)
        if safeguard_candidate is not None:
            score, exp, rationale = safeguard_candidate
            self._record_selection(audit, selected, selected_ids, exp, score, rationale)
            if exp.conflict_slot:
                blocked_conflict_slots.add(exp.conflict_slot)

        dataset_candidate = self._select_dataset_anchor(filtered_by_conflict, selected_ids, context)
        if dataset_candidate is not None and len(selected) < MAX_RETRIEVAL_CAPACITY:
            score, exp, rationale = dataset_candidate
            self._record_selection(audit, selected, selected_ids, exp, score, rationale)

        workflow_candidates = self._select_workflows(
            filtered_by_conflict,
            selected_ids=selected_ids,
            blocked_conflict_slots=blocked_conflict_slots,
            limit=min(WORKFLOW_LIMIT, MAX_RETRIEVAL_CAPACITY - len(selected)),
            audit=audit,
        )
        for score, exp, rationale in workflow_candidates:
            self._record_selection(audit, selected, selected_ids, exp, score, rationale)

        self._record_remaining_suppressions(
            filtered_by_conflict,
            selected_ids=selected_ids,
            blocked_conflict_slots=blocked_conflict_slots,
            context=context,
            audit=audit,
        )
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
                    payload = self._normalize_legacy_experience_payload(json.loads(line))
                    all_experiences.append(Experience(**payload))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning("Skipping corrupted line in experience.jsonl: {}", exc)
        return all_experiences

    async def _persist_experiences(self, experiences: list[Experience]) -> None:
        async with aiofiles.open(self.experience_path, mode="w", encoding="utf-8") as handle:
            for exp in experiences:
                await handle.write(exp.model_dump_json() + "\n")

    def _normalize_legacy_experience_payload(self, payload: dict) -> dict:
        normalized = dict(payload)
        if normalized.get("kind") == "execution":
            normalized["kind"] = MemoryKind.WORKFLOW.value
            normalized["applicability_scope"] = normalized.get("applicability_scope") or "workflow_generic"
            tags = [str(item) for item in normalized.get("tags", []) if str(item).strip()]
            if "legacy_execution" not in tags:
                tags.append("legacy_execution")
            normalized["tags"] = tags
        if "validation_count" not in normalized:
            helpful = int(normalized.get("times_helped", 0) or 0)
            harmful = int(normalized.get("times_hurt", 0) or 0)
            normalized["validation_count"] = max(0, helpful - harmful)
        return normalized

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
        seen_conflicts: dict[tuple[str, str, str, str], Experience] = {}
        for score, exp in scored_experiences:
            if not exp.conflict_slot:
                filtered.append((score, exp))
                continue
            key = (
                exp.kind.value,
                exp.conflict_slot,
                exp.applicability_scope,
                self._scope_target(exp),
            )
            if key in seen_conflicts:
                audit.suppressed_conflicts.append(
                    self._make_audit_entry(
                        exp,
                        score,
                        disposition="suppressed_conflict",
                        rationale=(
                            f"Suppressed because another {exp.kind.value} memory already occupied "
                            f"conflict slot '{exp.conflict_slot}' for the same scope."
                        ),
                    )
                )
                continue
            seen_conflicts[key] = exp
            filtered.append((score, exp))
        return filtered

    def _select_safeguard(
        self,
        candidates: list[tuple[MemoryScoreBreakdown, Experience]],
        selected_ids: set[int],
        context: RetrievalContext,
    ) -> tuple[MemoryScoreBreakdown, Experience, str] | None:
        for score, exp in candidates:
            if exp.kind != MemoryKind.SAFEGUARD or id(exp) in selected_ids:
                continue
            if not self._safeguard_matches_current_risks(exp, context):
                continue
            return score, exp, "Selected because its safeguard risk tags matched the current actionable task risks."
        return None

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

    def _select_workflows(
        self,
        candidates: list[tuple[MemoryScoreBreakdown, Experience]],
        selected_ids: set[int],
        blocked_conflict_slots: set[str],
        limit: int,
        audit: MemoryRetrievalAudit,
    ) -> list[tuple[MemoryScoreBreakdown, Experience, str]]:
        selected: list[tuple[MemoryScoreBreakdown, Experience, str]] = []
        for score, exp in candidates:
            if exp.kind != MemoryKind.WORKFLOW or id(exp) in selected_ids:
                continue
            if exp.conflict_slot in blocked_conflict_slots:
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
            selected.append(
                (score, exp, "Selected as a reusable workflow for the current task family and dataset context.")
            )
            if len(selected) >= limit:
                break
        return selected

    def _record_remaining_suppressions(
        self,
        filtered_by_conflict: list[tuple[MemoryScoreBreakdown, Experience]],
        selected_ids: set[int],
        blocked_conflict_slots: set[str],
        context: RetrievalContext,
        audit: MemoryRetrievalAudit,
    ) -> None:
        for score, exp in filtered_by_conflict:
            if id(exp) in selected_ids:
                continue
            if exp.kind == MemoryKind.WORKFLOW and exp.conflict_slot in blocked_conflict_slots:
                continue
            if exp.kind == MemoryKind.SAFEGUARD and not self._safeguard_matches_current_risks(exp, context):
                rationale = "Not selected because its safeguard risk tags did not match any actionable current task risk."
            elif exp.kind == MemoryKind.DATASET and exp.dataset_signature != context.dataset_signature:
                rationale = "Not selected because dataset memories are only retrieved as exact dataset anchors."
            else:
                rationale = "Not selected after the fixed safeguard, dataset, and workflow retrieval lanes were filled."
            audit.suppressed.append(
                self._make_audit_entry(
                    exp,
                    score,
                    disposition="suppressed",
                    rationale=rationale,
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
        task_family_bonus = 3 if exp.task_family == context.task_family else 0
        dataset_bonus = 4 if context.dataset_signature != "unknown" and exp.dataset_signature == context.dataset_signature else 0
        applicability_bonus = self._applicability_bonus(exp, context)
        schema_bonus = min(3, schema_overlap)
        risk_bonus = min(4, risk_overlap * 2)
        confidence_bonus = round(exp.confidence, 2)
        total_score = (
            overlap_score
            + task_family_bonus
            + dataset_bonus
            + applicability_bonus
            + schema_bonus
            + risk_bonus
            + confidence_bonus
        )
        return MemoryScoreBreakdown(
            overlap_score=overlap_score,
            task_family_bonus=task_family_bonus,
            dataset_bonus=dataset_bonus,
            applicability_bonus=applicability_bonus,
            schema_bonus=schema_bonus,
            risk_bonus=risk_bonus,
            confidence_bonus=confidence_bonus,
            total_score=round(total_score, 2),
        )

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def _applicability_bonus(self, exp: Experience, context: RetrievalContext) -> int:
        if exp.applicability_scope == "dataset_exact":
            return 3 if context.dataset_signature != "unknown" and exp.dataset_signature == context.dataset_signature else -2
        if exp.applicability_scope == "task_family":
            return 2 if exp.task_family == context.task_family else 0
        if exp.applicability_scope == "domain_ehr":
            return 1 if context.domain_focus == "ehr" else 0
        if exp.applicability_scope == "workflow_generic":
            return 1
        return 0

    def _safeguard_matches_current_risks(self, exp: Experience, context: RetrievalContext) -> bool:
        if not context.risk_tags or not exp.risk_tags:
            return False
        return bool(set(exp.risk_tags).intersection(context.risk_tags))

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
                self._scope_target(exp),
            ]
        )
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def _scope_target(self, exp: Experience) -> str:
        if exp.applicability_scope == "dataset_exact":
            return exp.dataset_signature.strip().lower()
        if exp.applicability_scope == "task_family":
            return exp.task_family.strip().lower()
        return ""

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

    def _retire_conflict_slot_memories(self, experiences: list[Experience], incoming: Experience) -> int:
        now = datetime.now(timezone.utc)
        retired_count = 0
        incoming_scope_target = self._scope_target(incoming)
        for exp in experiences:
            if exp.retired:
                continue
            if exp.kind != incoming.kind:
                continue
            if exp.conflict_slot != incoming.conflict_slot:
                continue
            if exp.applicability_scope != incoming.applicability_scope:
                continue
            if self._scope_target(exp) != incoming_scope_target:
                continue
            exp.retired = True
            exp.retired_reason = (
                f"Retired because a newer {incoming.kind.value} memory replaced the same conflict slot for the same scope."
            )
            exp.retired_at = now
            exp.last_validated_at = now
            retired_count += 1
        return retired_count
