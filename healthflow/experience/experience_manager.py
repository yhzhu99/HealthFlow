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


DEFAULT_TOTAL_RETRIEVAL_CAPACITY = 6
DEFAULT_RETRIEVAL_RANGES: dict[MemoryKind, tuple[int, int]] = {
    MemoryKind.SAFEGUARD: (0, 2),
    MemoryKind.WORKFLOW: (0, 2),
    MemoryKind.DATASET_ANCHOR: (0, 2),
    MemoryKind.CODE_SNIPPET: (0, 2),
}
SELECTION_ORDER: tuple[MemoryKind, ...] = (
    MemoryKind.SAFEGUARD,
    MemoryKind.DATASET_ANCHOR,
    MemoryKind.WORKFLOW,
    MemoryKind.CODE_SNIPPET,
)


class ExperienceManager:
    """
    Manages persistent structured experiences and retrieves explicit, inspectable
    memory for future runs.
    """

    def __init__(
        self,
        experience_path: Path,
        *,
        read_only: bool = False,
        retrieval_ranges: dict[MemoryKind, tuple[int, int]] | None = None,
        max_retrieval_capacity: int = DEFAULT_TOTAL_RETRIEVAL_CAPACITY,
    ):
        self.experience_path = experience_path
        self.read_only = read_only
        self.retrieval_ranges = self._normalize_retrieval_ranges(retrieval_ranges)
        self.max_retrieval_capacity = max(1, int(max_retrieval_capacity))
        if not self.read_only:
            self.experience_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.experience_path.exists():
                self.experience_path.touch()
        logger.info(
            "ExperienceManager initialized. Knowledge base at: {} (read_only={})",
            self.experience_path,
            self.read_only,
        )

    async def save_experiences(self, experiences: List[Experience]):
        if self.read_only:
            logger.info("Skipping experience save because memory is frozen: {}", self.experience_path)
            return
        if not experiences:
            return

        existing_experiences = await self._load_all_experiences()
        existing_signatures = {self._dedupe_signature(exp) for exp in existing_experiences}
        appended_count = 0
        retired_count = 0
        superseded_ids = {experience_id for exp in experiences for experience_id in exp.supersedes}

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
            existing_experiences.append(exp)
            existing_signatures.add(signature)
            appended_count += 1

        await self._persist_experiences(existing_experiences)
        if retired_count:
            logger.info("Retired {} explicitly superseded memories before saving new experiences.", retired_count)
        logger.info("Saved {} new experiences to the knowledge base.", appended_count)

    async def reset(self):
        if self.read_only:
            logger.info("Skipping reset because memory is frozen: {}", self.experience_path)
            return
        await self._persist_experiences([])
        logger.info("Reset experience memory at {}", self.experience_path)

    async def apply_memory_updates(self, updates: List[MemoryUpdate]) -> list[str]:
        if self.read_only:
            logger.info("Skipping memory lifecycle updates because memory is frozen: {}", self.experience_path)
            return []
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
            capacity=self.max_retrieval_capacity,
        )
        audit.selection_policy.extend(
            [
                "Retrieval ranks memories by task relevance, validation count, and confidence, without any recency term.",
                (
                    "Adaptive per-type retrieval ranges are bounded instead of fixed-top-k: "
                    f"safeguard {self._format_range(MemoryKind.SAFEGUARD)}, "
                    f"dataset_anchor {self._format_range(MemoryKind.DATASET_ANCHOR)}, "
                    f"workflow {self._format_range(MemoryKind.WORKFLOW)}, "
                    f"code_snippet {self._format_range(MemoryKind.CODE_SNIPPET)}."
                ),
                "Safeguards require current EHR risk-tag overlap; dataset anchors require exact dataset match.",
                "Workflows and code snippets are retrieved by task-family, schema, category, and implementation relevance.",
                "Competing memories are suppressed only within the same kind, category, and scope; cross-kind memories can coexist.",
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
                item[1].experience_id,
            ),
            reverse=True,
        )
        deduped = self._filter_duplicates(scored_experiences, audit)
        eligible, ineligible = self._partition_candidates(deduped, context)
        filtered_by_competition = self._filter_competing_candidates(eligible, audit)

        selected: list[Experience] = []
        selected_ids: set[int] = set()
        selected_counts: dict[MemoryKind, int] = defaultdict(int)

        for kind in SELECTION_ORDER:
            remaining_capacity = self.max_retrieval_capacity - len(selected)
            if remaining_capacity <= 0:
                break
            _, upper = self.retrieval_ranges[kind]
            limit = min(upper, remaining_capacity)
            if limit <= 0:
                continue
            for score, exp in filtered_by_competition:
                if exp.kind != kind or id(exp) in selected_ids:
                    continue
                self._record_selection(
                    audit,
                    selected,
                    selected_ids,
                    exp,
                    score,
                    self._selection_rationale(exp),
                )
                selected_counts[kind] += 1
                if selected_counts[kind] >= limit or len(selected) >= self.max_retrieval_capacity:
                    break

        self._record_remaining_suppressions(
            ineligible=ineligible,
            filtered_by_competition=filtered_by_competition,
            selected_ids=selected_ids,
            selected_counts=selected_counts,
            audit=audit,
        )
        if selected:
            for exp in selected:
                exp.times_retrieved += 1
            if not self.read_only:
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
            dataset_anchor_experiences=grouped[MemoryKind.DATASET_ANCHOR],
            code_snippet_experiences=grouped[MemoryKind.CODE_SNIPPET],
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
        if self.read_only:
            logger.debug("Ignoring persistence request for frozen memory: {}", self.experience_path)
            return
        async with aiofiles.open(self.experience_path, mode="w", encoding="utf-8") as handle:
            for exp in experiences:
                await handle.write(exp.model_dump_json() + "\n")

    def _normalize_retrieval_ranges(
        self,
        retrieval_ranges: dict[MemoryKind, tuple[int, int]] | None,
    ) -> dict[MemoryKind, tuple[int, int]]:
        normalized: dict[MemoryKind, tuple[int, int]] = {}
        for kind, default_range in DEFAULT_RETRIEVAL_RANGES.items():
            lower, upper = (retrieval_ranges or {}).get(kind, default_range)
            lower = max(0, int(lower))
            upper = max(lower, int(upper))
            normalized[kind] = (lower, upper)
        return normalized

    def _normalize_legacy_experience_payload(self, payload: dict) -> dict:
        normalized = dict(payload)
        legacy_kind = str(normalized.get("kind", "") or "").strip().lower()
        if legacy_kind == "execution":
            normalized["kind"] = MemoryKind.WORKFLOW.value
            normalized["applicability_scope"] = normalized.get("applicability_scope") or "workflow_generic"
            tags = [str(item) for item in normalized.get("tags", []) if str(item).strip()]
            if "legacy_execution" not in tags:
                tags.append("legacy_execution")
            normalized["tags"] = tags
        elif legacy_kind == "dataset":
            normalized["kind"] = MemoryKind.DATASET_ANCHOR.value
            normalized["applicability_scope"] = normalized.get("applicability_scope") or "dataset_exact"
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

    def _partition_candidates(
        self,
        scored_experiences: list[tuple[MemoryScoreBreakdown, Experience]],
        context: RetrievalContext,
    ) -> tuple[list[tuple[MemoryScoreBreakdown, Experience]], list[tuple[MemoryScoreBreakdown, Experience, str]]]:
        eligible: list[tuple[MemoryScoreBreakdown, Experience]] = []
        ineligible: list[tuple[MemoryScoreBreakdown, Experience, str]] = []
        for score, exp in scored_experiences:
            rationale = self._ineligibility_reason(exp, score, context)
            if rationale:
                ineligible.append((score, exp, rationale))
                continue
            eligible.append((score, exp))
        return eligible, ineligible

    def _filter_competing_candidates(
        self,
        scored_experiences: list[tuple[MemoryScoreBreakdown, Experience]],
        audit: MemoryRetrievalAudit,
    ) -> list[tuple[MemoryScoreBreakdown, Experience]]:
        filtered: list[tuple[MemoryScoreBreakdown, Experience]] = []
        seen_competitors: dict[tuple[str, str, str, str], Experience] = {}
        for score, exp in scored_experiences:
            key = self._competition_key(exp)
            if key in seen_competitors:
                audit.suppressed_competitors.append(
                    self._make_audit_entry(
                        exp,
                        score,
                        disposition="suppressed_competitor",
                        rationale=(
                            "Suppressed because a stronger active memory of the same type, category, "
                            "and scope had already been ranked higher."
                        ),
                    )
                )
                continue
            seen_competitors[key] = exp
            filtered.append((score, exp))
        return filtered

    def _record_remaining_suppressions(
        self,
        *,
        ineligible: list[tuple[MemoryScoreBreakdown, Experience, str]],
        filtered_by_competition: list[tuple[MemoryScoreBreakdown, Experience]],
        selected_ids: set[int],
        selected_counts: dict[MemoryKind, int],
        audit: MemoryRetrievalAudit,
    ) -> None:
        for score, exp, rationale in ineligible:
            audit.suppressed.append(
                self._make_audit_entry(
                    exp,
                    score,
                    disposition="suppressed",
                    rationale=rationale,
                )
            )

        for score, exp in filtered_by_competition:
            if id(exp) in selected_ids:
                continue
            _, upper = self.retrieval_ranges[exp.kind]
            if selected_counts[exp.kind] >= upper:
                rationale = (
                    f"Not selected because higher-ranked {exp.kind.value.replace('_', ' ')} memories "
                    "already filled the configured retrieval range."
                )
            else:
                rationale = "Not selected because higher-priority memories filled the bounded retrieval budget."
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
                        exp.applicability_scope,
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
        risk_bonus = min(4, risk_overlap * 2) if exp.kind == MemoryKind.SAFEGUARD else min(2, risk_overlap)
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
            return 3 if context.dataset_signature != "unknown" and exp.dataset_signature == context.dataset_signature else -3
        if exp.applicability_scope == "task_family":
            return 2 if exp.task_family == context.task_family else 0
        if exp.applicability_scope == "domain_ehr":
            return 1 if context.domain_focus == "ehr" else -1
        if exp.applicability_scope == "workflow_generic":
            return 1
        return 0

    def _ineligibility_reason(
        self,
        exp: Experience,
        score: MemoryScoreBreakdown,
        context: RetrievalContext,
    ) -> str | None:
        if exp.kind == MemoryKind.SAFEGUARD and not self._safeguard_matches_current_risks(exp, context):
            return "Not selected because its safeguard risk tags did not match any actionable current task risk."
        if exp.kind == MemoryKind.DATASET_ANCHOR and exp.dataset_signature != context.dataset_signature:
            return "Not selected because dataset anchors are only retrieved under exact dataset match."
        if exp.kind == MemoryKind.WORKFLOW and not self._workflow_matches(score):
            return "Not selected because no task-family, schema, or category relevance was detected for this workflow."
        if exp.kind == MemoryKind.CODE_SNIPPET and not self._code_snippet_matches(score):
            return "Not selected because no implementation relevance or schema compatibility was detected for this code snippet."
        return None

    def _workflow_matches(self, score: MemoryScoreBreakdown) -> bool:
        return bool(score.overlap_score or score.task_family_bonus or score.schema_bonus)

    def _code_snippet_matches(self, score: MemoryScoreBreakdown) -> bool:
        return bool(score.overlap_score or score.task_family_bonus or score.schema_bonus)

    def _safeguard_matches_current_risks(self, exp: Experience, context: RetrievalContext) -> bool:
        if not context.risk_tags or not exp.risk_tags:
            return False
        return bool(set(exp.risk_tags).intersection(context.risk_tags))

    def _selection_rationale(self, exp: Experience) -> str:
        if exp.kind == MemoryKind.SAFEGUARD:
            return "Selected because its safeguard risk tags matched the current EHR task risks."
        if exp.kind == MemoryKind.DATASET_ANCHOR:
            return "Selected as an exact dataset anchor for this dataset signature."
        if exp.kind == MemoryKind.WORKFLOW:
            return "Selected as a reusable workflow for the current task family and schema."
        return "Selected as an implementation-ready code snippet for the current task."

    def _format_range(self, kind: MemoryKind) -> str:
        lower, upper = self.retrieval_ranges[kind]
        return f"{lower}-{upper}"

    def _dedupe_signature(self, exp: Experience) -> str:
        normalized = "|".join(
            [
                exp.kind.value,
                exp.category.strip().lower(),
                re.sub(r"\s+", " ", exp.content.strip().lower()),
                exp.task_family.strip().lower(),
                exp.dataset_signature.strip().lower(),
                exp.applicability_scope.strip().lower(),
                self._scope_target(exp),
            ]
        )
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def _competition_key(self, exp: Experience) -> tuple[str, str, str, str]:
        return (
            exp.kind.value,
            exp.category.strip().lower(),
            exp.applicability_scope.strip().lower(),
            self._scope_target(exp),
        )

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
        scope_target = self._scope_target(exp)
        return MemoryAuditEntry(
            experience_id=exp.experience_id,
            source_task_id=exp.source_task_id,
            kind=exp.kind,
            source_outcome=exp.source_outcome,
            category=exp.category,
            content_preview=exp.content[:200],
            applicability_scope=exp.applicability_scope,
            scope_target=scope_target or None,
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
