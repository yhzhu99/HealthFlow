import json
from dataclasses import dataclass
from typing import Any, Dict, List

from loguru import logger

from ..core.contracts import EvaluationVerdict
from ..core.llm_provider import LLMMessage, LLMProvider, StructuredResponseError, parse_json_content
from ..experience.experience_models import (
    Experience,
    MemoryKind,
    MemoryUpdate,
    ReflectionSynthesisResult,
    SourceOutcome,
    SynthesizedExperience,
)
from ..prompts.templates import get_prompt, render_prompt


SAFEGUARD_RISK_TAGS = {
    "temporal_leakage",
    "target_leakage",
    "validation_strategy",
    "patient_linkage",
    "cohort_definition",
    "identifier_policy",
}
SAFEGUARD_CONTENT_TOKENS = [
    "leakage",
    "temporal",
    "patient-level split",
    "cohort",
    "identifier",
    "privacy",
    "unsafe",
    "label leakage",
]


class ReflectorAgent:
    """
    Analyze completed task trajectories and distill reusable memory items.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.last_usage: dict = {}
        self.last_model_name: str = llm_provider.model_name
        self.last_estimated_cost_usd: float | None = None
        self.last_trace: dict = {}

    async def synthesize_experience(
        self,
        full_history: Dict[str, Any],
        final_verdict: EvaluationVerdict,
    ) -> "ReflectionWriteback":
        system_prompt = get_prompt("reflector_system")
        history_for_prompt = {
            "user_request": full_history["user_request"],
            "data_profile": full_history.get("data_profile", {}),
            "risk_findings": full_history.get("risk_findings", []),
            "backend": full_history.get("backend", "unknown"),
            "task_success": final_verdict.status == "success",
            "evaluation_verdict": final_verdict.model_dump(mode="json"),
            "trajectory": self._trajectory_for_prompt(full_history),
        }
        history_str = json.dumps(history_for_prompt, indent=2)
        user_prompt = render_prompt("reflector_user", task_history=history_str)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        logger.info("Requesting experience synthesis from LLM...")
        experiences: List[Experience] = []
        memory_updates: list[MemoryUpdate] = []
        try:
            synthesis, response = await self.llm_provider.generate_structured(
                messages,
                lambda content: ReflectionSynthesisResult(**parse_json_content(content)),
                temperature=0.0,
            )
            self.last_usage = response.usage
            self.last_model_name = response.model_name
            self.last_estimated_cost_usd = response.estimated_cost_usd

            source_outcome = self._source_outcome(full_history, final_verdict)
            memory_updates = synthesis.memory_updates
            for item in synthesis.experiences:
                try:
                    proposed_kind = item.kind
                    if proposed_kind == MemoryKind.SAFEGUARD and not self._allow_safeguard_memory(
                        item,
                        full_history,
                        final_verdict,
                    ):
                        proposed_kind = MemoryKind.WORKFLOW
                    elif proposed_kind != MemoryKind.SAFEGUARD and self._should_promote_to_safeguard(
                        item,
                        full_history,
                        final_verdict,
                    ):
                        proposed_kind = MemoryKind.SAFEGUARD

                    experiences.append(
                        Experience(
                            kind=proposed_kind,
                            category=item.category,
                            content=item.content,
                            source_task_id=full_history["task_id"],
                            task_family=full_history.get("data_profile", {}).get("task_family", "general_analysis"),
                            dataset_signature=full_history.get("data_profile", {}).get("dataset_signature", "unknown"),
                            provenance={
                                "backend": full_history.get("backend", "unknown"),
                                "verdict_status": final_verdict.status,
                                "failure_type": final_verdict.failure_type,
                            },
                            stage="reflection",
                            backend=full_history.get("backend", "unknown"),
                            source_outcome=source_outcome,
                            confidence=float(item.confidence),
                            conflict_slot=item.conflict_slot,
                            applicability_scope=item.applicability_scope,
                            risk_tags=self._normalize_tags(item.risk_tags),
                            schema_tags=self._normalize_tags(item.schema_tags),
                            tags=self._normalize_tags(item.tags),
                            supersedes=item.supersedes,
                        )
                    )
                except ValueError as exc:
                    logger.warning("Skipping invalid experience item from LLM: {}. Error: {}", item, exc)

            logger.info("Successfully synthesized {} new experiences.", len(experiences))
            self.last_trace = self._build_trace(
                messages=messages,
                raw_output=response.content,
                parsed_output={
                    "experiences": [item.model_dump(mode="json") for item in synthesis.experiences],
                    "memory_updates": [item.model_dump(mode="json") for item in synthesis.memory_updates],
                },
                error=None,
            )
            return ReflectionWriteback(
                experiences=experiences,
                memory_updates=memory_updates,
            )
        except (StructuredResponseError, KeyError, ValueError) as exc:
            response = exc.response if isinstance(exc, StructuredResponseError) else None
            if response is not None:
                self.last_usage = response.usage
                self.last_model_name = response.model_name
                self.last_estimated_cost_usd = response.estimated_cost_usd
            logger.error("Failed to parse valid experiences from LLM response. Error: {}", exc)
            if response is not None:
                logger.debug("Invalid JSON response from LLM: {}", response.content)
            self.last_trace = self._build_trace(
                messages=messages,
                raw_output=response.content if response is not None else "",
                parsed_output={"experiences": [], "memory_updates": []},
                error=str(exc),
            )
            return ReflectionWriteback()

    def _allow_safeguard_memory(
        self,
        item: SynthesizedExperience,
        full_history: dict[str, Any],
        final_verdict: EvaluationVerdict,
    ) -> bool:
        if final_verdict.status != "success":
            return True
        if len(full_history.get("attempts", [])) > 1:
            return True
        return self._looks_like_safeguard(item, full_history)

    def _should_promote_to_safeguard(
        self,
        item: SynthesizedExperience,
        full_history: dict[str, Any],
        final_verdict: EvaluationVerdict,
    ) -> bool:
        if final_verdict.status == "success" and len(full_history.get("attempts", [])) == 1:
            return False
        return self._looks_like_safeguard(item, full_history)

    def _looks_like_safeguard(self, item: SynthesizedExperience, full_history: dict[str, Any]) -> bool:
        risk_tags = set(self._normalize_tags(item.risk_tags))
        if risk_tags.intersection(SAFEGUARD_RISK_TAGS):
            return True
        content = item.content.lower()
        if any(token in content for token in SAFEGUARD_CONTENT_TOKENS):
            return True
        category = item.category.lower()
        if category in SAFEGUARD_RISK_TAGS:
            return True
        observed_risks = {
            finding.get("category", "")
            for finding in full_history.get("risk_findings", [])
            if isinstance(finding, dict)
        }
        return bool(observed_risks.intersection(risk_tags))

    def _source_outcome(self, full_history: dict[str, Any], final_verdict: EvaluationVerdict) -> SourceOutcome:
        if final_verdict.status != "success":
            return SourceOutcome.FAILED
        if len(full_history.get("attempts", [])) > 1:
            return SourceOutcome.RECOVERED
        if any(
            finding.get("severity") in {"high", "medium"}
            for finding in full_history.get("risk_findings", [])
            if isinstance(finding, dict)
        ):
            return SourceOutcome.RECOVERED
        return SourceOutcome.SUCCESS

    def _normalize_tags(self, values: list[Any]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip().lower()
            if not text:
                continue
            normalized.append(text.replace(" ", "_"))
        return list(dict.fromkeys(normalized))

    def _trajectory_for_prompt(self, full_history: dict[str, Any]) -> list[dict[str, Any]]:
        trajectory: list[dict[str, Any]] = []
        for attempt in full_history.get("attempts", []):
            memory = attempt.get("memory", {})
            retrieval = memory.get("retrieval", {})
            execution = attempt.get("execution", {})
            evaluation = attempt.get("evaluation", {})
            trajectory.append(
                {
                    "attempt": attempt.get("attempt"),
                    "plan": attempt.get("plan", {}),
                    "selected_memory": [
                        {
                            "experience_id": item.get("experience_id"),
                            "kind": item.get("kind"),
                            "category": item.get("category"),
                            "content_preview": item.get("content_preview"),
                            "rationale": item.get("rationale"),
                        }
                        for item in retrieval.get("selected", [])
                    ],
                    "safeguard_overrides": [
                        {
                            "experience_id": item.get("experience_id"),
                            "category": item.get("category"),
                            "rationale": item.get("rationale"),
                        }
                        for item in retrieval.get("safeguard_overrides", [])
                    ],
                    "execution": {
                        "success": execution.get("success"),
                        "return_code": execution.get("return_code"),
                        "timed_out": execution.get("timed_out"),
                        "log_excerpt": str(execution.get("log", ""))[:3000],
                    },
                    "evaluation": {
                        "status": evaluation.get("status"),
                        "score": evaluation.get("score"),
                        "failure_type": evaluation.get("failure_type"),
                        "feedback": evaluation.get("feedback"),
                        "repair_instructions": evaluation.get("repair_instructions", []),
                        "violated_constraints": evaluation.get("violated_constraints", []),
                        "repair_hypotheses": evaluation.get("repair_hypotheses", []),
                    },
                }
            )
        return trajectory

    def _build_trace(
        self,
        *,
        messages: list[LLMMessage],
        raw_output: str,
        parsed_output: dict[str, Any],
        error: str | None,
    ) -> dict[str, Any]:
        provider_trace = getattr(self.llm_provider, "last_structured_trace", {}) or {}
        return {
            "input_messages": [message.model_dump() for message in messages],
            "output_raw": raw_output,
            "output_parsed": parsed_output,
            "call": {
                "model_name": self.last_model_name,
                "usage": self.last_usage,
                "estimated_cost_usd": self.last_estimated_cost_usd,
                "local_attempt_count": provider_trace.get("local_attempt_count", 0),
                "duration_seconds": provider_trace.get("duration_seconds"),
                "error": error,
            },
            "repair_trace": provider_trace,
        }


@dataclass
class ReflectionWriteback:
    experiences: List[Experience] = None
    memory_updates: List[MemoryUpdate] = None

    def __post_init__(self) -> None:
        if self.experiences is None:
            self.experiences = []
        if self.memory_updates is None:
            self.memory_updates = []
