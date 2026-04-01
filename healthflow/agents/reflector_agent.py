import json
from typing import Any, Dict, List

from loguru import logger

from ..core.contracts import EvaluationVerdict
from ..core.llm_provider import LLMMessage, LLMProvider, StructuredResponseError, parse_json_content
from ..experience.experience_models import Experience, MemoryKind, SourceOutcome
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

    async def synthesize_experience(self, full_history: Dict[str, Any], final_verdict: EvaluationVerdict) -> List[Experience]:
        system_prompt = get_prompt("reflector_system")

        final_attempt = full_history["attempts"][-1]
        history_for_prompt = {
            "user_request": full_history["user_request"],
            "data_profile": full_history.get("data_profile", {}),
            "risk_findings": full_history.get("risk_findings", []),
            "retrieved_memory": final_attempt.get("memory", {}),
            "backend": full_history.get("backend", "unknown"),
            "task_success": final_verdict.status == "success",
            "final_plan": final_attempt["plan_markdown"],
            "final_log": final_attempt["execution"]["log"][:8000],
            "evaluation_verdict": final_verdict.model_dump(mode="json"),
        }
        history_str = json.dumps(history_for_prompt, indent=2)
        user_prompt = render_prompt("reflector_user", task_history=history_str)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        logger.info("Requesting experience synthesis from LLM...")
        experiences: List[Experience] = []
        try:
            data, response = await self.llm_provider.generate_structured(
                messages,
                parse_json_content,
                temperature=0.0,
            )
            self.last_usage = response.usage
            self.last_model_name = response.model_name
            self.last_estimated_cost_usd = response.estimated_cost_usd

            raw_experiences = data.get("experiences", [])
            source_outcome = self._source_outcome(full_history, final_verdict)
            for item in raw_experiences:
                try:
                    proposed_kind = MemoryKind(item.get("kind", MemoryKind.WORKFLOW.value))
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
                            category=item["category"],
                            content=item["content"],
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
                            confidence=float(item.get("confidence", 0.6)),
                            conflict_slot=item.get("conflict_slot"),
                            applicability_scope=item.get("applicability_scope", "task_family"),
                            risk_tags=self._normalize_tags(item.get("risk_tags", [])),
                            schema_tags=self._normalize_tags(item.get("schema_tags", [])),
                            tags=self._normalize_tags(item.get("tags", [])),
                        )
                    )
                except (KeyError, ValueError) as exc:
                    logger.warning("Skipping invalid experience item from LLM: {}. Error: {}", item, exc)

            logger.info("Successfully synthesized {} new experiences.", len(experiences))
            return experiences
        except (StructuredResponseError, KeyError, ValueError) as exc:
            response = exc.response if isinstance(exc, StructuredResponseError) else None
            if response is not None:
                self.last_usage = response.usage
                self.last_model_name = response.model_name
                self.last_estimated_cost_usd = response.estimated_cost_usd
            logger.error("Failed to parse valid experiences from LLM response. Error: {}", exc)
            if response is not None:
                logger.debug("Invalid JSON response from LLM: {}", response.content)
            return []

    def _allow_safeguard_memory(
        self,
        item: dict[str, Any],
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
        item: dict[str, Any],
        full_history: dict[str, Any],
        final_verdict: EvaluationVerdict,
    ) -> bool:
        if final_verdict.status == "success" and len(full_history.get("attempts", [])) == 1:
            return False
        return self._looks_like_safeguard(item, full_history)

    def _looks_like_safeguard(self, item: dict[str, Any], full_history: dict[str, Any]) -> bool:
        risk_tags = set(self._normalize_tags(item.get("risk_tags", [])))
        if risk_tags.intersection(SAFEGUARD_RISK_TAGS):
            return True
        content = str(item.get("content", "")).lower()
        if any(token in content for token in SAFEGUARD_CONTENT_TOKENS):
            return True
        category = str(item.get("category", "")).lower()
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
