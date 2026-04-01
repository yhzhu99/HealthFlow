import json
from typing import List, Dict, Any
from loguru import logger

from ..core.contracts import EvaluationVerdict
from ..core.llm_provider import LLMProvider, LLMMessage, StructuredResponseError, parse_json_content
from ..prompts.templates import get_prompt, render_prompt
from ..experience.experience_models import (
    Experience,
    ExperiencePolarity,
    ExperienceType,
    MemoryLayer,
    ValidationStatus,
)


class ReflectorAgent:
    """
    This agent analyzes a task execution to synthesize generalizable knowledge,
    which is stored as structured `Experience` objects.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.last_usage: dict = {}
        self.last_model_name: str = llm_provider.model_name
        self.last_estimated_cost_usd: float | None = None

    async def synthesize_experience(self, full_history: Dict[str, Any], final_verdict: EvaluationVerdict) -> List[Experience]:
        """
        Analyze a completed execution history and synthesize reusable experiences.
        """
        system_prompt = get_prompt("reflector_system")

        final_attempt = full_history["attempts"][-1]
        was_success = final_verdict.status == "success"
        history_for_prompt = {
            "user_request": full_history["user_request"],
            "recommended_experiences": [exp["content"] for exp in full_history.get("recommended_experiences", [])],
            "avoidance_experiences": [exp["content"] for exp in full_history.get("avoidance_experiences", [])],
            "memory_retrieval": full_history.get("memory_retrieval", {}),
            "backend": full_history.get("backend", "unknown"),
            "task_success": was_success,
            "final_plan": final_attempt["plan_markdown"],
            "final_log": final_attempt["execution"]["log"][:8000],
            "evaluation_verdict": final_verdict.model_dump(mode="json"),
        }
        history_str = json.dumps(history_for_prompt, indent=2)

        user_prompt = render_prompt("reflector_user", task_history=history_str)

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
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
            for item in raw_experiences:
                try:
                    proposed_type = ExperienceType(item["type"])
                    proposed_layer = MemoryLayer(item.get("layer", "strategy"))
                    if not was_success:
                        proposed_layer = MemoryLayer.FAILURE
                        if proposed_type not in {ExperienceType.WARNING, ExperienceType.VERIFIER_RULE}:
                            proposed_type = ExperienceType.WARNING
                    exp = Experience(
                        type=proposed_type,
                        layer=proposed_layer,
                        polarity=ExperiencePolarity.AVOID if proposed_layer == MemoryLayer.FAILURE else ExperiencePolarity.RECOMMEND,
                        category=item["category"],
                        content=item["content"],
                        source_task_id=full_history["task_id"],
                        provenance={
                            "backend": full_history.get("backend", "unknown"),
                            "verdict_status": final_verdict.status,
                            "failure_type": final_verdict.failure_type,
                        },
                        stage="reflection",
                        backend=full_history.get("backend", "unknown"),
                        validation_status=ValidationStatus.VERIFIED if was_success else ValidationStatus.FAILED,
                        confidence=float(item.get("confidence", 0.6)),
                        conflict_group=item.get("conflict_group"),
                        applicability_scope=item.get(
                            "applicability_scope",
                            "task_family",
                        ),
                        safety_critical=self._is_safety_critical(item, proposed_type, proposed_layer),
                        verifier_supported=bool(item.get("verifier_supported", False)),
                        tags=item.get("tags", []),
                    )
                    experiences.append(exp)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid experience item from LLM: {item}. Error: {e}")

            logger.info(f"Successfully synthesized {len(experiences)} new experiences.")
            return experiences
        except (StructuredResponseError, KeyError, ValueError) as e:
            response = e.response if isinstance(e, StructuredResponseError) else None
            if response is not None:
                self.last_usage = response.usage
                self.last_model_name = response.model_name
                self.last_estimated_cost_usd = response.estimated_cost_usd
            logger.error(f"Failed to parse valid experiences from LLM response. Error: {e}")
            if response is not None:
                logger.debug(f"Invalid JSON response from LLM: {response.content}")
            return []

    def _is_safety_critical(self, item: dict, proposed_type: ExperienceType, proposed_layer: MemoryLayer) -> bool:
        if bool(item.get("safety_critical", False)):
            return True
        if proposed_layer != MemoryLayer.FAILURE:
            return False
        if proposed_type == ExperienceType.VERIFIER_RULE:
            return True
        content = str(item.get("content", "")).lower()
        critical_tokens = [
            "leakage",
            "temporal",
            "patient-level split",
            "privacy",
            "identifier",
            "cohort mismatch",
            "unsafe",
            "safety",
        ]
        return any(token in content for token in critical_tokens)
