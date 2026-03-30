import json
from typing import List, Dict, Any
from loguru import logger

from ..core.llm_provider import LLMProvider, LLMMessage
from ..prompts.templates import get_prompt
from ..experience.experience_models import Experience, ExperienceType, MemoryLayer, ValidationStatus

class ReflectorAgent:
    """
    This agent analyzes a successful task execution to synthesize generalizable knowledge,
    which is stored as structured `Experience` objects.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def synthesize_experience(self, full_history: Dict[str, Any], verified: bool) -> List[Experience]:
        """
        Analyzes a successful execution history and synthesizes reusable experiences.
        This method is used for all successfully completed tasks.
        """
        system_prompt = get_prompt("reflector_system")

        final_attempt = full_history["attempts"][-1]
        history_for_prompt = {
            "user_request": full_history["user_request"],
            "retrieved_experiences": [exp["content"] for exp in full_history.get("retrieved_experiences", [])],
            "task_family": full_history.get("task_family", "general"),
            "dataset_signature": full_history.get("dataset_signature", "unknown"),
            "backend": full_history.get("backend", "unknown"),
            "verification_passed": verified,
            "final_plan": final_attempt["task_list"],
            "final_log": final_attempt["execution"]["log"][:8000],
            "verification_summary": final_attempt.get("verification", {}),
            "evaluation_score": final_attempt["evaluation"]["score"],
            "evaluation_reasoning": final_attempt["evaluation"]["reasoning"],
        }
        history_str = json.dumps(history_for_prompt, indent=2)

        user_prompt = get_prompt("reflector_user").format(task_history=history_str)

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        logger.info("Requesting experience synthesis from LLM...")
        response = await self.llm_provider.generate(messages, json_mode=True)

        experiences: List[Experience] = []
        try:
            data = json.loads(response.content)
            raw_experiences = data.get("experiences", [])
            for item in raw_experiences:
                try:
                    exp = Experience(
                        type=ExperienceType(item["type"]),
                        layer=MemoryLayer(item.get("layer", "strategy")),
                        category=item["category"],
                        content=item["content"],
                        source_task_id=full_history["task_id"],
                        task_family=full_history.get("task_family", "general"),
                        dataset_signature=full_history.get("dataset_signature", "unknown"),
                        stage="reflection",
                        backend=full_history.get("backend", "unknown"),
                        validation_status=ValidationStatus.VERIFIED if verified else ValidationStatus.FAILED,
                        confidence=float(item.get("confidence", 0.6)),
                        conflict_group=item.get("conflict_group"),
                        tags=item.get("tags", []),
                    )
                    experiences.append(exp)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid experience item from LLM: {item}. Error: {e}")

            logger.info(f"Successfully synthesized {len(experiences)} new experiences.")
            return experiences
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse valid experiences from LLM response. Error: {e}")
            logger.debug(f"Invalid JSON response from LLM: {response.content}")
            return []
