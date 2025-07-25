import json
from typing import List, Dict, Any
from loguru import logger

from ..core.llm_provider import LLMProvider, LLMMessage
from ..prompts.templates import get_prompt
from ..experience.experience_models import Experience, ExperienceType

class ReflectorAgent:
    """
    This agent analyzes a successful task execution to synthesize generalizable knowledge.
    It distills heuristics, code snippets, and workflow patterns into structured Experience objects.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def synthesize_experience(self, full_history: Dict[str, Any]) -> List[Experience]:
        """
        Analyzes a successful task run and synthesizes reusable experiences.

        Args:
            full_history: A dictionary containing the complete history of the task execution.

        Returns:
            A list of Experience objects to be saved to the knowledge base.
        """
        system_prompt = get_prompt("reflector_system")

        # Prepare a condensed version of the history for the prompt to save tokens
        # We take the final, successful attempt for reflection.
        successful_attempt = full_history["attempts"][-1]
        history_for_prompt = {
            "user_request": full_history["user_request"],
            "final_plan": successful_attempt["task_list"],
            "final_log": successful_attempt["execution"]["log"][:8000] # Truncate log to avoid excessive length
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
                    # Validate and create Experience objects using Pydantic
                    exp = Experience(
                        type=ExperienceType(item["type"]),
                        category=item["category"],
                        content=item["content"],
                        source_task_id=full_history["task_id"]
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