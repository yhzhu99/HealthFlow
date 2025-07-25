import json
from typing import List, Dict, Any
from loguru import logger

from ..core.llm_provider import LLMProvider, LLMMessage
from ..prompts.templates import get_prompt
from ..experience.experience_models import Experience, ExperienceType

class ReflectorAgent:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def synthesize_experience(self, full_history: Dict[str, Any]) -> List[Experience]:
        """
        Analyzes a successful task run and synthesizes generalizable experiences.
        """
        system_prompt = get_prompt("reflector_system")

        # Prepare a condensed history for the prompt
        history_str = json.dumps({
            "user_request": full_history["user_request"],
            "final_plan": full_history["attempts"][-1]["task_list"],
            "final_log": full_history["attempts"][-1]["execution"]["log"][:5000] # Truncate log
        }, indent=2)

        user_prompt = get_prompt("reflector_user").format(task_history=history_str)

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        logger.info("Synthesizing experiences with ReflectorAgent...")
        response = await self.llm_provider.generate(messages, json_mode=True)

        experiences = []
        try:
            data = json.loads(response.content)
            raw_experiences = data.get("experiences", [])
            for item in raw_experiences:
                try:
                    exp = Experience(
                        type=ExperienceType(item["type"]),
                        category=item["category"],
                        content=item["content"],
                        source_task_id=full_history["task_id"]
                    )
                    experiences.append(exp)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid experience item: {item}. Error: {e}")
            return experiences
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse valid experiences from LLM response: {e}")
            logger.debug(f"Invalid JSON response: {response.content}")
            return []