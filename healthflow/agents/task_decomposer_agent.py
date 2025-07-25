import re
from typing import List, Optional
from loguru import logger

from ..core.llm_provider import LLMProvider, LLMMessage
from ..prompts.templates import get_prompt
from ..experience.experience_models import Experience

class TaskDecomposerAgent:
    """
    This agent takes a high-level user request and breaks it down into a detailed,
    step-by-step markdown plan suitable for execution by an agentic coder.
    It incorporates past experiences and feedback to improve its plans.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def generate_task_list(
        self,
        user_request: str,
        experiences: List[Experience],
        previous_feedback: Optional[str] = None
    ) -> str:
        """
        Generates a detailed, step-by-step markdown task list.

        Args:
            user_request: The high-level user request.
            experiences: A list of relevant experiences from past tasks.
            previous_feedback: Actionable feedback from a previous failed attempt, if any.

        Returns:
            A string containing the markdown plan.
        """
        system_prompt = get_prompt("task_decomposer_system")

        experience_str = "No relevant past experiences were found."
        if experiences:
            experience_str = "## Relevant Best Practices & Warnings from Past Tasks:\n\n"
            for exp in experiences:
                experience_str += f"- **Category**: {exp.category}\n"
                experience_str += f"  - **Type**: {exp.type.value}\n"
                experience_str += f"  - **Guideline**: {exp.content}\n\n"

        feedback_str = ""
        if previous_feedback:
            feedback_str = f"## CRITICAL FEEDBACK FROM PREVIOUS FAILED ATTEMPT (You MUST address this):\n{previous_feedback}"

        user_prompt = get_prompt("task_decomposer_user").format(
            user_request=user_request,
            experiences=experience_str,
            feedback=feedback_str
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        logger.info("Generating task list with TaskDecomposerAgent, incorporating experiences and feedback...")
        # Use a lower temperature for more deterministic and structured plans
        response = await self.llm_provider.generate(messages, temperature=0.0)

        # The prompt asks for a markdown block. This regex extracts it robustly.
        match = re.search(r"```markdown\n(.*?)\n```", response.content, re.DOTALL)
        if match:
            extracted_plan = match.group(1).strip()
            logger.info("Successfully extracted markdown plan from LLM response.")
            return extracted_plan
        else:
            logger.warning("Decomposer did not return content in a markdown block as requested. Returning raw content.")
            return response.content