import json
from typing import List, Optional
from loguru import logger

from ..core.llm_provider import LLMProvider, LLMMessage
from ..prompts.templates import get_prompt
from ..experience.experience_models import Experience

class TaskDecomposerAgent:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def generate_task_list(
        self,
        user_request: str,
        experiences: List[Experience],
        previous_feedback: Optional[str] = None
    ) -> str:
        """
        Generates a detailed, step-by-step markdown task list for Claude Code.
        """
        system_prompt = get_prompt("task_decomposer_system")

        experience_str = "No relevant past experiences found."
        if experiences:
            experience_str = "## Relevant Best Practices from Past Tasks:\n\n"
            for exp in experiences:
                experience_str += f"- **Category**: {exp.category}\n"
                experience_str += f"  - **Type**: {exp.type}\n"
                experience_str += f"  - **Content**: {exp.content}\n\n"

        feedback_str = ""
        if previous_feedback:
            feedback_str = f"## Feedback from Previous Failed Attempt (Address this!):\n{previous_feedback}"

        user_prompt = get_prompt("task_decomposer_user").format(
            user_request=user_request,
            experiences=experience_str,
            feedback=feedback_str
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        logger.info("Generating task list with TaskDecomposerAgent...")
        response = await self.llm_provider.generate(messages, temperature=0.1)

        # Extract markdown block
        import re
        match = re.search(r"```markdown\n(.*)\n```", response.content, re.DOTALL)
        if match:
            return match.group(1).strip()

        logger.warning("Decomposer did not return a markdown block. Returning raw content.")
        return response.content