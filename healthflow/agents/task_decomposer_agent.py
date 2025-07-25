import json
import re
from typing import List, Optional
from loguru import logger

from ..core.llm_provider import LLMProvider, LLMMessage
from ..prompts.templates import get_prompt
from ..experience.experience_models import Experience

class TaskDecomposerAgent:
    """
    This agent takes a high-level user request, triages it, and either breaks it down
    into a detailed markdown plan (for code execution) or provides a direct answer
    (for simple QA). It incorporates past experiences and feedback.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def triage_task(
        self,
        user_request: str,
        experiences: List[Experience],
        previous_feedback: Optional[str] = None
    ) -> dict:
        """
        Analyzes the user request and either generates a direct answer for simple QA
        or a detailed markdown plan for tasks requiring code execution.

        Returns:
            A dictionary containing the triage result, e.g.,
            {'task_type': 'simple_qa', 'answer': '...'} or
            {'task_type': 'code_execution', 'plan': '...'}
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

        logger.info("Triaging task with TaskDecomposerAgent...")
        # Use a lower temperature for more deterministic plans and JSON mode for structured output
        response = await self.llm_provider.generate(messages, temperature=0.0, json_mode=True)

        try:
            triage_result = json.loads(response.content)
            task_type = triage_result.get("task_type")
            if task_type not in ["simple_qa", "code_execution"]:
                raise ValueError("Response missing or has invalid 'task_type'.")

            if task_type == "simple_qa" and "answer" not in triage_result:
                raise ValueError("Task is 'simple_qa' but is missing 'answer'.")

            if task_type == "code_execution":
                plan = triage_result.get("plan", "")
                if not plan:
                    raise ValueError("Task is 'code_execution' but is missing 'plan'.")
                # The prompt asks for a markdown block. This regex extracts it robustly.
                match = re.search(r"```markdown\n(.*?)\n```", plan, re.DOTALL)
                if match:
                    triage_result["plan"] = match.group(1).strip()
                else: # Handle cases where LLM forgets the markdown block
                    logger.warning("Decomposer did not return plan in a markdown block as requested. Using raw plan content.")
                    triage_result["plan"] = plan.strip()

            logger.info(f"Task triaged as: {triage_result['task_type']}")
            return triage_result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse valid triage from LLM: {e}. Defaulting to code_execution.")
            logger.debug(f"Invalid JSON response from LLM: {response.content}")
            # Fallback to treat as a coding task with a basic plan
            return {
                "task_type": "code_execution",
                "plan": f"# Fallback Plan\n\n# Step 1: Attempt to fulfill user request directly.\n\n{user_request}"
            }