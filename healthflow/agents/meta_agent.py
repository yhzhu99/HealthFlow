import json
from typing import List, Optional
from loguru import logger

from ..core.llm_provider import LLMProvider, LLMMessage
from ..prompts.templates import get_prompt
from ..experience.experience_models import Experience

class MetaAgent:
    """
    This agent takes a high-level user request, synthesizes context from past
    experiences, and generates a detailed markdown plan for execution.
    It handles all tasks through this unified planning process.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.last_usage: dict = {}
        self.last_model_name: str = llm_provider.model_name
        self.last_estimated_cost_usd: float | None = None

    async def generate_plan(
        self,
        user_request: str,
        experiences: List[Experience],
        task_family: str,
        data_profile: str,
        risk_checks: List[str],
        tool_bundle: List[str],
        output_contract: List[str],
        previous_feedback: Optional[str] = None
    ) -> str:
        """
        Analyzes the user request and generates a detailed markdown plan.
        This is the universal planning step for all tasks.

        Returns:
            A string containing the markdown plan.
        """
        system_prompt = get_prompt("meta_agent_system")

        experience_str = "No relevant past experiences were found."
        if experiences:
            experience_str = "\n".join(
                (
                    f"- **{'Avoid' if exp.layer.value == 'failure' else 'Use'}** "
                    f"[{exp.layer.value}/{exp.validation_status.value}] "
                    f"{exp.type.value} | {exp.category}\n"
                    f"  - {exp.content}"
                )
                for exp in experiences
            )

        feedback_str = ""
        if previous_feedback:
            feedback_str = f"**Feedback from Previous Failed Attempt (Must be addressed):**\n{previous_feedback}"

        user_prompt = get_prompt("meta_agent_user").format(
            user_request=user_request,
            experiences=experience_str,
            task_family=task_family,
            data_profile=data_profile,
            risk_checks="\n".join(f"- {item}" for item in risk_checks) or "- None",
            tool_bundle="\n".join(f"- {item}" for item in tool_bundle) or "- Minimum required tooling only",
            output_contract="\n".join(f"- {item}" for item in output_contract) or "- Provide a final answer in stdout",
            feedback=feedback_str
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        logger.info("Generating plan with MetaAgent...")
        response = await self.llm_provider.generate(messages, temperature=0.0, json_mode=True)
        self.last_usage = response.usage
        self.last_model_name = response.model_name
        self.last_estimated_cost_usd = response.estimated_cost_usd

        try:
            result = json.loads(response.content)
            plan = result.get("plan")
            if not plan or not isinstance(plan, str):
                raise ValueError("Response JSON is missing or has an invalid 'plan' field.")

            logger.info("Plan generated successfully.")
            return plan.strip()
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse valid plan from LLM: {e}. Defaulting to a fallback plan.")
            logger.debug(f"Invalid JSON response from LLM: {response.content}")
            # Fallback to create a basic plan
            return f"# Fallback Plan\n\n## Step 1: Attempt to fulfill user request directly.\n\n- The system failed to generate a detailed plan. The executor should now attempt to address the following request based on its own capabilities:\n\n`{user_request}`"
