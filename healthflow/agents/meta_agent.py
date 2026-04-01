import json
from typing import List, Optional
from loguru import logger

from ..core.contracts import ExecutionPlan
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
        recommended_experiences: List[Experience],
        avoidance_experiences: List[Experience],
        available_tools: List[str],
        previous_feedback: Optional[str] = None,
    ) -> ExecutionPlan:
        """
        Analyze the user request and generate a structured execution plan.
        """
        system_prompt = get_prompt("meta_agent_system")

        recommended_str = "No recommended prior experience was retrieved."
        if recommended_experiences:
            recommended_str = "\n".join(
                (
                    f"- **Use** "
                    f"[{exp.layer.value}/{exp.validation_status.value}] "
                    f"{exp.type.value} | {exp.category}\n"
                    f"  - {exp.content}"
                )
                for exp in recommended_experiences
            )

        avoidance_str = "No avoidance memory was retrieved."
        if avoidance_experiences:
            avoidance_str = "\n".join(
                (
                    f"- **Avoid** "
                    f"[{exp.layer.value}/{exp.validation_status.value}] "
                    f"{exp.type.value} | {exp.category}\n"
                    f"  - {exp.content}"
                )
                for exp in avoidance_experiences
            )

        feedback_str = ""
        if previous_feedback:
            feedback_str = f"**Feedback from Previous Failed Attempt (Must be addressed):**\n{previous_feedback}"

        user_prompt = get_prompt("meta_agent_user").format(
            user_request=user_request,
            recommended_experiences=recommended_str,
            avoidance_experiences=avoidance_str,
            available_tools="\n".join(f"- {item}" for item in available_tools) or "- Minimum required tooling only",
            feedback=feedback_str,
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
            plan = ExecutionPlan(**result)

            logger.info("Plan generated successfully.")
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse valid plan from LLM: {e}. Defaulting to a fallback plan.")
            logger.debug(f"Invalid JSON response from LLM: {response.content}")
            return ExecutionPlan(
                objective=user_request,
                assumptions_to_check=["Inspect the workspace inputs before implementing a solution."],
                recommended_steps=[
                    "Inspect the workspace and available inputs.",
                    "Choose the most direct reproducible implementation path.",
                    "Produce the requested artifacts and a concise final answer.",
                ],
                preferred_tools=available_tools[:3],
                avoidances=["Do not ignore relevant avoidance memory or previous feedback."],
                success_signals=["The requested result is present in the workspace and summarized in the final answer."],
                executor_brief="The planner fallback triggered. Execute conservatively and keep the attempt auditable.",
            )
