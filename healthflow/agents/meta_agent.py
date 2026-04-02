from typing import List, Optional
from loguru import logger

from ..core.contracts import ExecutionPlan
from ..core.llm_provider import LLMProvider, LLMMessage, StructuredResponseError, parse_json_content
from ..prompts.templates import get_prompt, render_prompt
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
        safeguard_experiences: List[Experience],
        workflow_experiences: List[Experience],
        dataset_experiences: List[Experience],
        execution_experiences: List[Experience],
        execution_environment: List[str],
        available_project_cli_tools: List[str],
        workflow_recommendations: List[str],
        previous_feedback: Optional[str] = None,
    ) -> ExecutionPlan:
        """
        Analyze the user request and generate a structured execution plan.
        """
        system_prompt = get_prompt("meta_agent_system")
        user_prompt = self._build_user_prompt(
            user_request=user_request,
            safeguard_experiences=safeguard_experiences,
            workflow_experiences=workflow_experiences,
            dataset_experiences=dataset_experiences,
            execution_experiences=execution_experiences,
            execution_environment=execution_environment,
            available_project_cli_tools=available_project_cli_tools,
            workflow_recommendations=workflow_recommendations,
            previous_feedback=previous_feedback,
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        logger.info("Generating plan with MetaAgent...")
        try:
            plan, response = await self.llm_provider.generate_structured(
                messages,
                lambda content: ExecutionPlan(**parse_json_content(content)),
                temperature=0.0,
            )
            self.last_usage = response.usage
            self.last_model_name = response.model_name
            self.last_estimated_cost_usd = response.estimated_cost_usd
            logger.info("Plan generated successfully.")
            return plan
        except (StructuredResponseError, ValueError) as e:
            response = e.response if isinstance(e, StructuredResponseError) else None
            if response is not None:
                self.last_usage = response.usage
                self.last_model_name = response.model_name
                self.last_estimated_cost_usd = response.estimated_cost_usd
            logger.error(f"Failed to parse valid plan from LLM: {e}. Defaulting to a fallback plan.")
            if response is not None:
                logger.debug(f"Invalid JSON response from LLM: {response.content}")
            return ExecutionPlan(
                objective=user_request,
                assumptions_to_check=["Inspect the workspace inputs before implementing a solution."],
                recommended_steps=[
                    "Inspect the workspace and available inputs.",
                    "Choose the most direct reproducible implementation path.",
                    "Produce the requested artifacts and a concise final answer.",
                ],
                recommended_workflows=workflow_recommendations[:3],
                avoidances=["Do not ignore relevant safeguards or previous feedback."],
                success_signals=["The requested result is present in the workspace and summarized in the final answer."],
            )

    def _build_user_prompt(
        self,
        *,
        user_request: str,
        safeguard_experiences: List[Experience],
        workflow_experiences: List[Experience],
        dataset_experiences: List[Experience],
        execution_experiences: List[Experience],
        execution_environment: List[str],
        available_project_cli_tools: List[str],
        workflow_recommendations: List[str],
        previous_feedback: Optional[str],
    ) -> str:
        sections = [
            self._render_section("User request", user_request.strip()),
            self._render_section(
                "Execution environment",
                self._render_bullet_list(execution_environment) or "- Use the default executor environment.",
            ),
            self._render_section("Project CLI tools", self._render_bullet_list(available_project_cli_tools)),
            self._render_section("Workflow recommendations", self._render_bullet_list(workflow_recommendations)),
            self._render_section("EHR safeguards", self._render_memory_block(safeguard_experiences, prefix="Guardrail")),
            self._render_section("Workflow memories", self._render_memory_block(workflow_experiences, prefix="Workflow")),
            self._render_section("Dataset memories", self._render_memory_block(dataset_experiences, prefix="Dataset")),
            self._render_section("Execution memories", self._render_memory_block(execution_experiences, prefix="Execution")),
            self._render_section("Feedback from Previous Failed Attempt", previous_feedback or ""),
        ]
        return render_prompt("meta_agent_user", context_blocks="\n\n".join(section for section in sections if section)).strip()

    def _render_memory_block(self, experiences: List[Experience], prefix: str) -> str:
        if not experiences:
            return ""
        return "\n".join(
            (
                f"- **{prefix}** [{exp.kind.value}/{exp.source_outcome.value}] {exp.category}\n"
                f"  - {exp.content}"
            )
            for exp in experiences
        )

    def _render_bullet_list(self, items: List[str]) -> str:
        cleaned_items = [item.strip() for item in items if item.strip()]
        if not cleaned_items:
            return ""
        return "\n".join(f"- {item}" for item in cleaned_items)

    def _render_section(self, title: str, body: str) -> str:
        body = body.strip()
        if not body:
            return ""
        return f"{title}:\n---\n{body}\n---"
