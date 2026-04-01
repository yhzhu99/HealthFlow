import json
from loguru import logger

from ..core.contracts import EvaluationVerdict, ExecutionPlan
from ..core.llm_provider import LLMProvider, LLMMessage
from ..prompts.templates import get_prompt, render_prompt


class EvaluatorAgent:
    """
    This agent evaluates the execution of a task based on the original request,
    the plan, and the execution log. It provides a structured score and feedback.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.last_usage: dict = {}
        self.last_model_name: str = llm_provider.model_name
        self.last_estimated_cost_usd: float | None = None

    async def evaluate(
        self,
        user_request: str,
        plan: ExecutionPlan,
        execution_log: str,
        workspace_artifacts: list[str],
        generated_answer: str,
    ) -> EvaluationVerdict:
        """
        Evaluate an execution attempt and return a structured verdict.
        """
        system_prompt = get_prompt("evaluator_system")
        user_prompt = render_prompt(
            "evaluator_user",
            user_request=user_request,
            plan_markdown=plan.to_markdown(),
            execution_log=execution_log,
            workspace_artifacts="\n".join(f"- {item}" for item in workspace_artifacts) or "- No workspace artifacts found.",
            generated_answer=generated_answer or "No final answer was extracted.",
        )
        logger.info("Requesting code execution evaluation from LLM...")
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]
        return await self._get_evaluation(messages)

    async def _get_evaluation(self, messages: list[LLMMessage]) -> EvaluationVerdict:
        """Helper to call LLM and parse evaluation JSON."""
        response = await self.llm_provider.generate(messages, json_mode=True)
        self.last_usage = response.usage
        self.last_model_name = response.model_name
        self.last_estimated_cost_usd = response.estimated_cost_usd
        try:
            eval_data = json.loads(response.content)
            verdict = EvaluationVerdict(
                **eval_data,
                usage=response.usage,
                judge_model=response.model_name,
                estimated_cost_usd=response.estimated_cost_usd,
            )
            logger.info(f"Evaluation received successfully. Score: {verdict.score}")
            return verdict
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse valid evaluation from LLM. Error: {e}")
            logger.debug(f"Invalid JSON response from LLM: {response.content}")
            return EvaluationVerdict(
                status="failed",
                score=1.0,
                failure_type="evaluator_malformed_response",
                feedback="Evaluation Agent failed. The LLM evaluation response was malformed or empty.",
                repair_instructions=["Retry with a simpler execution path and ensure the final answer is explicit."],
                retry_recommended=True,
                memory_worthy_insights=["Malformed evaluator output should not be treated as a task success."],
                reasoning="Fallback due to a JSON parsing error.",
                usage=response.usage,
                judge_model=response.model_name,
                estimated_cost_usd=response.estimated_cost_usd,
            )
