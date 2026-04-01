import re
from loguru import logger

from ..core.contracts import EvaluationVerdict, ExecutionPlan
from ..core.llm_provider import LLMProvider, LLMMessage, StructuredResponseError, parse_json_content
from ..prompts.templates import get_prompt, render_prompt

_ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


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
        sanitized_execution_log = self._sanitize_execution_log(execution_log)
        user_prompt = render_prompt(
            "evaluator_user",
            user_request=user_request,
            plan_markdown=plan.to_markdown(),
            execution_log=sanitized_execution_log,
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
        try:
            verdict, response = await self.llm_provider.generate_structured(
                messages,
                lambda content: EvaluationVerdict(
                    **parse_json_content(content),
                ),
                temperature=0.0,
            )
            self.last_usage = response.usage
            self.last_model_name = response.model_name
            self.last_estimated_cost_usd = response.estimated_cost_usd
            verdict.usage = response.usage
            verdict.judge_model = response.model_name
            verdict.estimated_cost_usd = response.estimated_cost_usd
            logger.info(f"Evaluation received successfully. Score: {verdict.score}")
            return verdict
        except (StructuredResponseError, ValueError) as e:
            response = e.response if isinstance(e, StructuredResponseError) else None
            if response is not None:
                self.last_usage = response.usage
                self.last_model_name = response.model_name
                self.last_estimated_cost_usd = response.estimated_cost_usd
            logger.error(f"Failed to parse valid evaluation from LLM. Error: {e}")
            if response is not None:
                logger.debug(f"Invalid JSON response from LLM: {response.content}")
            return EvaluationVerdict(
                status="failed",
                score=0.0,
                failure_type="evaluator_malformed_response",
                feedback="Evaluation Agent failed. The LLM evaluation response was malformed or empty.",
                repair_instructions=["Retry with a simpler execution path and ensure the final answer is explicit."],
                retry_recommended=True,
                memory_worthy_insights=["Malformed evaluator output should not be treated as a task success."],
                reasoning="Fallback due to a JSON parsing error.",
                usage=response.usage if response is not None else {},
                judge_model=response.model_name if response is not None else None,
                estimated_cost_usd=response.estimated_cost_usd if response is not None else None,
            )

    def _sanitize_execution_log(self, execution_log: str, max_chars: int = 12000) -> str:
        normalized = _ANSI_ESCAPE_RE.sub("", execution_log).strip() or "No execution log was captured."
        if len(normalized) <= max_chars:
            return normalized

        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        omitted = len(normalized) - max_chars
        return (
            f"{normalized[:head_chars].rstrip()}\n\n"
            f"... [truncated {omitted} characters from the middle of the execution log] ...\n\n"
            f"{normalized[-tail_chars:].lstrip()}"
        )
