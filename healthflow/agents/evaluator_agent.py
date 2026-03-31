import json
from loguru import logger

from ..core.llm_provider import LLMProvider, LLMMessage
from ..prompts.templates import get_prompt

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
        task_list: str,
        execution_log: str,
        verification_summary: str,
        task_family: str = "general_analysis",
        domain_focus: str = "general",
    ) -> dict:
        """
        Evaluates the task's code execution and returns a structured dictionary.
        """
        system_prompt = get_prompt("evaluator_system")
        user_prompt = get_prompt("evaluator_user").format(
            user_request=user_request,
            task_list=task_list,
            execution_log=execution_log,
            verification_summary=verification_summary,
            task_family=task_family,
            domain_focus=domain_focus,
        )
        logger.info("Requesting code execution evaluation from LLM...")
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]
        return await self._get_evaluation(messages)

    async def _get_evaluation(self, messages: list[LLMMessage]) -> dict:
        """Helper to call LLM and parse evaluation JSON."""
        response = await self.llm_provider.generate(messages, json_mode=True)
        self.last_usage = response.usage
        self.last_model_name = response.model_name
        self.last_estimated_cost_usd = response.estimated_cost_usd
        try:
            eval_data = json.loads(response.content)
            # Basic validation
            if "score" in eval_data and "feedback" in eval_data and "reasoning" in eval_data:
                eval_data["usage"] = response.usage
                eval_data["judge_model"] = response.model_name
                eval_data["estimated_cost_usd"] = response.estimated_cost_usd
                logger.info(f"Evaluation received successfully. Score: {eval_data['score']}")
                return eval_data
            else:
                raise ValueError("Evaluation JSON is missing required keys: 'score', 'feedback', 'reasoning'.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse valid evaluation from LLM. Error: {e}")
            logger.debug(f"Invalid JSON response from LLM: {response.content}")
            # Fallback to prevent system crash
            return {
                "score": 1.0,
                "feedback": "Evaluation Agent failed. The LLM's evaluation response was malformed or empty.",
                "reasoning": "Fallback due to a JSON parsing error.",
                "usage": response.usage,
                "judge_model": response.model_name,
                "estimated_cost_usd": response.estimated_cost_usd,
            }
