import json
from loguru import logger

from ..core.llm_provider import LLMProvider, LLMMessage
from ..prompts.templates import get_prompt

class EvaluatorAgent:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def evaluate(self, user_request: str, task_list: str, execution_log: str) -> dict:
        """
        Evaluates the execution of a task and returns a structured score and feedback.
        """
        system_prompt = get_prompt("evaluator_system")
        user_prompt = get_prompt("evaluator_user").format(
            user_request=user_request,
            task_list=task_list,
            execution_log=execution_log
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        logger.info("Evaluating task execution with EvaluatorAgent...")
        response = await self.llm_provider.generate(messages, json_mode=True)

        try:
            eval_data = json.loads(response.content)
            # Basic validation
            if "score" in eval_data and "feedback" in eval_data:
                return eval_data
            else:
                raise ValueError("Evaluation JSON is missing required keys 'score' or 'feedback'.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse valid evaluation from LLM response: {e}")
            logger.debug(f"Invalid JSON response: {response.content}")
            return {
                "score": 1.0,
                "feedback": "Evaluation failed. The system was unable to parse the LLM's evaluation response. This often indicates a critical failure in the execution log.",
                "reasoning": "Fallback due to parsing error."
            }