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

    async def evaluate(self, user_request: str, task_list: str, execution_log: str) -> dict:
        """
        Evaluates the task's execution and returns a structured dictionary.

        Args:
            user_request: The initial high-level request from the user.
            task_list: The markdown plan that was executed.
            execution_log: The captured stdout/stderr from the execution.

        Returns:
            A dictionary containing the score, feedback, and reasoning.
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

        logger.info("Requesting evaluation from LLM...")
        response = await self.llm_provider.generate(messages, json_mode=True)

        try:
            eval_data = json.loads(response.content)
            # Basic validation of the received JSON structure
            if "score" in eval_data and "feedback" in eval_data and "reasoning" in eval_data:
                logger.info(f"Evaluation received successfully. Score: {eval_data['score']}")
                return eval_data
            else:
                raise ValueError("Evaluation JSON is missing required keys: 'score', 'feedback', 'reasoning'.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse valid evaluation from LLM. Error: {e}")
            logger.debug(f"Invalid JSON response from LLM: {response.content}")
            # Fallback in case of parsing failure to prevent system crash
            return {
                "score": 1.0,
                "feedback": "Evaluation Agent failed. The LLM's evaluation response was malformed or empty. This may indicate a severe failure in the execution log that confused the evaluator.",
                "reasoning": "Fallback due to a JSON parsing error."
            }