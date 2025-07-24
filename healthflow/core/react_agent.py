"""
ReAct (Reasoning and Acting) Agent implementation using OpenAI's native format.
"""
import logging
import re
from typing import List, Optional

from .llm_provider import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

class ReactAgent:
    """ReAct agent that uses reasoning and acting in an iterative loop."""

    def __init__(self, llm_provider: LLMProvider, tools_manager, max_rounds: int = 8):
        self.llm_provider = llm_provider
        self.tools_manager = tools_manager
        self.max_rounds = max_rounds

    async def run(self, task: str, system_prompt: str) -> tuple[str, List[LLMMessage]]:
        """Run the ReAct loop and return the final answer and the conversation history."""
        logger.info(f"ReAct Agent starting task: {task[:100]}...")

        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"Task: {task}")
        ]

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"ReAct Round {round_num}/{self.max_rounds}")

            try:
                response = await self.llm_provider.generate(messages=messages, temperature=0.1, max_tokens=2048)
                content = response.content.strip()
                messages.append(LLMMessage(role="assistant", content=content))

                # Case-insensitive check for final answer
                if "final_answer:" in content.lower():
                    final_answer = re.split(r"final_answer:", content, flags=re.IGNORECASE)[1].strip()
                    logger.info("ReAct agent finished by providing a final answer.")
                    return final_answer, messages

                # Case-insensitive and robust action parsing
                action_match = re.search(r"Action:\s*(.*)", content, re.IGNORECASE)
                input_match = re.search(r"Action Input:\s*(.*)", content, re.IGNORECASE | re.DOTALL)

                if action_match and input_match:
                    action_name = action_match.group(1).strip()
                    action_input = input_match.group(1).strip()

                    if not action_name:
                        logger.warning("Action name is empty. Continuing to next round for clarification.")
                        continue

                    logger.info(f"Executing action: {action_name}")
                    action_result = await self._execute_action(action_name, action_input)

                    observation = f"Observation: {action_result}"
                    # Use 'user' role for observation as per original simple design, to avoid model API errors
                    messages.append(LLMMessage(role="user", content=observation))
                    continue

                # If no action and no final answer, but it's the last round, treat the last response as the answer.
                if round_num == self.max_rounds:
                    logger.warning("Max rounds reached. No FINAL_ANSWER found. Returning last content.")
                    return content, messages

            except Exception as e:
                logger.error(f"Error in ReAct round {round_num}: {e}", exc_info=True)
                error_message = f"An error occurred in the ReAct loop: {e}"
                messages.append(LLMMessage(role="assistant", content=error_message))
                return error_message, messages

        final_thought = "I couldn't complete the task within the maximum number of reasoning steps."
        messages.append(LLMMessage(role="assistant", content=final_thought))
        return final_thought, messages

    async def _execute_action(self, action_name: str, action_input: str) -> str:
        """Executes a tool and returns the result as a string."""
        logger.debug(f"Attempting to execute tool '{action_name}' with input: '{action_input[:200]}'")
        try:
            result = await self.tools_manager.execute_tool(action_name, action_input)
            return str(result) if result is not None else "Tool executed successfully with no return value."
        except Exception as e:
            logger.error(f"Error executing action '{action_name}': {e}", exc_info=True)
            return f"Error: Failed to execute tool '{action_name}'. Reason: {e}"