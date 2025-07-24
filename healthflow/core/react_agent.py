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
                # Set temperature to 0.0 for more deterministic and structured output
                response = await self.llm_provider.generate(messages=messages, temperature=0.0, max_tokens=2048)
                content = response.content.strip()
                messages.append(LLMMessage(role="assistant", content=content))

                # 1. Check for explicit FINAL_ANSWER keyword.
                final_answer_match = re.search(r"FINAL_ANSWER:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                    logger.info("ReAct agent finished by providing an explicit FINAL_ANSWER.")
                    return final_answer, messages

                # 2. Check for an action to execute.
                action_match = re.search(r"Action:\s*([^\n]+)", content, re.IGNORECASE)
                input_match = re.search(r"Action Input:\s*(.*)", content, re.IGNORECASE | re.DOTALL)

                if action_match and input_match:
                    action_name = action_match.group(1).strip()
                    action_input = input_match.group(1).strip()

                    if not action_name:
                        observation = "Observation: Error - Action name was empty. Please specify a tool in the 'Action:' line and try again."
                        messages.append(LLMMessage(role="user", content=observation))
                        continue

                    logger.info(f"Executing action: {action_name}")
                    action_result = await self._execute_action(action_name, action_input)

                    observation = f"Observation: {action_result}"
                    messages.append(LLMMessage(role="user", content=observation))
                    continue

                # 3. If no FINAL_ANSWER and no valid Action, the format is wrong. Prompt for correction.
                logger.warning("ReAct response did not contain FINAL_ANSWER or a valid Action. Prompting for correction.")
                correction_prompt = (
                    "Observation: Your response was not formatted correctly. "
                    "You must either use the 'Action:' and 'Action Input:' format to call a tool, "
                    "or provide a final answer using 'FINAL_ANSWER:'. "
                    "Please respond in the correct format."
                )
                messages.append(LLMMessage(role="user", content=correction_prompt))

            except Exception as e:
                logger.error(f"Error in ReAct round {round_num}: {e}", exc_info=True)
                error_message = f"An error occurred in the ReAct loop: {e}. You can try to recover or abort."
                messages.append(LLMMessage(role="user", content=f"Observation: {error_message}"))

        # This part is reached if max rounds are exceeded.
        logger.warning("Max rounds reached without a definitive answer.")
        last_thought = "Agent failed to produce an answer within the round limit."
        # Find the last message from the assistant
        for msg in reversed(messages):
            if msg.role == 'assistant':
                last_thought = msg.content
                break

        final_message = f"Error: Agent reached maximum iterations ({self.max_rounds}) without providing a FINAL_ANSWER. The task could not be completed. Last thought: {last_thought}"
        return final_message, messages

    async def _execute_action(self, action_name: str, action_input: str) -> str:
        """Executes a tool and returns the result as a string."""
        logger.debug(f"Attempting to execute tool '{action_name}' with input: '{action_input[:200]}'")
        try:
            result = await self.tools_manager.execute_tool(action_name, action_input)
            return str(result) if result is not None else "Tool executed successfully with no return value."
        except Exception as e:
            logger.error(f"Error executing action '{action_name}': {e}", exc_info=True)
            return f"Error: Failed to execute tool '{action_name}'. Reason: {e}"