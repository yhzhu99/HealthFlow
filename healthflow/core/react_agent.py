"""
ReAct (Reasoning and Acting) Agent implementation using OpenAI's native format.
"""
import logging
from typing import List, Dict, Any, Optional
import json
import asyncio

from .llm_provider import LLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

class ReactAgent:
    """ReAct agent that uses reasoning and acting in an iterative loop."""

    def __init__(self, llm_provider: LLMProvider, tools_manager, max_rounds: int = 8):
        self.llm_provider = llm_provider
        self.tools_manager = tools_manager
        self.max_rounds = max_rounds

    async def run(self, task: str, system_prompt: str) -> str:
        """Run the ReAct loop for the given task."""
        logger.info(f"ReAct Agent starting task: {task[:100]}...")

        # Initialize conversation with system prompt and task
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"Task: {task}")
        ]

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"ReAct Round {round_num}/{self.max_rounds}")

            try:
                # Get LLM response
                response = await self.llm_provider.generate(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2048
                )

                content = response.content.strip()
                messages.append(LLMMessage(role="assistant", content=content))

                # Parse the response for actions
                if "FINAL_ANSWER:" in content:
                    # Extract final answer
                    final_answer = content.split("FINAL_ANSWER:", 1)[1].strip()
                    logger.info("ReAct agent finished by providing a final answer.")
                    return final_answer

                # Look for tool usage
                if "Action:" in content and "Action Input:" in content:
                    action_result = await self._execute_action(content)
                    if action_result:
                        # Add observation to conversation
                        observation = f"Observation: {action_result}"
                        messages.append(LLMMessage(role="user", content=observation))
                        continue

                # If no action and no final answer, treat as final answer
                logger.info("ReAct agent finished by providing a final text answer.")
                return content

            except Exception as e:
                logger.error(f"Error in ReAct round {round_num}: {e}")
                # Continue to next round or return error message
                if round_num == self.max_rounds:
                    return f"I encountered an error while processing your request: {str(e)}"
                continue

        return "I couldn't complete the task within the maximum number of reasoning steps."

    async def _execute_action(self, content: str) -> Optional[str]:
        """Execute an action from the LLM response."""
        try:
            # Parse action and input
            lines = content.split('\n')
            action_name = None
            action_input = None

            for line in lines:
                if line.startswith("Action:"):
                    action_name = line.split("Action:", 1)[1].strip()
                elif line.startswith("Action Input:"):
                    action_input = line.split("Action Input:", 1)[1].strip()

            if not action_name or not action_input:
                return None

            # Execute the tool
            if hasattr(self.tools_manager, 'execute_tool'):
                result = await self.tools_manager.execute_tool(action_name, action_input)
                return str(result) if result is not None else "Tool executed successfully."
            else:
                # Fallback for synchronous tool execution
                result = self.tools_manager.call_tool(action_name, action_input)
                return str(result) if result is not None else "Tool executed successfully."

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return f"Error executing action: {str(e)}"
