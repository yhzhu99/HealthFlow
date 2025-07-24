import logging
import json
from typing import Dict, Any, List, Tuple

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from healthflow.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

class ReactAgent:
    """
    Implements the ReAct (Reason-Act) loop to enable agentic behavior,
    especially for complex, multi-step computational tasks.
    """
    def __init__(self, chat_agent: ChatAgent, tool_manager: ToolManager, max_rounds: int = 5):
        self.agent = chat_agent
        self.tool_manager = tool_manager
        self.max_rounds = max_rounds
        self.agent.reset()

    async def solve_task(self, task_description: str) -> Dict[str, Any]:
        """
        Solves a task using an iterative think-act loop.
        """
        logger.info(f"ReAct Agent starting task: {task_description[:100]}...")
        self.agent.reset()

        history = [BaseMessage.make_user_message("User", task_description)]

        for i in range(self.max_rounds):
            logger.info(f"ReAct Round {i+1}/{self.max_rounds}")

            response = await self.agent.step(history[-1])
            history.append(response.msg)

            # Check for termination condition
            if "<DONE>" in response.msg.content:
                logger.info("ReAct agent finished with <DONE> tag.")
                final_answer = response.msg.content.replace("<DONE>", "").strip()
                return self._prepare_result(final_answer, history, True)

            # If agent wants to use a tool
            if response.msg.meta_dict and "tool_calls" in response.msg.meta_dict:
                tool_results = await self.tool_manager.execute_tool_calls(response.msg.meta_dict["tool_calls"])

                # Create a message with the tool results to continue the loop
                tool_response_msg = BaseMessage.make_tool_response_message(
                    tool_call_id=",".join([res['tool_call_id'] for res in tool_results]),
                    name=",".join([res['name'] for res in tool_results]),
                    content=json.dumps([res['result'] for res in tool_results])
                )
                history.append(tool_response_msg)
            else:
                # Agent provided a response without using a tool, likely the final answer
                logger.info("ReAct agent finished by providing a final text answer.")
                return self._prepare_result(response.msg.content, history, True)

        logger.warning("ReAct agent reached max rounds without finishing.")
        return self._prepare_result("Reached max rounds without completion.", history, False)

    def _prepare_result(self, final_result: str, trace: List, success: bool) -> Dict[str, Any]:
        return {
            "final_result": final_result,
            "trace": trace,
            "success": success
        }