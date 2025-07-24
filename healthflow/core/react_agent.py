import logging
import json
from typing import Dict, Any, List, Tuple

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType  # FIX: Import RoleType for proper role handling
from healthflow.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

class ReactAgent:
    """
    Implements the ReAct (Reason-Act) loop to enable agentic behavior,
    especially for complex, multi-step computational tasks.
    This version manages history manually to bypass issues in ChatAgent.step()
    with certain OpenAI-compatible APIs.
    """
    def __init__(self, chat_agent: ChatAgent, tool_manager: ToolManager, max_rounds: int = 5):
        self.agent = chat_agent
        self.tool_manager = tool_manager
        self.max_rounds = max_rounds

    async def solve_task(self, task_description: str) -> Dict[str, Any]:
        """
        Solves a task using an iterative think-act loop with manual history management.
        """
        logger.info(f"ReAct Agent starting task: {task_description[:100]}...")

        # Manually manage history, starting with the agent's system message.
        history: List[BaseMessage] = [self.agent.system_message]
        history.append(BaseMessage.make_user_message("User", task_description))

        for i in range(self.max_rounds):
            logger.info(f"ReAct Round {i+1}/{self.max_rounds}")

            # FIX: Better role handling for different message types
            openai_messages = []
            for msg in history:
                try:
                    # FIX: Handle different role types more robustly
                    if hasattr(msg.role_type, 'value'):
                        role_value = msg.role_type.value
                    else:
                        role_value = str(msg.role_type).lower()

                    # FIX: Map role types properly
                    if role_value in ['default', 'system']:
                        openai_role = 'system'
                    elif role_value == 'assistant':
                        openai_role = 'assistant'
                    elif role_value == 'tool':
                        openai_role = 'tool'
                    else:
                        openai_role = 'user'

                    if openai_role == 'assistant':
                        # Manually construct the dict for the 'assistant' role
                        msg_dict = {'role': 'assistant', 'content': msg.content}
                        if msg.meta_dict and 'tool_calls' in msg.meta_dict and msg.meta_dict['tool_calls']:
                            msg_dict['tool_calls'] = [tc.model_dump() for tc in msg.meta_dict['tool_calls']]
                            # Per OpenAI API spec, content can be null when tool_calls are present
                            if not msg_dict['content']:
                                msg_dict['content'] = None
                        openai_messages.append(msg_dict)
                    elif openai_role == 'tool':
                        # Handle tool response messages
                        msg_dict = {
                            'role': 'tool',
                            'content': msg.content,
                            'tool_call_id': msg.meta_dict.get('tool_call_id', 'unknown')
                        }
                        openai_messages.append(msg_dict)
                    else:
                        # For system and user messages
                        openai_messages.append({
                            'role': openai_role,
                            'content': msg.content or ""
                        })

                except Exception as e:
                    logger.error(f"Error processing message for OpenAI format: {e}")
                    # Ultimate fallback
                    openai_messages.append({'role': 'user', 'content': msg.content or ""})

            try:
                # FIX: Call the model backend without await since it's synchronous
                response_obj = self.agent.model_backend.run(openai_messages)

                if not response_obj or not response_obj.choices:
                    final_answer = "Error: Model returned no response or an empty response."
                    logger.error(final_answer)
                    return self._prepare_result(final_answer, history, False)

                # Create a BaseMessage from the raw OpenAI response
                try:
                    assistant_response_msg = BaseMessage.from_openai_message(
                        response_obj.choices[0].message
                    )
                except Exception as e:
                    logger.error(f"Failed to create BaseMessage from OpenAI response: {e}")
                    # FIX: Fallback manual message creation
                    assistant_response_msg = BaseMessage(
                        role_name="Assistant",
                        role_type=RoleType.ASSISTANT,
                        meta_dict={},
                        content=response_obj.choices[0].message.content or ""
                    )

                history.append(assistant_response_msg)

            except Exception as e:
                logger.error(f"Error calling model in ReAct loop: {e}", exc_info=True)
                return self._prepare_result(f"Failed to get model response: {e}", history, False)

            # Check for termination condition in the assistant's text content
            if assistant_response_msg.content and "<DONE>" in assistant_response_msg.content:
                logger.info("ReAct agent finished with <DONE> tag.")
                final_answer = assistant_response_msg.content.replace("<DONE>", "").strip()
                return self._prepare_result(final_answer, history, True)

            # Check if the agent wants to use a tool
            if assistant_response_msg.meta_dict and "tool_calls" in assistant_response_msg.meta_dict:
                tool_calls = assistant_response_msg.meta_dict["tool_calls"]

                # Execute the tool calls
                tool_results = await self.tool_manager.execute_tool_calls(tool_calls)

                # Create and append tool response messages to the history
                for res in tool_results:
                    try:
                        tool_response_msg = BaseMessage.make_tool_response_message(
                            tool_call_id=res['tool_call_id'],
                            name=res['name'],
                            content=str(res['result']) # Ensure content is a string
                        )
                    except Exception as e:
                        logger.error(f"Failed to create tool response message: {e}")
                        # FIX: Fallback manual tool response creation
                        tool_response_msg = BaseMessage(
                            role_name="Tool",
                            role_type=RoleType.TOOL,
                            meta_dict={"tool_call_id": res['tool_call_id'], "name": res['name']},
                            content=str(res['result'])
                        )
                    history.append(tool_response_msg)
            else:
                # Agent provided a response without using a tool, which we treat as the final answer.
                logger.info("ReAct agent finished by providing a final text answer.")
                final_answer = assistant_response_msg.content or "Agent finished without a clear textual answer."
                return self._prepare_result(final_answer, history, True)

        logger.warning("ReAct agent reached max rounds without finishing.")
        final_answer = "Reached maximum rounds of execution without a final answer. The last message was: "
        if history[-1].content:
            final_answer += history[-1].content
        else:
            final_answer += "(No text content in last message)"
        return self._prepare_result(final_answer, history, False)

    def _prepare_result(self, final_result: str, trace: List[BaseMessage], success: bool) -> Dict[str, Any]:
        return {
            "final_result": final_result,
            "trace": trace,
            "success": success
        }
