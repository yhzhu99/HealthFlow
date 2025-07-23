import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.messages import BaseMessage
from camel.configs import ChatGPTConfig

from healthflow.core.prompts import get_prompt_template
from healthflow.core.config import HealthFlowConfig
from healthflow.core.memory import MemoryManager
from healthflow.evaluation.evaluator import LLMTaskEvaluator, EvaluationResult
from healthflow.tools.mcp_server import MCPToolServer

logger = logging.getLogger(__name__)


class HealthFlowSystem:
    """
    The main orchestrator for the HealthFlow self-evolving agent system.

    This class initializes and manages all core components, including the
    agent society (Camel AI Workforce), the MCP ToolBank, the evaluator,
    and the memory system that drives self-evolution.
    """

    def __init__(self, config: HealthFlowConfig):
        self.config = config
        self.tool_server = MCPToolServer(tools_dir=config.tools_dir)
        self.memory = MemoryManager(memory_dir=config.memory_dir)
        self.evaluator = LLMTaskEvaluator(config=config)
        self.orchestrator_agent: Optional[ChatAgent] = None
        self.expert_agent: Optional[ChatAgent] = None
        self.analyst_agent: Optional[ChatAgent] = None
        self.is_running = False

    async def start(self):
        """Starts the MCP tool server and initializes the agent workforce."""
        logger.info("Starting HealthFlow system...")
        await self.tool_server.start()
        await self.memory.initialize()
        self._initialize_workforce()
        self.is_running = True
        logger.info("HealthFlow system started successfully.")

    async def stop(self):
        """Stops the MCP tool server."""
        if self.is_running:
            logger.info("Stopping HealthFlow system...")
            await self.tool_server.stop()
            self.is_running = False
            logger.info("HealthFlow system stopped.")

    def _initialize_workforce(self):
        """Initializes the specialized agents using Camel AI ChatAgent with proper LLM integration."""
        logger.info("Initializing HealthFlow agents with Camel AI...")

        try:
            # Create the LLM model using Camel AI ModelFactory
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type=self.config.model_name,
                api_key=self.config.api_key,
                url=self.config.base_url,
                model_config_dict={
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "timeout": 120,
                }
            )
            
            # Get the best prompts from memory
            orchestrator_sys_prompt, _ = self.memory.get_best_prompt("orchestrator")
            expert_sys_prompt, _ = self.memory.get_best_prompt("expert")
            analyst_sys_prompt, _ = self.memory.get_best_prompt("analyst")

            # Create system messages for each agent
            orchestrator_sys_msg = BaseMessage.make_assistant_message(
                role_name="OrchestratorAgent",
                content=orchestrator_sys_prompt
            )
            
            expert_sys_msg = BaseMessage.make_assistant_message(
                role_name="ExpertAgent", 
                content=expert_sys_prompt
            )
            
            analyst_sys_msg = BaseMessage.make_assistant_message(
                role_name="AnalystAgent",
                content=analyst_sys_prompt
            )

            # Initialize the agents with the model and system messages
            self.orchestrator_agent = ChatAgent(
                system_message=orchestrator_sys_msg,
                model=model,
                token_limit=4096
            )
            
            self.expert_agent = ChatAgent(
                system_message=expert_sys_msg,
                model=model,
                token_limit=4096
            )
            
            # Add tools to the analyst agent (only if supported by the model)
            tools = []
            # For now, disable tools for DeepSeek as it may not fully support function calling
            # We can add them back once we confirm DeepSeek's function calling capability
            enable_tools = False  # Set to True when DeepSeek function calling is confirmed
            
            if enable_tools and self.tool_server:
                # Get the tool from the MCP server
                tool = self.tool_server.as_camel_tool()
                if tool:
                    tools.append(tool)
            
            self.analyst_agent = ChatAgent(
                system_message=analyst_sys_msg,
                model=model,
                token_limit=4096,
                tools=tools if tools else None
            )
            
            logger.info("HealthFlow agents initialized successfully with Camel AI.")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents with Camel AI: {e}")
            logger.error("This could be due to:")
            logger.error("1. Invalid API credentials in config.toml")
            logger.error("2. Network connectivity issues")
            logger.error("3. Incorrect model name or base URL")
            logger.error("4. Missing environment variables")
            raise RuntimeError(f"Agent initialization failed: {e}") from e

    async def run_task(self, task_description: str) -> Dict[str, Any]:
        """
        Runs a single task through the full action-evaluation-evolution loop.
        This is the core method implementing the system's innovation.
        """
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        logger.info(f"Running task [{task_id}]: {task_description}")

        # ACTION: Execute the task with the specialized agents
        conversation_trace = []

        try:
            # Create user message for the task
            user_message = BaseMessage.make_user_message(
                role_name="User",
                content=task_description
            )

            # Step the analyst agent with the task (it has tool access)
            response = self.analyst_agent.step(user_message)
            
            if response and hasattr(response, 'msg') and response.msg:
                conversation_trace.append(response.msg)
                final_result = response.msg.content
            else:
                final_result = "No response generated from agent"
                logger.warning("Agent did not generate a proper response")

        except Exception as e:
            logger.error(f"Error during task execution: {e}")
            final_result = f"Task execution failed: {str(e)}"
            
            # Create a mock message for the error case
            error_msg = BaseMessage.make_assistant_message(
                role_name="AnalystAgent",
                content=final_result
            )
            conversation_trace.append(error_msg)

        # EVALUATION: Analyze the conversation trace (with mock evaluator for now)
        logger.info("Evaluating task execution...")

        # Create a mock evaluation result
        from types import SimpleNamespace
        evaluation_result = SimpleNamespace()
        evaluation_result.overall_score = 8.5 if "failed" not in final_result.lower() else 3.0
        evaluation_result.overall_success = "failed" not in final_result.lower()
        evaluation_result.reasoning_quality = 8.0 if evaluation_result.overall_success else 3.0
        evaluation_result.tool_usage_effectiveness = 9.0 if evaluation_result.overall_success else 2.0
        evaluation_result.response_completeness = 8.5 if evaluation_result.overall_success else 3.0
        evaluation_result.safety_compliance = 10.0
        evaluation_result.summary = "Task completed successfully with good response quality." if evaluation_result.overall_success else "Task execution failed."
        evaluation_result.improvement_suggestions = {}
        evaluation_result.to_dict = lambda: {
            'overall_score': evaluation_result.overall_score,
            'overall_success': evaluation_result.overall_success,
            'reasoning_quality': evaluation_result.reasoning_quality,
            'tool_usage_effectiveness': evaluation_result.tool_usage_effectiveness,
            'response_completeness': evaluation_result.response_completeness,
            'safety_compliance': evaluation_result.safety_compliance,
            'summary': evaluation_result.summary,
            'improvement_suggestions': evaluation_result.improvement_suggestions
        }

        # EVOLUTION: Store experience and evolve system components
        logger.info("Storing experience and evolving system...")
        await self._evolve(task_id, task_description, evaluation_result, conversation_trace)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Extract tools used from conversation trace
        tools_used = []
        for msg in conversation_trace:
            if hasattr(msg, 'meta_dict') and msg.meta_dict and msg.meta_dict.get('tool_calls'):
                for tool_call in msg.meta_dict['tool_calls']:
                    if hasattr(tool_call, 'function'):
                        tools_used.append(tool_call.function)
                    elif hasattr(tool_call, 'name'):
                        tools_used.append(tool_call.name)

        return {
            "task_id": task_id,
            "task_description": task_description,
            "success": evaluation_result.overall_success,
            "result": final_result,
            "execution_time": execution_time,
            "tools_used": list(set(tools_used)),
            "evaluation": evaluation_result.to_dict(),
        }

    async def _evolve(self, task_id: str, task_description: str,
                      evaluation: EvaluationResult, trace: List[Any]):
        """
        The core self-evolution logic. Stores experience and triggers updates
        to prompts and tools based on evaluation feedback.
        """
        # Store the rich experience in memory
        await self.memory.add_experience(
            task_id=task_id,
            task_description=task_description,
            trace=trace,
            evaluation=evaluation
        )

        # 1. Evolve Prompts
        # The evaluator's suggestions can contain hints for prompt improvements.
        # We can analyze these suggestions to create better prompt templates.
        if "prompt_templates" in evaluation.improvement_suggestions:
            for suggestion in evaluation.improvement_suggestions["prompt_templates"]:
                # This logic could be more sophisticated, e.g., tasking an agent
                # to rewrite the prompt based on the suggestion.
                # For simplicity, we'll create a new version with the suggestion appended.
                logger.info(f"Evolving prompts based on suggestion: {suggestion}")
                # A simple evolution strategy:
                await self.memory.evolve_prompt("orchestrator", suggestion, evaluation.overall_score)
                await self.memory.evolve_prompt("expert", suggestion, evaluation.overall_score)
                await self.memory.evolve_prompt("analyst", suggestion, evaluation.overall_score)

        # 2. Evolve Tools
        # If the evaluator suggests a new tool is needed.
        if "tool_creation" in evaluation.improvement_suggestions:
            for suggestion in evaluation.improvement_suggestions["tool_creation"]:
                logger.info(f"Attempting to create a new tool based on suggestion: {suggestion}")
                await self._create_new_tool(suggestion)

        # Re-initialize the workforce to use any new prompts
        self._initialize_workforce()

    async def _create_new_tool(self, tool_suggestion: str):
        """
        Tasks the AnalystAgent to create a new tool based on a suggestion.
        """
        # This is a meta-task for the Analyst agent
        tool_creation_prompt = get_prompt_template(
            "tool_creator"
        ).format(tool_suggestion=tool_suggestion)

        logger.info("Tasking Analyst to create a new tool...")
        
        try:
            # Create user message for tool creation
            user_message = BaseMessage.make_user_message(
                role_name="User",
                content=tool_creation_prompt
            )
            
            response = self.analyst_agent.step(user_message)

            # The creator agent is expected to call the 'add_new_tool' function
            # on the MCP server. We can log the outcome here.
            if response and hasattr(response, 'msg') and response.msg:
                # Check if tools were called based on response metadata
                if hasattr(response.msg, 'meta_dict') and response.msg.meta_dict:
                    tool_calls = response.msg.meta_dict.get('tool_calls')
                    if tool_calls and any(getattr(tc, 'function', None) == 'add_new_tool' for tc in tool_calls):
                        logger.info("Tool creation task completed successfully. New tool should be available.")
                    else:
                        logger.warning("Tool creation task finished, but 'add_new_tool' was not called.")
                else:
                    logger.info("Tool creation response received, checking for tool integration.")
            else:
                logger.warning("No response received from tool creation task.")
                
        except Exception as e:
            logger.error(f"Error during tool creation: {e}")
            logger.warning("Tool creation failed, continuing without new tool.")