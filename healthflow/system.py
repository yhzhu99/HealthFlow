import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from camel.agents import ChatAgent
from camel.societies.workforce import Workforce
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.tasks import Task

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
        self.workforce: Optional[Workforce] = None
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
        """Initializes the Camel AI Workforce with specialized agents."""
        logger.info("Initializing Camel AI Workforce...")
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=self.config.model_name,
            model_config_dict={
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
            },
            api_key = self.config.api_key,
        )

        # 1. Orchestrator (Coordinator)
        orchestrator_sys_prompt, _ = self.memory.get_best_prompt("orchestrator")
        orchestrator_agent = ChatAgent(
            system_message=orchestrator_sys_prompt, model=model
        )

        # 2. Expert (Worker)
        expert_sys_prompt, _ = self.memory.get_best_prompt("expert")
        expert_agent = ChatAgent(system_message=expert_sys_prompt, model=model)

        # 3. Analyst (Worker)
        analyst_sys_prompt, _ = self.memory.get_best_prompt("analyst")
        # The analyst is the only agent that directly interacts with tools.
        analyst_agent = ChatAgent(
            system_message=analyst_sys_prompt,
            model=model,
            tools=[self.tool_server.as_langchain_tool()],
        )

        self.workforce = Workforce(
            agents=[orchestrator_agent, expert_agent, analyst_agent],
            agents_role_names=["Coordinator", "Medical Expert", "Data Analyst"],
            task_prompt_template="""
Here is a task: {task}
The user wants a comprehensive answer to this medical query.
As the Coordinator, create a plan and delegate tasks to the Medical Expert and Data Analyst.
The Data Analyst has access to a code interpreter and other tools.
The Medical Expert provides clinical knowledge.
Synthesize their findings into a final, accurate, and safe response.
"""
        )

    async def run_task(self, task_description: str) -> Dict[str, Any]:
        """
        Runs a single task through the full action-evaluation-evolution loop.
        This is the core method implementing the system's innovation.
        """
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        logger.info(f"Running task [{task_id}]: {task_description}")

        # ACTION: Execute the task with the agent workforce
        task = Task(content=task_description)
        # The `run` method in Workforce returns a list of messages (conversation trace)
        conversation_trace = self.workforce.run(task)
        final_response_msg = conversation_trace[-1]
        final_result = final_response_msg.content

        # EVALUATION: Analyze the conversation trace
        logger.info("Evaluating task execution...")
        evaluation_result = await self.evaluator.evaluate_task(
            task_id, task_description, conversation_trace
        )

        # EVOLUTION: Store experience and evolve system components
        logger.info("Storing experience and evolving system...")
        await self._evolve(task_id, task_description, evaluation_result, conversation_trace)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        tools_used = [
            tool_call.function for msg in conversation_trace if msg.meta_dict and msg.meta_dict.get('tool_calls')
            for tool_call in msg.meta_dict['tool_calls']
        ]


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

        model = self.workforce.agents[2].model # Get Analyst's model
        creator_agent = ChatAgent(
            system_message=get_prompt_template("tool_creator_system"),
            model=model,
            tools=[self.tool_server.as_langchain_tool(include_management=True)],
        )

        logger.info("Tasking Analyst to create a new tool...")
        response = creator_agent.step(tool_creation_prompt)

        # The creator agent is expected to call the 'add_new_tool' function
        # on the MCP server. We can log the outcome here.
        if response and response.msgs:
            tool_calls = response.msgs[0].meta_dict.get('tool_calls')
            if tool_calls and any(tc.function == 'add_new_tool' for tc in tool_calls):
                 logger.info(f"Tool creation task completed successfully. New tool should be available.")
            else:
                 logger.warning(f"Tool creation task finished, but 'add_new_tool' was not called.")