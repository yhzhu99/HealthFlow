import logging
import uuid
import json
import re
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, RoleType
from camel.messages import BaseMessage

from healthflow.core.config import HealthFlowConfig
from healthflow.core.prompts import get_prompt
from healthflow.core.react_agent import ReactAgent
from healthflow.core.evolution import EvolutionManager
from healthflow.core.memory import MemoryManager
from healthflow.evaluation.evaluator import LLMTaskEvaluator, EvaluationResult
from healthflow.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

class HealthFlowSystem:
    """
    The central class of the HealthFlow framework. It orchestrates agent
    collaboration, task execution, and the self-evolution loop.
    """

    def __init__(self, config: HealthFlowConfig):
        self.config = config
        self.task_count = 0
        self.is_running = False

        # Core Components
        self.evolution_manager = EvolutionManager(Path(config.evolution_dir))
        self.memory_manager = MemoryManager(Path(config.memory_dir))
        self.tool_manager = ToolManager(tools_dir=Path(config.tools_dir))
        self.evaluator = LLMTaskEvaluator(config=config)

        # Agent Placeholders
        self.model: Optional[Any] = None
        self.orchestrator_agent: Optional[ChatAgent] = None
        self.expert_agent: Optional[ChatAgent] = None
        self.analyst_agent: Optional[ChatAgent] = None

    async def start(self):
        """Initializes the system, loads evolving assets, and prepares agents."""
        logger.info("🚀 Starting HealthFlow System...")
        await self.tool_manager.load_tools()
        self._initialize_agents()
        self.is_running = True
        logger.info("✅ HealthFlow started successfully.")

    async def stop(self):
        """Shuts down the system and saves any pending evolution data."""
        if self.is_running:
            logger.info("🛑 Stopping HealthFlow...")
            self.evolution_manager.save_all()
            self.is_running = False
            logger.info("✅ HealthFlow stopped.")

    def _initialize_agents(self):
        """Initializes or re-initializes agents with the best-performing prompts."""
        logger.info("🤖 Initializing agents with the best evolved prompts...")

        # BUG FIX: Added max_tokens to prevent the warning and ensure consistent behavior.
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=self.config.model_name,
            api_key=self.config.api_key,
            url=self.config.base_url,
            model_config_dict={"temperature": 0.2, "timeout": 180, "max_tokens": 4096}
        )

        # Get best prompts for each role
        orchestrator_prompt = self.evolution_manager.get_best_prompt("orchestrator")
        expert_prompt = self.evolution_manager.get_best_prompt("expert")
        analyst_prompt = self.evolution_manager.get_best_prompt("analyst")

        logger.info(f"Loaded prompts. Orchestrator score: {orchestrator_prompt[1]:.2f}, Expert score: {expert_prompt[1]:.2f}, Analyst score: {analyst_prompt[1]:.2f}")

        # Create agents
        self.orchestrator_agent = self._create_agent("Orchestrator", orchestrator_prompt[0])
        self.expert_agent = self._create_agent("Expert", expert_prompt[0])

        # BUG FIX: The `tools` parameter now receives a list of FunctionTool objects
        # directly from the new `get_camel_tools` method.
        self.analyst_agent = self._create_agent(
            "Analyst",
            analyst_prompt[0],
            tools=self.tool_manager.get_camel_tools()
        )
        logger.info("✅ Agents initialized.")

    def _create_agent(self, role: str, system_prompt: str, tools: Optional[List[Any]] = None) -> ChatAgent:
        """Helper to create a ChatAgent."""
        # FIX: Changed RoleType.SYSTEM to RoleType.DEFAULT since SYSTEM doesn't exist
        sys_msg = BaseMessage(
            role_name=f"{role}Agent",
            role_type=RoleType.DEFAULT,
            meta_dict={},
            content=system_prompt,
        )
        return ChatAgent(system_message=sys_msg, model=self.model, tools=tools)

    async def run_task(self, task_description: str) -> Dict[str, Any]:
        """
        Executes a complete task, including planning, execution, and evolution.
        """
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        self.task_count += 1
        logger.info(f"--- Starting Task #{self.task_count}: {task_description} ---")

        trace = []
        final_result = "Task failed to produce a result."
        success = False
        evaluation_result = None
        strategy = "analyst_only" # Default strategy

        try:
            # Step 1: Orchestrator plans the approach
            plan, strategy, trace = await self._plan_task(task_description, trace)
            self.evolution_manager.record_strategy_usage(strategy)

            # Step 2: Execute the plan using the chosen strategy
            execution_result, trace = await self._execute_plan(plan, strategy, task_description, trace)
            final_result = execution_result

            # FIX: Better success detection based on actual execution results
            success = (
                "error" not in final_result.lower() and
                "failed" not in final_result.lower() and
                "exception" not in final_result.lower() and
                "can't be used in" not in final_result.lower() and
                "traceback" not in final_result.lower()
            )

            if not success:
                logger.warning(f"Execution appears to have failed based on result: {final_result[:200]}...")

        except Exception as e:
            logger.error(f"Task execution failed with an exception: {e}", exc_info=True)
            final_result = f"Task failed due to a critical error: {e}"
            success = False

        # Step 3: Evaluate and Evolve
        try:
            evaluation_result = await self._evaluate_and_evolve(task_id, task_description, trace, success)
            if evaluation_result:
                # FIX: Use evaluation result to determine actual success, not just execution success
                success = evaluation_result.overall_success
                self.evolution_manager.update_strategy_performance(strategy, success)
        except Exception as e:
            logger.error(f"Evaluation and evolution step failed: {e}", exc_info=True)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        logger.info(f"--- Finished Task #{self.task_count} in {execution_time:.2f}s. Success: {success} ---")

        return {
            "task_id": task_id,
            "success": success,
            "result": final_result,
            "execution_time": execution_time,
            "evaluation": evaluation_result.to_dict() if evaluation_result else None,
        }

    async def _plan_task(self, task_description: str, trace: List) -> Tuple[str, str, List]:
        """Lets the orchestrator create a plan and choose a strategy."""
        logger.info("Orchestrator is planning...")
        user_msg = BaseMessage.make_user_message("User", content=f"Analyze this task and create a plan. Your output must be a JSON object with 'plan' and 'strategy' keys. Strategy must be one of: ['analyst_only', 'expert_only', 'expert_then_analyst'].\n\nTask: {task_description}")
        trace.append(user_msg)

        response = self.orchestrator_agent.step(user_msg)
        trace.append(response.msg)

        plan_content = response.msg.content
        # Try to find JSON in the response
        try:
            json_match = re.search(r"\{.*\}", plan_content, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
                plan = plan_data.get("plan", "No plan provided.")
                strategy = plan_data.get("strategy", "analyst_only")
                if strategy not in ['analyst_only', 'expert_only', 'expert_then_analyst']:
                    strategy = 'analyst_only' # Default fallback
                logger.info(f"Orchestrator decided on strategy: {strategy}")
                return plan, strategy, trace
        except json.JSONDecodeError:
            pass # Fall through to default

        logger.warning("Orchestrator failed to produce valid JSON plan. Defaulting to analyst_only.")
        return plan_content, "analyst_only", trace


    async def _execute_plan(self, plan: str, strategy: str, task_desc: str, trace: List) -> Tuple[str, List]:
        """Executes the task based on the chosen strategy."""
        logger.info(f"Executing with strategy: {strategy}")
        if strategy == "expert_only":
            return await self._execute_with_expert(plan, task_desc, trace)

        if strategy == "expert_then_analyst":
            expert_opinion, trace = await self._execute_with_expert(plan, task_desc, trace)
            context = f"Here is the initial medical expert opinion:\n{expert_opinion}\n\nNow, perform the computational part of the plan."
            return await self._execute_with_analyst(plan, task_desc, trace, context)

        return await self._execute_with_analyst(plan, task_desc, trace)

    async def _execute_with_analyst(self, plan: str, task_desc: str, trace: List, context: str = "") -> Tuple[str, List]:
        """Executes a task using the Analyst agent with ReAct capabilities."""
        logger.info("Engaging Analyst Agent with ReAct...")
        react_agent = ReactAgent(
            chat_agent=self.analyst_agent,
            tool_manager=self.tool_manager,
            max_rounds=self.config.max_react_rounds
        )

        react_prompt = f"Based on this plan: '{plan}', solve the following task: '{task_desc}'."
        if context:
            react_prompt += f"\n\nAdditional context: {context}"

        react_result = await react_agent.solve_task(react_prompt)
        trace.extend(react_result['trace'])

        return react_result['final_result'], trace

    async def _execute_with_expert(self, plan: str, task_desc: str, trace: List) -> Tuple[str, List]:
        """Executes a task using the Expert agent."""
        logger.info("Engaging Expert Agent...")
        expert_prompt = f"Based on this plan: '{plan}', provide your medical expertise on the following task: '{task_desc}'."
        user_msg = BaseMessage.make_user_message("User", content=expert_prompt)
        trace.append(user_msg)
        response = self.expert_agent.step(user_msg)
        trace.append(response.msg)
        return response.msg.content, trace

    async def _evaluate_and_evolve(self, task_id: str, task_desc: str, trace: List, success: bool) -> Optional[EvaluationResult]:
        """Evaluates the task and triggers system evolution if needed."""
        logger.info("Evaluating task performance...")
        evaluation_result = await self.evaluator.evaluate_task(task_id, task_desc, trace)
        score = evaluation_result.overall_score

        logger.info(f"Evaluation complete. Score: {score:.2f}/10.0")

        await self.memory_manager.add_experience(task_id, task_desc, trace, evaluation_result)

        if score < self.config.evolution_trigger_score:
            logger.info(f"Low score detected ({score:.2f}). Triggering prompt evolution.")
            await self._evolve_prompts(evaluation_result)

        tool_suggestions = evaluation_result.improvement_suggestions.get("tool_creation", [])
        if tool_suggestions:
            logger.info("Tool creation suggestions found. Attempting to evolve tools.")
            await self._evolve_tools(tool_suggestions)

        return evaluation_result

    async def _evolve_prompts(self, eval_result: EvaluationResult):
        """Evolves agent prompts based on evaluation feedback."""
        feedback = f"Evaluation Summary: {eval_result.executive_summary}. Suggestions: {eval_result.improvement_suggestions.get('prompt_templates', [])}"

        for role in ["orchestrator", "expert", "analyst"]:
            prompt_id, current_prompt, old_score = self.evolution_manager.get_best_prompt_with_id(role)
            evolver_agent = self._create_agent("Evolver", get_prompt("evolve_prompt_system"))
            evolution_request = get_prompt("evolve_prompt_task").format(
                role=role,
                current_prompt=current_prompt,
                feedback=feedback
            )
            user_msg = BaseMessage.make_user_message("User", content=evolution_request)
            response = evolver_agent.step(user_msg)
            new_prompt = response.msg.content

            if new_prompt and new_prompt != current_prompt:
                # FIX: Use the evaluation score to update the prompt score, not a default
                self.evolution_manager.add_prompt_version(role, new_prompt, eval_result.overall_score, feedback)
                logger.info(f"Evolved prompt for role: {role} with score {eval_result.overall_score:.2f}")

        self._initialize_agents()

    async def _evolve_tools(self, suggestions: List[str]):
        """Triggers the Analyst agent to create new tools based on suggestions."""
        creator_agent = ReactAgent(
            chat_agent=self.analyst_agent,
            tool_manager=self.tool_manager,
            max_rounds=3
        )
        for suggestion in suggestions:
            logger.info(f"Attempting to create tool for suggestion: {suggestion}")
            tool_creation_task = f"A new tool is needed. Suggestion: '{suggestion}'. Your task is to write the Python code for this tool and register it using the `add_new_tool` function. Analyze the request, write a single Python function, and then call `add_new_tool` with the correct arguments (name, code, description)."
            await creator_agent.solve_task(tool_creation_task)

        await self.tool_manager.load_tools()

    async def get_system_status(self) -> Dict[str, Any]:
        """Retrieves and formats the current system status for display."""
        return {
            "task_count": self.task_count,
            "prompt_status": self.evolution_manager.get_prompt_status(),
            "strategy_performance": self.evolution_manager.get_strategy_performance(),
            "tool_status": self.tool_manager.get_tool_info(),
        }
