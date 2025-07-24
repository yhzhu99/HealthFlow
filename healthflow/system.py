"""
Main HealthFlow system orchestration using OpenAI's native format.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
import json
import time
import uuid
from datetime import datetime

from .core.llm_provider import LLMProvider, LLMMessage, create_llm_provider
from .core.react_agent import ReactAgent
from .core.memory import MemoryManager
from .core.evolution import EvolutionManager
from .tools.tool_manager import ToolManager
from .evaluation.evaluator import LLMTaskEvaluator, EvaluationResult

logger = logging.getLogger(__name__)

class HealthFlowSystem:
    """Main HealthFlow system coordinator."""

    def __init__(self, config):
        self.config = config
        self.llm_provider = self._create_llm_provider()
        self.memory_manager = MemoryManager(self.config.memory_dir)
        self.evolution_manager = EvolutionManager(self.config.evolution_dir)
        self.tool_manager = ToolManager()
        self.react_agent = ReactAgent(self.llm_provider, self.tool_manager)
        self.evaluator = LLMTaskEvaluator(self.config, self.llm_provider)
        self._task_counter = 0

        # Load evolved prompts
        self.prompts = {
            'orchestrator': {
                'prompt': self.evolution_manager.get_best_prompt('orchestrator')[0],
                'score': self.evolution_manager.get_best_prompt('orchestrator')[1]
            },
            'expert': {
                'prompt': self.evolution_manager.get_best_prompt('expert')[0],
                'score': self.evolution_manager.get_best_prompt('expert')[1]
            },
            'analyst': {
                'prompt': self.evolution_manager.get_best_prompt('analyst')[0],
                'score': self.evolution_manager.get_best_prompt('analyst')[1]
            }
        }
        logger.info(f"Loaded prompts. Orchestrator score: {self.prompts['orchestrator']['score']:.2f}, "
                   f"Expert score: {self.prompts['expert']['score']:.2f}, "
                   f"Analyst score: {self.prompts['analyst']['score']:.2f}")

    async def start(self):
        """Start the HealthFlow system."""
        logger.info("HealthFlow system started successfully")

    async def stop(self):
        """Stop the HealthFlow system and clean up resources."""
        logger.info("HealthFlow system stopped")

    def _create_llm_provider(self) -> LLMProvider:
        """Create LLM provider from config."""
        return create_llm_provider(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model_name=self.config.model_name
        )

    async def run_task(self, user_input: str) -> Dict[str, Any]:
        """Run a task and return the result."""
        result = await self.process_task(user_input)
        # Convert final_answer to result to match expected interface
        if 'final_answer' in result:
            result['result'] = result.pop('final_answer')
        return result

    async def get_system_status(self) -> Dict[str, Any]:
        """Get the current system status."""
        return {
            'task_count': self._task_counter,
            'prompt_status': self.evolution_manager.get_prompt_status(),
            'strategy_performance': self.evolution_manager.get_strategy_performance(),
            'tool_status': self._get_tool_status()
        }

    def _get_tool_status(self) -> Dict[str, str]:
        """Get tool status information."""
        tools = self.tool_manager.get_available_tools()
        status = {}
        for tool_name in tools:
            tool_info = self.tool_manager.get_tool_info(tool_name)
            status[tool_name] = tool_info.get('description', 'No description')
        return status

    async def process_task(self, user_input: str) -> Dict[str, Any]:
        """Process a user task through the HealthFlow pipeline."""
        task_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        self._task_counter += 1

        logger.info(f"--- Starting Task #{self._task_counter}: {user_input} ---")

        try:
            # Step 1: Orchestrator planning
            logger.info("Orchestrator is planning...")
            plan_result = await self._orchestrator_plan(user_input)

            if not plan_result:
                return {"success": False, "error": "Failed to create execution plan"}

            strategy = plan_result.get('strategy')
            plan = plan_result.get('plan')

            logger.info(f"Orchestrator decided on strategy: {strategy}")
            logger.info(f"Executing with strategy: {strategy}")

            # Record strategy usage
            self.evolution_manager.record_strategy_usage(strategy)

            # Step 2: Execute based on strategy
            conversation_trace = []
            if strategy == "expert_only":
                result, trace = await self._execute_expert_only(plan, user_input)
                conversation_trace = trace
            elif strategy == "analyst_only":
                result, trace = await self._execute_analyst_only(plan, user_input)
                conversation_trace = trace
            elif strategy == "expert_then_analyst":
                result, trace = await self._execute_expert_then_analyst(plan, user_input)
                conversation_trace = trace
            else:
                return {"success": False, "error": f"Unknown strategy: {strategy}"}

            # Step 3: Evaluate and store
            evaluation = await self._evaluate_performance(task_id, user_input, conversation_trace)

            # Update strategy performance
            self.evolution_manager.update_strategy_performance(strategy, evaluation.overall_success)

            # Store experience
            await self.memory_manager.add_experience(task_id, user_input, conversation_trace, evaluation)

            # Save evolution data
            self.evolution_manager.save_all()

            execution_time = time.time() - start_time
            logger.info(f"--- Finished Task #{self._task_counter} in {execution_time:.2f}s. Success: {evaluation.overall_success} ---")

            return {
                "success": evaluation.overall_success,
                "final_answer": result,
                "task_id": task_id,
                "execution_time": execution_time,
                "evaluation": evaluation.to_dict()
            }

        except Exception as e:
            logger.error(f"Task processing failed: {e}", exc_info=True)
            execution_time = time.time() - start_time
            logger.info(f"--- Finished Task #{self._task_counter} in {execution_time:.2f}s. Success: False ---")

            return {
                "success": False,
                "error": str(e),
                "task_id": task_id,
                "execution_time": execution_time
            }

    async def _orchestrator_plan(self, user_input: str) -> Optional[Dict[str, str]]:
        """Get execution plan from orchestrator."""
        try:
            messages = [
                LLMMessage(role="system", content=self.prompts['orchestrator']['prompt']),
                LLMMessage(role="user", content=f"Analyze this task and create a plan. Your output must be a JSON object with 'plan' and 'strategy' keys. Strategy must be one of: ['analyst_only', 'expert_only', 'expert_then_analyst'].\n\nTask: {user_input}")
            ]

            response = await self.llm_provider.generate(messages, temperature=0.1)

            # Parse JSON response
            result = json.loads(response.content)
            return result

        except Exception as e:
            logger.error(f"Orchestrator planning failed: {e}")
            return None

    async def _execute_expert_only(self, plan: str, user_input: str) -> tuple[str, List]:
        """Execute using expert agent only."""
        logger.info("Engaging Expert Agent...")

        messages = [
            LLMMessage(role="system", content=self.prompts['expert']['prompt']),
            LLMMessage(role="user", content=f"Based on this plan: '{plan}', provide your medical expertise on the following task: '{user_input}'.")
        ]

        response = await self.llm_provider.generate(messages, temperature=0.3)

        # Create conversation trace
        trace = [
            self._create_trace_message("system", self.prompts['expert']['prompt']),
            self._create_trace_message("user", f"Based on this plan: '{plan}', provide your medical expertise on the following task: '{user_input}'."),
            self._create_trace_message("assistant", response.content)
        ]

        return response.content, trace

    async def _execute_analyst_only(self, plan: str, user_input: str) -> tuple[str, List]:
        """Execute using analyst agent (ReAct) only."""
        logger.info("Engaging Analyst Agent with ReAct...")

        system_prompt = f"""You are an Analyst Agent. Your role is to solve analytical tasks using available tools.

AVAILABLE TOOLS:
{self._get_tools_description()}

INSTRUCTIONS:
1. Think step by step about what you need to do
2. Use tools when necessary by formatting: Action: tool_name, Action Input: parameters
3. After using a tool, wait for the Observation before proceeding
4. When you have the final answer, respond with: FINAL_ANSWER: [your answer]

Your task plan: {plan}"""

        result = await self.react_agent.run(user_input, system_prompt)

        # Create conversation trace (simplified)
        trace = [
            self._create_trace_message("system", system_prompt),
            self._create_trace_message("user", user_input),
            self._create_trace_message("assistant", result)
        ]

        return result, trace

    async def _execute_expert_then_analyst(self, plan: str, user_input: str) -> tuple[str, List]:
        """Execute expert first, then analyst."""
        # First get expert knowledge
        expert_result, expert_trace = await self._execute_expert_only(plan, user_input)

        # Then use analyst with expert context
        logger.info("Engaging Analyst Agent with expert context...")
        enhanced_task = f"Expert context: {expert_result}\n\nNow complete this analytical task: {user_input}"
        analyst_result, analyst_trace = await self._execute_analyst_only(plan, enhanced_task)

        # Combine traces
        combined_trace = expert_trace + analyst_trace

        return analyst_result, combined_trace

    def _create_trace_message(self, role: str, content: str):
        """Create a trace message object."""
        class TraceMessage:
            def __init__(self, role_name, content):
                self.role_name = role_name
                self.content = content
                self.meta_dict = {}

        return TraceMessage(role, content)

    def _get_tools_description(self) -> str:
        """Get description of available tools."""
        tools = self.tool_manager.get_available_tools()
        descriptions = []
        for tool_name in tools:
            tool_info = self.tool_manager.get_tool_info(tool_name)
            descriptions.append(f"- {tool_name}: {tool_info.get('description', 'No description')}")
        return "\n".join(descriptions)

    async def _evaluate_performance(self, task_id: str, user_input: str, conversation_trace: List) -> EvaluationResult:
        """Evaluate task performance using the LLM evaluator."""
        logger.info("Evaluating task performance...")

        try:
            evaluation = await self.evaluator.evaluate_task(task_id, user_input, conversation_trace)
            logger.info(f"Evaluation complete. Score: {evaluation.overall_score}/10.0")
            return evaluation
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return a fallback evaluation
            return EvaluationResult(
                evaluation_id=str(uuid.uuid4()),
                task_id=task_id,
                timestamp=datetime.now(),
                overall_success=False,
                overall_score=3.0,
                scores={"success_accuracy": 3.0, "strategy_reasoning": 3.0, "tool_usage_agentic_skill": 3.0, "safety_clarity": 3.0},
                executive_summary=f"Evaluation failed: {str(e)}",
                improvement_suggestions={"prompt_templates": [], "tool_creation": [], "collaboration_strategy": []}
            )

    def _get_task_counter(self) -> int:
        """Get current task counter."""
        return self._task_counter
