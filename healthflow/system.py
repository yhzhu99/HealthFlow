"""
Main HealthFlow system orchestration using OpenAI's native format.
"""
import logging
import asyncio
from typing import Dict, Any, Optional
import json
import time
import uuid

from .core.llm_provider import LLMProvider, LLMMessage, create_llm_provider
from .core.react_agent import ReactAgent
from .core.memory import MemoryManager
from .core.evolution import EvolutionManager
from .tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

class HealthFlowSystem:
    """Main HealthFlow system coordinator."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_provider = self._create_llm_provider()
        self.memory_manager = MemoryManager(config.get('memory', {}))
        self.evolution_manager = EvolutionManager()
        self.tool_manager = ToolManager()
        self.react_agent = ReactAgent(self.llm_provider, self.tool_manager)

        # Load evolved prompts
        self.prompts = self.evolution_manager.load_best_prompts()
        logger.info(f"Loaded prompts. Orchestrator score: {self.prompts['orchestrator']['score']:.2f}, "
                   f"Expert score: {self.prompts['expert']['score']:.2f}, "
                   f"Analyst score: {self.prompts['analyst']['score']:.2f}")

    def _create_llm_provider(self) -> LLMProvider:
        """Create LLM provider from config."""
        llm_config = self.config['llm']
        return create_llm_provider(
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url'],
            model_name=llm_config['model']
        )

    async def process_task(self, user_input: str) -> Dict[str, Any]:
        """Process a user task through the HealthFlow pipeline."""
        task_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(f"--- Starting Task #{self._get_task_counter()}: {user_input} ---")

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

            # Step 2: Execute based on strategy
            if strategy == "expert_only":
                result = await self._execute_expert_only(plan, user_input)
            elif strategy == "analyst_only":
                result = await self._execute_analyst_only(plan, user_input)
            elif strategy == "expert_then_analyst":
                result = await self._execute_expert_then_analyst(plan, user_input)
            else:
                return {"success": False, "error": f"Unknown strategy: {strategy}"}

            # Step 3: Evaluate and store
            evaluation = await self._evaluate_performance(user_input, result, strategy)

            # Store experience
            experience = {
                "task_id": task_id,
                "user_input": user_input,
                "strategy": strategy,
                "plan": plan,
                "result": result,
                "evaluation": evaluation,
                "execution_time": time.time() - start_time
            }

            self.memory_manager.store_experience(experience)
            self.evolution_manager.save_evolution_data()

            execution_time = time.time() - start_time
            logger.info(f"--- Finished Task #{self._get_task_counter()} in {execution_time:.2f}s. Success: True ---")

            return {
                "success": True,
                "final_answer": result,
                "task_id": task_id,
                "execution_time": execution_time,
                "evaluation": evaluation
            }

        except Exception as e:
            logger.error(f"Task processing failed: {e}", exc_info=True)
            execution_time = time.time() - start_time
            logger.info(f"--- Finished Task #{self._get_task_counter()} in {execution_time:.2f}s. Success: False ---")

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

    async def _execute_expert_only(self, plan: str, user_input: str) -> str:
        """Execute using expert agent only."""
        logger.info("Engaging Expert Agent...")

        messages = [
            LLMMessage(role="system", content=self.prompts['expert']['prompt']),
            LLMMessage(role="user", content=f"Based on this plan: '{plan}', provide your medical expertise on the following task: '{user_input}'.")
        ]

        response = await self.llm_provider.generate(messages, temperature=0.3)
        return response.content

    async def _execute_analyst_only(self, plan: str, user_input: str) -> str:
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

        return await self.react_agent.run(user_input, system_prompt)

    async def _execute_expert_then_analyst(self, plan: str, user_input: str) -> str:
        """Execute expert first, then analyst."""
        # First get expert knowledge
        expert_result = await self._execute_expert_only(plan, user_input)

        # Then use analyst with expert context
        logger.info("Engaging Analyst Agent with expert context...")
        enhanced_task = f"Expert context: {expert_result}\n\nNow complete this analytical task: {user_input}"
        return await self._execute_analyst_only(plan, enhanced_task)

    def _get_tools_description(self) -> str:
        """Get description of available tools."""
        tools = self.tool_manager.get_available_tools()
        descriptions = []
        for tool_name in tools:
            tool_info = self.tool_manager.get_tool_info(tool_name)
            descriptions.append(f"- {tool_name}: {tool_info.get('description', 'No description')}")
        return "\n".join(descriptions)

    async def _evaluate_performance(self, user_input: str, result: str, strategy: str) -> Dict[str, Any]:
        """Evaluate task performance."""
        logger.info("Evaluating task performance...")

        eval_prompt = f"""Evaluate this AI agent's performance on the following task:

Task: {user_input}
Strategy Used: {strategy}
Result: {result}

Rate the performance from 1-10 considering:
- Accuracy and correctness
- Completeness of the answer
- Appropriateness of strategy chosen
- Efficiency of execution

Respond with JSON: {{"score": X.X, "summary": "brief explanation"}}"""

        try:
            messages = [LLMMessage(role="user", content=eval_prompt)]
            response = await self.llm_provider.generate(messages, temperature=0.1)

            evaluation = json.loads(response.content)
            logger.info(f"Evaluation complete. Score: {evaluation['score']}/10.0")
            return evaluation

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"score": 5.0, "summary": "Evaluation failed"}

    def _get_task_counter(self) -> int:
        """Get current task counter (simplified)."""
        return getattr(self, '_task_counter', 0) + 1
