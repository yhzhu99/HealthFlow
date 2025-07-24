"""
HealthFlow System V2 - Simplified and Self-Evolving

A clean, LLM-driven healthcare AI system that emphasizes simplicity and self-improvement.
Features ReAct loops, simplified prompts, and transparent evolution tracking.
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.messages import BaseMessage

from healthflow.core.config import HealthFlowConfig
from healthflow.core.simple_prompts import generate_evolved_prompt
from healthflow.core.simple_interpreter import SimpleHealthcareInterpreter
from healthflow.core.react_agent import ReactAgent
from healthflow.core.evolution_config import EvolutionConfig
from healthflow.core.enhanced_logging import get_enhanced_logger
from healthflow.evaluation.evaluator import LLMTaskEvaluator

logger = logging.getLogger(__name__)
enhanced_logger = get_enhanced_logger()


class HealthFlowSystemV2:
    """
    Simple, self-evolving healthcare AI system.
    
    Core principles:
    - Let LLM think and adapt rather than rigid hardcoded prompts
    - Use ReAct loops for iterative problem solving
    - Track evolution transparently in external config files
    - Keep the framework simple yet effective
    """
    
    def __init__(self, config: HealthFlowConfig):
        self.config = config
        self.interpreter = SimpleHealthcareInterpreter()
        
        # Evolution management
        self.evolution_config = EvolutionConfig(config.memory_dir / "evolution")
        
        # Simple evaluator
        self.evaluator = LLMTaskEvaluator(config=config)
        
        # Agents
        self.orchestrator_agent: Optional[ChatAgent] = None
        self.expert_agent: Optional[ChatAgent] = None
        self.analyst_agent: Optional[ChatAgent] = None
        
        # Task tracking
        self.task_count = 0
        
        self.is_running = False
    
    async def start(self):
        """Start the system."""
        logger.info("🚀 Starting HealthFlow V2 - Simple & Self-Evolving")
        
        # Initialize agents with evolved prompts
        self._initialize_agents()
        
        self.is_running = True
        logger.info("✅ HealthFlow V2 started successfully")
    
    async def stop(self):
        """Stop the system."""
        if self.is_running:
            logger.info("🛑 Stopping HealthFlow V2...")
            # Save evolution state
            self.evolution_config.save_all()
            self.is_running = False
            logger.info("✅ HealthFlow V2 stopped")
    
    def _initialize_agents(self):
        """Initialize agents with the best evolved prompts."""
        logger.info("🤖 Initializing agents with evolved prompts...")
        
        try:
            # Create LLM model
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
            
            # Get best prompts for each role
            orchestrator_prompt, orchestrator_score = self.evolution_config.get_best_prompt("orchestrator")
            expert_prompt, expert_score = self.evolution_config.get_best_prompt("expert")
            analyst_prompt, analyst_score = self.evolution_config.get_best_prompt("analyst")
            
            logger.info(f"📊 Prompt scores - Orchestrator: {orchestrator_score:.2f}, Expert: {expert_score:.2f}, Analyst: {analyst_score:.2f}")
            
            # Create system messages
            orchestrator_sys_msg = BaseMessage.make_assistant_message(
                role_name="OrchestratorAgent",
                content=orchestrator_prompt
            )
            
            expert_sys_msg = BaseMessage.make_assistant_message(
                role_name="ExpertAgent", 
                content=expert_prompt
            )
            
            analyst_sys_msg = BaseMessage.make_assistant_message(
                role_name="AnalystAgent",
                content=analyst_prompt
            )
            
            # Initialize agents
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
            
            self.analyst_agent = ChatAgent(
                system_message=analyst_sys_msg,
                model=model,
                token_limit=4096
            )
            
            logger.info("✅ Agents initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
            raise RuntimeError(f"Agent initialization failed: {e}") from e
    
    async def run_task(self, task_description: str) -> Dict[str, Any]:
        """
        Execute a task using the simplified, adaptive approach.
        """
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        self.task_count += 1
        
        # Start enhanced logging
        enhanced_logger.start_task(task_id, task_description)
        
        try:
            # Step 1: Orchestrator plans the approach
            enhanced_logger.step("Planning", "Orchestrator analyzing task and creating approach")
            enhanced_logger.agent_thinking("Orchestrator", task_description)
            plan_result = await self._plan_task(task_description)
            enhanced_logger.agent_result("Orchestrator", plan_result is not None, f"Approach: {plan_result.get('approach', 'unknown')}")
            
            # Step 2: Execute based on plan
            enhanced_logger.step("Executing", f"Using {plan_result.get('approach', 'unknown')} approach")
            execution_result = await self._execute_plan(plan_result, task_description)
            enhanced_logger.agent_result("Execution", execution_result.get("success", False), "Task completed")
            
            # Step 3: Synthesize final result
            enhanced_logger.step("Synthesizing", "Creating final comprehensive answer")
            enhanced_logger.agent_thinking("Orchestrator", "Synthesizing all information")
            final_result = await self._synthesize_result(plan_result, execution_result, task_description)
            enhanced_logger.agent_result("Orchestrator", True, "Final answer ready")
            
            success = True
            
        except Exception as e:
            enhanced_logger.error("Task execution failed", e)
            final_result = f"Task failed: {str(e)}"
            plan_result = {"plan": "Failed to create plan", "approach": "error"}
            execution_result = {"result": f"Execution error: {str(e)}", "success": False}
            success = False
        
        # Step 4: Evaluate and evolve
        enhanced_logger.step("Learning", "Evaluating performance and evolving system")
        await self._evaluate_and_evolve(task_id, task_description, plan_result, execution_result, final_result, success)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result = {
            "task_id": task_id,
            "task_description": task_description,
            "success": success,
            "result": final_result,
            "execution_time": execution_time,
            "task_count": self.task_count
        }
        
        # Finish enhanced logging
        enhanced_logger.finish_task(success, execution_time, f"Task #{self.task_count}")
        
        return result
    
    async def _plan_task(self, task_description: str) -> Dict[str, Any]:
        """Let the orchestrator plan the task approach."""
        user_message = BaseMessage.make_user_message(
            role_name="User",
            content=f"Plan how to solve this task: {task_description}"
        )
        
        response = self.orchestrator_agent.step(user_message)
        if not response or not hasattr(response, 'msg') or not response.msg:
            raise Exception("Orchestrator failed to create plan")
        
        plan_content = response.msg.content
        
        # Determine approach based on plan
        approach = "expert"  # default
        if any(word in plan_content.lower() for word in ['code', 'python', 'calculate', 'compute', 'model', 'data']):
            approach = "analyst"
        if any(word in plan_content.lower() for word in ['both', 'expert', 'medical']) and approach == "analyst":
            approach = "collaborative"
        
        return {
            "plan": plan_content,
            "approach": approach
        }
    
    async def _execute_plan(self, plan_result: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Execute the plan using the appropriate agent(s)."""
        approach = plan_result["approach"]
        plan = plan_result["plan"]
        
        if approach == "analyst":
            return await self._execute_with_analyst(task_description, plan)
        elif approach == "expert":
            return await self._execute_with_expert(task_description, plan)
        elif approach == "collaborative":
            return await self._execute_collaborative(task_description, plan)
        else:
            return {"result": "Unknown approach", "success": False}
    
    async def _execute_with_analyst(self, task_description: str, plan: str) -> Dict[str, Any]:
        """Execute using analyst with ReAct capability."""
        enhanced_logger.agent_thinking("Analyst", "Computational analysis with ReAct approach")
        
        # Create ReAct agent for iterative problem solving
        react_agent = ReactAgent(
            chat_agent=self.analyst_agent, 
            max_rounds=self.evolution_config.get_system_param("max_react_rounds", 3)
        )
        
        context = f"Plan: {plan}\n\nUse your computational skills to solve this step by step."
        react_result = react_agent.solve_task(task_description, context)
        
        enhanced_logger.agent_action("Analyst", "ReAct complete", f"{react_result['rounds_used']}/{react_result['max_rounds']} rounds")
        
        # Update strategy performance
        self.evolution_config.update_strategy_performance("computational_routing", react_result["success"])
        
        return {
            "result": react_result["final_result"],
            "success": react_result["success"],
            "details": react_result
        }
    
    async def _execute_with_expert(self, task_description: str, plan: str) -> Dict[str, Any]:
        """Execute using medical expert."""
        enhanced_logger.agent_thinking("Expert", "Medical analysis and clinical reasoning")
        
        user_message = BaseMessage.make_user_message(
            role_name="User",
            content=f"Based on this plan: {plan}\n\nProvide your medical expertise for: {task_description}"
        )
        
        response = self.expert_agent.step(user_message)
        success = response is not None and hasattr(response, 'msg') and response.msg
        
        enhanced_logger.agent_action("Expert", "Medical analysis", "Complete" if success else "Failed")
        
        # Update strategy performance
        self.evolution_config.update_strategy_performance("medical_routing", success)
        
        if success:
            return {"result": response.msg.content, "success": True}
        else:
            return {"result": "Expert agent failed to respond", "success": False}
    
    async def _execute_collaborative(self, task_description: str, plan: str) -> Dict[str, Any]:
        """Execute using both expert and analyst."""
        # Get expert input first
        expert_result = await self._execute_with_expert(task_description, plan)
        
        # Then get analyst input
        analyst_context = f"Plan: {plan}\n\nExpert input: {expert_result['result']}\n\nNow provide computational analysis."
        analyst_result = await self._execute_with_analyst(task_description, analyst_context)
        
        success = expert_result["success"] and analyst_result["success"]
        
        # Update strategy performance
        self.evolution_config.update_strategy_performance("collaborative_approach", success)
        
        combined_result = f"Expert Analysis:\n{expert_result['result']}\n\nComputational Analysis:\n{analyst_result['result']}"
        
        return {
            "result": combined_result,
            "success": success,
            "expert_result": expert_result,
            "analyst_result": analyst_result
        }
    
    async def _synthesize_result(self, plan_result: Dict[str, Any], execution_result: Dict[str, Any], task_description: str) -> str:
        """Synthesize the final result."""
        synthesis_prompt = f"""
Original task: {task_description}
Plan: {plan_result['plan']}
Execution result: {execution_result['result']}

Provide a clear, concise final answer that directly addresses the original task.
        """
        
        user_message = BaseMessage.make_user_message(
            role_name="User",
            content=synthesis_prompt
        )
        
        response = self.orchestrator_agent.step(user_message)
        if response and hasattr(response, 'msg') and response.msg:
            return response.msg.content
        else:
            # Fallback to execution result
            return execution_result.get("result", "No result generated")
    
    async def _evaluate_and_evolve(self, task_id: str, task_description: str, 
                                  plan_result: Dict[str, Any], execution_result: Dict[str, Any], 
                                  final_result: str, success: bool):
        """Evaluate performance and evolve the system."""
        
        # Simple scoring based on success and result quality
        base_score = 8.0 if success else 3.0
        
        # Adjust based on result quality
        if success:
            if len(final_result) > 100 and "error" not in final_result.lower():
                base_score += 1.0
            if "```python" in final_result and "result" in final_result.lower():
                base_score += 0.5
        
        score = min(base_score, 10.0)
        
        # Check if we should evolve
        if self.evolution_config.should_evolve(self.task_count):
            logger.info("🧬 Evolution triggered - improving system...")
            await self._evolve_system(score, final_result, success)
        
        logger.info(f"📊 Task evaluation: Score {score:.1f}/10.0, Success: {success}")
    
    async def _evolve_system(self, score: float, result: str, success: bool):
        """Evolve system components based on performance."""
        feedback = f"Score: {score}, Success: {success}, Result quality: {'good' if success else 'needs improvement'}"
        
        # Evolve prompts if score is below threshold
        threshold = self.evolution_config.get_system_param("success_threshold", 7.5)
        
        if score < threshold:
            enhanced_logger.evolution_event("Prompt Evolution", f"Score {score:.1f} below threshold {threshold}")
            
            # Evolve each role's prompt
            for role in ["orchestrator", "expert", "analyst"]:
                current_prompt, _ = self.evolution_config.get_best_prompt(role)
                
                try:
                    # Use LLM to evolve the prompt
                    new_prompt = generate_evolved_prompt(current_prompt, feedback, score, self.orchestrator_agent)
                    
                    if new_prompt and new_prompt != current_prompt:
                        self.evolution_config.add_prompt_evolution(role, new_prompt, score + 1.0, feedback)
                        enhanced_logger.evolution_event("Prompt Updated", f"{role.title()} prompt evolved")
                
                except Exception as e:
                    enhanced_logger.warning(f"Failed to evolve {role} prompt: {e}")
            
            # Re-initialize agents with new prompts
            self._initialize_agents()
            enhanced_logger.evolution_event("Agents Reinitialized", "Using evolved prompts")
        
        else:
            enhanced_logger.evolution_event("Performance Good", f"Score {score:.1f} above threshold - no evolution needed")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        return {
            "task_count": self.task_count,
            "is_running": self.is_running,
            "prompt_versions": {
                role: len(prompts) for role, prompts in self.evolution_config.prompts.items()
            },
            "best_prompt_scores": {
                role: self.evolution_config.get_best_prompt(role)[1] 
                for role in ["orchestrator", "expert", "analyst"]
            },
            "strategy_performance": {
                name: strategy.success_rate 
                for name, strategy in self.evolution_config.strategies.items()
            }
        }