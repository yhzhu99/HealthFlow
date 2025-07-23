import logging
import uuid
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.messages import BaseMessage
from camel.interpreters import InternalPythonInterpreter

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
        self.interpreter = InternalPythonInterpreter()  # Add Python interpreter
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
            
            # For DeepSeek API compatibility, disable function calling tools
            # Instead, we'll use a prompt-based approach for Python code execution
            self.analyst_agent = ChatAgent(
                system_message=analyst_sys_msg,
                model=model,
                token_limit=4096
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
        This implements proper multi-agent collaboration: Orchestrator → Expert/Analyst → Final synthesis.
        """
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        logger.info(f"Running task [{task_id}]: {task_description}")

        # ACTION: Execute the task with proper agent collaboration
        conversation_trace = []

        try:
            # STEP 1: Orchestrator analyzes the task and creates a plan
            logger.info("Step 1: Orchestrator analyzing task and creating plan...")
            orchestrator_input = BaseMessage.make_user_message(
                role_name="User",
                content=f"Analyze this task and create a detailed plan: {task_description}"
            )

            orchestrator_response = self.orchestrator_agent.step(orchestrator_input)
            if orchestrator_response and hasattr(orchestrator_response, 'msg') and orchestrator_response.msg:
                conversation_trace.append(orchestrator_response.msg)
                orchestrator_plan = orchestrator_response.msg.content
                logger.info(f"Orchestrator created plan: {orchestrator_plan[:100]}...")
            else:
                raise Exception("Orchestrator failed to create a plan")

            # STEP 2: Determine which specialist agent(s) to use based on the task
            needs_medical_expert = any(word in task_description.lower() for word in [
                'medical', 'disease', 'symptom', 'diagnosis', 'treatment', 'health', 
                'medication', 'patient', 'clinical', 'hypertension', 'diabetes'
            ])
            
            needs_analyst = any(word in task_description.lower() for word in [
                'calculate', 'compute', 'analyze', 'data', 'python', 'code', 'math',
                'statistics', 'plot', 'graph', 'number'
            ])

            specialist_results = []

            # STEP 3: Engage Expert Agent if needed
            if needs_medical_expert:
                logger.info("Step 3a: Engaging Expert Agent for medical expertise...")
                expert_input = BaseMessage.make_user_message(
                    role_name="Orchestrator",
                    content=f"Based on this plan: {orchestrator_plan}\n\nPlease provide your medical expertise for: {task_description}"
                )
                
                expert_response = self.expert_agent.step(expert_input)
                if expert_response and hasattr(expert_response, 'msg') and expert_response.msg:
                    conversation_trace.append(expert_response.msg)
                    specialist_results.append(f"Medical Expert Analysis:\n{expert_response.msg.content}")
                    logger.info("Expert provided medical analysis")

            # STEP 4: Engage Analyst Agent if needed
            if needs_analyst:
                logger.info("Step 4a: Engaging Analyst Agent for computational tasks...")
                analyst_input = BaseMessage.make_user_message(
                    role_name="Orchestrator", 
                    content=f"Based on this plan: {orchestrator_plan}\n\nPlease perform the analytical/computational work for: {task_description}"
                )
                
                analyst_response = self.analyst_agent.step(analyst_input)
                if analyst_response and hasattr(analyst_response, 'msg') and analyst_response.msg:
                    # Execute any Python code in the analyst response
                    enhanced_content = self._execute_python_code_in_response(analyst_response.msg.content)
                    analyst_response.msg.content = enhanced_content
                    
                    conversation_trace.append(analyst_response.msg)
                    specialist_results.append(f"Data Analyst Results:\n{enhanced_content}")
                    logger.info("Analyst provided computational analysis with code execution")

            # STEP 5: If no specialist was clearly needed, use the most appropriate agent
            if not specialist_results:
                logger.info("Step 5: Using Analyst as default for general task...")
                analyst_input = BaseMessage.make_user_message(
                    role_name="User",
                    content=task_description
                )
                
                analyst_response = self.analyst_agent.step(analyst_input)
                if analyst_response and hasattr(analyst_response, 'msg') and analyst_response.msg:
                    # Execute any Python code in the analyst response
                    enhanced_content = self._execute_python_code_in_response(analyst_response.msg.content)
                    analyst_response.msg.content = enhanced_content
                    
                    conversation_trace.append(analyst_response.msg)
                    specialist_results.append(f"General Analysis:\n{enhanced_content}")

            # STEP 6: Orchestrator synthesizes final result
            logger.info("Step 6: Orchestrator synthesizing final result...")
            synthesis_content = f"""
            Original task: {task_description}
            Initial plan: {orchestrator_plan}
            
            Specialist findings:
            {chr(10).join(specialist_results)}
            
            Please provide a comprehensive final answer that synthesizes all the above information."""

            synthesis_input = BaseMessage.make_user_message(
                role_name="User", 
                content=synthesis_content
            )
            
            final_response = self.orchestrator_agent.step(synthesis_input)
            if final_response and hasattr(final_response, 'msg') and final_response.msg:
                conversation_trace.append(final_response.msg)
                final_result = final_response.msg.content
                logger.info("Orchestrator provided final synthesis")
            else:
                # Fallback to the last specialist result if synthesis fails
                final_result = specialist_results[-1] if specialist_results else "No result generated"

        except Exception as e:
            logger.error(f"Error during task execution: {e}")
            final_result = f"Task execution failed: {str(e)}"
            
            # Create an error message for the trace
            error_msg = BaseMessage.make_assistant_message(
                role_name="System",
                content=final_result
            )
            conversation_trace.append(error_msg)

        # EVALUATION: Analyze the conversation trace
        logger.info("Evaluating task execution...")
        evaluation_result = self._create_evaluation_result(final_result, conversation_trace)

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
                        tools_used.append(tool_call.function.get('name', 'unknown'))
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
            "conversation_trace_length": len(conversation_trace)
        }

    def _create_evaluation_result(self, final_result: str, conversation_trace: List[Any]):
        """Create a mock evaluation result based on the task outcome."""
        from types import SimpleNamespace
        
        # Determine success based on result quality
        has_error = "failed" in final_result.lower() or "error" in final_result.lower()
        has_content = len(final_result.strip()) > 20
        has_collaboration = len(conversation_trace) > 1
        
        success = not has_error and has_content
        score = 8.5 if success else 3.0
        
        # Adjust score based on collaboration quality
        if has_collaboration:
            score += 0.5
            
        evaluation_result = SimpleNamespace()
        evaluation_result.overall_score = min(score, 10.0)
        evaluation_result.overall_success = success
        evaluation_result.reasoning_quality = score - 0.5 if success else 3.0
        evaluation_result.tool_usage_effectiveness = 9.0 if success and "tool" in str(conversation_trace) else 5.0
        evaluation_result.response_completeness = score if success else 3.0
        evaluation_result.safety_compliance = 10.0
        evaluation_result.summary = "Task completed successfully with agent collaboration." if success else "Task execution encountered issues."
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
        
        return evaluation_result

    def _execute_python_code_in_response(self, response_content: str) -> str:
        """
        Process analyst responses to find and execute Python code blocks.
        Returns the response with executed code results.
        """
        # Find Python code blocks in the response
        code_pattern = r"```python\n(.*?)\n```"
        matches = re.findall(code_pattern, response_content, re.DOTALL)
        
        if not matches:
            return response_content
        
        enhanced_response = response_content
        
        for code_block in matches:
            try:
                # Execute the Python code
                result = self.interpreter.run(code_block, "python")
                
                # Replace the code block with code + executed result
                executed_block = f"```python\n{code_block}\n```\n**Executed Result:**\n```\n{result}\n```"
                
                # Find and replace the original code block
                original_block = f"```python\n{code_block}\n```"
                enhanced_response = enhanced_response.replace(original_block, executed_block)
                
                logger.info(f"Executed Python code successfully, result: {result}")
                
            except Exception as e:
                # If execution fails, add error message
                error_msg = f"**Code Execution Error:** {str(e)}"
                original_block = f"```python\n{code_block}\n```"
                executed_block = f"{original_block}\n{error_msg}"
                enhanced_response = enhanced_response.replace(original_block, executed_block)
                
                logger.warning(f"Python code execution failed: {e}")
        
        return enhanced_response

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