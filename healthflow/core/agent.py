"""
HealthFlow Agent System
Core agent implementation with self-evolution, multi-agent collaboration, and medical task capabilities
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .llm_provider import LLMProvider, LLMMessage, LLMResponse, create_llm_provider
from .memory import MemoryManager, MemoryEntry, MemoryType
from .config import HealthFlowConfig
from .rewards import calculate_mi_reward, calculate_final_reward
from ..tools.toolbank import ToolBank
from ..evaluation.evaluator import TaskEvaluator


class AgentRole(Enum):
    """Agent roles in the multi-agent system"""
    COORDINATOR = "coordinator"
    MEDICAL_EXPERT = "medical_expert"
    DATA_ANALYST = "data_analyst"
    RESEARCHER = "researcher"
    DIAGNOSIS_SPECIALIST = "diagnosis_specialist"
    TREATMENT_PLANNER = "treatment_planner"
    CODE_EXECUTOR = "code_executor"


@dataclass
class AgentMessage:
    """Message exchanged between agents"""
    sender_id: str
    receiver_id: str
    message_type: str  # "request", "response", "notification", "collaboration"
    content: Dict[str, Any]
    timestamp: datetime
    conversation_id: str
    priority: int = 1  # 1-5, higher is more urgent

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'conversation_id': self.conversation_id,
            'priority': self.priority
        }


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    success: bool
    result: Any
    error: Optional[str]
    execution_time: float
    agent_id: str
    tools_used: List[str]
    memory_entries: List[str]
    evaluation: Optional[Dict[str, Any]]


class HealthFlowAgent:
    """
    Core HealthFlow Agent with self-evolution and multi-agent capabilities
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        config: HealthFlowConfig,
        llm_provider: Optional[LLMProvider] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.config = config

        # Initialize LLM provider
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            self.llm_provider = create_llm_provider(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model_name=self.config.model_name
            )

        # Initialize components
        self.memory_manager = MemoryManager(
            config.memory_dir / agent_id,
            max_memory_size=config.memory_window
        )
        self.tool_bank = ToolBank(config.tools_dir / agent_id)
        self.evaluator = TaskEvaluator(config.evaluation_dir / agent_id, config=config)

        # Agent state
        self.is_active = True
        self.current_task = None
        self.conversation_history: List[AgentMessage] = []
        self.collaboration_network: Dict[str, 'HealthFlowAgent'] = {}

        # Performance tracking
        self.task_history: List[TaskResult] = []
        self.success_rate = 1.0
        self.specialization_areas: List[str] = []

        # Self-evolution parameters
        self.experience_threshold = 10  # Tasks before evolution
        self.evolution_generation = 1

        # Initialize system prompts based on role
        self.system_prompts = self._initialize_system_prompts()

    async def initialize(self):
        """Initialize agent components"""
        await self.memory_manager.initialize()
        await self.tool_bank.initialize()

        # Load agent-specific specializations from memory
        await self._load_specializations()

    def _initialize_system_prompts(self) -> Dict[str, str]:
        """Initialize role-specific system prompts"""
        base_prompt = f"""
You are a {self.role.value} agent in the HealthFlow medical AI system.
Your ID is {self.agent_id}.

Core capabilities:
- Medical knowledge reasoning and analysis
- Tool creation and usage
- Code execution and data processing
- Multi-agent collaboration
- Self-improvement through experience

Always prioritize patient safety, medical accuracy, and evidence-based practice.
Communicate clearly with other agents and maintain detailed records of your reasoning.
"""

        role_specific_prompts = {
            AgentRole.COORDINATOR: base_prompt + """
As a Coordinator, you orchestrate multi-agent collaborations, assign tasks to appropriate specialists,
and ensure efficient workflow management. You have a broad overview of all medical domains.
""",
            AgentRole.MEDICAL_EXPERT: base_prompt + """
As a Medical Expert, you provide specialized medical knowledge, clinical reasoning,
and evidence-based recommendations. You stay current with medical literature and best practices.
""",
            AgentRole.DATA_ANALYST: base_prompt + """
As a Data Analyst, you specialize in processing medical data, statistical analysis,
pattern recognition, and generating insights from complex healthcare datasets.
""",
            AgentRole.RESEARCHER: base_prompt + """
As a Researcher, you conduct literature reviews, analyze clinical studies,
and provide evidence-based insights to support medical decision making.
""",
            AgentRole.DIAGNOSIS_SPECIALIST: base_prompt + """
As a Diagnosis Specialist, you excel at differential diagnosis, symptom analysis,
and using diagnostic tools to identify medical conditions accurately.
""",
            AgentRole.TREATMENT_PLANNER: base_prompt + """
As a Treatment Planner, you develop comprehensive treatment strategies,
consider drug interactions, and create personalized care plans.
""",
            AgentRole.CODE_EXECUTOR: base_prompt + """
As a Code Executor, you run medical analysis code, create visualizations,
process data, and integrate computational tools into medical workflows.
"""
        }

        return {
            "base": role_specific_prompts.get(self.role, base_prompt),
            "task_execution": "Focus on executing the given task efficiently and accurately.",
            "collaboration": "Collaborate effectively with other agents, sharing relevant insights.",
            "self_reflection": "Reflect on your performance and identify areas for improvement."
        }

    async def execute_task(
        self,
        task_description: str,
        task_context: Dict[str, Any] = None,
        max_iterations: int = None
    ) -> TaskResult:
        """Execute a medical task with self-evolution capabilities"""

        task_id = str(uuid.uuid4())
        task_context = task_context or {}
        max_iterations = max_iterations or self.config.max_iterations

        start_time = datetime.now()
        tools_used = []
        memory_entries_created = []

        try:
            self.current_task = task_id

            # Store initial task memory
            initial_memory = MemoryEntry(
                id=str(uuid.uuid4()),
                memory_type=MemoryType.INTERACTION,
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                content={
                    "task_id": task_id,
                    "task_description": task_description,
                    "task_context": task_context,
                    "status": "started"
                }
            )
            await self.memory_manager.add_memory(initial_memory)
            memory_entries_created.append(initial_memory.id)

            # Retrieve relevant experiences
            relevant_memories = await self.memory_manager.get_recent_memories(
                limit=20,
                memory_type=MemoryType.EXPERIENCE
            )

            # Plan task execution approach
            execution_plan = await self._plan_task_execution(
                task_description, task_context, relevant_memories
            )

            result = None
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Execute current step
                step_result = await self._execute_task_step(
                    task_description,
                    task_context,
                    execution_plan,
                    iteration,
                    relevant_memories
                )

                if step_result["tools_used"]:
                    tools_used.extend(step_result["tools_used"])

                if step_result["memory_entries"]:
                    memory_entries_created.extend(step_result["memory_entries"])

                # Check if task is complete
                if step_result["completed"]:
                    result = step_result["result"]
                    break

                # Update execution plan if needed
                if step_result.get("plan_update"):
                    execution_plan = step_result["plan_update"]

            # Evaluate result
            evaluation = await self.evaluator.evaluate_task_result(
                task_description, result, task_context
            )

            success = evaluation.get("success", False)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Store final result memory
            result_memory = MemoryEntry(
                id=str(uuid.uuid4()),
                memory_type=MemoryType.EXPERIENCE,
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                content={
                    "task_id": task_id,
                    "task_description": task_description,
                    "result": result,
                    "evaluation": evaluation,
                    "tools_used": tools_used,
                    "iterations": iteration,
                    "execution_time": execution_time
                },
                success=success,
                reward=evaluation.get("reward", 0.0)
            )
            await self.memory_manager.add_memory(result_memory)
            memory_entries_created.append(result_memory.id)

            # Create task result
            task_result = TaskResult(
                task_id=task_id,
                success=success,
                result=result,
                error=None,
                execution_time=execution_time,
                agent_id=self.agent_id,
                tools_used=tools_used,
                memory_entries=memory_entries_created,
                evaluation=evaluation
            )

            # Update agent statistics
            self.task_history.append(task_result)
            await self._update_success_rate()

            # Trigger self-evolution if needed
            if len(self.task_history) >= self.experience_threshold:
                await self._trigger_self_evolution()

            self.current_task = None
            return task_result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            error_memory = MemoryEntry(
                id=str(uuid.uuid4()),
                memory_type=MemoryType.FAILURE_ANALYSIS,
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                content={
                    "task_id": task_id,
                    "task_description": task_description,
                    "error": str(e),
                    "execution_time": execution_time
                },
                success=False
            )
            await self.memory_manager.add_memory(error_memory)
            memory_entries_created.append(error_memory.id)

            task_result = TaskResult(
                task_id=task_id,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                agent_id=self.agent_id,
                tools_used=tools_used,
                memory_entries=memory_entries_created,
                evaluation=None
            )

            self.task_history.append(task_result)
            await self._update_success_rate()

            self.current_task = None
            return task_result

    async def _plan_task_execution(
        self,
        task_description: str,
        task_context: Dict[str, Any],
        relevant_memories: List[MemoryEntry]
    ) -> Dict[str, Any]:
        """Plan the execution approach for a task"""

        # Get the best prompt from memory evolution
        best_prompt = await self.memory_manager.get_best_prompt("task_planning")
        if not best_prompt:
            best_prompt = self.system_prompts["task_execution"]

        # Create planning prompt
        memory_context = "\n".join([
            f"- {mem.content.get('task_description', '')}: {'Success' if mem.success else 'Failure'}"
            for mem in relevant_memories[:5]
        ])

        planning_prompt = f"""
{best_prompt}

Task: {task_description}
Context: {json.dumps(task_context, indent=2)}

Relevant past experiences:
{memory_context}

Create an execution plan with:
1. Approach strategy
2. Required tools or capabilities
3. Potential challenges
4. Success criteria
5. Collaboration needs (if any)

Return your plan as a JSON object.
"""

        try:
            messages = [LLMMessage(role="user", content=planning_prompt)]
            response = await self.llm_provider.generate(
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )

            # Try to parse JSON response
            try:
                plan = json.loads(response.content)
                return plan
            except json.JSONDecodeError:
                # Extract structured information if JSON parsing fails
                return {
                    "approach": "adaptive",
                    "reasoning": response.content,
                    "tools_needed": [],
                    "collaboration_needed": False
                }

        except Exception as e:
            return {
                "approach": "fallback",
                "error": str(e),
                "tools_needed": [],
                "collaboration_needed": False
            }

    async def _execute_task_step(
        self,
        task_description: str,
        task_context: Dict[str, Any],
        execution_plan: Dict[str, Any],
        iteration: int,
        relevant_memories: List[MemoryEntry]
    ) -> Dict[str, Any]:
        """Execute a single step of task execution"""

        step_prompt = f"""
{self.system_prompts["base"]}

Current task: {task_description}
Execution plan: {json.dumps(execution_plan, indent=2)}
Iteration: {iteration}

Based on the plan and your role as {self.role.value}, execute the next step.
If you need specific tools, indicate what tools are required.
If you need to collaborate with other agents, specify the collaboration request.

Provide your response in this format:
{{
    "action": "description of action taken",
    "result": "result or output",
    "completed": true/false,
    "tools_needed": ["tool1", "tool2"],
    "collaboration_request": {{
        "agent_role": "role_needed",
        "request": "what help is needed"
    }},
    "next_steps": "what to do next if not completed"
}}
"""

        try:
            messages = [LLMMessage(role="user", content=step_prompt)]
            response = await self.llm_provider.generate(
                messages=messages,
                max_tokens=1500,
                temperature=0.5
            )

            # Try to parse structured response
            try:
                step_result = json.loads(response.content)
            except json.JSONDecodeError:
                step_result = {
                    "action": "reasoning",
                    "result": response.content,
                    "completed": False,
                    "tools_needed": [],
                    "next_steps": "continue analysis"
                }

            # Handle tool usage
            tools_used = []
            if step_result.get("tools_needed"):
                for tool_name in step_result["tools_needed"]:
                    # Check if tool exists or needs to be created
                    if await self._handle_tool_requirement(tool_name, task_context):
                        tools_used.append(tool_name)

            # Handle collaboration request
            if step_result.get("collaboration_request"):
                await self._handle_collaboration_request(
                    step_result["collaboration_request"],
                    task_description,
                    task_context
                )

            # Store step memory
            step_memory = MemoryEntry(
                id=str(uuid.uuid4()),
                memory_type=MemoryType.INTERACTION,
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                content={
                    "step_iteration": iteration,
                    "action": step_result.get("action"),
                    "result": step_result.get("result"),
                    "tools_used": tools_used
                }
            )
            await self.memory_manager.add_memory(step_memory)

            return {
                "completed": step_result.get("completed", False),
                "result": step_result.get("result"),
                "tools_used": tools_used,
                "memory_entries": [step_memory.id],
                "plan_update": step_result.get("plan_update")
            }

        except Exception as e:
            return {
                "completed": False,
                "result": f"Error in step execution: {str(e)}",
                "tools_used": [],
                "memory_entries": [],
                "error": str(e)
            }

    async def _handle_tool_requirement(
        self,
        tool_name: str,
        task_context: Dict[str, Any]
    ) -> bool:
        """Handle tool requirement - find existing or create new tool"""

        # Search for existing tools
        existing_tools = self.tool_bank.search_tools(
            query=tool_name,
            min_success_rate=0.5
        )

        if existing_tools:
            return True

        # Tool doesn't exist - create it
        try:
            # Use LLM to generate tool specification
            tool_spec = await self._generate_tool_specification(tool_name, task_context)

            if tool_spec:
                tool_id = await self.tool_bank.create_python_tool(
                    name=tool_name,
                    description=tool_spec["description"],
                    implementation=tool_spec["implementation"],
                    parameters=tool_spec["parameters"],
                    tags=tool_spec.get("tags", [])
                )

                # Store tool creation memory
                tool_memory = MemoryEntry(
                    id=str(uuid.uuid4()),
                    memory_type=MemoryType.TOOL_CREATION,
                    timestamp=datetime.now(),
                    agent_id=self.agent_id,
                    content={
                        "tool_name": tool_name,
                        "tool_id": tool_id,
                        "specification": tool_spec,
                        "created_for_task": task_context.get("task_id")
                    },
                    success=True
                )
                await self.memory_manager.add_memory(tool_memory)

                return True

        except Exception as e:
            print(f"Error creating tool {tool_name}: {e}")

        return False

    async def _generate_tool_specification(
        self,
        tool_name: str,
        task_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate specification for a new tool"""

        spec_prompt = f"""
Create a Python tool specification for: {tool_name}

Context: This tool is needed for a medical task in the HealthFlow system.
Task context: {json.dumps(task_context, indent=2)}

Generate a complete tool specification with:
1. description: Clear description of what the tool does
2. implementation: Complete Python code with a main() function
3. parameters: Dict describing input parameters and types
4. tags: List of relevant tags

The tool should be focused, reliable, and follow medical best practices.
Return as JSON format.
"""

        try:
            messages = [LLMMessage(role="user", content=spec_prompt)]
            response = await self.llm_provider.generate(
                messages=messages,
                max_tokens=2000,
                temperature=0.3
            )

            return json.loads(response.content)
        except Exception as e:
            print(f"Error generating tool specification: {e}")
            return None

    async def _handle_collaboration_request(
        self,
        collaboration_request: Dict[str, Any],
        task_description: str,
        task_context: Dict[str, Any]
    ):
        """Handle request for collaboration with other agents"""

        requested_role = collaboration_request.get("agent_role")
        request_content = collaboration_request.get("request")

        # Find appropriate agent for collaboration
        collaborator = None
        for agent in self.collaboration_network.values():
            if agent.role.value == requested_role and agent.is_active:
                collaborator = agent
                break

        if collaborator:
            # Send collaboration message
            message = AgentMessage(
                sender_id=self.agent_id,
                receiver_id=collaborator.agent_id,
                message_type="collaboration",
                content={
                    "original_task": task_description,
                    "collaboration_request": request_content,
                    "context": task_context
                },
                timestamp=datetime.now(),
                conversation_id=str(uuid.uuid4()),
                priority=2
            )

            await collaborator.receive_message(message)

    async def receive_message(self, message: AgentMessage):
        """Receive and process message from another agent"""

        self.conversation_history.append(message)

        # Store collaboration memory
        collab_memory = MemoryEntry(
            id=str(uuid.uuid4()),
            memory_type=MemoryType.INTERACTION,
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            content={
                "message_type": message.message_type,
                "sender": message.sender_id,
                "content": message.content,
                "conversation_id": message.conversation_id
            }
        )
        await self.memory_manager.add_memory(collab_memory)

        # Process based on message type
        if message.message_type == "collaboration":
            await self._process_collaboration_request(message)
        elif message.message_type == "request":
            await self._process_request(message)
        elif message.message_type == "response":
            await self._process_response(message)

    async def _process_collaboration_request(self, message: AgentMessage):
        """Process collaboration request from another agent"""

        request_content = message.content.get("collaboration_request")
        context = message.content.get("context", {})

        # Generate response to collaboration request
        collab_prompt = f"""
{self.system_prompts["collaboration"]}

You received a collaboration request from agent {message.sender_id}:
Request: {request_content}
Context: {json.dumps(context, indent=2)}

Based on your role as {self.role.value}, provide helpful assistance.
Be specific and actionable in your response.
"""

        try:
            messages = [LLMMessage(role="user", content=collab_prompt)]
            response = await self.llm_provider.generate(
                messages=messages,
                max_tokens=1000,
                temperature=0.4
            )

            # Send response back
            response_message = AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="response",
                content={
                    "collaboration_response": response.content,
                    "original_request": request_content
                },
                timestamp=datetime.now(),
                conversation_id=message.conversation_id,
                priority=message.priority
            )

            # Find sender agent and send response
            sender_agent = self.collaboration_network.get(message.sender_id)
            if sender_agent:
                await sender_agent.receive_message(response_message)

        except Exception as e:
            print(f"Error processing collaboration request: {e}")

    async def _trigger_self_evolution(self):
        """Trigger self-evolution based on accumulated experience"""

        # Analyze recent performance
        recent_tasks = self.task_history[-self.experience_threshold:]
        success_count = sum(1 for task in recent_tasks if task.success)
        success_rate = success_count / len(recent_tasks)

        # Get successful and failed experiences
        successful_experiences = await self.memory_manager.get_successful_experiences(limit=20)
        failed_experiences = await self.memory_manager.get_failed_experiences(limit=10)

        # Extract patterns
        if successful_experiences:
            success_pattern = await self.memory_manager.extract_experience_pattern(successful_experiences)

        if failed_experiences:
            failure_pattern = await self.memory_manager.extract_experience_pattern(failed_experiences)

        # Evolve prompts based on experience
        if success_rate < 0.7:  # If performance is below threshold
            await self._evolve_prompts(successful_experiences, failed_experiences)

        # Update specialization areas
        await self._update_specializations()

        self.evolution_generation += 1

        # Store evolution memory
        evolution_memory = MemoryEntry(
            id=str(uuid.uuid4()),
            memory_type=MemoryType.PROMPT_EVOLUTION,
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            content={
                "generation": self.evolution_generation,
                "success_rate": success_rate,
                "improvements": "Evolved prompts and specializations based on experience"
            },
            success=True
        )
        await self.memory_manager.add_memory(evolution_memory)

    async def _evolve_prompts(
        self,
        successful_experiences: List[MemoryEntry],
        failed_experiences: List[MemoryEntry]
    ):
        """Evolve agent prompts based on experience"""

        # Analyze what works and what doesn't
        success_patterns = [exp.content for exp in successful_experiences[:5]]
        failure_patterns = [exp.content for exp in failed_experiences[:3]]

        evolution_prompt = f"""
Analyze the following successful and failed task experiences to improve agent performance.

Successful experiences:
{json.dumps(success_patterns, indent=2)}

Failed experiences:
{json.dumps(failure_patterns, indent=2)}

Based on this analysis, suggest improvements to the agent's approach:
1. What patterns lead to success?
2. What should be avoided?
3. How can the agent's reasoning be improved?
4. What additional capabilities might be needed?

Provide specific improvements as a structured response.
"""

        try:
            messages = [LLMMessage(role="user", content=evolution_prompt)]
            response = await self.llm_provider.generate(
                messages=messages,
                max_tokens=1500,
                temperature=0.3
            )

            # Store evolved prompt
            performance_metrics = {"success_rate": self.success_rate}
            improvements = [response.content]

            await self.memory_manager.evolve_prompt(
                current_prompt=self.system_prompts["base"],
                performance_metrics=performance_metrics,
                improvements=improvements
            )

        except Exception as e:
            print(f"Error evolving prompts: {e}")

    async def _update_specializations(self):
        """Update agent specialization areas based on successful tasks"""

        # Analyze successful tasks to identify specialization patterns
        successful_tasks = [task for task in self.task_history if task.success]

        if len(successful_tasks) >= 5:
            # Extract common patterns from successful tasks
            task_descriptions = [task.result for task in successful_tasks[-10:]]
            # Simple keyword extraction for specialization areas
            # In a real implementation, this would use more sophisticated NLP

            specialization_areas = []
            medical_keywords = [
                'diagnosis', 'treatment', 'analysis', 'research', 'planning',
                'cardiology', 'oncology', 'neurology', 'radiology'
            ]

            for keyword in medical_keywords:
                count = sum(1 for desc in task_descriptions
                           if keyword.lower() in str(desc).lower())
                if count >= 3:  # If keyword appears in multiple successful tasks
                    specialization_areas.append(keyword)

            self.specialization_areas = specialization_areas[:5]  # Keep top 5

    async def _load_specializations(self):
        """Load agent specializations from memory"""

        # Load previous specializations from memory
        specialization_memories = await self.memory_manager.get_recent_memories(
            limit=5,
            memory_type=MemoryType.SUCCESS_PATTERN
        )

        for memory in specialization_memories:
            if 'specializations' in memory.content:
                self.specialization_areas.extend(memory.content['specializations'])

        # Remove duplicates
        self.specialization_areas = list(set(self.specialization_areas))

    async def _update_success_rate(self):
        """Update agent success rate with exponential moving average"""

        if not self.task_history:
            return

        recent_task = self.task_history[-1]
        alpha = 0.1  # Learning rate

        if recent_task.success:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 1.0
        else:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 0.0

    def add_collaborator(self, agent: 'HealthFlowAgent'):
        """Add another agent to collaboration network"""
        self.collaboration_network[agent.agent_id] = agent

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "is_active": self.is_active,
            "current_task": self.current_task,
            "success_rate": self.success_rate,
            "total_tasks": len(self.task_history),
            "successful_tasks": sum(1 for task in self.task_history if task.success),
            "specialization_areas": self.specialization_areas,
            "evolution_generation": self.evolution_generation,
            "collaboration_network_size": len(self.collaboration_network)
        }