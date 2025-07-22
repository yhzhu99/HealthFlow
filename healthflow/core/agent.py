"""
Core HealthFlow Agent - The main orchestrator for self-evolving healthcare agent system.

Integrates experience accumulation, memory management, sensitive data protection,
and multi-agent collaboration to create a continuously improving healthcare agent.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from .memory import MemoryManager
from .evolution import ExperienceAccumulator
from .security import DataProtector
from ..tools.toolbank import ToolBank
from ..evaluation.evaluator import TaskEvaluator


@dataclass
class TaskResult:
    """Represents the result of a task execution"""
    task_id: str
    success: bool
    output: Any
    feedback: str
    execution_time: float
    memory_used: int
    tools_used: List[str]
    timestamp: datetime
    error_msg: Optional[str] = None


@dataclass
class AgentState:
    """Current state of the agent"""
    agent_id: str
    active: bool
    current_task: Optional[str]
    experience_level: int
    specialized_domains: List[str]
    performance_metrics: Dict[str, float]


class HealthFlowAgent:
    """
    Main HealthFlow Agent that orchestrates the self-evolving healthcare system.
    
    Key features:
    - Experience accumulation and continuous learning
    - Multi-agent collaboration with role assignment
    - Sensitive healthcare data protection
    - Dynamic tool creation and management
    - Memory management for long-term context
    """
    
    def __init__(
        self,
        agent_id: str,
        openai_api_key: str,
        specialized_domains: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.openai_api_key = openai_api_key
        self.config = config or {}
        
        # Initialize core components
        self.memory_manager = MemoryManager(agent_id)
        self.experience_accumulator = ExperienceAccumulator(agent_id)
        self.data_protector = DataProtector()
        self.toolbank = ToolBank(openai_api_key)
        self.evaluator = TaskEvaluator()
        
        # Agent state
        self.state = AgentState(
            agent_id=agent_id,
            active=True,
            current_task=None,
            experience_level=0,
            specialized_domains=specialized_domains or [],
            performance_metrics={}
        )
        
        # Task history
        self.task_history: List[TaskResult] = []
        self.active_collaborations: Dict[str, List[str]] = {}
        
        self.logger = logging.getLogger(f"HealthFlowAgent-{agent_id}")
        self.logger.info(f"HealthFlow Agent {agent_id} initialized")
    
    async def execute_task(
        self,
        task_description: str,
        task_type: str,
        data: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """
        Execute a healthcare task with full self-evolution capabilities.
        
        Args:
            task_description: Natural language description of the task
            task_type: Type of task (diagnosis, analysis, research, etc.)
            data: Input data (will be protected if sensitive)
            context: Additional context for the task
            
        Returns:
            TaskResult with execution details and outcomes
        """
        start_time = datetime.now()
        task_id = f"{self.agent_id}_{start_time.isoformat()}"
        
        self.logger.info(f"Starting task {task_id}: {task_description}")
        self.state.current_task = task_id
        
        try:
            # Step 1: Protect sensitive data
            protected_data = await self.data_protector.protect_data(data) if data else None
            
            # Step 2: Retrieve relevant memories and experiences
            relevant_memories = await self.memory_manager.retrieve_relevant_memories(
                task_description, task_type
            )
            
            # Step 3: Get experience-based recommendations
            experience_insights = await self.experience_accumulator.get_task_insights(
                task_type, task_description
            )
            
            # Step 4: Identify and create required tools
            required_tools = await self.toolbank.identify_required_tools(
                task_description, task_type, experience_insights
            )
            
            created_tools = []
            for tool_spec in required_tools:
                if not await self.toolbank.tool_exists(tool_spec['name']):
                    new_tool = await self.toolbank.create_tool(tool_spec)
                    created_tools.append(new_tool)
            
            # Step 5: Execute the task using available tools
            execution_context = {
                'task_id': task_id,
                'memories': relevant_memories,
                'experience_insights': experience_insights,
                'available_tools': await self.toolbank.get_available_tools(),
                'protected_data': protected_data,
                **(context or {})
            }
            
            result = await self._execute_with_tools(
                task_description, task_type, execution_context
            )
            
            # Step 6: Evaluate performance
            evaluation = await self.evaluator.evaluate_task_result(
                task_description, result, task_type
            )
            
            # Step 7: Accumulate experience
            await self.experience_accumulator.add_experience(
                task_type=task_type,
                task_description=task_description,
                execution_context=execution_context,
                result=result,
                evaluation=evaluation,
                tools_used=created_tools + [t for t in required_tools if t.get('existing')]
            )
            
            # Step 8: Update memory
            await self.memory_manager.store_task_memory(
                task_id=task_id,
                task_description=task_description,
                result=result,
                evaluation=evaluation,
                context=execution_context
            )
            
            # Step 9: Create task result
            execution_time = (datetime.now() - start_time).total_seconds()
            task_result = TaskResult(
                task_id=task_id,
                success=evaluation.get('success', False),
                output=result,
                feedback=evaluation.get('feedback', ''),
                execution_time=execution_time,
                memory_used=len(json.dumps(asdict(self.state))),
                tools_used=[t['name'] for t in created_tools],
                timestamp=start_time
            )
            
            # Step 10: Update agent state
            self._update_agent_state(task_result, evaluation)
            
            self.task_history.append(task_result)
            self.state.current_task = None
            
            self.logger.info(f"Task {task_id} completed successfully")
            return task_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_result = TaskResult(
                task_id=task_id,
                success=False,
                output=None,
                feedback=f"Task failed: {str(e)}",
                execution_time=execution_time,
                memory_used=0,
                tools_used=[],
                timestamp=start_time,
                error_msg=str(e)
            )
            
            self.task_history.append(error_result)
            self.state.current_task = None
            
            self.logger.error(f"Task {task_id} failed: {e}")
            return error_result
    
    async def _execute_with_tools(
        self,
        task_description: str,
        task_type: str,
        context: Dict[str, Any]
    ) -> Any:
        """Execute the actual task using available tools and context"""
        
        # This is where the main LLM reasoning and tool orchestration happens
        # For now, we'll implement a basic structure that can be expanded
        
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=self.openai_api_key)
        
        # Construct system prompt with context
        system_prompt = self._build_system_prompt(context)
        
        # Create messages with task description
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Execute this healthcare task: {task_description}"}
        ]
        
        # Add relevant memories to context
        if context.get('memories'):
            memory_context = "Relevant past experiences:\n" + "\n".join(
                [f"- {mem['summary']}" for mem in context['memories'][:5]]
            )
            messages.append({"role": "assistant", "content": memory_context})
        
        # Execute with OpenAI
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        
        # Post-process result if needed
        processed_result = await self._post_process_result(result, context)
        
        return processed_result
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build comprehensive system prompt with all available context"""
        
        prompt_parts = [
            "You are HealthFlow, an advanced self-evolving healthcare AI agent.",
            "You have access to a comprehensive toolbank, past experiences, and memory.",
            "\nCORE CAPABILITIES:",
            "- Medical reasoning and diagnosis support",
            "- EHR analysis and pattern recognition", 
            "- Drug discovery and repurposing",
            "- Clinical decision support",
            "- Medical literature analysis",
            "- Sensitive data protection compliance",
            "\nEXPERIENCE INSIGHTS:"
        ]
        
        # Add experience insights
        if context.get('experience_insights'):
            for insight in context['experience_insights'][:3]:
                prompt_parts.append(f"- {insight}")
        
        prompt_parts.extend([
            "\nAVAILABLE TOOLS:"
        ])
        
        # Add available tools
        if context.get('available_tools'):
            for tool in context['available_tools'][:10]:  # Limit to prevent context overflow
                prompt_parts.append(f"- {tool['name']}: {tool['description']}")
        
        prompt_parts.extend([
            "\nIMPORTANT:",
            "- Always protect patient privacy and sensitive data",
            "- Use evidence-based approaches",
            "- Provide clear reasoning for recommendations",
            "- Flag any data that appears sensitive or personal",
            "- Continuously learn from each interaction"
        ])
        
        return "\n".join(prompt_parts)
    
    async def _post_process_result(self, result: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process LLM result to extract structured information"""
        
        # For now, return a structured format
        # This can be enhanced to parse LLM output more intelligently
        
        return {
            "response": result,
            "reasoning": "Generated based on available context and tools",
            "confidence": 0.8,  # This should be computed based on actual factors
            "data_protection_applied": bool(context.get('protected_data')),
            "tools_utilized": len(context.get('available_tools', [])),
            "memory_references": len(context.get('memories', []))
        }
    
    def _update_agent_state(self, task_result: TaskResult, evaluation: Dict[str, Any]):
        """Update agent state based on task performance"""
        
        # Update experience level
        if task_result.success:
            self.state.experience_level += 1
        
        # Update performance metrics
        if 'performance_score' in evaluation:
            score = evaluation['performance_score']
            if 'avg_performance' not in self.state.performance_metrics:
                self.state.performance_metrics['avg_performance'] = score
            else:
                current_avg = self.state.performance_metrics['avg_performance']
                self.state.performance_metrics['avg_performance'] = (current_avg + score) / 2
        
        # Update success rate
        total_tasks = len(self.task_history)
        successful_tasks = sum(1 for t in self.task_history if t.success)
        self.state.performance_metrics['success_rate'] = successful_tasks / total_tasks if total_tasks > 0 else 0
    
    async def collaborate_with_agent(
        self,
        other_agent_id: str,
        collaboration_type: str,
        shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collaborate with another HealthFlow agent.
        This enables multi-agent workflows for complex healthcare tasks.
        """
        
        collaboration_id = f"{self.agent_id}_{other_agent_id}_{collaboration_type}"
        
        self.logger.info(f"Starting collaboration {collaboration_id}")
        
        # For now, return a placeholder structure
        # This should be implemented to enable real agent-to-agent communication
        
        collaboration_result = {
            "collaboration_id": collaboration_id,
            "type": collaboration_type,
            "status": "initiated",
            "shared_insights": {},
            "combined_output": None
        }
        
        # Track active collaborations
        if collaboration_type not in self.active_collaborations:
            self.active_collaborations[collaboration_type] = []
        self.active_collaborations[collaboration_type].append(other_agent_id)
        
        return collaboration_result
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the agent's performance and state"""
        
        return {
            "agent_state": asdict(self.state),
            "task_history_summary": {
                "total_tasks": len(self.task_history),
                "successful_tasks": sum(1 for t in self.task_history if t.success),
                "average_execution_time": sum(t.execution_time for t in self.task_history) / len(self.task_history) if self.task_history else 0,
                "most_used_tools": self._get_most_used_tools()
            },
            "memory_status": self.memory_manager.get_memory_statistics(),
            "toolbank_status": {
                "total_tools": len(self.toolbank.tools) if hasattr(self.toolbank, 'tools') else 0,
                "custom_tools_created": 0  # Will be implemented
            },
            "collaboration_status": self.active_collaborations
        }
    
    def _get_most_used_tools(self) -> List[str]:
        """Get the most frequently used tools"""
        
        tool_usage = {}
        for task in self.task_history:
            for tool in task.tools_used:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        return sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]