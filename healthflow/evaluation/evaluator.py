"""
Advanced LLM-Based Task Evaluator for HealthFlow
The core innovation of HealthFlow: monitors the entire process lifecycle and provides 
rich supervision signals for evaluation-driven self-improvement.

Key Features:
- Process monitoring throughout task execution
- Multi-dimensional evaluation across critical criteria
- Rich structured feedback for system evolution
- NeurIPS-quality comprehensive evaluation framework
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import uuid

from ..core.llm_provider import LLMProvider, LLMMessage, create_llm_provider
from ..core.config import HealthFlowConfig


class ProcessStage(Enum):
    """Stages of task execution that can be monitored"""
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    AGENT_COLLABORATION = "agent_collaboration"
    REASONING = "reasoning" 
    RESULT_SYNTHESIS = "result_synthesis"
    FINAL_OUTPUT = "final_output"


@dataclass
class ProcessStep:
    """Individual step in the task execution process"""
    stage: ProcessStage
    timestamp: datetime
    agent_id: str
    action: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    tools_used: List[str]
    collaboration_messages: List[Dict[str, Any]]
    reasoning_trace: str
    success: bool
    execution_time: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'stage': self.stage.value,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'action': self.action,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'tools_used': self.tools_used,
            'collaboration_messages': self.collaboration_messages,
            'reasoning_trace': self.reasoning_trace,
            'success': self.success,
            'execution_time': self.execution_time,
            'metadata': self.metadata or {}
        }


@dataclass 
class ExecutionTrace:
    """Complete trace of task execution process - the key input to evaluation"""
    task_id: str
    initial_plan: Dict[str, Any]
    process_steps: List[ProcessStep]
    final_result: Any
    total_execution_time: float
    agents_involved: List[str]
    tools_used: List[str]
    collaboration_patterns: Dict[str, Any]
    error_incidents: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'task_id': self.task_id,
            'initial_plan': self.initial_plan,
            'process_steps': [step.to_dict() for step in self.process_steps],
            'final_result': self.final_result,
            'total_execution_time': self.total_execution_time,
            'agents_involved': self.agents_involved,
            'tools_used': self.tools_used,
            'collaboration_patterns': self.collaboration_patterns,
            'error_incidents': self.error_incidents
        }


@dataclass
class EvaluationCriteria:
    """Multi-dimensional evaluation criteria with scores and detailed feedback"""
    medical_accuracy: float  # 0-10: Correctness of medical information
    safety: float           # 0-10: Adherence to safety protocols  
    reasoning_quality: float # 0-10: Logical soundness of reasoning
    tool_usage_efficiency: float # 0-10: Appropriateness of tool selection/usage
    collaboration_effectiveness: float # 0-10: Quality of inter-agent collaboration
    completeness: float     # 0-10: How completely the task was addressed
    clarity: float          # 0-10: Clarity of final output
    
    # Detailed textual feedback for each dimension
    medical_accuracy_feedback: str = ""
    safety_feedback: str = ""
    reasoning_quality_feedback: str = ""
    tool_usage_feedback: str = ""
    collaboration_feedback: str = ""
    completeness_feedback: str = ""
    clarity_feedback: str = ""
    
    def get_composite_score(self) -> float:
        """Calculate weighted composite score across all criteria"""
        weights = {
            'medical_accuracy': 0.20,
            'safety': 0.20, 
            'reasoning_quality': 0.15,
            'tool_usage_efficiency': 0.15,
            'collaboration_effectiveness': 0.10,
            'completeness': 0.15,
            'clarity': 0.05
        }
        
        weighted_sum = (
            self.medical_accuracy * weights['medical_accuracy'] +
            self.safety * weights['safety'] +
            self.reasoning_quality * weights['reasoning_quality'] +
            self.tool_usage_efficiency * weights['tool_usage_efficiency'] +
            self.collaboration_effectiveness * weights['collaboration_effectiveness'] +
            self.completeness * weights['completeness'] +
            self.clarity * weights['clarity']
        )
        
        return weighted_sum


@dataclass
class EvaluationResult:
    """Rich evaluation result with process monitoring and improvement suggestions"""
    evaluation_id: str
    task_id: str
    timestamp: datetime
    
    # Overall assessment
    overall_success: bool
    overall_score: float  # 0-10 composite score
    confidence: float     # 0-1 confidence in evaluation
    
    # Multi-dimensional criteria scores  
    criteria: EvaluationCriteria
    
    # Rich textual feedback
    executive_summary: str
    detailed_feedback: str
    
    # Actionable improvement suggestions (the key supervision signal)
    improvement_suggestions: Dict[str, List[str]]  # Category -> suggestions
    
    # Process insights
    process_insights: Dict[str, Any]
    collaboration_analysis: Dict[str, Any]
    efficiency_analysis: Dict[str, Any]
    
    # LLM evaluator reasoning
    evaluator_reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
        


class LLMTaskEvaluator:
    """
    Advanced LLM-based Task Evaluator - The Core Innovation of HealthFlow
    
    This evaluator monitors the ENTIRE process lifecycle and provides rich supervision 
    signals for evaluation-driven self-improvement. Key to NeurIPS contribution.
    
    Features:
    - Process monitoring throughout task execution (not just final outcomes)
    - Multi-dimensional evaluation across critical medical criteria
    - Rich structured feedback for system evolution 
    - Actionable improvement suggestions as supervision signals
    - Comprehensive collaboration and efficiency analysis
    """

    def __init__(
        self,
        evaluation_dir: Optional[Path] = None,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[HealthFlowConfig] = None
    ):
        self.evaluation_dir = evaluation_dir or Path("./data/evaluation")
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM provider for evaluation
        if llm_provider:
            self.llm_provider = llm_provider
        elif config:
            self.llm_provider = create_llm_provider(
                api_key=config.api_key,
                base_url=config.base_url,
                model_name=config.model_name
            )
        else:
            # Fallback: load config from default location
            try:
                config = HealthFlowConfig.from_toml()
                self.llm_provider = create_llm_provider(
                    api_key=config.api_key,
                    base_url=config.base_url,
                    model_name=config.model_name
                )
            except Exception as e:
                raise RuntimeError(f"Could not initialize LLM provider for evaluation: {e}")

        # Storage paths - all JSON format
        self.evaluations_path = self.evaluation_dir / "evaluations.json"
        self.process_traces_path = self.evaluation_dir / "process_traces.json"
        self.improvement_history_path = self.evaluation_dir / "improvement_history.json"
        self.performance_analytics_path = self.evaluation_dir / "performance_analytics.json"
        
        # In-memory stores
        self.evaluation_history: List[EvaluationResult] = []
        self.process_traces: List[ExecutionTrace] = []
        
        # Load existing data
        self._load_evaluation_data()
    
    async def evaluate_task(self, execution_trace: ExecutionTrace) -> EvaluationResult:
        """
        Main evaluation method - The core of HealthFlow's innovation.
        
        Monitors the ENTIRE process lifecycle and provides rich supervision signals.
        This is what enables the evaluation-driven self-improvement loop.
        
        Args:
            execution_trace: Complete trace of task execution process
            
        Returns:
            Rich evaluation result with multi-dimensional feedback
        """
        
        evaluation_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Step 1: Analyze the complete execution process
        process_analysis = await self._analyze_execution_process(execution_trace)
        
        # Step 2: Multi-dimensional evaluation using LLM judge
        criteria_evaluation = await self._evaluate_criteria(execution_trace, process_analysis)
        
        # Step 3: Generate actionable improvement suggestions
        improvement_suggestions = await self._generate_improvement_suggestions(
            execution_trace, criteria_evaluation, process_analysis
        )
        
        # Step 4: Analyze collaboration patterns and efficiency
        collaboration_analysis = self._analyze_collaboration_patterns(execution_trace)
        efficiency_analysis = self._analyze_efficiency_patterns(execution_trace)
        
        # Step 5: Generate executive summary and detailed feedback
        executive_summary, detailed_feedback = await self._generate_evaluation_summary(
            execution_trace, criteria_evaluation, improvement_suggestions
        )
        
        # Step 6: Calculate overall score and success
        overall_score = criteria_evaluation.get_composite_score()
        overall_success = overall_score >= 7.0  # Configurable threshold
        
        # Create comprehensive evaluation result
        evaluation_result = EvaluationResult(
            evaluation_id=evaluation_id,
            task_id=execution_trace.task_id,
            timestamp=timestamp,
            overall_success=overall_success,
            overall_score=overall_score,
            confidence=0.85,  # Could be computed based on consistency of criteria
            criteria=criteria_evaluation,
            executive_summary=executive_summary,
            detailed_feedback=detailed_feedback,
            improvement_suggestions=improvement_suggestions,
            process_insights=process_analysis,
            collaboration_analysis=collaboration_analysis,
            efficiency_analysis=efficiency_analysis,
            evaluator_reasoning=f"Evaluated {len(execution_trace.process_steps)} process steps across {len(execution_trace.agents_involved)} agents"
        )
        
        # Store evaluation and trace
        self.evaluation_history.append(evaluation_result)
        self.process_traces.append(execution_trace)
        
        # Persist to storage
        await self._save_evaluation_data()
        
        return evaluation_result
    
    async def _analyze_execution_process(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """Analyze the execution process for patterns and insights"""
        
        process_insights = {
            "total_steps": len(trace.process_steps),
            "execution_time_distribution": {},
            "stage_analysis": {},
            "error_analysis": {},
            "tool_usage_patterns": {},
            "agent_activity_patterns": {}
        }
        
        # Analyze execution time distribution
        stage_times = {}
        for step in trace.process_steps:
            stage = step.stage.value
            if stage not in stage_times:
                stage_times[stage] = []
            stage_times[stage].append(step.execution_time)
        
        for stage, times in stage_times.items():
            process_insights["execution_time_distribution"][stage] = {
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "step_count": len(times)
            }
        
        # Analyze stage progression
        stages_sequence = [step.stage.value for step in trace.process_steps]
        process_insights["stage_analysis"] = {
            "stages_used": list(set(stages_sequence)),
            "stage_transitions": len(set(zip(stages_sequence, stages_sequence[1:]))),
            "repeated_stages": len(stages_sequence) - len(set(stages_sequence))
        }
        
        # Analyze errors and failures
        failed_steps = [step for step in trace.process_steps if not step.success]
        process_insights["error_analysis"] = {
            "total_errors": len(failed_steps),
            "error_stages": [step.stage.value for step in failed_steps],
            "error_recovery": len([step for step in trace.process_steps 
                                  if step.stage == ProcessStage.REASONING and "error" in step.reasoning_trace.lower()])
        }
        
        return process_insights
    
    async def _evaluate_criteria(self, trace: ExecutionTrace, process_analysis: Dict[str, Any]) -> EvaluationCriteria:
        """Multi-dimensional evaluation using LLM judge"""
        
        # Build comprehensive evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt(trace, process_analysis)
        
        try:
            messages = [LLMMessage(role="user", content=evaluation_prompt)]
            response = await self.llm_provider.generate(
                messages=messages,
                max_tokens=2000,
                temperature=0.1  # Low temperature for consistent evaluation
            )
            
            # Parse structured evaluation response
            evaluation_data = self._parse_evaluation_response(response.content)
            
            return EvaluationCriteria(
                medical_accuracy=evaluation_data.get("medical_accuracy", 5.0),
                safety=evaluation_data.get("safety", 5.0),
                reasoning_quality=evaluation_data.get("reasoning_quality", 5.0),
                tool_usage_efficiency=evaluation_data.get("tool_usage_efficiency", 5.0),
                collaboration_effectiveness=evaluation_data.get("collaboration_effectiveness", 5.0),
                completeness=evaluation_data.get("completeness", 5.0),
                clarity=evaluation_data.get("clarity", 5.0),
                medical_accuracy_feedback=evaluation_data.get("medical_accuracy_feedback", ""),
                safety_feedback=evaluation_data.get("safety_feedback", ""),
                reasoning_quality_feedback=evaluation_data.get("reasoning_quality_feedback", ""),
                tool_usage_feedback=evaluation_data.get("tool_usage_feedback", ""),
                collaboration_feedback=evaluation_data.get("collaboration_feedback", ""),
                completeness_feedback=evaluation_data.get("completeness_feedback", ""),
                clarity_feedback=evaluation_data.get("clarity_feedback", "")
            )
        
        except Exception as e:
            # Fallback to default scores if LLM evaluation fails
            return EvaluationCriteria(
                medical_accuracy=5.0, safety=5.0, reasoning_quality=5.0,
                tool_usage_efficiency=5.0, collaboration_effectiveness=5.0,
                completeness=5.0, clarity=5.0,
                medical_accuracy_feedback=f"Evaluation failed: {e}",
                safety_feedback="", reasoning_quality_feedback="",
                tool_usage_feedback="", collaboration_feedback="",
                completeness_feedback="", clarity_feedback=""
            )
    
    def _build_evaluation_prompt(self, trace: ExecutionTrace, process_analysis: Dict[str, Any]) -> str:
        """Build comprehensive evaluation prompt for LLM judge"""
        
        # Summarize execution trace
        process_summary = f"""
TASK EXECUTION ANALYSIS:

Task ID: {trace.task_id}
Total Execution Time: {trace.total_execution_time:.2f}s
Agents Involved: {', '.join(trace.agents_involved)}
Tools Used: {', '.join(trace.tools_used)}
Process Steps: {len(trace.process_steps)}

INITIAL PLAN:
{json.dumps(trace.initial_plan, indent=2)}

PROCESS STEPS SUMMARY:
"""
        
        for i, step in enumerate(trace.process_steps):
            process_summary += f"""
Step {i+1} - {step.stage.value}:
- Agent: {step.agent_id}
- Action: {step.action}
- Success: {step.success}
- Execution Time: {step.execution_time:.2f}s
- Tools Used: {', '.join(step.tools_used)}
- Reasoning: {step.reasoning_trace[:200]}...
"""
        
        process_summary += f"""
FINAL RESULT:
{json.dumps(trace.final_result, indent=2)}

PROCESS INSIGHTS:
{json.dumps(process_analysis, indent=2)}
"""
        
        return f"""
You are an expert medical AI evaluator. Your task is to comprehensively evaluate the task execution process across multiple critical dimensions.

{process_summary}

Please evaluate this task execution across the following criteria (0-10 scale):

1. MEDICAL ACCURACY (0-10): Correctness of medical information and clinical reasoning
2. SAFETY (0-10): Adherence to medical safety protocols and risk management  
3. REASONING QUALITY (0-10): Logical soundness and coherence of reasoning steps
4. TOOL USAGE EFFICIENCY (0-10): Appropriateness and effectiveness of tool selection/usage
5. COLLABORATION EFFECTIVENESS (0-10): Quality of inter-agent collaboration and communication
6. COMPLETENESS (0-10): How completely the task requirements were addressed
7. CLARITY (0-10): Clarity and understandability of the final output

For each criterion, provide:
- A score (0-10)
- Detailed feedback explaining the score
- Specific suggestions for improvement

Respond with the following JSON structure:
{{
    "medical_accuracy": <score>,
    "medical_accuracy_feedback": "<detailed feedback>",
    "safety": <score>, 
    "safety_feedback": "<detailed feedback>",
    "reasoning_quality": <score>,
    "reasoning_quality_feedback": "<detailed feedback>",
    "tool_usage_efficiency": <score>,
    "tool_usage_feedback": "<detailed feedback>",
    "collaboration_effectiveness": <score>,
    "collaboration_feedback": "<detailed feedback>",
    "completeness": <score>,
    "completeness_feedback": "<detailed feedback>",
    "clarity": <score>,
    "clarity_feedback": "<detailed feedback>"
}}
"""
    
    def _parse_evaluation_response(self, response_content: str) -> Dict[str, Any]:
        """Parse structured evaluation response from LLM"""
        try:
            # Try to extract JSON from response
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback parsing if JSON structure not found
                return self._fallback_parse_evaluation(response_content)
                
        except json.JSONDecodeError:
            return self._fallback_parse_evaluation(response_content)
    
    def _fallback_parse_evaluation(self, content: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        # Simple regex-based parsing as fallback
        import re
        
        result = {}
        
        # Extract scores
        score_patterns = {
            "medical_accuracy": r"medical[_\s]accuracy[:\s]*(\d+(?:\.\d+)?)",
            "safety": r"safety[:\s]*(\d+(?:\.\d+)?)",
            "reasoning_quality": r"reasoning[_\s]quality[:\s]*(\d+(?:\.\d+)?)",
            "tool_usage_efficiency": r"tool[_\s]usage[_\s]efficiency[:\s]*(\d+(?:\.\d+)?)",
            "collaboration_effectiveness": r"collaboration[_\s]effectiveness[:\s]*(\d+(?:\.\d+)?)",
            "completeness": r"completeness[:\s]*(\d+(?:\.\d+)?)",
            "clarity": r"clarity[:\s]*(\d+(?:\.\d+)?)?"
        }
        
        for criterion, pattern in score_patterns.items():
            match = re.search(pattern, content.lower())
            if match:
                result[criterion] = float(match.group(1))
            else:
                result[criterion] = 5.0  # Default score
            
            # Set empty feedback as fallback
            result[f"{criterion}_feedback"] = f"Extracted from evaluation response (score: {result[criterion]})"
        
        return result

    async def _generate_improvement_suggestions(
        self, 
        trace: ExecutionTrace, 
        criteria: EvaluationCriteria, 
        process_analysis: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate actionable improvement suggestions - the key supervision signal"""
        
        suggestions = {
            "agent_collaboration": [],
            "tool_usage": [],
            "reasoning_process": [],
            "prompt_templates": [],
            "system_architecture": []
        }
        
        # Generate suggestions based on evaluation criteria
        if criteria.medical_accuracy < 7.0:
            suggestions["prompt_templates"].append(
                "Improve medical knowledge prompts with more specific clinical guidelines"
            )
            suggestions["tool_usage"].append(
                "Integrate medical knowledge base tools for fact verification"
            )
        
        if criteria.safety < 8.0:
            suggestions["prompt_templates"].append(
                "Add explicit safety checks and contraindication warnings to prompts"
            )
            suggestions["system_architecture"].append(
                "Implement mandatory safety review stage before final output"
            )
        
        if criteria.collaboration_effectiveness < 6.0:
            suggestions["agent_collaboration"].append(
                "Improve inter-agent communication protocols with structured message formats"
            )
        
        if criteria.tool_usage_efficiency < 6.0:
            suggestions["tool_usage"].append(
                "Enhance tool selection algorithm with better context matching"
            )
        
        # Process-based suggestions
        if process_analysis["error_analysis"]["total_errors"] > 2:
            suggestions["system_architecture"].append(
                "Implement better error recovery and retry mechanisms"
            )
        
        return suggestions
    
    def _analyze_collaboration_patterns(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """Analyze inter-agent collaboration patterns"""
        
        collaboration_data = {
            "total_messages": 0,
            "agent_interactions": {},
            "communication_efficiency": 0.0,
            "collaboration_stages": []
        }
        
        # Count collaboration messages across all steps
        for step in trace.process_steps:
            collaboration_data["total_messages"] += len(step.collaboration_messages)
            
            if step.stage == ProcessStage.AGENT_COLLABORATION:
                collaboration_data["collaboration_stages"].append({
                    "agent": step.agent_id,
                    "timestamp": step.timestamp.isoformat(),
                    "success": step.success,
                    "execution_time": step.execution_time
                })
        
        # Analyze agent interaction patterns
        agent_pairs = set()
        for step in trace.process_steps:
            for msg in step.collaboration_messages:
                sender = msg.get("sender", "unknown")
                receiver = msg.get("receiver", "unknown")
                if sender != receiver:
                    agent_pairs.add(tuple(sorted([sender, receiver])))
        
        collaboration_data["unique_agent_pairs"] = len(agent_pairs)
        collaboration_data["communication_efficiency"] = (
            collaboration_data["total_messages"] / max(len(trace.process_steps), 1)
        )
        
        return collaboration_data
    
    def _analyze_efficiency_patterns(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """Analyze execution efficiency patterns"""
        
        efficiency_data = {
            "total_execution_time": trace.total_execution_time,
            "avg_step_time": trace.total_execution_time / max(len(trace.process_steps), 1),
            "tool_switching_overhead": 0.0,
            "stage_efficiency": {},
            "bottleneck_analysis": {}
        }
        
        # Analyze stage efficiency
        stage_times = {}
        for step in trace.process_steps:
            stage = step.stage.value
            if stage not in stage_times:
                stage_times[stage] = []
            stage_times[stage].append(step.execution_time)
        
        for stage, times in stage_times.items():
            efficiency_data["stage_efficiency"][stage] = {
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "efficiency_score": 10.0 - min(sum(times) / 10.0, 9.0)  # Simple efficiency metric
            }
        
        # Identify bottlenecks
        if stage_times:
            slowest_stage = max(stage_times.keys(), key=lambda s: sum(stage_times[s]))
            efficiency_data["bottleneck_analysis"]["slowest_stage"] = slowest_stage
            efficiency_data["bottleneck_analysis"]["bottleneck_time"] = sum(stage_times[slowest_stage])
        
        return efficiency_data
    
    async def _generate_evaluation_summary(
        self, 
        trace: ExecutionTrace, 
        criteria: EvaluationCriteria, 
        suggestions: Dict[str, List[str]]
    ) -> Tuple[str, str]:
        """Generate executive summary and detailed feedback"""
        
        # Executive summary
        overall_score = criteria.get_composite_score()
        executive_summary = f"""
TASK EVALUATION SUMMARY

Overall Score: {overall_score:.1f}/10.0
Task Success: {'✓' if overall_score >= 7.0 else '✗'}

Key Strengths:
- Medical Accuracy: {criteria.medical_accuracy:.1f}/10
- Safety Adherence: {criteria.safety:.1f}/10  
- Reasoning Quality: {criteria.reasoning_quality:.1f}/10

Areas for Improvement:
- Tool Usage: {criteria.tool_usage_efficiency:.1f}/10
- Collaboration: {criteria.collaboration_effectiveness:.1f}/10
- Completeness: {criteria.completeness:.1f}/10

Total Process Steps: {len(trace.process_steps)}
Execution Time: {trace.total_execution_time:.2f}s
Agents Involved: {len(trace.agents_involved)}
"""
        
        # Detailed feedback combining all criteria feedback
        detailed_feedback = f"""
DETAILED EVALUATION FEEDBACK

MEDICAL ACCURACY ({criteria.medical_accuracy:.1f}/10):
{criteria.medical_accuracy_feedback}

SAFETY ASSESSMENT ({criteria.safety:.1f}/10):
{criteria.safety_feedback}

REASONING QUALITY ({criteria.reasoning_quality:.1f}/10):  
{criteria.reasoning_quality_feedback}

TOOL USAGE EFFICIENCY ({criteria.tool_usage_efficiency:.1f}/10):
{criteria.tool_usage_feedback}

COLLABORATION EFFECTIVENESS ({criteria.collaboration_effectiveness:.1f}/10):
{criteria.collaboration_feedback}

COMPLETENESS ({criteria.completeness:.1f}/10):
{criteria.completeness_feedback}

CLARITY ({criteria.clarity:.1f}/10):
{criteria.clarity_feedback}

IMPROVEMENT RECOMMENDATIONS:
"""
        
        for category, suggestion_list in suggestions.items():
            if suggestion_list:
                detailed_feedback += f"\n{category.replace('_', ' ').title()}:\n"
                for suggestion in suggestion_list:
                    detailed_feedback += f"  • {suggestion}\n"
        
        return executive_summary, detailed_feedback
    
    def _load_evaluation_data(self):
        """Load existing evaluation data from JSON files"""
        try:
            if self.evaluations_path.exists():
                with open(self.evaluations_path, 'r') as f:
                    data = json.load(f)
                    # Convert back to EvaluationResult objects would require more complex deserialization
                    # For now, keep as dicts for simplicity
                    pass
        except Exception as e:
            pass  # Start with empty data if loading fails
    
    async def _save_evaluation_data(self):
        """Save evaluation data to JSON files"""
        try:
            # Save evaluation results
            evaluation_dicts = [result.to_dict() for result in self.evaluation_history]
            with open(self.evaluations_path, 'w') as f:
                json.dump(evaluation_dicts, f, indent=2, ensure_ascii=False)
            
            # Save process traces
            trace_dicts = [trace.to_dict() for trace in self.process_traces]  
            with open(self.process_traces_path, 'w') as f:
                json.dump(trace_dicts, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving evaluation data: {e}")
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics"""
        
        if not self.evaluation_history:
            return {"message": "No evaluations available"}
        
        # Calculate aggregate statistics
        total_evaluations = len(self.evaluation_history)
        successful_evaluations = sum(1 for eval in self.evaluation_history if eval.overall_success)
        
        avg_scores = {
            "medical_accuracy": sum(eval.criteria.medical_accuracy for eval in self.evaluation_history) / total_evaluations,
            "safety": sum(eval.criteria.safety for eval in self.evaluation_history) / total_evaluations,
            "reasoning_quality": sum(eval.criteria.reasoning_quality for eval in self.evaluation_history) / total_evaluations,
            "tool_usage_efficiency": sum(eval.criteria.tool_usage_efficiency for eval in self.evaluation_history) / total_evaluations,
            "collaboration_effectiveness": sum(eval.criteria.collaboration_effectiveness for eval in self.evaluation_history) / total_evaluations,
            "completeness": sum(eval.criteria.completeness for eval in self.evaluation_history) / total_evaluations,
            "clarity": sum(eval.criteria.clarity for eval in self.evaluation_history) / total_evaluations,
            "overall": sum(eval.overall_score for eval in self.evaluation_history) / total_evaluations
        }
        
        return {
            "total_evaluations": total_evaluations,
            "success_rate": successful_evaluations / total_evaluations,
            "average_scores": avg_scores,
            "recent_trend": "improving" if len(self.evaluation_history) > 1 and 
                           self.evaluation_history[-1].overall_score > self.evaluation_history[-2].overall_score 
                           else "stable"
        }


# Alias for compatibility
TaskEvaluator = LLMTaskEvaluator