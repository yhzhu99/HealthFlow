"""
LLM-Based Task Evaluator for HealthFlow
Evaluates task execution results using LLM judges and provides performance feedback
Uses JSON format for all persistence - no pandas/pickle dependencies
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from ..core.llm_provider import LLMProvider, LLMMessage, create_llm_provider


@dataclass
class EvaluationResult:
    """Result of LLM-based task evaluation"""
    success: bool
    confidence: float
    performance_score: float
    accuracy: float
    completeness: float
    medical_safety: float
    evidence_based: float
    feedback: str
    improvement_suggestions: List[str]
    llm_reasoning: str
    novel_insights: Optional[List[str]] = None
    evaluation_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'evaluation_id': self.evaluation_id,
            'success': self.success,
            'confidence': self.confidence,
            'performance_score': self.performance_score,
            'accuracy': self.accuracy,
            'completeness': self.completeness,
            'medical_safety': self.medical_safety,
            'evidence_based': self.evidence_based,
            'feedback': self.feedback,
            'improvement_suggestions': self.improvement_suggestions,
            'llm_reasoning': self.llm_reasoning,
            'novel_insights': self.novel_insights
        }


class LLMTaskEvaluator:
    """
    Advanced LLM-based task evaluator with comprehensive medical assessment
    
    Features:
    - Uses LLM judges for nuanced evaluation
    - Multiple evaluation criteria with medical focus
    - Continuous improvement based on feedback
    - JSON-only storage (no pandas/pickle)
    - Code execution error analysis
    - QA task judgment
    """
    
    def __init__(
        self, 
        evaluation_dir: Optional[Path] = None,
        llm_provider: Optional[LLMProvider] = None
    ):
        self.evaluation_dir = evaluation_dir or Path("./data/evaluation")
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM provider for evaluation
        self.llm_provider = llm_provider or create_llm_provider(
            provider_type="auto",
            model_name="gpt-4"  # Use powerful model for evaluation
        )
        
        # Storage paths - all JSON format
        self.evaluations_path = self.evaluation_dir / "evaluations.json"
        self.evaluation_history_path = self.evaluation_dir / "evaluation_history.json"
        self.judge_feedback_path = self.evaluation_dir / "judge_feedback.json"
        self.performance_trends_path = self.evaluation_dir / "performance_trends.json"
        
        # In-memory stores
        self.evaluation_history: List[Dict[str, Any]] = []
        self.judge_feedback_history: List[Dict[str, Any]] = []
        self.performance_trends: Dict[str, List[float]] = {}
        
        # Evaluation thresholds
        self.performance_thresholds = {
            "success": 0.7,
            "confidence": 0.6,
            "medical_safety": 0.9,  # High threshold for safety
            "evidence_based": 0.7,
            "code_execution": 0.8
        }
        
        # Initialize evaluation prompts
        self.evaluation_prompts = self._initialize_evaluation_prompts()
    
    def _initialize_evaluation_prompts(self) -> Dict[str, str]:
        """Initialize LLM evaluation prompts for different aspects"""
        return {
            "medical_qa_judge": """
You are an expert medical AI evaluator. Your task is to assess the quality and safety of medical responses.

Task Description: {task_description}
Agent Response: {response}
Context: {context}

Evaluate the response across these dimensions (score 0.0-1.0 for each):

1. MEDICAL ACCURACY: Is the information medically accurate and up-to-date?
2. COMPLETENESS: Does the response fully address the question/task?
3. SAFETY: Is the advice safe? Does it include appropriate warnings/disclaimers?
4. EVIDENCE-BASED: Is the response based on clinical evidence and guidelines?
5. CLARITY: Is the response clear and understandable?

Provide your evaluation in this JSON format:
{{
    "medical_accuracy": 0.0-1.0,
    "completeness": 0.0-1.0,
    "medical_safety": 0.0-1.0,
    "evidence_based": 0.0-1.0,
    "clarity": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "reasoning": "Detailed explanation of your assessment",
    "improvement_suggestions": ["suggestion 1", "suggestion 2"],
    "safety_concerns": ["concern 1", "concern 2"] or [],
    "strengths": ["strength 1", "strength 2"]
}}
""",
            
            "code_execution_judge": """
You are an expert code execution evaluator for medical AI tasks.

Task Description: {task_description}
Code Execution Result: {execution_result}
Error Messages: {error_messages}
Context: {context}

Evaluate the code execution across these dimensions (score 0.0-1.0 for each):

1. EXECUTION_SUCCESS: Did the code execute without errors?
2. RESULT_QUALITY: Is the output meaningful and correct?
3. ERROR_HANDLING: How well were errors handled and reported?
4. PERFORMANCE: Was the execution efficient and timely?
5. MEDICAL_RELEVANCE: Is the result medically relevant and useful?

Provide your evaluation in this JSON format:
{{
    "execution_success": 0.0-1.0,
    "result_quality": 0.0-1.0,
    "error_handling": 0.0-1.0,
    "performance": 0.0-1.0,
    "medical_relevance": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "reasoning": "Detailed explanation of your assessment",
    "improvement_suggestions": ["suggestion 1", "suggestion 2"],
    "error_analysis": "Analysis of any errors encountered",
    "code_quality_feedback": "Feedback on code quality and best practices"
}}
""",
            
            "task_completion_judge": """
You are an expert task completion evaluator for healthcare AI systems.

Original Task: {task_description}
Agent Result: {result}
Tools Used: {tools_used}
Execution Time: {execution_time}
Context: {context}

Evaluate the task completion across these dimensions (score 0.0-1.0 for each):

1. TASK_COMPLETION: Was the original task fully completed?
2. APPROACH_QUALITY: Was the approach logical and well-structured?
3. TOOL_USAGE: Were appropriate tools selected and used effectively?
4. EFFICIENCY: Was the task completed efficiently?
5. OUTPUT_QUALITY: Is the final output high-quality and useful?

Provide your evaluation in this JSON format:
{{
    "task_completion": 0.0-1.0,
    "approach_quality": 0.0-1.0,
    "tool_usage": 0.0-1.0,
    "efficiency": 0.0-1.0,
    "output_quality": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "reasoning": "Detailed explanation of your assessment",
    "improvement_suggestions": ["suggestion 1", "suggestion 2"],
    "best_practices": ["practice 1", "practice 2"],
    "alternative_approaches": ["approach 1", "approach 2"] or []
}}
"""
        }
    
    async def initialize(self):
        """Initialize evaluator and load historical data"""
        await self._load_evaluation_data()
    
    async def _load_evaluation_data(self):
        """Load evaluation data from JSON files"""
        # Load evaluation history
        if self.evaluation_history_path.exists():
            try:
                with open(self.evaluation_history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.evaluation_history = data.get('evaluations', [])
            except Exception as e:
                print(f"Error loading evaluation history: {e}")
        
        # Load judge feedback history
        if self.judge_feedback_path.exists():
            try:
                with open(self.judge_feedback_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.judge_feedback_history = data.get('feedback', [])
            except Exception as e:
                print(f"Error loading judge feedback: {e}")
        
        # Load performance trends
        if self.performance_trends_path.exists():
            try:
                with open(self.performance_trends_path, 'r', encoding='utf-8') as f:
                    self.performance_trends = json.load(f)
            except Exception as e:
                print(f"Error loading performance trends: {e}")
    
    async def evaluate_task_result(
        self,
        task_description: str,
        result: Any,
        task_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive LLM-based evaluation of task execution result
        
        Args:
            task_description: Original task description
            result: Task execution result (can be text, code output, etc.)
            task_context: Additional context including tools used, errors, etc.
            
        Returns:
            Detailed evaluation metrics and feedback
        """
        
        task_context = task_context or {}
        evaluation_id = str(uuid.uuid4())
        
        try:
            # Determine evaluation type based on context
            evaluation_type = self._determine_evaluation_type(task_description, result, task_context)
            
            # Run LLM-based evaluation
            llm_evaluation = await self._run_llm_evaluation(
                evaluation_type, task_description, result, task_context
            )
            
            # Calculate composite scores
            composite_scores = self._calculate_composite_scores(llm_evaluation, evaluation_type)
            
            # Generate overall assessment
            overall_assessment = await self._generate_overall_assessment(
                task_description, result, llm_evaluation, composite_scores
            )
            
            # Create evaluation result
            evaluation_result = {
                "evaluation_id": evaluation_id,
                "timestamp": datetime.now().isoformat(),
                "task_description": task_description,
                "result_summary": str(result)[:500],  # Truncate for storage
                "evaluation_type": evaluation_type,
                "context": task_context,
                
                # LLM evaluation results
                "llm_scores": llm_evaluation,
                
                # Composite scores
                "success": composite_scores["success"],
                "confidence": composite_scores["confidence"],
                "performance_score": composite_scores["performance_score"],
                "accuracy": composite_scores["accuracy"],
                "completeness": composite_scores["completeness"],
                "medical_safety": composite_scores["medical_safety"],
                "evidence_based": composite_scores["evidence_based"],
                
                # Feedback and suggestions
                "feedback": overall_assessment["feedback"],
                "improvement_suggestions": overall_assessment["improvement_suggestions"],
                "llm_reasoning": llm_evaluation.get("reasoning", ""),
                
                # Reward for agent learning
                "reward": self._calculate_reward(composite_scores),
                
                # Novel insights detection
                "novel_insights": overall_assessment.get("novel_insights"),
                
                # Safety analysis
                "safety_concerns": llm_evaluation.get("safety_concerns", []),
                "strengths": llm_evaluation.get("strengths", [])
            }
            
            # Store evaluation
            await self._store_evaluation(evaluation_result)
            
            # Update performance trends
            await self._update_performance_trends(evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            # Return error evaluation
            error_evaluation = {
                "evaluation_id": evaluation_id,
                "timestamp": datetime.now().isoformat(),
                "task_description": task_description,
                "result_summary": str(result)[:500] if result else "",
                "evaluation_type": "error",
                "success": False,
                "confidence": 0.0,
                "performance_score": 0.0,
                "accuracy": 0.0,
                "completeness": 0.0,
                "medical_safety": 0.5,
                "evidence_based": 0.0,
                "feedback": f"Evaluation failed: {str(e)}",
                "improvement_suggestions": ["Fix evaluation system error"],
                "llm_reasoning": f"Error during evaluation: {str(e)}",
                "reward": 0.0,
                "error": str(e)
            }
            
            await self._store_evaluation(error_evaluation)
            return error_evaluation
    
    def _determine_evaluation_type(
        self, 
        task_description: str, 
        result: Any, 
        context: Dict[str, Any]
    ) -> str:
        """Determine the type of evaluation needed"""
        
        # Check for code execution context
        if context.get("tools_used") or context.get("error_messages") or context.get("code_execution"):
            return "code_execution"
        
        # Check for QA context
        if any(word in task_description.lower() for word in ["question", "answer", "explain", "what", "how", "why"]):
            return "medical_qa"
        
        # Default to general task completion evaluation
        return "task_completion"
    
    async def _run_llm_evaluation(
        self,
        evaluation_type: str,
        task_description: str,
        result: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run LLM-based evaluation using appropriate prompt"""
        
        # Select appropriate evaluation prompt
        prompt_template = self.evaluation_prompts.get(f"{evaluation_type}_judge")
        if not prompt_template:
            prompt_template = self.evaluation_prompts["task_completion_judge"]
        
        # Format prompt with specific context
        prompt = self._format_evaluation_prompt(
            prompt_template, evaluation_type, task_description, result, context
        )
        
        try:
            # Call LLM for evaluation
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self.llm_provider.generate(
                messages=messages,
                max_tokens=1500,
                temperature=0.1  # Low temperature for consistent evaluation
            )
            
            # Parse LLM response
            try:
                evaluation_result = json.loads(response.content)
                return evaluation_result
            except json.JSONDecodeError:
                # Fallback: extract structured information from text
                return self._extract_evaluation_from_text(response.content, evaluation_type)
                
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return self._get_fallback_evaluation(evaluation_type)
    
    def _format_evaluation_prompt(
        self,
        prompt_template: str,
        evaluation_type: str,
        task_description: str,
        result: Any,
        context: Dict[str, Any]
    ) -> str:
        """Format evaluation prompt with specific details"""
        
        format_dict = {
            "task_description": task_description,
            "response": str(result),
            "result": str(result),
            "context": json.dumps(context, indent=2),
            "tools_used": context.get("tools_used", []),
            "execution_time": context.get("execution_time", "N/A"),
            "error_messages": context.get("error_messages", "None"),
            "execution_result": str(result)
        }
        
        return prompt_template.format(**format_dict)
    
    def _extract_evaluation_from_text(self, text: str, evaluation_type: str) -> Dict[str, Any]:
        """Extract evaluation metrics from unstructured LLM text response"""
        
        # Basic fallback evaluation based on text analysis
        evaluation = {
            "overall_score": 0.6,  # Neutral score
            "reasoning": text[:500],
            "improvement_suggestions": ["Review LLM evaluation output format"],
            "strengths": [],
            "safety_concerns": []
        }
        
        # Try to extract scores from text
        import re
        
        # Look for score patterns
        score_patterns = [
            r"(\w+):\s*([0-9]*\.?[0-9]+)",
            r"(\w+)\s*=\s*([0-9]*\.?[0-9]+)",
            r"(\w+)\s*score:\s*([0-9]*\.?[0-9]+)"
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, text.lower())
            for metric, score_str in matches:
                try:
                    score = float(score_str)
                    if 0 <= score <= 1:
                        evaluation[metric] = score
                    elif 0 <= score <= 10:  # Scale from 0-10 to 0-1
                        evaluation[metric] = score / 10
                except ValueError:
                    continue
        
        return evaluation
    
    def _get_fallback_evaluation(self, evaluation_type: str) -> Dict[str, Any]:
        """Get fallback evaluation when LLM evaluation fails"""
        
        fallback = {
            "overall_score": 0.5,
            "reasoning": f"Fallback evaluation due to {evaluation_type} assessment failure",
            "improvement_suggestions": ["Fix evaluation system", "Retry with better context"],
            "strengths": [],
            "safety_concerns": []
        }
        
        # Add type-specific defaults
        if evaluation_type == "medical_qa":
            fallback.update({
                "medical_accuracy": 0.5,
                "completeness": 0.5,
                "medical_safety": 0.7,  # Conservative safety score
                "evidence_based": 0.4,
                "clarity": 0.5
            })
        elif evaluation_type == "code_execution":
            fallback.update({
                "execution_success": 0.3,
                "result_quality": 0.4,
                "error_handling": 0.5,
                "performance": 0.5,
                "medical_relevance": 0.4
            })
        else:  # task_completion
            fallback.update({
                "task_completion": 0.4,
                "approach_quality": 0.5,
                "tool_usage": 0.4,
                "efficiency": 0.5,
                "output_quality": 0.4
            })
        
        return fallback
    
    def _calculate_composite_scores(
        self, 
        llm_evaluation: Dict[str, Any], 
        evaluation_type: str
    ) -> Dict[str, Any]:
        """Calculate composite scores from LLM evaluation results"""
        
        # Extract individual scores based on evaluation type
        if evaluation_type == "medical_qa":
            accuracy = llm_evaluation.get("medical_accuracy", 0.5)
            completeness = llm_evaluation.get("completeness", 0.5)
            medical_safety = llm_evaluation.get("medical_safety", 0.7)
            evidence_based = llm_evaluation.get("evidence_based", 0.4)
            clarity = llm_evaluation.get("clarity", 0.5)
            
            # Calculate performance score
            performance_score = (
                accuracy * 0.3 +
                completeness * 0.25 +
                medical_safety * 0.25 +
                evidence_based * 0.15 +
                clarity * 0.05
            )
            
        elif evaluation_type == "code_execution":
            execution_success = llm_evaluation.get("execution_success", 0.3)
            result_quality = llm_evaluation.get("result_quality", 0.4)
            error_handling = llm_evaluation.get("error_handling", 0.5)
            performance = llm_evaluation.get("performance", 0.5)
            medical_relevance = llm_evaluation.get("medical_relevance", 0.4)
            
            accuracy = (execution_success + result_quality) / 2
            completeness = result_quality
            medical_safety = min(medical_relevance, error_handling)
            evidence_based = medical_relevance
            
            performance_score = (
                execution_success * 0.3 +
                result_quality * 0.25 +
                error_handling * 0.2 +
                performance * 0.15 +
                medical_relevance * 0.1
            )
            
        else:  # task_completion
            task_completion = llm_evaluation.get("task_completion", 0.4)
            approach_quality = llm_evaluation.get("approach_quality", 0.5)
            tool_usage = llm_evaluation.get("tool_usage", 0.4)
            efficiency = llm_evaluation.get("efficiency", 0.5)
            output_quality = llm_evaluation.get("output_quality", 0.4)
            
            accuracy = (approach_quality + output_quality) / 2
            completeness = task_completion
            medical_safety = 0.7  # Default safe assumption
            evidence_based = output_quality
            
            performance_score = (
                task_completion * 0.3 +
                approach_quality * 0.2 +
                tool_usage * 0.2 +
                efficiency * 0.15 +
                output_quality * 0.15
            )
        
        # Calculate overall success
        success = (
            performance_score >= self.performance_thresholds["success"] and
            medical_safety >= self.performance_thresholds["medical_safety"]
        )
        
        # Calculate confidence based on score consistency
        all_scores = [accuracy, completeness, medical_safety, evidence_based]
        confidence = self._calculate_confidence(all_scores)
        
        return {
            "success": success,
            "confidence": confidence,
            "performance_score": performance_score,
            "accuracy": accuracy,
            "completeness": completeness,
            "medical_safety": medical_safety,
            "evidence_based": evidence_based
        }
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on consistency of scores"""
        
        if not scores:
            return 0.0
        
        # Calculate standard deviation
        mean_score = sum(scores) / len(scores)
        if len(scores) == 1:
            return mean_score
            
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Higher consistency = higher confidence
        confidence = max(0.0, 1.0 - (std_dev * 2))
        
        # Adjust based on mean score
        confidence = (confidence + mean_score) / 2
        
        return min(1.0, confidence)
    
    async def _generate_overall_assessment(
        self,
        task_description: str,
        result: Any,
        llm_evaluation: Dict[str, Any],
        composite_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall assessment and improvement suggestions"""
        
        feedback_parts = []
        improvements = []
        
        # Performance-based feedback
        performance = composite_scores["performance_score"]
        if performance >= 0.8:
            feedback_parts.append("Excellent task execution with high-quality results.")
        elif performance >= 0.6:
            feedback_parts.append("Good task execution with room for improvement.")
        else:
            feedback_parts.append("Task execution needs significant improvement.")
        
        # Safety feedback (critical)
        safety = composite_scores["medical_safety"]
        if safety < 0.8:
            feedback_parts.append("⚠️ IMPORTANT: Medical safety considerations need attention.")
            improvements.append("Prioritize patient safety and include appropriate medical disclaimers")
        
        # Accuracy feedback
        accuracy = composite_scores["accuracy"]
        if accuracy < 0.7:
            feedback_parts.append("Accuracy could be improved with better evidence and validation.")
            improvements.append("Enhance accuracy through evidence-based approaches and fact-checking")
        
        # Completeness feedback
        completeness = composite_scores["completeness"]
        if completeness < 0.7:
            feedback_parts.append("Response could be more comprehensive and complete.")
            improvements.append("Provide more detailed and thorough responses")
        
        # Evidence-based feedback
        evidence = composite_scores["evidence_based"]
        if evidence < 0.6:
            feedback_parts.append("Response would benefit from more evidence-based information.")
            improvements.append("Include references to medical literature and clinical guidelines")
        
        # Add LLM-specific suggestions
        llm_suggestions = llm_evaluation.get("improvement_suggestions", [])
        improvements.extend(llm_suggestions)
        
        # Identify novel insights
        novel_insights = None
        if result:
            result_str = str(result).lower()
            insight_keywords = ["novel", "new", "innovative", "unique", "breakthrough", "unprecedented"]
            if any(keyword in result_str for keyword in insight_keywords):
                novel_insights = ["Potential novel insight detected in response"]
        
        return {
            "feedback": " ".join(feedback_parts),
            "improvement_suggestions": list(set(improvements)),  # Remove duplicates
            "novel_insights": novel_insights
        }
    
    def _calculate_reward(self, composite_scores: Dict[str, Any]) -> float:
        """Calculate reward signal for agent learning and optimization"""
        
        performance = composite_scores["performance_score"]
        safety = composite_scores["medical_safety"]
        success = composite_scores["success"]
        
        # Base reward from performance
        reward = performance
        
        # Safety is critical - heavily weight it
        safety_factor = min(safety * 1.5, 1.0)
        reward *= safety_factor
        
        # Success bonus
        if success:
            reward += 0.2
        
        # High performance bonus
        if performance > 0.8:
            reward += (performance - 0.8) * 0.5
        
        # Safety penalty for low safety scores
        if safety < 0.7:
            reward *= 0.5
        
        return max(0.0, min(1.0, reward))
    
    async def _store_evaluation(self, evaluation_result: Dict[str, Any]):
        """Store evaluation result to JSON files"""
        
        # Add to evaluation history
        self.evaluation_history.append(evaluation_result)
        
        # Keep history manageable (last 1000 evaluations)
        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-1000:]
        
        # Save to JSON
        try:
            history_data = {
                "evaluations": self.evaluation_history,
                "last_updated": datetime.now().isoformat(),
                "total_count": len(self.evaluation_history)
            }
            
            with open(self.evaluation_history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"Error saving evaluation: {e}")
    
    async def _update_performance_trends(self, evaluation_result: Dict[str, Any]):
        """Update performance trends for monitoring improvement"""
        
        timestamp = datetime.now().isoformat()
        performance_score = evaluation_result["performance_score"]
        
        # Update trends by day
        day_key = datetime.now().strftime("%Y-%m-%d")
        if day_key not in self.performance_trends:
            self.performance_trends[day_key] = []
        
        self.performance_trends[day_key].append(performance_score)
        
        # Keep only recent trends (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        cutoff_key = cutoff_date.strftime("%Y-%m-%d")
        
        self.performance_trends = {
            k: v for k, v in self.performance_trends.items() 
            if k >= cutoff_key
        }
        
        # Save trends
        try:
            with open(self.performance_trends_path, 'w', encoding='utf-8') as f:
                json.dump(self.performance_trends, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving performance trends: {e}")
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics"""
        
        if not self.evaluation_history:
            return {"message": "No evaluations available"}
        
        evaluations = self.evaluation_history
        total_evals = len(evaluations)
        
        # Calculate basic statistics
        avg_performance = sum(e.get("performance_score", 0) for e in evaluations) / total_evals
        success_rate = sum(1 for e in evaluations if e.get("success", False)) / total_evals
        avg_confidence = sum(e.get("confidence", 0) for e in evaluations) / total_evals
        avg_safety = sum(e.get("medical_safety", 0) for e in evaluations) / total_evals
        
        # Safety analysis
        safety_concerns_count = sum(
            len(e.get("safety_concerns", [])) for e in evaluations
        )
        
        # Novel insights count
        novel_insights_count = sum(
            1 for e in evaluations if e.get("novel_insights")
        )
        
        # Performance trend analysis
        trend_analysis = self._analyze_performance_trends()
        
        # Evaluation type distribution
        type_distribution = {}
        for eval_result in evaluations:
            eval_type = eval_result.get("evaluation_type", "unknown")
            type_distribution[eval_type] = type_distribution.get(eval_type, 0) + 1
        
        return {
            "total_evaluations": total_evals,
            "average_performance": avg_performance,
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "average_safety_score": avg_safety,
            "safety_concerns_total": safety_concerns_count,
            "novel_insights_count": novel_insights_count,
            "evaluation_type_distribution": type_distribution,
            "performance_trend": trend_analysis,
            "last_evaluation": evaluations[-1]["timestamp"] if evaluations else None
        }
    
    def _analyze_performance_trends(self) -> str:
        """Analyze performance trends over time"""
        
        if len(self.evaluation_history) < 10:
            return "insufficient_data"
        
        # Get recent performance scores
        recent_scores = [
            e.get("performance_score", 0) 
            for e in self.evaluation_history[-20:]
        ]
        
        if len(recent_scores) < 10:
            return "insufficient_recent_data"
        
        # Compare first and second half
        mid_point = len(recent_scores) // 2
        early_avg = sum(recent_scores[:mid_point]) / mid_point
        late_avg = sum(recent_scores[mid_point:]) / (len(recent_scores) - mid_point)
        
        if late_avg > early_avg + 0.1:
            return "improving"
        elif late_avg < early_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    async def get_improvement_recommendations(
        self, 
        task_type: str = None, 
        limit: int = 5
    ) -> List[str]:
        """Get improvement recommendations based on evaluation history"""
        
        if not self.evaluation_history:
            return ["No evaluation history available for recommendations"]
        
        # Filter evaluations by task type if specified
        relevant_evaluations = self.evaluation_history
        if task_type:
            relevant_evaluations = [
                e for e in self.evaluation_history 
                if task_type.lower() in e.get("task_description", "").lower()
            ]
        
        if not relevant_evaluations:
            return ["No relevant evaluations found for this task type"]
        
        # Aggregate improvement suggestions
        suggestion_counts = {}
        for evaluation in relevant_evaluations[-50:]:  # Last 50 evaluations
            suggestions = evaluation.get("improvement_suggestions", [])
            for suggestion in suggestions:
                suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        # Sort by frequency and return top suggestions
        top_suggestions = sorted(
            suggestion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [suggestion for suggestion, count in top_suggestions[:limit]]
    
    async def export_evaluation_report(self, output_path: Path):
        """Export comprehensive evaluation report in JSON format"""
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate comprehensive report
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "statistics": self.get_evaluation_statistics(),
            "recent_evaluations": self.evaluation_history[-50:],  # Last 50 evaluations
            "performance_trends": self.performance_trends,
            "improvement_recommendations": await self.get_improvement_recommendations(),
            "evaluation_thresholds": self.performance_thresholds,
            "report_version": "1.0"
        }
        
        # Save report
        with open(output_path / "evaluation_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Save detailed evaluation history
        with open(output_path / "detailed_evaluation_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_history, f, ensure_ascii=False, indent=2)
    
    async def backup_evaluations(self, backup_path: Path):
        """Create comprehensive backup of all evaluation data"""
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all JSON files
        import shutil
        
        for file_path in [self.evaluation_history_path, self.judge_feedback_path, 
                         self.performance_trends_path]:
            if file_path.exists():
                shutil.copy2(file_path, backup_path / file_path.name)
        
        # Create backup manifest
        manifest = {
            "backup_timestamp": datetime.now().isoformat(),
            "total_evaluations": len(self.evaluation_history),
            "files_backed_up": [f.name for f in backup_path.iterdir() if f.is_file()],
            "evaluation_date_range": {
                "earliest": self.evaluation_history[0]["timestamp"] if self.evaluation_history else None,
                "latest": self.evaluation_history[-1]["timestamp"] if self.evaluation_history else None
            }
        }
        
        with open(backup_path / "backup_manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


# Alias for compatibility
TaskEvaluator = LLMTaskEvaluator