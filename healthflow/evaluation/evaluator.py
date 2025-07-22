"""
Task Evaluator for HealthFlow
Evaluates task execution results and provides performance metrics
Uses jsonl/parquet/pkl for persistence (no SQLite)
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle


@dataclass
class EvaluationResult:
    """Result of task evaluation"""
    success: bool
    confidence: float
    performance_score: float
    accuracy: float
    completeness: float
    medical_safety: float
    evidence_based: float
    feedback: str
    improvement_suggestions: List[str]
    novel_insights: Optional[List[str]] = None


class TaskEvaluator:
    """Evaluates medical task execution results using file-based persistence"""
    
    def __init__(self, evaluation_dir: Optional[Path] = None):
        self.evaluation_dir = evaluation_dir or Path("./data/evaluation")
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.evaluations_jsonl = self.evaluation_dir / "evaluations.jsonl"
        self.evaluations_parquet = self.evaluation_dir / "evaluations.parquet"
        self.statistics_pkl = self.evaluation_dir / "statistics.pkl"
        
        self.evaluation_history: List[Dict[str, Any]] = []
        self.performance_thresholds = {
            "success": 0.7,
            "confidence": 0.6,
            "medical_safety": 0.9,  # High threshold for safety
            "evidence_based": 0.7
        }
        
        # Load existing evaluations
        asyncio.create_task(self._load_evaluations())
    
    async def _load_evaluations(self):
        """Load existing evaluations from storage"""
        # Load from JSONL
        if self.evaluations_jsonl.exists():
            try:
                with open(self.evaluations_jsonl, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self.evaluation_history.append(data)
            except Exception as e:
                print(f"Error loading evaluations from JSONL: {e}")
        
        # Load from parquet as backup
        if self.evaluations_parquet.exists() and not self.evaluation_history:
            try:
                df = pd.read_parquet(self.evaluations_parquet)
                self.evaluation_history = df.to_dict('records')
            except Exception as e:
                print(f"Error loading evaluations from parquet: {e}")
    
    async def _save_evaluation(self, evaluation_data: Dict[str, Any]):
        """Save evaluation to persistent storage"""
        # Append to JSONL
        try:
            with open(self.evaluations_jsonl, 'a') as f:
                f.write(json.dumps(evaluation_data) + '\n')
        except Exception as e:
            print(f"Error saving evaluation to JSONL: {e}")
        
        # Save to parquet periodically
        if len(self.evaluation_history) % 100 == 0:  # Every 100 evaluations
            await self._save_to_parquet()
    
    async def _save_to_parquet(self):
        """Save evaluations to parquet format"""
        try:
            if self.evaluation_history:
                df = pd.DataFrame(self.evaluation_history)
                df.to_parquet(self.evaluations_parquet, index=False)
        except Exception as e:
            print(f"Error saving evaluations to parquet: {e}")
    
    async def evaluate_task_result(
        self,
        task_description: str,
        result: Any,
        task_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate task execution result
        
        Args:
            task_description: Original task description
            result: Task execution result
            task_context: Additional context
            
        Returns:
            Evaluation metrics and feedback
        """
        
        task_context = task_context or {}
        
        try:
            # Evaluate different aspects
            completeness_score = await self._evaluate_completeness(
                task_description, result, task_context
            )
            
            accuracy_score = await self._evaluate_accuracy(
                task_description, result, task_context
            )
            
            medical_safety_score = await self._evaluate_medical_safety(
                result, task_context
            )
            
            evidence_based_score = await self._evaluate_evidence_based(
                result, task_context
            )
            
            # Calculate overall performance
            performance_score = (
                completeness_score * 0.3 +
                accuracy_score * 0.3 +
                medical_safety_score * 0.25 +
                evidence_based_score * 0.15
            )
            
            # Determine success
            success = (
                performance_score >= self.performance_thresholds["success"] and
                medical_safety_score >= self.performance_thresholds["medical_safety"]
            )
            
            # Calculate confidence based on consistency of metrics
            confidence = self._calculate_confidence([
                completeness_score, accuracy_score, 
                medical_safety_score, evidence_based_score
            ])
            
            # Generate feedback
            feedback, improvements = await self._generate_feedback(
                task_description, result, {
                    "completeness": completeness_score,
                    "accuracy": accuracy_score,
                    "medical_safety": medical_safety_score,
                    "evidence_based": evidence_based_score,
                    "performance": performance_score
                }
            )
            
            # Check for novel insights
            novel_insights = await self._identify_novel_insights(result, task_context)
            
            # Create evaluation result
            evaluation = {
                "success": success,
                "confidence": confidence,
                "performance_score": performance_score,
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "medical_safety": medical_safety_score,
                "evidence_based": evidence_based_score,
                "feedback": feedback,
                "improvement_suggestions": improvements,
                "reward": self._calculate_reward(performance_score, success),
                "timestamp": datetime.now().isoformat(),
                "task_description": task_description,
                "result": str(result)[:500]  # Truncate for storage
            }
            
            if novel_insights:
                evaluation["novel_insights"] = novel_insights
            
            # Store evaluation history
            self.evaluation_history.append(evaluation)
            await self._save_evaluation(evaluation)
            
            return evaluation
            
        except Exception as e:
            # Return minimal evaluation on error
            error_evaluation = {
                "success": False,
                "confidence": 0.0,
                "performance_score": 0.0,
                "accuracy": 0.0,
                "completeness": 0.0,
                "medical_safety": 0.5,  # Neutral safety score
                "evidence_based": 0.0,
                "feedback": f"Evaluation error: {str(e)}",
                "improvement_suggestions": ["Fix evaluation error"],
                "reward": 0.0,
                "timestamp": datetime.now().isoformat(),
                "task_description": task_description,
                "result": str(result)[:500] if result else ""
            }
            
            self.evaluation_history.append(error_evaluation)
            await self._save_evaluation(error_evaluation)
            
            return error_evaluation
    
    async def _evaluate_completeness(
        self,
        task_description: str,
        result: Any,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate how complete the result is"""
        
        if not result:
            return 0.0
        
        result_str = str(result).lower()
        task_str = task_description.lower()
        
        # Check for key medical components
        medical_components = [
            'diagnosis', 'treatment', 'symptoms', 'analysis',
            'recommendation', 'assessment', 'evaluation'
        ]
        
        # Count how many expected components are present
        present_components = sum(
            1 for component in medical_components
            if component in task_str and component in result_str
        )
        
        # Base completeness score
        completeness = min(1.0, present_components / max(1, len(medical_components) * 0.5))
        
        # Bonus for comprehensive responses
        if len(result_str) > 200:  # Detailed response
            completeness = min(1.0, completeness + 0.1)
        
        # Penalty for very short responses
        if len(result_str) < 50:
            completeness *= 0.7
        
        return completeness
    
    async def _evaluate_accuracy(
        self,
        task_description: str,
        result: Any,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate accuracy of the result"""
        
        if not result:
            return 0.0
        
        result_str = str(result).lower()
        
        # Check for medical accuracy indicators
        accuracy_indicators = [
            'evidence-based', 'clinical guidelines', 'peer-reviewed',
            'systematic review', 'meta-analysis', 'randomized controlled trial',
            'clinical trial', 'scientific literature'
        ]
        
        accuracy_count = sum(
            1 for indicator in accuracy_indicators
            if indicator in result_str
        )
        
        # Check for uncertainty acknowledgment (good practice)
        uncertainty_phrases = [
            'may', 'might', 'could', 'suggests', 'indicates',
            'further evaluation needed', 'consult', 'specialist'
        ]
        
        uncertainty_count = sum(
            1 for phrase in uncertainty_phrases
            if phrase in result_str
        )
        
        # Base accuracy score
        accuracy = min(1.0, (accuracy_count * 0.3 + uncertainty_count * 0.1))
        
        # Bonus for structured responses
        if any(marker in result_str for marker in ['1.', '2.', '•', '-']):
            accuracy = min(1.0, accuracy + 0.2)
        
        # Default reasonable accuracy if no specific indicators
        if accuracy < 0.3:
            accuracy = 0.6  # Assume reasonable baseline
        
        return accuracy
    
    async def _evaluate_medical_safety(
        self,
        result: Any,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate medical safety of the result"""
        
        if not result:
            return 0.5  # Neutral safety
        
        result_str = str(result).lower()
        
        # Check for safety red flags
        safety_red_flags = [
            'self-medicate', 'ignore symptoms', 'delay treatment',
            'skip medication', 'avoid doctor', 'home remedy only',
            'definitely', 'certainly', 'guaranteed cure'
        ]
        
        red_flag_count = sum(
            1 for flag in safety_red_flags
            if flag in result_str
        )
        
        # Check for safety best practices
        safety_practices = [
            'consult healthcare provider', 'seek medical attention',
            'follow up', 'monitor', 'emergency', 'call doctor',
            'professional medical advice', 'healthcare professional'
        ]
        
        safety_count = sum(
            1 for practice in safety_practices
            if practice in result_str
        )
        
        # Calculate safety score
        safety_score = 0.8  # Start with reasonable baseline
        
        # Penalties for red flags
        safety_score -= (red_flag_count * 0.3)
        
        # Bonuses for safety practices
        safety_score += (safety_count * 0.1)
        
        return max(0.0, min(1.0, safety_score))
    
    async def _evaluate_evidence_based(
        self,
        result: Any,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate how evidence-based the result is"""
        
        if not result:
            return 0.0
        
        result_str = str(result).lower()
        
        # Check for evidence-based indicators
        evidence_indicators = [
            'studies show', 'research indicates', 'clinical evidence',
            'published research', 'medical literature', 'guidelines recommend',
            'according to', 'based on research', 'evidence suggests'
        ]
        
        evidence_count = sum(
            1 for indicator in evidence_indicators
            if indicator in result_str
        )
        
        # Check for specific evidence types
        high_quality_evidence = [
            'systematic review', 'meta-analysis', 'randomized controlled trial',
            'cochrane review', 'clinical practice guidelines'
        ]
        
        high_quality_count = sum(
            1 for evidence in high_quality_evidence
            if evidence in result_str
        )
        
        # Calculate evidence-based score
        evidence_score = (evidence_count * 0.2) + (high_quality_count * 0.4)
        
        # Bonus for citing specific sources or years
        if any(year in result_str for year in ['2020', '2021', '2022', '2023', '2024']):
            evidence_score += 0.1
        
        return min(1.0, evidence_score)
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on consistency of scores"""
        
        if not scores:
            return 0.0
        
        # Calculate standard deviation
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Higher consistency = higher confidence
        # Lower standard deviation = higher confidence
        confidence = max(0.0, 1.0 - (std_dev * 2))
        
        # Adjust based on mean score
        confidence = (confidence + mean_score) / 2
        
        return min(1.0, confidence)
    
    async def _generate_feedback(
        self,
        task_description: str,
        result: Any,
        scores: Dict[str, float]
    ) -> Tuple[str, List[str]]:
        """Generate feedback and improvement suggestions"""
        
        feedback_parts = []
        improvements = []
        
        # Overall performance feedback
        performance = scores["performance"]
        if performance >= 0.8:
            feedback_parts.append("Excellent task execution with high quality results.")
        elif performance >= 0.6:
            feedback_parts.append("Good task execution with room for improvement.")
        else:
            feedback_parts.append("Task execution needs significant improvement.")
        
        # Specific aspect feedback
        if scores["completeness"] < 0.7:
            feedback_parts.append("Response could be more comprehensive.")
            improvements.append("Provide more detailed and complete responses")
        
        if scores["accuracy"] < 0.7:
            feedback_parts.append("Accuracy could be improved with better evidence.")
            improvements.append("Use more evidence-based sources and references")
        
        if scores["medical_safety"] < 0.8:
            feedback_parts.append("IMPORTANT: Medical safety considerations need attention.")
            improvements.append("Always prioritize patient safety and include appropriate warnings")
        
        if scores["evidence_based"] < 0.6:
            feedback_parts.append("Response would benefit from more evidence-based information.")
            improvements.append("Include references to medical literature and clinical guidelines")
        
        # Positive reinforcement
        best_aspect = max(scores.items(), key=lambda x: x[1])
        if best_aspect[1] > 0.7:
            feedback_parts.append(f"Strong performance in {best_aspect[0].replace('_', ' ')}.")
        
        feedback = " ".join(feedback_parts)
        
        return feedback, improvements
    
    async def _identify_novel_insights(
        self,
        result: Any,
        context: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Identify novel insights in the result"""
        
        if not result:
            return None
        
        result_str = str(result).lower()
        
        # Look for novel insight indicators
        insight_indicators = [
            'new approach', 'novel method', 'innovative',
            'unprecedented', 'breakthrough', 'discovery',
            'first time', 'unique finding', 'unexpected'
        ]
        
        insights = []
        for indicator in insight_indicators:
            if indicator in result_str:
                # Extract the sentence containing the insight
                sentences = result_str.split('.')
                for sentence in sentences:
                    if indicator in sentence:
                        insights.append(sentence.strip().capitalize())
                        break
        
        return insights if insights else None
    
    def _calculate_reward(self, performance_score: float, success: bool) -> float:
        """Calculate reward signal for reinforcement learning"""
        
        base_reward = performance_score
        
        # Bonus for successful completion
        if success:
            base_reward += 0.2
        
        # Exponential scaling for high performance
        if performance_score > 0.8:
            base_reward += (performance_score - 0.8) * 0.5
        
        # Penalty for poor performance
        if performance_score < 0.5:
            base_reward *= 0.5
        
        return min(1.0, max(0.0, base_reward))
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about evaluation history"""
        
        if not self.evaluation_history:
            return {}
        
        evaluations = self.evaluation_history
        
        # Calculate basic statistics
        total_evals = len(evaluations)
        avg_performance = sum(e.get("performance_score", 0) for e in evaluations) / total_evals
        success_rate = sum(1 for e in evaluations if e.get("success", False)) / total_evals
        avg_confidence = sum(e.get("confidence", 0) for e in evaluations) / total_evals
        avg_safety = sum(e.get("medical_safety", 0) for e in evaluations) / total_evals
        novel_insights_count = sum(1 for e in evaluations if e.get("novel_insights"))
        
        return {
            "total_evaluations": total_evals,
            "average_performance": avg_performance,
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "average_safety_score": avg_safety,
            "novel_insights_count": novel_insights_count,
            "recent_performance_trend": self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend"""
        
        if len(self.evaluation_history) < 10:
            return "insufficient_data"
        
        recent_scores = [
            e.get("performance_score", 0) 
            for e in self.evaluation_history[-10:]
        ]
        
        early_avg = sum(recent_scores[:5]) / 5
        late_avg = sum(recent_scores[5:]) / 5
        
        if late_avg > early_avg + 0.1:
            return "improving"
        elif late_avg < early_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    async def backup_evaluations(self, backup_path: Path):
        """Backup evaluations to specified path"""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(backup_path / "evaluations_backup.json", 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        
        # Save as parquet if we have data
        if self.evaluation_history:
            df = pd.DataFrame(self.evaluation_history)
            df.to_parquet(backup_path / "evaluations_backup.parquet", index=False)
        
        # Save statistics
        stats = self.get_evaluation_statistics()
        with open(backup_path / "evaluation_statistics.pkl", 'wb') as f:
            pickle.dump(stats, f)