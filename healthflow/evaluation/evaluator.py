"""
Task Evaluation System for HealthFlow

This module implements comprehensive evaluation mechanisms for healthcare tasks:
- Performance assessment
- Clinical accuracy evaluation  
- Safety and compliance checking
- Quality metrics calculation
- Comparative benchmarking
- User satisfaction tracking
"""

import json
import sqlite3
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import re


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating task performance"""
    accuracy: float = 1.0
    completeness: float = 1.0
    clinical_relevance: float = 1.0
    safety: float = 1.0
    compliance: float = 1.0
    efficiency: float = 1.0
    user_satisfaction: float = 1.0


@dataclass
class EvaluationResult:
    """Result of task evaluation"""
    evaluation_id: str
    task_description: str
    task_type: str
    result: Any
    criteria_scores: EvaluationCriteria
    overall_score: float
    success: bool
    feedback: str
    recommendations: List[str]
    safety_flags: List[str]
    compliance_issues: List[str]
    timestamp: datetime


@dataclass
class BenchmarkResult:
    """Benchmark comparison result"""
    benchmark_name: str
    score: float
    percentile: float
    comparison_data: Dict[str, Any]
    improvement_suggestions: List[str]


class TaskEvaluator:
    """
    Comprehensive task evaluation system for healthcare tasks.
    
    Features:
    - Multi-dimensional performance assessment
    - Clinical accuracy validation
    - Safety and compliance monitoring
    - Benchmarking against standards
    - Continuous improvement suggestions
    """
    
    def __init__(self, evaluation_dir: str = "evaluation"):
        self.evaluation_dir = Path(evaluation_dir)
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.evaluation_dir / "evaluations.db"
        self._init_database()
        
        self.logger = logging.getLogger("TaskEvaluator")
        
        # Load evaluation templates and benchmarks
        self.evaluation_templates = self._load_evaluation_templates()
        self.benchmarks = self._load_benchmarks()
        
        self.logger.info("Task evaluator initialized")
    
    def _init_database(self):
        """Initialize SQLite database for evaluation storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    task_description TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    result TEXT NOT NULL,
                    criteria_scores TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    feedback TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    safety_flags TEXT NOT NULL,
                    compliance_issues TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    benchmark_id TEXT PRIMARY KEY,
                    benchmark_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    percentile REAL NOT NULL,
                    comparison_data TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_task_type ON evaluations (task_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_success ON evaluations (success)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_timestamp ON evaluations (timestamp)")
    
    def _load_evaluation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load evaluation templates for different task types"""
        
        # In production, these would be loaded from configuration files
        templates = {
            "diagnosis": {
                "primary_metrics": ["accuracy", "clinical_relevance", "safety"],
                "weight_accuracy": 0.4,
                "weight_clinical_relevance": 0.3,
                "weight_safety": 0.3,
                "safety_critical": True,
                "compliance_required": ["HIPAA", "clinical_guidelines"]
            },
            "treatment_planning": {
                "primary_metrics": ["accuracy", "safety", "compliance"],
                "weight_accuracy": 0.35,
                "weight_safety": 0.4,
                "weight_compliance": 0.25,
                "safety_critical": True,
                "compliance_required": ["FDA_guidelines", "clinical_protocols"]
            },
            "data_analysis": {
                "primary_metrics": ["accuracy", "completeness", "efficiency"],
                "weight_accuracy": 0.4,
                "weight_completeness": 0.35,
                "weight_efficiency": 0.25,
                "safety_critical": False,
                "compliance_required": ["data_privacy", "HIPAA"]
            },
            "literature_review": {
                "primary_metrics": ["completeness", "clinical_relevance", "efficiency"],
                "weight_completeness": 0.4,
                "weight_clinical_relevance": 0.35,
                "weight_efficiency": 0.25,
                "safety_critical": False,
                "compliance_required": ["copyright", "attribution"]
            },
            "drug_interaction": {
                "primary_metrics": ["accuracy", "safety", "completeness"],
                "weight_accuracy": 0.4,
                "weight_safety": 0.5,
                "weight_completeness": 0.1,
                "safety_critical": True,
                "compliance_required": ["FDA_drug_database", "clinical_guidelines"]
            }
        }
        
        return templates
    
    def _load_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Load benchmark data for different task types"""
        
        # In production, these would be loaded from actual benchmark datasets
        benchmarks = {
            "diagnosis": {
                "expert_accuracy": 0.85,
                "average_time": 120,  # seconds
                "safety_incidents": 0.02,  # 2% rate
                "patient_satisfaction": 0.8
            },
            "treatment_planning": {
                "guideline_adherence": 0.9,
                "safety_score": 0.95,
                "patient_outcomes": 0.82,
                "cost_effectiveness": 0.75
            },
            "data_analysis": {
                "accuracy_threshold": 0.95,
                "processing_speed": 60,  # seconds per 1000 records
                "completeness_rate": 0.98
            }
        }
        
        return benchmarks
    
    async def evaluate_task_result(
        self,
        task_description: str,
        result: Any,
        task_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of a task result"""
        
        self.logger.info(f"Evaluating task result for: {task_description[:50]}...")
        
        # Get evaluation template for task type
        template = self.evaluation_templates.get(task_type, self.evaluation_templates["data_analysis"])
        
        # Evaluate different criteria
        criteria_scores = await self._evaluate_criteria(
            task_description, result, task_type, template, context
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(criteria_scores, template)
        
        # Determine success
        success = overall_score >= 0.7 and not await self._has_critical_issues(result, template)
        
        # Generate feedback and recommendations
        feedback = self._generate_feedback(criteria_scores, overall_score, template)
        recommendations = self._generate_recommendations(criteria_scores, template)
        
        # Check safety and compliance
        safety_flags = await self._check_safety(result, template)
        compliance_issues = await self._check_compliance(result, template)
        
        # Create evaluation result
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        evaluation_result = EvaluationResult(
            evaluation_id=evaluation_id,
            task_description=task_description,
            task_type=task_type,
            result=result,
            criteria_scores=criteria_scores,
            overall_score=overall_score,
            success=success,
            feedback=feedback,
            recommendations=recommendations,
            safety_flags=safety_flags,
            compliance_issues=compliance_issues,
            timestamp=datetime.now()
        )
        
        # Store evaluation
        await self._store_evaluation(evaluation_result)
        
        # Return evaluation as dict
        return {
            "success": success,
            "overall_score": overall_score,
            "performance_score": overall_score,  # Alias for compatibility
            "confidence": criteria_scores.accuracy,
            "feedback": feedback,
            "recommendations": recommendations,
            "safety_flags": safety_flags,
            "compliance_issues": compliance_issues,
            "criteria_breakdown": asdict(criteria_scores),
            "evaluation_id": evaluation_id
        }
    
    async def _evaluate_criteria(
        self,
        task_description: str,
        result: Any,
        task_type: str,
        template: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> EvaluationCriteria:
        """Evaluate specific criteria for the task"""
        
        # Accuracy assessment
        accuracy = await self._assess_accuracy(result, task_type, context)
        
        # Completeness assessment
        completeness = await self._assess_completeness(result, task_description)
        
        # Clinical relevance assessment
        clinical_relevance = await self._assess_clinical_relevance(result, task_type)
        
        # Safety assessment
        safety = await self._assess_safety(result, template)
        
        # Compliance assessment
        compliance = await self._assess_compliance(result, template)
        
        # Efficiency assessment
        efficiency = await self._assess_efficiency(context)
        
        # User satisfaction (placeholder - would be based on feedback)
        user_satisfaction = 0.8  # Default value
        
        return EvaluationCriteria(
            accuracy=accuracy,
            completeness=completeness,
            clinical_relevance=clinical_relevance,
            safety=safety,
            compliance=compliance,
            efficiency=efficiency,
            user_satisfaction=user_satisfaction
        )
    
    async def _assess_accuracy(
        self,
        result: Any,
        task_type: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Assess the accuracy of the result"""
        
        if not result:
            return 0.0
        
        # Basic accuracy assessment based on result structure and content
        score = 0.5  # Base score
        
        result_str = str(result).lower()
        
        # Check for specific indicators of accuracy
        if task_type == "diagnosis":
            # Look for medical terminology and structured output
            medical_terms = ['symptom', 'diagnosis', 'condition', 'disease', 'treatment']
            medical_term_count = sum(1 for term in medical_terms if term in result_str)
            score += min(medical_term_count * 0.1, 0.3)
            
            # Check for confidence indicators
            if 'confidence' in result_str or 'certainty' in result_str:
                score += 0.1
        
        elif task_type == "data_analysis":
            # Look for statistical measures and structured results
            analysis_terms = ['average', 'mean', 'standard deviation', 'correlation', 'p-value']
            analysis_term_count = sum(1 for term in analysis_terms if term in result_str)
            score += min(analysis_term_count * 0.1, 0.3)
        
        # Check for reasoning and explanation
        if len(result_str) > 100:  # Detailed response
            score += 0.1
        
        # Check for structured output
        if isinstance(result, dict) and len(result) > 1:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _assess_completeness(self, result: Any, task_description: str) -> float:
        """Assess completeness of the result"""
        
        if not result:
            return 0.0
        
        result_str = str(result)
        task_keywords = self._extract_task_keywords(task_description)
        
        # Check how many task keywords are addressed in the result
        addressed_keywords = sum(1 for keyword in task_keywords if keyword.lower() in result_str.lower())
        
        if len(task_keywords) == 0:
            return 0.8  # Default if no specific keywords identified
        
        keyword_coverage = addressed_keywords / len(task_keywords)
        
        # Adjust based on result length (longer results generally more complete)
        length_factor = min(len(result_str) / 500, 1.0)  # Cap at 500 chars
        
        completeness_score = (keyword_coverage * 0.7) + (length_factor * 0.3)
        
        return min(completeness_score, 1.0)
    
    def _extract_task_keywords(self, task_description: str) -> List[str]:
        """Extract key terms from task description"""
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', task_description.lower())
        
        # Filter for meaningful words (length > 3, not common words)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'does', 'each', 'she', 'that', 'their', 'what', 'will', 'with'}
        
        keywords = [word for word in words if len(word) > 3 and word not in common_words]
        
        return list(set(keywords))  # Remove duplicates
    
    async def _assess_clinical_relevance(self, result: Any, task_type: str) -> float:
        """Assess clinical relevance of the result"""
        
        if not result:
            return 0.0
        
        result_str = str(result).lower()
        
        # Check for clinical terminology
        clinical_terms = [
            'patient', 'clinical', 'medical', 'diagnosis', 'treatment', 'therapy',
            'symptom', 'disease', 'condition', 'medication', 'drug', 'dose',
            'laboratory', 'test', 'result', 'examination', 'assessment'
        ]
        
        clinical_term_count = sum(1 for term in clinical_terms if term in result_str)
        
        # Base score based on clinical terminology
        base_score = min(clinical_term_count * 0.1, 0.8)
        
        # Boost for specific task types
        if task_type in ['diagnosis', 'treatment_planning', 'drug_interaction']:
            base_score = min(base_score * 1.2, 1.0)
        
        # Check for evidence-based language
        evidence_terms = ['evidence', 'study', 'research', 'guideline', 'protocol']
        evidence_count = sum(1 for term in evidence_terms if term in result_str)
        
        if evidence_count > 0:
            base_score = min(base_score + 0.1, 1.0)
        
        return base_score
    
    async def _assess_safety(self, result: Any, template: Dict[str, Any]) -> float:
        """Assess safety aspects of the result"""
        
        if not result:
            return 0.5  # Neutral if no result
        
        result_str = str(result).lower()
        
        # Check for safety warnings or considerations
        safety_terms = ['warning', 'caution', 'risk', 'adverse', 'contraindication', 'side effect']
        safety_mentions = sum(1 for term in safety_terms if term in result_str)
        
        base_score = 0.7  # Start with good safety score
        
        # Boost for mentioning safety considerations
        if safety_mentions > 0:
            base_score = min(base_score + 0.2, 1.0)
        
        # Check for dangerous recommendations (simplified)
        dangerous_terms = ['high dose', 'overdose', 'toxic', 'lethal']
        dangerous_mentions = sum(1 for term in dangerous_terms if term in result_str)
        
        if dangerous_mentions > 0:
            base_score = max(base_score - 0.3, 0.0)
        
        # For safety-critical tasks, be more stringent
        if template.get('safety_critical', False):
            if safety_mentions == 0:
                base_score = min(base_score - 0.1, 0.9)
        
        return base_score
    
    async def _assess_compliance(self, result: Any, template: Dict[str, Any]) -> float:
        """Assess compliance with regulations and guidelines"""
        
        if not result:
            return 0.5
        
        result_str = str(result).lower()
        
        base_score = 0.8  # Start with good compliance score
        
        # Check for compliance-related language
        compliance_terms = ['guideline', 'protocol', 'standard', 'regulation', 'approved']
        compliance_mentions = sum(1 for term in compliance_terms if term in result_str)
        
        if compliance_mentions > 0:
            base_score = min(base_score + 0.1, 1.0)
        
        # Check for privacy protection language (HIPAA compliance)
        privacy_terms = ['confidential', 'privacy', 'protected', 'anonymous', 'de-identified']
        privacy_mentions = sum(1 for term in privacy_terms if term in result_str)
        
        if privacy_mentions > 0:
            base_score = min(base_score + 0.1, 1.0)
        
        return base_score
    
    async def _assess_efficiency(self, context: Optional[Dict[str, Any]]) -> float:
        """Assess efficiency of task execution"""
        
        if not context:
            return 0.7  # Default efficiency score
        
        # Check execution time if available
        if 'execution_time' in context:
            execution_time = context['execution_time']
            
            # Score based on execution time (lower is better)
            if execution_time < 30:  # Less than 30 seconds
                return 1.0
            elif execution_time < 60:  # Less than 1 minute
                return 0.9
            elif execution_time < 120:  # Less than 2 minutes
                return 0.8
            elif execution_time < 300:  # Less than 5 minutes
                return 0.7
            else:
                return 0.6
        
        # Check resource usage if available
        if 'memory_used' in context and 'tools_used' in context:
            memory_used = context['memory_used']
            tools_count = len(context['tools_used'])
            
            # Lower memory and fewer tools generally indicate efficiency
            memory_score = 1.0 - min(memory_used / 10000, 0.3)  # Cap penalty at 0.3
            tools_score = 1.0 - min(tools_count / 10, 0.2)  # Cap penalty at 0.2
            
            return (memory_score + tools_score) / 2
        
        return 0.7  # Default
    
    def _calculate_overall_score(
        self,
        criteria_scores: EvaluationCriteria,
        template: Dict[str, Any]
    ) -> float:
        """Calculate overall weighted score"""
        
        # Get weights from template or use defaults
        weights = {
            'accuracy': template.get('weight_accuracy', 0.3),
            'completeness': template.get('weight_completeness', 0.2),
            'clinical_relevance': template.get('weight_clinical_relevance', 0.2),
            'safety': template.get('weight_safety', 0.15),
            'compliance': template.get('weight_compliance', 0.1),
            'efficiency': template.get('weight_efficiency', 0.05)
        }
        
        # Ensure weights sum to 1.0
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        overall_score = (
            criteria_scores.accuracy * weights['accuracy'] +
            criteria_scores.completeness * weights['completeness'] +
            criteria_scores.clinical_relevance * weights['clinical_relevance'] +
            criteria_scores.safety * weights['safety'] +
            criteria_scores.compliance * weights['compliance'] +
            criteria_scores.efficiency * weights['efficiency']
        )
        
        return min(overall_score, 1.0)
    
    async def _has_critical_issues(self, result: Any, template: Dict[str, Any]) -> bool:
        """Check for critical issues that would fail the task"""
        
        if not result:
            return True  # No result is a critical issue
        
        # For safety-critical tasks, check for safety violations
        if template.get('safety_critical', False):
            safety_score = await self._assess_safety(result, template)
            if safety_score < 0.5:
                return True
        
        # Check for obvious errors in result
        result_str = str(result).lower()
        error_indicators = ['error', 'failed', 'exception', 'invalid', 'unable to', 'cannot']
        
        if any(indicator in result_str for indicator in error_indicators):
            return True
        
        return False
    
    def _generate_feedback(
        self,
        criteria_scores: EvaluationCriteria,
        overall_score: float,
        template: Dict[str, Any]
    ) -> str:
        """Generate human-readable feedback"""
        
        feedback_parts = []
        
        # Overall performance
        if overall_score >= 0.9:
            feedback_parts.append("Excellent performance across all criteria.")
        elif overall_score >= 0.8:
            feedback_parts.append("Very good performance with minor areas for improvement.")
        elif overall_score >= 0.7:
            feedback_parts.append("Good performance but with some areas needing attention.")
        elif overall_score >= 0.6:
            feedback_parts.append("Acceptable performance but significant improvements needed.")
        else:
            feedback_parts.append("Performance below standards, major improvements required.")
        
        # Specific criteria feedback
        if criteria_scores.accuracy < 0.7:
            feedback_parts.append("Accuracy needs improvement - verify information sources and methods.")
        
        if criteria_scores.completeness < 0.7:
            feedback_parts.append("Response completeness is lacking - address all aspects of the task.")
        
        if criteria_scores.clinical_relevance < 0.7:
            feedback_parts.append("Clinical relevance could be stronger - focus on practical applications.")
        
        if criteria_scores.safety < 0.8:
            feedback_parts.append("Safety considerations need more attention - include relevant warnings.")
        
        if criteria_scores.compliance < 0.8:
            feedback_parts.append("Ensure compliance with relevant guidelines and regulations.")
        
        if criteria_scores.efficiency < 0.7:
            feedback_parts.append("Efficiency could be improved - optimize resource usage and execution time.")
        
        return " ".join(feedback_parts)
    
    def _generate_recommendations(
        self,
        criteria_scores: EvaluationCriteria,
        template: Dict[str, Any]
    ) -> List[str]:
        """Generate specific improvement recommendations"""
        
        recommendations = []
        
        if criteria_scores.accuracy < 0.8:
            recommendations.append("Verify information against authoritative medical sources")
            recommendations.append("Implement additional fact-checking mechanisms")
        
        if criteria_scores.completeness < 0.8:
            recommendations.append("Develop comprehensive checklists for task requirements")
            recommendations.append("Implement systematic review of all task components")
        
        if criteria_scores.clinical_relevance < 0.8:
            recommendations.append("Consult clinical guidelines and evidence-based practices")
            recommendations.append("Include practical implementation considerations")
        
        if criteria_scores.safety < 0.9:
            recommendations.append("Implement additional safety checks and warnings")
            recommendations.append("Review contraindications and adverse effects")
        
        if criteria_scores.compliance < 0.9:
            recommendations.append("Review applicable regulations and guidelines")
            recommendations.append("Implement compliance verification procedures")
        
        if criteria_scores.efficiency < 0.8:
            recommendations.append("Optimize tool usage and resource allocation")
            recommendations.append("Streamline execution workflow")
        
        return recommendations
    
    async def _check_safety(self, result: Any, template: Dict[str, Any]) -> List[str]:
        """Check for specific safety flags"""
        
        safety_flags = []
        
        if not result:
            return safety_flags
        
        result_str = str(result).lower()
        
        # Check for potentially dangerous recommendations
        if 'high dose' in result_str or 'maximum dose' in result_str:
            safety_flags.append("High dosage recommendations detected - verify safety")
        
        if 'off-label' in result_str:
            safety_flags.append("Off-label use mentioned - ensure proper precautions")
        
        # Check for missing safety information in safety-critical tasks
        if template.get('safety_critical', False):
            safety_terms = ['warning', 'caution', 'contraindication', 'adverse']
            if not any(term in result_str for term in safety_terms):
                safety_flags.append("Safety-critical task lacks safety considerations")
        
        return safety_flags
    
    async def _check_compliance(self, result: Any, template: Dict[str, Any]) -> List[str]:
        """Check for compliance issues"""
        
        compliance_issues = []
        
        if not result:
            return compliance_issues
        
        result_str = str(result).lower()
        
        # Check for required compliance elements
        required_compliance = template.get('compliance_required', [])
        
        for requirement in required_compliance:
            if 'hipaa' in requirement.lower():
                privacy_terms = ['confidential', 'privacy', 'protected', 'de-identified']
                if not any(term in result_str for term in privacy_terms):
                    compliance_issues.append("HIPAA privacy considerations not addressed")
            
            elif 'fda' in requirement.lower():
                if 'approved' not in result_str and 'fda' not in result_str:
                    compliance_issues.append("FDA approval status not mentioned")
            
            elif 'clinical' in requirement.lower():
                if 'guideline' not in result_str and 'protocol' not in result_str:
                    compliance_issues.append("Clinical guidelines not referenced")
        
        return compliance_issues
    
    async def _store_evaluation(self, evaluation_result: EvaluationResult):
        """Store evaluation result in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO evaluations
                (evaluation_id, task_description, task_type, result, criteria_scores,
                 overall_score, success, feedback, recommendations, safety_flags,
                 compliance_issues, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation_result.evaluation_id,
                evaluation_result.task_description,
                evaluation_result.task_type,
                json.dumps(evaluation_result.result, default=str),
                json.dumps(asdict(evaluation_result.criteria_scores)),
                evaluation_result.overall_score,
                evaluation_result.success,
                evaluation_result.feedback,
                json.dumps(evaluation_result.recommendations),
                json.dumps(evaluation_result.safety_flags),
                json.dumps(evaluation_result.compliance_issues),
                evaluation_result.timestamp.isoformat()
            ))
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(overall_score) as avg_score,
                    AVG(CAST(success AS FLOAT)) as success_rate
                FROM evaluations
            """)
            
            total, avg_score, success_rate = cursor.fetchone()
            
            # Statistics by task type
            cursor = conn.execute("""
                SELECT 
                    task_type,
                    COUNT(*) as count,
                    AVG(overall_score) as avg_score,
                    AVG(CAST(success AS FLOAT)) as success_rate
                FROM evaluations
                GROUP BY task_type
                ORDER BY count DESC
            """)
            
            task_type_stats = {}
            for row in cursor.fetchall():
                task_type_stats[row[0]] = {
                    "count": row[1],
                    "avg_score": row[2],
                    "success_rate": row[3]
                }
            
            # Recent performance trend
            cursor = conn.execute("""
                SELECT overall_score
                FROM evaluations
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            
            recent_scores = [row[0] for row in cursor.fetchall()]
            performance_trend = "stable"
            
            if len(recent_scores) >= 5:
                recent_avg = np.mean(recent_scores[:5])
                older_avg = np.mean(recent_scores[5:])
                
                if recent_avg > older_avg + 0.05:
                    performance_trend = "improving"
                elif recent_avg < older_avg - 0.05:
                    performance_trend = "declining"
        
        return {
            "total_evaluations": total or 0,
            "average_score": avg_score or 0.0,
            "overall_success_rate": success_rate or 0.0,
            "task_type_breakdown": task_type_stats,
            "performance_trend": performance_trend,
            "database_path": str(self.db_path)
        }