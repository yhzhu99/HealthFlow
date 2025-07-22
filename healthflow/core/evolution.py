"""
Experience Accumulation and Evolution System for HealthFlow

This module implements the core self-evolution capabilities:
- Prompt evolution based on past failures and successes
- Experience-based learning and adaptation
- Performance pattern recognition
- Agent collaboration pattern learning
- Continuous improvement of reasoning strategies
"""

import json
import sqlite3
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import logging
from pathlib import Path
from collections import defaultdict, Counter

from .security import DataProtector


@dataclass
class Experience:
    """Individual experience record"""
    experience_id: str
    task_type: str
    task_description: str
    context: Dict[str, Any]
    result: Any
    evaluation: Dict[str, Any]
    tools_used: List[str]
    success: bool
    performance_score: float
    timestamp: datetime
    lessons_learned: List[str]


@dataclass
class PromptEvolution:
    """Tracks evolution of prompts over time"""
    prompt_id: str
    task_type: str
    prompt_version: int
    prompt_content: str
    performance_history: List[float]
    success_rate: float
    usage_count: int
    created_at: datetime
    last_updated: datetime


@dataclass
class PatternInsight:
    """Insights derived from experience patterns"""
    pattern_id: str
    pattern_type: str  # failure_pattern, success_pattern, tool_usage_pattern
    description: str
    confidence: float
    supporting_experiences: List[str]
    actionable_recommendations: List[str]
    discovered_at: datetime


class ExperienceAccumulator:
    """
    Manages experience accumulation and learning for continuous improvement.
    
    Key features:
    - Tracks all task executions and outcomes
    - Identifies patterns in successes and failures
    - Evolves prompts based on experience
    - Generates insights for future tasks
    - Learns optimal tool usage patterns
    - Develops domain-specific expertise
    """
    
    def __init__(self, agent_id: str, experience_dir: str = "experience"):
        self.agent_id = agent_id
        self.experience_dir = Path(experience_dir)
        self.experience_dir.mkdir(exist_ok=True)
        
        # Initialize database for experience storage
        self.db_path = self.experience_dir / f"{agent_id}_experience.db"
        self._init_database()
        
        # Configuration
        self.config = {
            "min_experiences_for_pattern": 5,
            "prompt_evolution_threshold": 0.7,  # Success rate below which prompt evolves
            "pattern_confidence_threshold": 0.8,
            "max_prompt_versions": 10,
            "insight_decay_days": 30
        }
        
        self.logger = logging.getLogger(f"ExperienceAccumulator-{agent_id}")
        self.logger.info(f"Experience accumulator initialized for agent {agent_id}")
    
    def _init_database(self):
        """Initialize SQLite database for experience storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Experiences table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    task_description TEXT NOT NULL,
                    context TEXT NOT NULL,
                    result TEXT NOT NULL,
                    evaluation TEXT NOT NULL,
                    tools_used TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    performance_score REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    lessons_learned TEXT NOT NULL
                )
            """)
            
            # Prompt evolution table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_evolution (
                    prompt_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    prompt_version INTEGER NOT NULL,
                    prompt_content TEXT NOT NULL,
                    performance_history TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)
            
            # Pattern insights table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_insights (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    supporting_experiences TEXT NOT NULL,
                    actionable_recommendations TEXT NOT NULL,
                    discovered_at TEXT NOT NULL
                )
            """)
            
            # Performance metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Create indexes for faster queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_type ON experiences (task_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_success ON experiences (success)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance ON experiences (performance_score)")
    
    async def add_experience(
        self,
        task_type: str,
        task_description: str,
        execution_context: Dict[str, Any],
        result: Any,
        evaluation: Dict[str, Any],
        tools_used: List[str]
    ) -> str:
        """Add a new experience to the accumulator"""
        
        # Generate unique experience ID
        timestamp = datetime.now()
        experience_id = hashlib.md5(
            f"{self.agent_id}_{timestamp.isoformat()}_{task_type}".encode()
        ).hexdigest()
        
        # Extract key metrics
        success = evaluation.get('success', False)
        performance_score = evaluation.get('performance_score', 0.5)
        
        # Extract lessons learned
        lessons_learned = self._extract_lessons_learned(
            task_description, result, evaluation, tools_used
        )
        
        # Create experience object
        experience = Experience(
            experience_id=experience_id,
            task_type=task_type,
            task_description=task_description,
            context=execution_context,
            result=result,
            evaluation=evaluation,
            tools_used=tools_used,
            success=success,
            performance_score=performance_score,
            timestamp=timestamp,
            lessons_learned=lessons_learned
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiences 
                (experience_id, task_type, task_description, context, result, 
                 evaluation, tools_used, success, performance_score, timestamp, lessons_learned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experience.experience_id,
                experience.task_type,
                experience.task_description,
                json.dumps(experience.context, default=str),
                json.dumps(experience.result, default=str),
                json.dumps(experience.evaluation, default=str),
                json.dumps(experience.tools_used),
                experience.success,
                experience.performance_score,
                experience.timestamp.isoformat(),
                json.dumps(experience.lessons_learned)
            ))
        
        # Update performance metrics
        await self._update_performance_metrics(task_type, performance_score, timestamp)
        
        # Trigger pattern analysis
        await self._analyze_patterns()
        
        # Trigger prompt evolution if needed
        await self._check_prompt_evolution(task_type)
        
        self.logger.info(f"Added experience {experience_id} for task type {task_type}")
        return experience_id
    
    async def get_task_insights(
        self,
        task_type: str,
        task_description: str
    ) -> List[str]:
        """Get experience-based insights for a task"""
        
        insights = []
        
        # Get successful patterns for this task type
        successful_experiences = await self._get_successful_experiences(task_type)
        if successful_experiences:
            insights.append(
                f"Based on {len(successful_experiences)} successful experiences in {task_type}, "
                f"consider using tools: {self._get_common_successful_tools(successful_experiences)}"
            )
        
        # Get failure patterns to avoid
        failure_patterns = await self._get_failure_patterns(task_type)
        for pattern in failure_patterns:
            insights.append(f"Avoid: {pattern['description']} (confidence: {pattern['confidence']:.2f})")
        
        # Get recent performance trends
        performance_trend = await self._get_performance_trend(task_type)
        if performance_trend:
            insights.append(f"Recent performance trend: {performance_trend}")
        
        # Get evolved prompts
        evolved_prompt = await self._get_best_prompt(task_type)
        if evolved_prompt:
            insights.append(f"Use evolved prompt strategy: {evolved_prompt[:100]}...")
        
        return insights[:5]  # Limit to top 5 insights
    
    async def _get_successful_experiences(self, task_type: str) -> List[Dict[str, Any]]:
        """Get successful experiences for a task type"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT tools_used, performance_score, lessons_learned
                FROM experiences 
                WHERE task_type = ? AND success = 1
                ORDER BY performance_score DESC
                LIMIT 10
            """, (task_type,))
            
            experiences = []
            for row in cursor.fetchall():
                experiences.append({
                    'tools_used': json.loads(row[0]),
                    'performance_score': row[1],
                    'lessons_learned': json.loads(row[2])
                })
            
            return experiences
    
    def _get_common_successful_tools(self, experiences: List[Dict[str, Any]]) -> List[str]:
        """Get most commonly used tools in successful experiences"""
        
        tool_counts = Counter()
        for exp in experiences:
            for tool in exp['tools_used']:
                tool_counts[tool] += 1
        
        # Return top 3 most common tools
        return [tool for tool, _ in tool_counts.most_common(3)]
    
    async def _get_failure_patterns(self, task_type: str) -> List[Dict[str, Any]]:
        """Get failure patterns for a task type"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT pattern_id, description, confidence, actionable_recommendations
                FROM pattern_insights
                WHERE pattern_type = 'failure_pattern'
                ORDER BY confidence DESC
                LIMIT 3
            """)
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'pattern_id': row[0],
                    'description': row[1],
                    'confidence': row[2],
                    'recommendations': json.loads(row[3])
                })
            
            return patterns
    
    async def _get_performance_trend(self, task_type: str) -> Optional[str]:
        """Get recent performance trend for a task type"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT performance_score, timestamp
                FROM experiences
                WHERE task_type = ?
                ORDER BY timestamp DESC
                LIMIT 10
            """, (task_type,))
            
            scores = []
            for row in cursor.fetchall():
                scores.append(row[0])
            
            if len(scores) < 3:
                return None
            
            # Simple trend analysis
            recent_avg = np.mean(scores[:5])
            older_avg = np.mean(scores[5:])
            
            if recent_avg > older_avg + 0.1:
                return "improving"
            elif recent_avg < older_avg - 0.1:
                return "declining"
            else:
                return "stable"
    
    async def _get_best_prompt(self, task_type: str) -> Optional[str]:
        """Get the best evolved prompt for a task type"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT prompt_content
                FROM prompt_evolution
                WHERE task_type = ?
                ORDER BY success_rate DESC, prompt_version DESC
                LIMIT 1
            """, (task_type,))
            
            row = cursor.fetchone()
            return row[0] if row else None
    
    async def _analyze_patterns(self):
        """Analyze experience patterns to generate insights"""
        
        # Get recent experiences for analysis
        cutoff_date = datetime.now() - timedelta(days=7)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT task_type, tools_used, success, performance_score, evaluation
                FROM experiences
                WHERE timestamp >= ?
            """, (cutoff_date.isoformat(),))
            
            recent_experiences = cursor.fetchall()
        
        if len(recent_experiences) < self.config["min_experiences_for_pattern"]:
            return
        
        # Analyze failure patterns
        await self._analyze_failure_patterns(recent_experiences)
        
        # Analyze success patterns
        await self._analyze_success_patterns(recent_experiences)
        
        # Analyze tool usage patterns
        await self._analyze_tool_patterns(recent_experiences)
    
    async def _analyze_failure_patterns(self, experiences: List[Tuple]):
        """Analyze patterns in failed experiences"""
        
        failed_experiences = [exp for exp in experiences if not exp[2]]  # success = False
        
        if len(failed_experiences) < 3:
            return
        
        # Group failures by task type
        failures_by_type = defaultdict(list)
        for exp in failed_experiences:
            task_type, tools_used, _, performance_score, evaluation = exp
            failures_by_type[task_type].append({
                'tools_used': json.loads(tools_used),
                'performance_score': performance_score,
                'evaluation': json.loads(evaluation)
            })
        
        # Identify common failure patterns
        for task_type, failures in failures_by_type.items():
            if len(failures) >= 3:
                pattern_description = self._identify_failure_pattern(failures)
                if pattern_description:
                    await self._store_pattern_insight(
                        pattern_type="failure_pattern",
                        description=pattern_description,
                        confidence=0.8,
                        supporting_experiences=[],  # Would store actual experience IDs
                        recommendations=self._generate_failure_recommendations(pattern_description)
                    )
    
    def _identify_failure_pattern(self, failures: List[Dict[str, Any]]) -> Optional[str]:
        """Identify common patterns in failures"""
        
        # Analyze common tools in failures
        tool_counts = Counter()
        for failure in failures:
            for tool in failure['tools_used']:
                tool_counts[tool] += 1
        
        # Check if certain tools are overrepresented in failures
        total_failures = len(failures)
        problematic_tools = [
            tool for tool, count in tool_counts.items() 
            if count / total_failures > 0.6
        ]
        
        if problematic_tools:
            return f"Frequent failures when using tools: {', '.join(problematic_tools)}"
        
        # Analyze common error types
        error_types = Counter()
        for failure in failures:
            evaluation = failure['evaluation']
            if 'error_type' in evaluation:
                error_types[evaluation['error_type']] += 1
        
        if error_types:
            most_common_error = error_types.most_common(1)[0][0]
            return f"Common failure type: {most_common_error}"
        
        return None
    
    def _generate_failure_recommendations(self, pattern_description: str) -> List[str]:
        """Generate recommendations based on failure patterns"""
        
        recommendations = []
        
        if "tools" in pattern_description.lower():
            recommendations.append("Consider alternative tools for this task type")
            recommendations.append("Validate tool prerequisites before execution")
        
        if "error" in pattern_description.lower():
            recommendations.append("Add additional error checking and validation")
            recommendations.append("Implement fallback strategies for common errors")
        
        recommendations.append("Review successful experiences for alternative approaches")
        
        return recommendations
    
    async def _analyze_success_patterns(self, experiences: List[Tuple]):
        """Analyze patterns in successful experiences"""
        
        successful_experiences = [exp for exp in experiences if exp[2]]  # success = True
        
        if len(successful_experiences) < 3:
            return
        
        # Group successes by task type and high performance
        high_performance_successes = [
            exp for exp in successful_experiences if exp[3] > 0.8  # performance_score > 0.8
        ]
        
        if high_performance_successes:
            # Analyze common tools in high-performance successes
            tool_patterns = self._analyze_successful_tool_patterns(high_performance_successes)
            
            for pattern_description, confidence in tool_patterns:
                await self._store_pattern_insight(
                    pattern_type="success_pattern",
                    description=pattern_description,
                    confidence=confidence,
                    supporting_experiences=[],
                    recommendations=[f"Apply this successful pattern: {pattern_description}"]
                )
    
    def _analyze_successful_tool_patterns(self, successes: List[Tuple]) -> List[Tuple[str, float]]:
        """Analyze tool usage patterns in successful experiences"""
        
        patterns = []
        
        # Count tool combinations
        tool_combinations = Counter()
        for exp in successes:
            tools = tuple(sorted(json.loads(exp[1])))  # tools_used
            if len(tools) > 1:
                tool_combinations[tools] += 1
        
        # Identify frequently successful combinations
        total_successes = len(successes)
        for combination, count in tool_combinations.most_common(3):
            if count / total_successes > 0.3:  # Appears in 30%+ of successes
                confidence = count / total_successes
                description = f"Successful tool combination: {', '.join(combination)}"
                patterns.append((description, confidence))
        
        return patterns
    
    async def _analyze_tool_patterns(self, experiences: List[Tuple]):
        """Analyze overall tool usage patterns"""
        
        # This would implement more sophisticated tool pattern analysis
        # For now, just log that we're doing it
        self.logger.info("Analyzing tool usage patterns...")
    
    async def _store_pattern_insight(
        self,
        pattern_type: str,
        description: str,
        confidence: float,
        supporting_experiences: List[str],
        recommendations: List[str]
    ):
        """Store a discovered pattern insight"""
        
        pattern_id = hashlib.md5(
            f"{pattern_type}_{description}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pattern_insights
                (pattern_id, pattern_type, description, confidence,
                 supporting_experiences, actionable_recommendations, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id,
                pattern_type,
                description,
                confidence,
                json.dumps(supporting_experiences),
                json.dumps(recommendations),
                datetime.now().isoformat()
            ))
        
        self.logger.info(f"Stored pattern insight: {description}")
    
    async def _check_prompt_evolution(self, task_type: str):
        """Check if prompt evolution is needed for a task type"""
        
        # Get recent performance for this task type
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT AVG(CAST(success AS FLOAT)) as success_rate
                FROM experiences
                WHERE task_type = ?
                AND timestamp >= ?
            """, (task_type, (datetime.now() - timedelta(days=7)).isoformat()))
            
            row = cursor.fetchone()
            recent_success_rate = row[0] if row[0] is not None else 1.0
        
        # If success rate is below threshold, evolve the prompt
        if recent_success_rate < self.config["prompt_evolution_threshold"]:
            await self._evolve_prompt(task_type, recent_success_rate)
    
    async def _evolve_prompt(self, task_type: str, current_success_rate: float):
        """Evolve the prompt for a task type based on experience"""
        
        # Get current best prompt
        current_prompt = await self._get_best_prompt(task_type)
        
        # Analyze recent failures to understand what to improve
        recent_failures = await self._get_recent_failures(task_type)
        
        # Generate evolved prompt (simplified version)
        evolved_prompt = await self._generate_evolved_prompt(
            task_type, current_prompt, recent_failures
        )
        
        # Store evolved prompt
        await self._store_evolved_prompt(task_type, evolved_prompt, current_success_rate)
        
        self.logger.info(f"Evolved prompt for task type {task_type}")
    
    async def _get_recent_failures(self, task_type: str) -> List[Dict[str, Any]]:
        """Get recent failures for prompt evolution"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT task_description, evaluation, lessons_learned
                FROM experiences
                WHERE task_type = ? AND success = 0
                AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 5
            """, (task_type, (datetime.now() - timedelta(days=7)).isoformat()))
            
            failures = []
            for row in cursor.fetchall():
                failures.append({
                    'description': row[0],
                    'evaluation': json.loads(row[1]),
                    'lessons_learned': json.loads(row[2])
                })
            
            return failures
    
    async def _generate_evolved_prompt(
        self,
        task_type: str,
        current_prompt: Optional[str],
        recent_failures: List[Dict[str, Any]]
    ) -> str:
        """Generate an evolved prompt based on failure analysis"""
        
        # Base prompt if none exists
        base_prompt = current_prompt or f"""
        You are a healthcare AI assistant specializing in {task_type}.
        Approach each task systematically and provide evidence-based recommendations.
        """
        
        # Add learning from failures
        if recent_failures:
            common_issues = self._extract_common_failure_issues(recent_failures)
            
            evolution_additions = []
            for issue in common_issues:
                if "data quality" in issue.lower():
                    evolution_additions.append(
                        "- Always validate data quality and completeness before analysis"
                    )
                elif "tool" in issue.lower():
                    evolution_additions.append(
                        "- Verify tool prerequisites and validate inputs before execution"
                    )
                elif "context" in issue.lower():
                    evolution_additions.append(
                        "- Consider broader clinical context and patient history"
                    )
            
            if evolution_additions:
                evolved_prompt = base_prompt + "\n\nBased on recent experience, pay special attention to:\n" + "\n".join(evolution_additions)
            else:
                evolved_prompt = base_prompt
        else:
            evolved_prompt = base_prompt
        
        return evolved_prompt
    
    def _extract_common_failure_issues(self, failures: List[Dict[str, Any]]) -> List[str]:
        """Extract common issues from recent failures"""
        
        issue_counts = Counter()
        
        for failure in failures:
            # Extract issues from lessons learned
            for lesson in failure['lessons_learned']:
                if "data" in lesson.lower():
                    issue_counts["data quality"] += 1
                elif "tool" in lesson.lower():
                    issue_counts["tool usage"] += 1
                elif "context" in lesson.lower():
                    issue_counts["context understanding"] += 1
        
        # Return issues that appear in multiple failures
        return [issue for issue, count in issue_counts.items() if count >= 2]
    
    async def _store_evolved_prompt(
        self,
        task_type: str,
        prompt_content: str,
        current_success_rate: float
    ):
        """Store an evolved prompt"""
        
        # Get current version number
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT MAX(prompt_version) FROM prompt_evolution
                WHERE task_type = ?
            """, (task_type,))
            
            row = cursor.fetchone()
            next_version = (row[0] or 0) + 1
        
        # Create new prompt entry
        prompt_id = f"{task_type}_v{next_version}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO prompt_evolution
                (prompt_id, task_type, prompt_version, prompt_content,
                 performance_history, success_rate, usage_count,
                 created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt_id,
                task_type,
                next_version,
                prompt_content,
                json.dumps([current_success_rate]),
                current_success_rate,
                0,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
    
    async def _update_performance_metrics(
        self,
        task_type: str,
        performance_score: float,
        timestamp: datetime
    ):
        """Update performance metrics"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_metrics
                (task_type, metric_name, metric_value, timestamp)
                VALUES (?, ?, ?, ?)
            """, (task_type, "performance_score", performance_score, timestamp.isoformat()))
    
    def _extract_lessons_learned(
        self,
        task_description: str,
        result: Any,
        evaluation: Dict[str, Any],
        tools_used: List[str]
    ) -> List[str]:
        """Extract lessons learned from task execution"""
        
        lessons = []
        
        # Success-based lessons
        if evaluation.get('success', False):
            if evaluation.get('performance_score', 0) > 0.8:
                lessons.append(f"High-performance approach: {tools_used[:2]}")
            
            if evaluation.get('confidence', 0) > 0.9:
                lessons.append("High confidence indicates robust methodology")
        else:
            # Failure-based lessons
            if evaluation.get('error_msg'):
                lessons.append(f"Error to avoid: {evaluation['error_msg'][:100]}")
            
            if not tools_used:
                lessons.append("Consider using appropriate tools for this task type")
        
        # Tool-specific lessons
        if len(tools_used) > 5:
            lessons.append("Consider simplifying tool usage for better performance")
        elif len(tools_used) == 0:
            lessons.append("Task may benefit from specialized tool usage")
        
        return lessons
    
    def get_experience_statistics(self) -> Dict[str, Any]:
        """Get comprehensive experience statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_experiences,
                    AVG(CAST(success AS FLOAT)) as overall_success_rate,
                    AVG(performance_score) as avg_performance
                FROM experiences
            """)
            
            total, success_rate, avg_performance = cursor.fetchone()
            
            # Statistics by task type
            cursor = conn.execute("""
                SELECT 
                    task_type,
                    COUNT(*) as count,
                    AVG(CAST(success AS FLOAT)) as success_rate,
                    AVG(performance_score) as avg_performance
                FROM experiences
                GROUP BY task_type
                ORDER BY count DESC
            """)
            
            task_type_stats = {}
            for row in cursor.fetchall():
                task_type_stats[row[0]] = {
                    "count": row[1],
                    "success_rate": row[2],
                    "avg_performance": row[3]
                }
            
            # Pattern insights count
            cursor = conn.execute("SELECT COUNT(*) FROM pattern_insights")
            pattern_count = cursor.fetchone()[0]
            
            # Evolved prompts count
            cursor = conn.execute("SELECT COUNT(*) FROM prompt_evolution")
            prompt_count = cursor.fetchone()[0]
        
        return {
            "total_experiences": total,
            "overall_success_rate": success_rate,
            "average_performance": avg_performance,
            "task_type_breakdown": task_type_stats,
            "pattern_insights_discovered": pattern_count,
            "evolved_prompts_created": prompt_count,
            "database_path": str(self.db_path)
        }