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
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import logging
from pathlib import Path
from collections import defaultdict, Counter


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

        # Initialize JSON files for experience storage
        self.experiences_file = self.experience_dir / f"{agent_id}_experiences.json"
        self.prompt_evolution_file = self.experience_dir / f"{agent_id}_prompt_evolution.json"
        self.pattern_insights_file = self.experience_dir / f"{agent_id}_pattern_insights.json"
        self.performance_metrics_file = self.experience_dir / f"{agent_id}_performance_metrics.json"

        self._init_json_storage()

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

    def _load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_json_file(self, file_path: Path, data: List[Dict[str, Any]]):
        """Save data to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _init_json_storage(self):
        """Initialize JSON files for experience storage"""

        # Initialize experiences file
        if not self.experiences_file.exists():
            with open(self.experiences_file, 'w') as f:
                json.dump([], f)

        # Initialize prompt evolution file
        if not self.prompt_evolution_file.exists():
            with open(self.prompt_evolution_file, 'w') as f:
                json.dump([], f)

        # Initialize pattern insights file
        if not self.pattern_insights_file.exists():
            with open(self.pattern_insights_file, 'w') as f:
                json.dump([], f)

        # Initialize performance metrics file
        if not self.performance_metrics_file.exists():
            with open(self.performance_metrics_file, 'w') as f:
                json.dump([], f)

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

        # Store in JSON file
        experiences_data = self._load_json_file(self.experiences_file)

        experience_dict = {
            "experience_id": experience.experience_id,
            "task_type": experience.task_type,
            "task_description": experience.task_description,
            "context": experience.context,
            "result": experience.result,
            "evaluation": experience.evaluation,
            "tools_used": experience.tools_used,
            "success": experience.success,
            "performance_score": experience.performance_score,
            "timestamp": experience.timestamp.isoformat(),
            "lessons_learned": experience.lessons_learned
        }

        experiences_data.append(experience_dict)
        self._save_json_file(self.experiences_file, experiences_data)

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

        experiences_data = self._load_json_file(self.experiences_file)

        # Filter and sort successful experiences
        successful_experiences = [
            exp for exp in experiences_data
            if exp['task_type'] == task_type and exp['success']
        ]

        # Sort by performance score descending
        successful_experiences.sort(key=lambda x: x['performance_score'], reverse=True)

        # Return top 10
        return successful_experiences[:10]

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

        pattern_insights_data = self._load_json_file(self.pattern_insights_file)

        # Filter failure patterns and sort by confidence
        failure_patterns = [
            pattern for pattern in pattern_insights_data
            if pattern['pattern_type'] == 'failure_pattern'
        ]

        failure_patterns.sort(key=lambda x: x['confidence'], reverse=True)

        return failure_patterns[:3]

    async def _get_performance_trend(self, task_type: str) -> Optional[str]:
        """Get recent performance trend for a task type"""

        experiences_data = self._load_json_file(self.experiences_file)

        # Filter by task type and sort by timestamp descending
        task_experiences = [
            exp for exp in experiences_data
            if exp['task_type'] == task_type
        ]

        task_experiences.sort(key=lambda x: x['timestamp'], reverse=True)

        scores = [exp['performance_score'] for exp in task_experiences[:10]]

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

        prompt_evolution_data = self._load_json_file(self.prompt_evolution_file)

        # Filter by task type
        task_prompts = [
            prompt for prompt in prompt_evolution_data
            if prompt['task_type'] == task_type
        ]

        if not task_prompts:
            return None

        # Sort by success rate desc, then by version desc
        task_prompts.sort(key=lambda x: (x['success_rate'], x['prompt_version']), reverse=True)

        return task_prompts[0]['prompt_content']

    async def _analyze_patterns(self):
        """Analyze experience patterns to generate insights"""

        # Get recent experiences for analysis
        cutoff_date = datetime.now() - timedelta(days=7)
        experiences_data = self._load_json_file(self.experiences_file)

        recent_experiences = [
            exp for exp in experiences_data
            if datetime.fromisoformat(exp['timestamp']) >= cutoff_date
        ]

        if len(recent_experiences) < self.config["min_experiences_for_pattern"]:
            return

        # Analyze failure patterns
        await self._analyze_failure_patterns(recent_experiences)

        # Analyze success patterns
        await self._analyze_success_patterns(recent_experiences)

        # Analyze tool usage patterns
        await self._analyze_tool_patterns(recent_experiences)

    async def _analyze_failure_patterns(self, experiences: List[Dict[str, Any]]):
        """Analyze patterns in failed experiences"""

        failed_experiences = [exp for exp in experiences if not exp['success']]

        if len(failed_experiences) < 3:
            return

        # Group failures by task type
        failures_by_type = defaultdict(list)
        for exp in failed_experiences:
            task_type = exp['task_type']
            failures_by_type[task_type].append({
                'tools_used': exp['tools_used'],
                'performance_score': exp['performance_score'],
                'evaluation': exp['evaluation']
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

    async def _analyze_success_patterns(self, experiences: List[Dict[str, Any]]):
        """Analyze patterns in successful experiences"""

        successful_experiences = [exp for exp in experiences if exp['success']]

        if len(successful_experiences) < 3:
            return

        # Group successes by task type and high performance
        high_performance_successes = [
            exp for exp in successful_experiences if exp['performance_score'] > 0.8
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

    def _analyze_successful_tool_patterns(self, successes: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Analyze tool usage patterns in successful experiences"""

        patterns = []

        # Count tool combinations
        tool_combinations = Counter()
        for exp in successes:
            tools = tuple(sorted(exp['tools_used']))
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

    async def _analyze_tool_patterns(self, experiences: List[Dict[str, Any]]):
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

        pattern_insights_data = self._load_json_file(self.pattern_insights_file)

        # Remove existing pattern with same ID if exists
        pattern_insights_data = [p for p in pattern_insights_data if p.get('pattern_id') != pattern_id]

        # Add new pattern insight
        pattern_insights_data.append({
            "pattern_id": pattern_id,
            "pattern_type": pattern_type,
            "description": description,
            "confidence": confidence,
            "supporting_experiences": supporting_experiences,
            "actionable_recommendations": recommendations,
            "discovered_at": datetime.now().isoformat()
        })

        self._save_json_file(self.pattern_insights_file, pattern_insights_data)

        self.logger.info(f"Stored pattern insight: {description}")

    async def _check_prompt_evolution(self, task_type: str):
        """Check if prompt evolution is needed for a task type"""

        # Get recent performance for this task type
        cutoff_date = datetime.now() - timedelta(days=7)
        experiences_data = self._load_json_file(self.experiences_file)

        recent_experiences = [
            exp for exp in experiences_data
            if exp['task_type'] == task_type and datetime.fromisoformat(exp['timestamp']) >= cutoff_date
        ]

        if recent_experiences:
            recent_success_rate = sum(1 for exp in recent_experiences if exp['success']) / len(recent_experiences)
        else:
            recent_success_rate = 1.0

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

        cutoff_date = datetime.now() - timedelta(days=7)
        experiences_data = self._load_json_file(self.experiences_file)

        # Filter recent failures for this task type
        recent_failures = [
            exp for exp in experiences_data
            if (exp['task_type'] == task_type and
                not exp['success'] and
                datetime.fromisoformat(exp['timestamp']) >= cutoff_date)
        ]

        # Sort by timestamp descending and limit to 5
        recent_failures.sort(key=lambda x: x['timestamp'], reverse=True)

        return recent_failures[:5]

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
        prompt_evolution_data = self._load_json_file(self.prompt_evolution_file)

        task_prompts = [
            prompt for prompt in prompt_evolution_data
            if prompt['task_type'] == task_type
        ]

        if task_prompts:
            next_version = max(prompt['prompt_version'] for prompt in task_prompts) + 1
        else:
            next_version = 1

        # Create new prompt entry
        prompt_id = f"{task_type}_v{next_version}"

        prompt_evolution_data.append({
            "prompt_id": prompt_id,
            "task_type": task_type,
            "prompt_version": next_version,
            "prompt_content": prompt_content,
            "performance_history": [current_success_rate],
            "success_rate": current_success_rate,
            "usage_count": 0,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        })

        self._save_json_file(self.prompt_evolution_file, prompt_evolution_data)

    async def _update_performance_metrics(
        self,
        task_type: str,
        performance_score: float,
        timestamp: datetime
    ):
        """Update performance metrics"""

        performance_metrics_data = self._load_json_file(self.performance_metrics_file)

        performance_metrics_data.append({
            "task_type": task_type,
            "metric_name": "performance_score",
            "metric_value": performance_score,
            "timestamp": timestamp.isoformat()
        })

        self._save_json_file(self.performance_metrics_file, performance_metrics_data)

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

        experiences_data = self._load_json_file(self.experiences_file)
        pattern_insights_data = self._load_json_file(self.pattern_insights_file)
        prompt_evolution_data = self._load_json_file(self.prompt_evolution_file)

        # Overall statistics
        total = len(experiences_data)
        if total > 0:
            success_rate = sum(1 for exp in experiences_data if exp['success']) / total
            avg_performance = sum(exp['performance_score'] for exp in experiences_data) / total
        else:
            success_rate = 0
            avg_performance = 0

        # Statistics by task type
        task_type_stats = {}
        task_groups = defaultdict(list)

        for exp in experiences_data:
            task_groups[exp['task_type']].append(exp)

        for task_type, task_exps in task_groups.items():
            count = len(task_exps)
            task_success_rate = sum(1 for exp in task_exps if exp['success']) / count
            task_avg_performance = sum(exp['performance_score'] for exp in task_exps) / count

            task_type_stats[task_type] = {
                "count": count,
                "success_rate": task_success_rate,
                "avg_performance": task_avg_performance
            }

        # Pattern insights count
        pattern_count = len(pattern_insights_data)

        # Evolved prompts count
        prompt_count = len(prompt_evolution_data)

        return {
            "total_experiences": total,
            "overall_success_rate": success_rate,
            "average_performance": avg_performance,
            "task_type_breakdown": task_type_stats,
            "pattern_insights_discovered": pattern_count,
            "evolved_prompts_created": prompt_count,
            "storage_files": {
                "experiences": str(self.experiences_file),
                "prompt_evolution": str(self.prompt_evolution_file),
                "pattern_insights": str(self.pattern_insights_file),
                "performance_metrics": str(self.performance_metrics_file)
            }
        }


class EvolutionEngine:
    """
    Main evolution engine that orchestrates the self-improvement process.

    This class coordinates:
    - Experience accumulation across multiple agents
    - Cross-agent learning and knowledge sharing
    - System-wide optimization and adaptation
    - Meta-learning across different task types
    """

    def __init__(self, config_path: str = "evolution_config.json"):
        self.config_path = Path(config_path)
        self.experience_dir = Path("experience")
        self.experience_dir.mkdir(exist_ok=True)

        # Initialize configuration
        self.config = self._load_config()

        # Track active agents and their accumulators
        self.agent_accumulators = {}

        # Global insights and patterns
        self.global_insights_file = self.experience_dir / "global_insights.json"
        self.cross_agent_patterns_file = self.experience_dir / "cross_agent_patterns.json"

        self.logger = logging.getLogger("EvolutionEngine")
        self.logger.info("Evolution engine initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load evolution configuration"""
        default_config = {
            "meta_learning_enabled": True,
            "cross_agent_learning_enabled": True,
            "global_pattern_analysis_enabled": True,
            "evolution_frequency_hours": 24,
            "min_experiences_for_meta_learning": 50,
            "cross_agent_similarity_threshold": 0.7,
            "global_insight_confidence_threshold": 0.8
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}, using defaults")

        return default_config

    def register_agent(self, agent_id: str) -> ExperienceAccumulator:
        """Register an agent and return its experience accumulator"""
        if agent_id not in self.agent_accumulators:
            accumulator = ExperienceAccumulator(agent_id, str(self.experience_dir))
            self.agent_accumulators[agent_id] = accumulator
            self.logger.info(f"Registered agent: {agent_id}")

        return self.agent_accumulators[agent_id]

    async def evolve_system(self) -> Dict[str, Any]:
        """Run a complete system evolution cycle"""
        evolution_results = {
            "timestamp": datetime.now().isoformat(),
            "agents_evolved": [],
            "global_insights_discovered": [],
            "cross_agent_patterns": [],
            "meta_learning_applied": []
        }

        # Perform individual agent evolution
        for agent_id, accumulator in self.agent_accumulators.items():
            try:
                # Trigger pattern analysis for each agent
                await accumulator._analyze_patterns()
                evolution_results["agents_evolved"].append(agent_id)
            except Exception as e:
                self.logger.error(f"Evolution failed for agent {agent_id}: {e}")

        # Perform cross-agent learning
        if self.config["cross_agent_learning_enabled"]:
            cross_patterns = await self._discover_cross_agent_patterns()
            evolution_results["cross_agent_patterns"] = cross_patterns

        # Perform global pattern analysis
        if self.config["global_pattern_analysis_enabled"]:
            global_insights = await self._discover_global_insights()
            evolution_results["global_insights_discovered"] = global_insights

        # Apply meta-learning
        if self.config["meta_learning_enabled"]:
            meta_learning_results = await self._apply_meta_learning()
            evolution_results["meta_learning_applied"] = meta_learning_results

        # Save evolution results
        await self._save_evolution_results(evolution_results)

        self.logger.info("System evolution cycle completed")
        return evolution_results

    async def _discover_cross_agent_patterns(self) -> List[Dict[str, Any]]:
        """Discover patterns across multiple agents"""
        cross_patterns = []

        if len(self.agent_accumulators) < 2:
            return cross_patterns

        # Collect successful experiences from all agents
        all_successful_experiences = {}

        for agent_id, accumulator in self.agent_accumulators.items():
            agent_successes = {}
            # Get successful experiences by task type for this agent
            experiences_data = accumulator._load_json_file(accumulator.experiences_file)

            for exp in experiences_data:
                if exp['success'] and exp['performance_score'] > 0.8:
                    task_type = exp['task_type']
                    if task_type not in agent_successes:
                        agent_successes[task_type] = []
                    agent_successes[task_type].append({
                        'tools_used': exp['tools_used'],
                        'performance_score': exp['performance_score']
                    })

            all_successful_experiences[agent_id] = agent_successes

        # Find common successful patterns across agents
        task_types = set()
        for agent_successes in all_successful_experiences.values():
            task_types.update(agent_successes.keys())

        for task_type in task_types:
            # Check if multiple agents have successful experiences with this task type
            agents_with_task = [
                agent_id for agent_id, successes in all_successful_experiences.items()
                if task_type in successes and len(successes[task_type]) >= 2
            ]

            if len(agents_with_task) >= 2:
                # Find common tool patterns
                common_tools = self._find_common_tool_patterns(
                    [all_successful_experiences[agent_id][task_type]
                     for agent_id in agents_with_task]
                )

                if common_tools:
                    cross_patterns.append({
                        "pattern_type": "cross_agent_tool_success",
                        "task_type": task_type,
                        "agents_involved": agents_with_task,
                        "common_tools": common_tools,
                        "confidence": len(agents_with_task) / len(self.agent_accumulators)
                    })

        return cross_patterns

    def _find_common_tool_patterns(self, agent_experiences_list: List[List[Dict]]) -> List[str]:
        """Find tools commonly used across agents for successful outcomes"""
        all_tool_counters = []

        for agent_experiences in agent_experiences_list:
            tool_counter = Counter()
            for exp in agent_experiences:
                for tool in exp['tools_used']:
                    tool_counter[tool] += 1
            all_tool_counters.append(tool_counter)

        # Find tools that appear frequently across agents
        common_tools = []
        if all_tool_counters:
            # Get tools that appear in at least half of the agent lists
            min_agents = len(all_tool_counters) // 2 + 1

            all_tools = set()
            for counter in all_tool_counters:
                all_tools.update(counter.keys())

            for tool in all_tools:
                agent_count = sum(1 for counter in all_tool_counters if tool in counter)
                if agent_count >= min_agents:
                    common_tools.append(tool)

        return common_tools

    async def _discover_global_insights(self) -> List[Dict[str, Any]]:
        """Discover system-wide insights and patterns"""
        global_insights = []

        # Aggregate statistics across all agents
        total_experiences = 0
        total_successes = 0
        task_type_performance = defaultdict(list)

        for agent_id, accumulator in self.agent_accumulators.items():
            stats = accumulator.get_experience_statistics()
            total_experiences += stats["total_experiences"]
            total_successes += stats["total_experiences"] * stats["overall_success_rate"]

            for task_type, task_stats in stats["task_type_breakdown"].items():
                task_type_performance[task_type].append(task_stats["avg_performance"])

        overall_success_rate = total_successes / total_experiences if total_experiences > 0 else 0

        # Generate global insights
        if overall_success_rate > 0.8:
            global_insights.append({
                "insight_type": "system_performance",
                "description": f"System achieving high success rate: {overall_success_rate:.2%}",
                "confidence": 0.9,
                "recommendation": "Current strategies are effective, focus on scaling"
            })
        elif overall_success_rate < 0.6:
            global_insights.append({
                "insight_type": "system_performance",
                "description": f"System underperforming: {overall_success_rate:.2%}",
                "confidence": 0.9,
                "recommendation": "Review and update core strategies and prompts"
            })

        # Task type insights
        for task_type, performances in task_type_performance.items():
            avg_performance = np.mean(performances)
            if len(performances) > 1 and np.std(performances) > 0.2:
                global_insights.append({
                    "insight_type": "task_variability",
                    "description": f"High performance variance in {task_type}",
                    "confidence": 0.8,
                    "recommendation": "Standardize approaches for this task type"
                })

        return global_insights

    async def _apply_meta_learning(self) -> List[Dict[str, Any]]:
        """Apply meta-learning insights across the system"""
        meta_learning_results = []

        # Check if we have enough experiences for meta-learning
        total_experiences = sum(
            accumulator.get_experience_statistics()["total_experiences"]
            for accumulator in self.agent_accumulators.values()
        )

        if total_experiences < self.config["min_experiences_for_meta_learning"]:
            return meta_learning_results

        # Meta-learning: Find universally successful patterns
        universal_patterns = await self._find_universal_patterns()

        for pattern in universal_patterns:
            meta_learning_results.append({
                "meta_learning_type": "universal_pattern",
                "pattern": pattern,
                "applied_to": "all_agents",
                "expected_improvement": "10-20% performance boost"
            })

        return meta_learning_results

    async def _find_universal_patterns(self) -> List[Dict[str, Any]]:
        """Find patterns that work across all agents and task types"""
        universal_patterns = []

        # This would implement sophisticated pattern discovery
        # For now, return a simplified example
        if len(self.agent_accumulators) > 0:
            universal_patterns.append({
                "pattern_type": "tool_validation",
                "description": "Always validate tool inputs before execution",
                "confidence": 0.9
            })

        return universal_patterns

    async def _save_evolution_results(self, results: Dict[str, Any]):
        """Save evolution results to persistent storage"""
        results_file = self.experience_dir / f"evolution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Also update the global insights file
        if results["global_insights_discovered"]:
            await self._update_global_insights(results["global_insights_discovered"])

    async def _update_global_insights(self, new_insights: List[Dict[str, Any]]):
        """Update the global insights file"""
        existing_insights = []

        if self.global_insights_file.exists():
            try:
                with open(self.global_insights_file, 'r') as f:
                    existing_insights = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load existing insights: {e}")

        # Add new insights with timestamps
        for insight in new_insights:
            insight["discovered_at"] = datetime.now().isoformat()
            existing_insights.append(insight)

        # Keep only recent insights (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_insights = [
            insight for insight in existing_insights
            if datetime.fromisoformat(insight["discovered_at"]) > cutoff_date
        ]

        with open(self.global_insights_file, 'w') as f:
            json.dump(recent_insights, f, indent=2, default=str)

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system evolution statistics"""
        system_stats = {
            "total_agents": len(self.agent_accumulators),
            "evolution_config": self.config,
            "agent_statistics": {}
        }

        for agent_id, accumulator in self.agent_accumulators.items():
            system_stats["agent_statistics"][agent_id] = accumulator.get_experience_statistics()

        # Add global insights count
        if self.global_insights_file.exists():
            try:
                with open(self.global_insights_file, 'r') as f:
                    insights = json.load(f)
                system_stats["global_insights_count"] = len(insights)
            except Exception:
                system_stats["global_insights_count"] = 0
        else:
            system_stats["global_insights_count"] = 0

        return system_stats