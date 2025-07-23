"""
Enhanced Logging System for HealthFlow
Tracks agent interactions, memory evolution, tool evolution, and system dynamics
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"


class LogCategory(Enum):
    AGENT_INTERACTION = "agent_interaction"
    MEMORY_EVOLUTION = "memory_evolution"
    TOOL_EVOLUTION = "tool_evolution"
    TASK_EXECUTION = "task_execution"
    COLLABORATION = "collaboration"
    SYSTEM_PERFORMANCE = "system_performance"
    ERROR_TRACKING = "error_tracking"


@dataclass
class LogEntry:
    """Enhanced log entry with rich contextual information"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    agent_id: Optional[str]
    event_id: str
    message: str
    context: Dict[str, Any]
    correlation_id: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'category': self.category.value,
            'agent_id': self.agent_id,
            'event_id': self.event_id,
            'message': self.message,
            'context': self.context,
            'correlation_id': self.correlation_id,
            'performance_metrics': self.performance_metrics or {}
        }


@dataclass
class AgentInteractionLog:
    """Detailed log of agent interactions"""
    timestamp: datetime
    interaction_id: str
    sender_agent: str
    receiver_agent: str
    interaction_type: str  # collaboration, delegation, information_sharing
    message_content: Dict[str, Any]
    response_time: float
    success: bool
    context: Dict[str, Any]


@dataclass
class MemoryEvolutionLog:
    """Track how agent memory evolves over time"""
    timestamp: datetime
    agent_id: str
    memory_type: str
    operation: str  # create, update, delete, retrieve
    memory_id: str
    content_summary: str
    impact_score: float  # How much this affects agent behavior
    memory_size: int
    retention_score: float


@dataclass  
class ToolEvolutionLog:
    """Track tool creation, usage, and evolution"""
    timestamp: datetime
    tool_id: str
    agent_id: str
    operation: str  # create, execute, modify, delete
    tool_name: str
    success_rate: float
    usage_frequency: int
    performance_improvement: float
    context: Dict[str, Any]


class EnhancedHealthFlowLogger:
    """
    Advanced logging system for comprehensive HealthFlow monitoring
    
    Features:
    - Multi-dimensional logging (agents, memory, tools, performance)
    - Real-time analytics and pattern detection
    - Correlation tracking across system components
    - Performance monitoring and bottleneck detection
    - Rich contextual information for debugging
    """
    
    def __init__(
        self, 
        log_dir: Path,
        max_log_entries: int = 10000,
        analytics_window_hours: int = 24
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_log_entries = max_log_entries
        self.analytics_window = timedelta(hours=analytics_window_hours)
        
        # In-memory log stores with rotation
        self.log_entries: deque = deque(maxlen=max_log_entries)
        self.agent_interactions: deque = deque(maxlen=max_log_entries)
        self.memory_evolution: deque = deque(maxlen=max_log_entries) 
        self.tool_evolution: deque = deque(maxlen=max_log_entries)
        
        # Analytics caches
        self.interaction_patterns = defaultdict(list)
        self.performance_trends = defaultdict(list)
        self.error_patterns = defaultdict(int)
        
        # File paths
        self.main_log_file = self.log_dir / "healthflow_main.log"
        self.interactions_file = self.log_dir / "agent_interactions.jsonl"
        self.memory_file = self.log_dir / "memory_evolution.jsonl"
        self.tools_file = self.log_dir / "tool_evolution.jsonl"
        self.analytics_file = self.log_dir / "system_analytics.json"
        
        # Set up Python logging integration  
        self.python_logger = self._setup_python_logger()
        
    def _setup_python_logger(self) -> logging.Logger:
        """Set up Python logging with custom formatting"""
        logger = logging.getLogger("HealthFlow")
        logger.setLevel(logging.DEBUG)
        
        # Create file handler
        file_handler = logging.FileHandler(self.main_log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)15s | %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    async def log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        agent_id: Optional[str] = None,
        context: Dict[str, Any] = None,
        correlation_id: Optional[str] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ):
        """Log an enhanced entry with rich context"""
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            agent_id=agent_id,
            event_id=str(uuid.uuid4()),
            message=message,
            context=context or {},
            correlation_id=correlation_id,
            performance_metrics=performance_metrics
        )
        
        # Add to in-memory store
        self.log_entries.append(entry)
        
        # Log to Python logger as well
        python_level = getattr(logging, level.value.upper())
        extra_info = f"[{category.value}]"
        if agent_id:
            extra_info += f"[{agent_id}]"
        
        self.python_logger.log(python_level, f"{extra_info} {message}")
        
        # Update analytics
        await self._update_analytics(entry)
        
        # Periodic persistence
        if len(self.log_entries) % 100 == 0:
            await self._persist_logs()
    
    async def log_agent_interaction(
        self,
        sender_agent: str,
        receiver_agent: str,
        interaction_type: str,
        message_content: Dict[str, Any],
        response_time: float,
        success: bool,
        context: Dict[str, Any] = None
    ):
        """Log detailed agent interaction"""
        
        interaction = AgentInteractionLog(
            timestamp=datetime.now(),
            interaction_id=str(uuid.uuid4()),
            sender_agent=sender_agent,
            receiver_agent=receiver_agent,
            interaction_type=interaction_type,
            message_content=message_content,
            response_time=response_time,
            success=success,
            context=context or {}
        )
        
        self.agent_interactions.append(interaction)
        
        # Log to main logger
        await self.log(
            LogLevel.INFO,
            LogCategory.AGENT_INTERACTION,
            f"Agent interaction: {sender_agent} -> {receiver_agent} ({interaction_type})",
            agent_id=sender_agent,
            context={
                "receiver": receiver_agent,
                "type": interaction_type,
                "success": success,
                "response_time": response_time
            },
            performance_metrics={"response_time": response_time}
        )
        
        # Update interaction patterns
        pattern_key = f"{sender_agent}->{receiver_agent}"
        self.interaction_patterns[pattern_key].append({
            "timestamp": interaction.timestamp,
            "type": interaction_type,
            "success": success,
            "response_time": response_time
        })
    
    async def log_memory_evolution(
        self,
        agent_id: str,
        memory_type: str,
        operation: str,
        memory_id: str,
        content_summary: str,
        impact_score: float = 0.0,
        memory_size: int = 0,
        retention_score: float = 1.0
    ):
        """Log memory evolution events"""
        
        memory_log = MemoryEvolutionLog(
            timestamp=datetime.now(),
            agent_id=agent_id,
            memory_type=memory_type,
            operation=operation,
            memory_id=memory_id,
            content_summary=content_summary,
            impact_score=impact_score,
            memory_size=memory_size,
            retention_score=retention_score
        )
        
        self.memory_evolution.append(memory_log)
        
        # Log to main logger
        await self.log(
            LogLevel.INFO,
            LogCategory.MEMORY_EVOLUTION,
            f"Memory {operation}: {memory_type} (impact: {impact_score:.2f})",
            agent_id=agent_id,
            context={
                "memory_id": memory_id,
                "memory_type": memory_type,
                "operation": operation,
                "size": memory_size,
                "summary": content_summary[:100] + "..." if len(content_summary) > 100 else content_summary
            },
            performance_metrics={
                "impact_score": impact_score,
                "memory_size": memory_size,
                "retention_score": retention_score
            }
        )
    
    async def log_tool_evolution(
        self,
        tool_id: str,
        agent_id: str,
        operation: str,
        tool_name: str,
        success_rate: float = 1.0,
        usage_frequency: int = 0,
        performance_improvement: float = 0.0,
        context: Dict[str, Any] = None
    ):
        """Log tool evolution events"""
        
        tool_log = ToolEvolutionLog(
            timestamp=datetime.now(),
            tool_id=tool_id,
            agent_id=agent_id,
            operation=operation,
            tool_name=tool_name,
            success_rate=success_rate,
            usage_frequency=usage_frequency,
            performance_improvement=performance_improvement,
            context=context or {}
        )
        
        self.tool_evolution.append(tool_log)
        
        # Log to main logger
        await self.log(
            LogLevel.INFO,
            LogCategory.TOOL_EVOLUTION,
            f"Tool {operation}: {tool_name} (success: {success_rate:.2f})",
            agent_id=agent_id,
            context={
                "tool_id": tool_id,
                "tool_name": tool_name,
                "operation": operation,
                "usage_frequency": usage_frequency,
                **context
            },
            performance_metrics={
                "success_rate": success_rate,
                "usage_frequency": usage_frequency,
                "performance_improvement": performance_improvement
            }
        )
    
    async def log_task_execution(
        self,
        agent_id: str,
        task_id: str,
        task_description: str,
        execution_time: float,
        success: bool,
        tools_used: List[str],
        collaboration_count: int = 0,
        memory_updates: int = 0,
        context: Dict[str, Any] = None
    ):
        """Log comprehensive task execution details"""
        
        await self.log(
            LogLevel.INFO,
            LogCategory.TASK_EXECUTION,
            f"Task execution: {task_description[:50]}... ({'SUCCESS' if success else 'FAILED'})",
            agent_id=agent_id,
            context={
                "task_id": task_id,
                "description": task_description,
                "tools_used": tools_used,
                "collaboration_count": collaboration_count,
                "memory_updates": memory_updates,
                **(context or {})
            },
            correlation_id=task_id,
            performance_metrics={
                "execution_time": execution_time,
                "tools_count": len(tools_used),
                "collaboration_count": collaboration_count,
                "memory_updates": memory_updates
            }
        )
    
    async def log_error(
        self,
        error: str,
        agent_id: Optional[str] = None,
        context: Dict[str, Any] = None,
        correlation_id: Optional[str] = None
    ):
        """Log errors with enhanced context"""
        
        await self.log(
            LogLevel.ERROR,
            LogCategory.ERROR_TRACKING,
            f"Error: {error}",
            agent_id=agent_id,
            context=context,
            correlation_id=correlation_id
        )
        
        # Track error patterns
        error_key = error.split(':')[0] if ':' in error else error[:50]
        self.error_patterns[error_key] += 1
    
    async def _update_analytics(self, entry: LogEntry):
        """Update real-time analytics based on log entry"""
        
        # Performance trends
        if entry.performance_metrics:
            for metric, value in entry.performance_metrics.items():
                self.performance_trends[metric].append({
                    "timestamp": entry.timestamp,
                    "value": value,
                    "agent_id": entry.agent_id,
                    "category": entry.category.value
                })
    
    async def _persist_logs(self):
        """Persist logs to files"""
        
        try:
            # Persist agent interactions
            if self.agent_interactions:
                with open(self.interactions_file, 'a', encoding='utf-8') as f:
                    for interaction in list(self.agent_interactions):
                        f.write(json.dumps(asdict(interaction), default=str) + '\n')
            
            # Persist memory evolution
            if self.memory_evolution:
                with open(self.memory_file, 'a', encoding='utf-8') as f:
                    for memory_log in list(self.memory_evolution):
                        f.write(json.dumps(asdict(memory_log), default=str) + '\n')
            
            # Persist tool evolution  
            if self.tool_evolution:
                with open(self.tools_file, 'a', encoding='utf-8') as f:
                    for tool_log in list(self.tool_evolution):
                        f.write(json.dumps(asdict(tool_log), default=str) + '\n')
            
            # Persist analytics
            analytics = await self.get_system_analytics()
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, indent=2, default=str)
                
        except Exception as e:
            self.python_logger.error(f"Failed to persist logs: {e}")
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive system analytics"""
        
        current_time = datetime.now()
        window_start = current_time - self.analytics_window
        
        # Filter recent entries
        recent_entries = [
            entry for entry in self.log_entries 
            if entry.timestamp >= window_start
        ]
        
        recent_interactions = [
            interaction for interaction in self.agent_interactions
            if interaction.timestamp >= window_start
        ]
        
        # Calculate metrics
        total_interactions = len(recent_interactions)
        successful_interactions = sum(1 for i in recent_interactions if i.success)
        avg_response_time = (
            sum(i.response_time for i in recent_interactions) / total_interactions
            if total_interactions > 0 else 0
        )
        
        # Agent activity
        agent_activity = defaultdict(int)
        for entry in recent_entries:
            if entry.agent_id:
                agent_activity[entry.agent_id] += 1
        
        # Error analysis
        error_entries = [entry for entry in recent_entries if entry.level == LogLevel.ERROR]
        error_rate = len(error_entries) / len(recent_entries) if recent_entries else 0
        
        # Collaboration patterns
        collaboration_matrix = defaultdict(lambda: defaultdict(int))
        for interaction in recent_interactions:
            collaboration_matrix[interaction.sender_agent][interaction.receiver_agent] += 1
        
        return {
            "analytics_period": {
                "start": window_start.isoformat(),
                "end": current_time.isoformat(),
                "duration_hours": self.analytics_window.total_seconds() / 3600
            },
            "system_activity": {
                "total_log_entries": len(recent_entries),
                "total_interactions": total_interactions,
                "successful_interactions": successful_interactions,
                "interaction_success_rate": successful_interactions / total_interactions if total_interactions > 0 else 0,
                "average_response_time": avg_response_time,
                "error_rate": error_rate
            },
            "agent_activity": dict(agent_activity),
            "collaboration_matrix": {
                sender: dict(receivers) 
                for sender, receivers in collaboration_matrix.items()
            },
            "error_patterns": dict(self.error_patterns),
            "performance_trends": {
                metric: trends[-100:] if len(trends) > 100 else trends
                for metric, trends in self.performance_trends.items()
            },
            "memory_statistics": {
                "total_memory_operations": len([
                    log for log in self.memory_evolution 
                    if log.timestamp >= window_start
                ]),
                "average_impact_score": sum(
                    log.impact_score for log in self.memory_evolution 
                    if log.timestamp >= window_start
                ) / len([log for log in self.memory_evolution if log.timestamp >= window_start])
                if any(log.timestamp >= window_start for log in self.memory_evolution) else 0
            },
            "tool_statistics": {
                "total_tool_operations": len([
                    log for log in self.tool_evolution
                    if log.timestamp >= window_start
                ]),
                "average_success_rate": sum(
                    log.success_rate for log in self.tool_evolution
                    if log.timestamp >= window_start
                ) / len([log for log in self.tool_evolution if log.timestamp >= window_start])
                if any(log.timestamp >= window_start for log in self.tool_evolution) else 0
            }
        }
    
    async def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed performance metrics for a specific agent"""
        
        current_time = datetime.now()
        window_start = current_time - self.analytics_window
        
        # Filter agent-specific entries
        agent_entries = [
            entry for entry in self.log_entries
            if entry.agent_id == agent_id and entry.timestamp >= window_start
        ]
        
        agent_interactions_sent = [
            interaction for interaction in self.agent_interactions
            if interaction.sender_agent == agent_id and interaction.timestamp >= window_start
        ]
        
        agent_interactions_received = [
            interaction for interaction in self.agent_interactions
            if interaction.receiver_agent == agent_id and interaction.timestamp >= window_start
        ]
        
        # Calculate metrics
        task_executions = [
            entry for entry in agent_entries 
            if entry.category == LogCategory.TASK_EXECUTION
        ]
        
        successful_tasks = [
            entry for entry in task_executions
            if "SUCCESS" in entry.message
        ]
        
        return {
            "agent_id": agent_id,
            "analysis_period": {
                "start": window_start.isoformat(),
                "end": current_time.isoformat()
            },
            "activity_metrics": {
                "total_log_entries": len(agent_entries),
                "task_executions": len(task_executions),
                "successful_tasks": len(successful_tasks),
                "task_success_rate": len(successful_tasks) / len(task_executions) if task_executions else 0,
                "interactions_sent": len(agent_interactions_sent),
                "interactions_received": len(agent_interactions_received)
            },
            "collaboration_metrics": {
                "outbound_success_rate": (
                    sum(1 for i in agent_interactions_sent if i.success) / len(agent_interactions_sent)
                    if agent_interactions_sent else 0
                ),
                "average_response_time": (
                    sum(i.response_time for i in agent_interactions_sent) / len(agent_interactions_sent)
                    if agent_interactions_sent else 0
                ),
                "collaboration_partners": list(set(
                    [i.receiver_agent for i in agent_interactions_sent] +
                    [i.sender_agent for i in agent_interactions_received]
                ))
            },
            "memory_evolution": [
                asdict(log) for log in self.memory_evolution
                if log.agent_id == agent_id and log.timestamp >= window_start
            ][-20:],  # Last 20 memory operations
            "tool_usage": [
                asdict(log) for log in self.tool_evolution
                if log.agent_id == agent_id and log.timestamp >= window_start
            ][-20:]  # Last 20 tool operations
        }
    
    async def export_analytics(self, output_dir: Path):
        """Export comprehensive analytics to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export system analytics
        system_analytics = await self.get_system_analytics()
        with open(output_dir / "system_analytics.json", 'w') as f:
            json.dump(system_analytics, f, indent=2, default=str)
        
        # Export individual agent analytics
        agents = set(entry.agent_id for entry in self.log_entries if entry.agent_id)
        agent_analytics = {}
        
        for agent_id in agents:
            agent_perf = await self.get_agent_performance(agent_id)
            agent_analytics[agent_id] = agent_perf
        
        with open(output_dir / "agent_analytics.json", 'w') as f:
            json.dump(agent_analytics, f, indent=2, default=str)
        
        # Export raw logs in different formats
        with open(output_dir / "raw_logs.jsonl", 'w') as f:
            for entry in self.log_entries:
                f.write(json.dumps(entry.to_dict()) + '\n')
        
        print(f"📊 Analytics exported to {output_dir}")


# Global logger instance
_global_logger: Optional[EnhancedHealthFlowLogger] = None


def get_logger() -> EnhancedHealthFlowLogger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = EnhancedHealthFlowLogger(Path("./logs"))
    return _global_logger


def set_logger(logger: EnhancedHealthFlowLogger):
    """Set the global logger instance"""
    global _global_logger
    _global_logger = logger