"""
Memory Management System for HealthFlow
Supports unified JSON format for all persistent storage
Implements experience accumulation and self-evolving capabilities
"""

import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import uuid

import numpy as np
from pydantic import BaseModel


class MemoryType(Enum):
    """Types of memory entries"""
    INTERACTION = "interaction"
    EXPERIENCE = "experience" 
    PROMPT_EVOLUTION = "prompt_evolution"
    TOOL_CREATION = "tool_creation"
    FAILURE_ANALYSIS = "failure_analysis"
    SUCCESS_PATTERN = "success_pattern"


@dataclass
class MemoryEntry:
    """Individual memory entry with metadata"""
    id: str
    memory_type: MemoryType
    timestamp: datetime
    agent_id: str
    content: Dict[str, Any]
    success: Optional[bool] = None
    reward: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'memory_type': self.memory_type.value,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'content': self.content,
            'success': self.success,
            'reward': self.reward,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            memory_type=MemoryType(data['memory_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            agent_id=data['agent_id'],
            content=data['content'],
            success=data.get('success'),
            reward=data.get('reward'),
            metadata=data.get('metadata', {})
        )


class ExperiencePattern(BaseModel):
    """Pattern extracted from successful/failed experiences"""
    pattern_id: str
    pattern_type: str  # "success", "failure", "optimization"
    description: str
    conditions: Dict[str, Any]
    actions: List[str]
    outcomes: Dict[str, Any]
    confidence: float
    usage_count: int = 0
    last_updated: datetime
    
    class Config:
        arbitrary_types_allowed = True


class PromptEvolution(BaseModel):
    """Evolution of prompts over time"""
    prompt_id: str
    version: int
    prompt_text: str
    performance_metrics: Dict[str, float]
    improvements: List[str]
    created_at: datetime
    parent_prompt_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class MemoryManager:
    """Advanced memory management with JSON storage and self-evolving capabilities"""
    
    def __init__(self, memory_dir: Path, max_memory_size: int = 10000):
        self.memory_dir = Path(memory_dir)
        self.max_memory_size = max_memory_size
        
        # Storage paths - all JSON format
        self.interactions_path = self.memory_dir / "interactions.jsonl"
        self.experiences_path = self.memory_dir / "experiences.json"
        self.patterns_path = self.memory_dir / "patterns.json"
        self.prompts_path = self.memory_dir / "prompt_evolution.json"
        self.analytics_path = self.memory_dir / "analytics.json"
        
        # In-memory stores
        self.recent_memories: List[MemoryEntry] = []
        self.experience_patterns: Dict[str, ExperiencePattern] = {}
        self.prompt_evolution: Dict[str, List[PromptEvolution]] = {}
        self._archived_experiences_count = 0
        
        # Create directory
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialize and load all data"""
        await self._load_all_data()
    
    async def _load_all_data(self):
        """Load all persistent data"""
        await asyncio.gather(
            self._load_interactions(),
            self._load_experiences(), 
            self._load_patterns(),
            self._load_prompt_evolution()
        )
    
    async def _load_interactions(self):
        """Load recent interactions from JSONL"""
        if not self.interactions_path.exists():
            return
            
        try:
            with open(self.interactions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        memory = MemoryEntry.from_dict(data)
                        self.recent_memories.append(memory)
        except Exception as e:
            print(f"Error loading interactions data: {e}")
    
    async def _load_experiences(self):
        """Load archived experience data from JSON"""
        if not self.experiences_path.exists():
            return
            
        try:
            with open(self.experiences_path, 'r', encoding='utf-8') as f:
                archived_data = json.load(f)
                # Store for analytics queries without loading into memory
                self._archived_experiences_count = len(archived_data.get('experiences', []))
        except Exception as e:
            print(f"Error loading experiences data: {e}")
            self._archived_experiences_count = 0
    
    async def _load_patterns(self):
        """Load experience patterns from JSON"""
        if not self.patterns_path.exists():
            return
            
        try:
            with open(self.patterns_path, 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
                for pattern_id, pattern_dict in patterns_data.items():
                    # Convert datetime strings back to datetime objects
                    pattern_dict['last_updated'] = datetime.fromisoformat(pattern_dict['last_updated'])
                    self.experience_patterns[pattern_id] = ExperiencePattern(**pattern_dict)
        except Exception as e:
            print(f"Error loading patterns: {e}")
    
    async def _load_prompt_evolution(self):
        """Load prompt evolution history from JSON"""
        if not self.prompts_path.exists():
            return
            
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                evolution_data = json.load(f)
                for prompt_id, evolutions_list in evolution_data.items():
                    evolutions = []
                    for evo_dict in evolutions_list:
                        # Convert datetime strings back to datetime objects
                        evo_dict['created_at'] = datetime.fromisoformat(evo_dict['created_at'])
                        evolutions.append(PromptEvolution(**evo_dict))
                    self.prompt_evolution[prompt_id] = evolutions
        except Exception as e:
            print(f"Error loading prompt evolution: {e}")
    
    async def add_memory(self, memory: MemoryEntry):
        """Add new memory entry"""
        self.recent_memories.append(memory)
        
        # Write to JSONL immediately for durability
        await self._append_to_interactions(memory)
        
        # Maintain memory size limit
        if len(self.recent_memories) > self.max_memory_size:
            # Archive oldest memories to JSON
            await self._archive_old_memories()
    
    async def _append_to_interactions(self, memory: MemoryEntry):
        """Append memory to interactions JSONL file"""
        try:
            with open(self.interactions_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(memory.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error writing to interactions: {e}")
    
    async def _archive_old_memories(self):
        """Archive old memories to JSON format"""
        try:
            # Take oldest 20% of memories for archival
            archive_count = len(self.recent_memories) // 5
            to_archive = self.recent_memories[:archive_count]
            
            # Convert to JSON format
            new_archived_data = [memory.to_dict() for memory in to_archive]
            
            # Load existing archived data or create new
            archived_data = {'experiences': []}
            if self.experiences_path.exists():
                with open(self.experiences_path, 'r', encoding='utf-8') as f:
                    archived_data = json.load(f)
            
            # Append new archived memories
            archived_data['experiences'].extend(new_archived_data)
            archived_data['last_updated'] = datetime.now().isoformat()
            archived_data['total_count'] = len(archived_data['experiences'])
            
            # Save updated archive
            with open(self.experiences_path, 'w', encoding='utf-8') as f:
                json.dump(archived_data, f, ensure_ascii=False, indent=2)
            
            # Remove from recent memories
            self.recent_memories = self.recent_memories[archive_count:]
            
            # Rewrite interactions file with remaining memories
            await self._rewrite_interactions()
            
        except Exception as e:
            print(f"Error archiving memories: {e}")
    
    async def _rewrite_interactions(self):
        """Rewrite interactions file with current recent memories"""
        try:
            with open(self.interactions_path, 'w', encoding='utf-8') as f:
                for memory in self.recent_memories:
                    f.write(json.dumps(memory.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error rewriting interactions: {e}")
    
    async def get_recent_memories(self, limit: int = 100, memory_type: Optional[MemoryType] = None) -> List[MemoryEntry]:
        """Get recent memories with optional filtering"""
        memories = self.recent_memories
        
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        return memories[-limit:] if memories else []
    
    async def get_successful_experiences(self, limit: int = 50) -> List[MemoryEntry]:
        """Get successful experiences for learning"""
        successful = [m for m in self.recent_memories if m.success is True and m.reward and m.reward > 0]
        return sorted(successful, key=lambda x: x.reward or 0, reverse=True)[:limit]
    
    async def get_failed_experiences(self, limit: int = 50) -> List[MemoryEntry]:
        """Get failed experiences for learning"""
        failed = [m for m in self.recent_memories if m.success is False]
        return failed[-limit:]
    
    async def extract_experience_pattern(self, experiences: List[MemoryEntry]) -> Optional[ExperiencePattern]:
        """Extract common patterns from experiences"""
        if not experiences:
            return None
        
        # Analyze common conditions, actions, and outcomes
        conditions = {}
        actions = []
        outcomes = {}
        
        for exp in experiences:
            content = exp.content
            if 'conditions' in content:
                for k, v in content['conditions'].items():
                    conditions[k] = conditions.get(k, []) + [v]
            if 'actions' in content:
                actions.extend(content['actions'])
            if 'outcomes' in content:
                for k, v in content['outcomes'].items():
                    outcomes[k] = outcomes.get(k, []) + [v]
        
        # Create pattern
        pattern_id = str(uuid.uuid4())
        pattern_type = "success" if experiences[0].success else "failure"
        
        pattern = ExperiencePattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            description=f"Pattern extracted from {len(experiences)} {pattern_type} experiences",
            conditions=conditions,
            actions=list(set(actions)),
            outcomes=outcomes,
            confidence=min(1.0, len(experiences) / 10.0),
            last_updated=datetime.now()
        )
        
        self.experience_patterns[pattern_id] = pattern
        await self._save_patterns()
        
        return pattern
    
    async def evolve_prompt(self, current_prompt: str, performance_metrics: Dict[str, float], 
                          improvements: List[str]) -> PromptEvolution:
        """Create evolved version of prompt based on experience"""
        prompt_id = str(uuid.uuid4())
        
        # Find parent prompt if exists
        parent_id = None
        for pid, evolutions in self.prompt_evolution.items():
            if evolutions and evolutions[-1].prompt_text == current_prompt:
                parent_id = pid
                break
        
        # Create new evolution
        version = 1
        if parent_id and parent_id in self.prompt_evolution:
            version = len(self.prompt_evolution[parent_id]) + 1
        
        evolution = PromptEvolution(
            prompt_id=prompt_id,
            version=version,
            prompt_text=current_prompt,
            performance_metrics=performance_metrics,
            improvements=improvements,
            created_at=datetime.now(),
            parent_prompt_id=parent_id
        )
        
        # Store evolution
        if parent_id:
            self.prompt_evolution[parent_id].append(evolution)
        else:
            self.prompt_evolution[prompt_id] = [evolution]
        
        await self._save_prompt_evolution()
        return evolution
    
    async def get_best_prompt(self, task_type: str) -> Optional[str]:
        """Get best performing prompt for a task type"""
        best_prompt = None
        best_score = -1
        
        for evolutions in self.prompt_evolution.values():
            for evolution in evolutions:
                if task_type in evolution.performance_metrics:
                    score = evolution.performance_metrics[task_type]
                    if score > best_score:
                        best_score = score
                        best_prompt = evolution.prompt_text
        
        return best_prompt
    
    async def _save_patterns(self):
        """Save experience patterns to JSON"""
        try:
            patterns_data = {}
            for pattern_id, pattern in self.experience_patterns.items():
                pattern_dict = pattern.dict()
                pattern_dict['last_updated'] = pattern_dict['last_updated'].isoformat()
                patterns_data[pattern_id] = pattern_dict
            
            with open(self.patterns_path, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving patterns: {e}")
    
    async def _save_prompt_evolution(self):
        """Save prompt evolution to JSON"""
        try:
            evolution_data = {}
            for prompt_id, evolutions in self.prompt_evolution.items():
                evolution_list = []
                for evo in evolutions:
                    evo_dict = evo.dict()
                    evo_dict['created_at'] = evo_dict['created_at'].isoformat()
                    evolution_list.append(evo_dict)
                evolution_data[prompt_id] = evolution_list
            
            with open(self.prompts_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving prompt evolution: {e}")
    
    async def analyze_memory_trends(self) -> Dict[str, Any]:
        """Analyze trends in memory data"""
        if not self.recent_memories:
            return {}
        
        # Success rate over time
        recent_success = [m for m in self.recent_memories[-100:] if m.success is not None]
        success_rate = sum(1 for m in recent_success if m.success) / len(recent_success) if recent_success else 0
        
        # Average reward trend
        recent_rewards = [m.reward for m in self.recent_memories[-100:] if m.reward is not None]
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        
        # Memory type distribution
        type_counts = {}
        for memory in self.recent_memories[-200:]:
            type_counts[memory.memory_type.value] = type_counts.get(memory.memory_type.value, 0) + 1
        
        analytics = {
            'success_rate': success_rate,
            'average_reward': avg_reward,
            'memory_type_distribution': type_counts,
            'total_memories': len(self.recent_memories),
            'archived_memories': self._archived_experiences_count,
            'experience_patterns': len(self.experience_patterns),
            'prompt_evolutions': sum(len(evols) for evols in self.prompt_evolution.values()),
            'last_updated': datetime.now().isoformat()
        }
        
        # Save analytics for future reference
        await self._save_analytics(analytics)
        
        return analytics
    
    async def _save_analytics(self, analytics: Dict[str, Any]):
        """Save analytics data to JSON"""
        try:
            with open(self.analytics_path, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving analytics: {e}")
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Remove old memories
        self.recent_memories = [m for m in self.recent_memories if m.timestamp > cutoff_date]
        
        # Rewrite files
        await self._rewrite_interactions()
        
        # Clean up archived experiences
        if self.experiences_path.exists():
            try:
                with open(self.experiences_path, 'r', encoding='utf-8') as f:
                    archived_data = json.load(f)
                
                # Filter out old experiences
                if 'experiences' in archived_data:
                    filtered_experiences = []
                    for exp in archived_data['experiences']:
                        exp_timestamp = datetime.fromisoformat(exp['timestamp'])
                        if exp_timestamp > cutoff_date:
                            filtered_experiences.append(exp)
                    
                    archived_data['experiences'] = filtered_experiences
                    archived_data['total_count'] = len(filtered_experiences)
                    archived_data['last_cleaned'] = datetime.now().isoformat()
                    
                    with open(self.experiences_path, 'w', encoding='utf-8') as f:
                        json.dump(archived_data, f, ensure_ascii=False, indent=2)
                        
            except Exception as e:
                print(f"Error cleaning archived data: {e}")
    
    async def export_analytics_data(self, output_path: Path):
        """Export data for analytics in JSON format"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export recent memories as JSON
        with open(output_path / "recent_memories.json", 'w', encoding='utf-8') as f:
            json.dump([m.to_dict() for m in self.recent_memories], f, ensure_ascii=False, indent=2)
        
        # Export patterns as JSON
        patterns_dict = {}
        for k, v in self.experience_patterns.items():
            pattern_data = v.dict()
            pattern_data['last_updated'] = pattern_data['last_updated'].isoformat()
            patterns_dict[k] = pattern_data
        with open(output_path / "experience_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(patterns_dict, f, ensure_ascii=False, indent=2)
        
        # Export analytics summary
        trends = await self.analyze_memory_trends()
        with open(output_path / "analytics_summary.json", 'w', encoding='utf-8') as f:
            json.dump(trends, f, ensure_ascii=False, indent=2)

    async def search_memories(
        self, 
        query: str = None,
        memory_type: MemoryType = None,
        success_only: bool = False,
        limit: int = 50
    ) -> List[MemoryEntry]:
        """Search memories with various filters"""
        results = []
        
        # Search recent memories
        for memory in self.recent_memories:
            # Filter by memory type
            if memory_type and memory.memory_type != memory_type:
                continue
                
            # Filter by success
            if success_only and not memory.success:
                continue
                
            # Simple text search
            if query:
                memory_text = json.dumps(memory.content, default=str).lower()
                if query.lower() not in memory_text:
                    continue
            
            results.append(memory)
        
        # Sort by timestamp (most recent first) and limit
        results.sort(key=lambda m: m.timestamp, reverse=True)
        return results[:limit]