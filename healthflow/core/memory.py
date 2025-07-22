"""
Memory Management System for HealthFlow
Supports persistent storage using jsonl, parquet, and pickle formats
Implements experience accumulation and self-evolving capabilities
"""

import json
import pickle
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import pandas as pd
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
    """Advanced memory management with multiple storage formats and self-evolving capabilities"""
    
    def __init__(self, memory_dir: Path, max_memory_size: int = 10000):
        self.memory_dir = Path(memory_dir)
        self.max_memory_size = max_memory_size
        
        # Storage paths
        self.jsonl_path = self.memory_dir / "interactions.jsonl"
        self.parquet_path = self.memory_dir / "experiences.parquet"
        self.patterns_path = self.memory_dir / "patterns.pkl"
        self.prompts_path = self.memory_dir / "prompt_evolution.pkl"
        
        # In-memory stores
        self.recent_memories: List[MemoryEntry] = []
        self.experience_patterns: Dict[str, ExperiencePattern] = {}
        self.prompt_evolution: Dict[str, List[PromptEvolution]] = {}
        
        # Create directory
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialize and load all data"""
        await self._load_all_data()
    
    async def _load_all_data(self):
        """Load all persistent data"""
        await asyncio.gather(
            self._load_jsonl_data(),
            self._load_parquet_data(), 
            self._load_patterns(),
            self._load_prompt_evolution()
        )
    
    async def _load_jsonl_data(self):
        """Load recent interactions from JSONL"""
        if not self.jsonl_path.exists():
            return
            
        try:
            with open(self.jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        memory = MemoryEntry.from_dict(data)
                        self.recent_memories.append(memory)
        except Exception as e:
            print(f"Error loading JSONL data: {e}")
    
    async def _load_parquet_data(self):
        """Load experience data from Parquet"""
        if not self.parquet_path.exists():
            return
            
        try:
            df = pd.read_parquet(self.parquet_path)
            # Convert back to memory entries if needed
            # This is for analytical queries, main storage is in recent_memories
        except Exception as e:
            print(f"Error loading Parquet data: {e}")
    
    async def _load_patterns(self):
        """Load experience patterns from pickle"""
        if not self.patterns_path.exists():
            return
            
        try:
            with open(self.patterns_path, 'rb') as f:
                self.experience_patterns = pickle.load(f)
        except Exception as e:
            print(f"Error loading patterns: {e}")
    
    async def _load_prompt_evolution(self):
        """Load prompt evolution history from pickle"""
        if not self.prompts_path.exists():
            return
            
        try:
            with open(self.prompts_path, 'rb') as f:
                self.prompt_evolution = pickle.load(f)
        except Exception as e:
            print(f"Error loading prompt evolution: {e}")
    
    async def add_memory(self, memory: MemoryEntry):
        """Add new memory entry"""
        self.recent_memories.append(memory)
        
        # Write to JSONL immediately for durability
        await self._append_to_jsonl(memory)
        
        # Maintain memory size limit
        if len(self.recent_memories) > self.max_memory_size:
            # Archive oldest memories to parquet
            await self._archive_old_memories()
    
    async def _append_to_jsonl(self, memory: MemoryEntry):
        """Append memory to JSONL file"""
        try:
            with open(self.jsonl_path, 'a') as f:
                f.write(json.dumps(memory.to_dict()) + '\n')
        except Exception as e:
            print(f"Error writing to JSONL: {e}")
    
    async def _archive_old_memories(self):
        """Archive old memories to parquet format"""
        try:
            # Take oldest 20% of memories for archival
            archive_count = len(self.recent_memories) // 5
            to_archive = self.recent_memories[:archive_count]
            
            # Convert to DataFrame
            data = [memory.to_dict() for memory in to_archive]
            df = pd.DataFrame(data)
            
            # Append to existing parquet or create new
            if self.parquet_path.exists():
                existing_df = pd.read_parquet(self.parquet_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_parquet(self.parquet_path, index=False)
            
            # Remove from recent memories
            self.recent_memories = self.recent_memories[archive_count:]
            
            # Rewrite JSONL with remaining memories
            await self._rewrite_jsonl()
            
        except Exception as e:
            print(f"Error archiving memories: {e}")
    
    async def _rewrite_jsonl(self):
        """Rewrite JSONL file with current recent memories"""
        try:
            with open(self.jsonl_path, 'w') as f:
                for memory in self.recent_memories:
                    f.write(json.dumps(memory.to_dict()) + '\n')
        except Exception as e:
            print(f"Error rewriting JSONL: {e}")
    
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
        """Save experience patterns to pickle"""
        try:
            with open(self.patterns_path, 'wb') as f:
                pickle.dump(self.experience_patterns, f)
        except Exception as e:
            print(f"Error saving patterns: {e}")
    
    async def _save_prompt_evolution(self):
        """Save prompt evolution to pickle"""
        try:
            with open(self.prompts_path, 'wb') as f:
                pickle.dump(self.prompt_evolution, f)
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
        
        return {
            'success_rate': success_rate,
            'average_reward': avg_reward,
            'memory_type_distribution': type_counts,
            'total_memories': len(self.recent_memories),
            'experience_patterns': len(self.experience_patterns),
            'prompt_evolutions': sum(len(evols) for evols in self.prompt_evolution.values())
        }
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data beyond retention period"""
        cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
        
        # Remove old memories
        self.recent_memories = [m for m in self.recent_memories if m.timestamp > cutoff_date]
        
        # Rewrite files
        await self._rewrite_jsonl()
        
        # Clean up parquet file
        if self.parquet_path.exists():
            try:
                df = pd.read_parquet(self.parquet_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[df['timestamp'] > cutoff_date]
                df.to_parquet(self.parquet_path, index=False)
            except Exception as e:
                print(f"Error cleaning parquet data: {e}")
    
    async def export_analytics_data(self, output_path: Path):
        """Export data for analytics in multiple formats"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export recent memories as JSON
        with open(output_path / "recent_memories.json", 'w') as f:
            json.dump([m.to_dict() for m in self.recent_memories], f, indent=2)
        
        # Export patterns as JSON
        patterns_dict = {k: v.dict() for k, v in self.experience_patterns.items()}
        with open(output_path / "experience_patterns.json", 'w') as f:
            json.dump(patterns_dict, f, indent=2, default=str)
        
        # Export analytics summary
        trends = await self.analyze_memory_trends()
        with open(output_path / "analytics_summary.json", 'w') as f:
            json.dump(trends, f, indent=2)