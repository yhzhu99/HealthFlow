"""
Memory Management System for HealthFlow Agents

Implements sophisticated memory mechanisms including:
- Short-term and long-term memory
- Episode memory for task sequences
- Semantic memory for knowledge storage
- Patient-specific memory (with privacy protection)
- Memory consolidation and retrieval
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

from .security import DataProtector


@dataclass
class MemoryItem:
    """Individual memory item"""
    memory_id: str
    content: Any
    memory_type: str  # short_term, long_term, episodic, semantic, patient_specific
    importance_score: float
    access_count: int
    created_at: datetime
    last_accessed: datetime
    tags: List[str]
    patient_id: Optional[str] = None  # For patient-specific memories
    privacy_level: str = "low"  # low, medium, high


@dataclass
class MemoryQuery:
    """Memory query structure"""
    query_text: str
    memory_types: List[str]
    time_range: Optional[Tuple[datetime, datetime]] = None
    patient_id: Optional[str] = None
    min_importance: float = 0.0
    max_results: int = 10


class MemoryManager:
    """
    Advanced memory management system for healthcare agents.
    
    Features:
    - Multi-level memory hierarchy
    - Privacy-aware patient memory
    - Automatic memory consolidation
    - Similarity-based retrieval
    - Memory importance scoring
    """
    
    def __init__(self, agent_id: str, memory_dir: str = "memory"):
        self.agent_id = agent_id
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Initialize data protector for sensitive memories
        self.data_protector = DataProtector()
        
        # Initialize databases
        self.db_path = self.memory_dir / f"{agent_id}_memory.db"
        self._init_database()
        
        # Memory configuration
        self.config = {
            "short_term_capacity": 100,
            "long_term_capacity": 10000,
            "consolidation_threshold": 24,  # hours
            "importance_decay_rate": 0.95,
            "similarity_threshold": 0.7
        }
        
        self.logger = logging.getLogger(f"MemoryManager-{agent_id}")
        self.logger.info(f"Memory manager initialized for agent {agent_id}")
    
    def _init_database(self):
        """Initialize SQLite database for memory storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    patient_id TEXT,
                    privacy_level TEXT DEFAULT 'low'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories (memory_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id_1 TEXT NOT NULL,
                    memory_id_2 TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    FOREIGN KEY (memory_id_1) REFERENCES memories (memory_id),
                    FOREIGN KEY (memory_id_2) REFERENCES memories (memory_id)
                )
            """)
            
            # Create indexes for faster queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories (memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patient_id ON memories (patient_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories (importance_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories (created_at)")
    
    async def store_memory(
        self,
        content: Any,
        memory_type: str,
        importance_score: float,
        tags: List[str],
        patient_id: Optional[str] = None,
        privacy_level: str = "low"
    ) -> str:
        """Store a new memory item"""
        
        # Generate unique memory ID
        content_str = json.dumps(content, default=str)
        memory_id = hashlib.md5(
            f"{self.agent_id}_{datetime.now().isoformat()}_{content_str}".encode()
        ).hexdigest()
        
        # Protect sensitive content if needed
        if privacy_level in ["medium", "high"]:
            content = await self.data_protector.protect_data(content)
        
        # Create memory item
        memory_item = MemoryItem(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance_score=importance_score,
            access_count=0,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            tags=tags,
            patient_id=patient_id,
            privacy_level=privacy_level
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memories 
                (memory_id, content, memory_type, importance_score, access_count, 
                 created_at, last_accessed, tags, patient_id, privacy_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_item.memory_id,
                json.dumps(memory_item.content, default=str),
                memory_item.memory_type,
                memory_item.importance_score,
                memory_item.access_count,
                memory_item.created_at.isoformat(),
                memory_item.last_accessed.isoformat(),
                json.dumps(memory_item.tags),
                memory_item.patient_id,
                memory_item.privacy_level
            ))
        
        # Generate and store embedding for similarity search
        await self._store_embedding(memory_id, content_str)
        
        # Trigger memory consolidation if needed
        await self._check_consolidation()
        
        self.logger.info(f"Stored memory {memory_id} of type {memory_type}")
        return memory_id
    
    async def retrieve_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """Retrieve memories based on query"""
        
        # Start with database query
        with sqlite3.connect(self.db_path) as conn:
            # Build SQL query
            conditions = []
            params = []
            
            if query.memory_types:
                placeholders = ",".join("?" * len(query.memory_types))
                conditions.append(f"memory_type IN ({placeholders})")
                params.extend(query.memory_types)
            
            if query.patient_id:
                conditions.append("patient_id = ?")
                params.append(query.patient_id)
            
            if query.min_importance > 0:
                conditions.append("importance_score >= ?")
                params.append(query.min_importance)
            
            if query.time_range:
                conditions.append("created_at >= ? AND created_at <= ?")
                params.extend([t.isoformat() for t in query.time_range])
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            sql = f"""
                SELECT memory_id, content, memory_type, importance_score, 
                       access_count, created_at, last_accessed, tags, 
                       patient_id, privacy_level
                FROM memories 
                WHERE {where_clause}
                ORDER BY importance_score DESC, last_accessed DESC
                LIMIT ?
            """
            params.append(query.max_results)
            
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
        
        # Convert to MemoryItem objects
        memories = []
        for row in rows:
            memory = MemoryItem(
                memory_id=row[0],
                content=json.loads(row[1]),
                memory_type=row[2],
                importance_score=row[3],
                access_count=row[4],
                created_at=datetime.fromisoformat(row[5]),
                last_accessed=datetime.fromisoformat(row[6]),
                tags=json.loads(row[7]),
                patient_id=row[8],
                privacy_level=row[9]
            )
            memories.append(memory)
        
        # Update access counts
        memory_ids = [m.memory_id for m in memories]
        if memory_ids:
            await self._update_access_counts(memory_ids)
        
        # If we have query text, do semantic similarity search
        if query.query_text and memories:
            memories = await self._rank_by_similarity(query.query_text, memories)
        
        self.logger.info(f"Retrieved {len(memories)} memories for query: {query.query_text[:50]}...")
        return memories
    
    async def retrieve_relevant_memories(
        self,
        task_description: str,
        task_type: str,
        patient_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to a specific task"""
        
        # Create query
        memory_query = MemoryQuery(
            query_text=task_description,
            memory_types=["episodic", "semantic", "long_term"],
            patient_id=patient_id,
            min_importance=0.3,
            max_results=10
        )
        
        # Add patient-specific memories if available
        if patient_id:
            patient_query = MemoryQuery(
                query_text=task_description,
                memory_types=["patient_specific"],
                patient_id=patient_id,
                min_importance=0.1,
                max_results=5
            )
            patient_memories = await self.retrieve_memories(patient_query)
        else:
            patient_memories = []
        
        # Retrieve general memories
        general_memories = await self.retrieve_memories(memory_query)
        
        # Combine and format results
        all_memories = general_memories + patient_memories
        
        # Convert to simple dict format for easy consumption
        formatted_memories = []
        for memory in all_memories:
            formatted_memories.append({
                "memory_id": memory.memory_id,
                "summary": str(memory.content)[:200] + "..." if len(str(memory.content)) > 200 else str(memory.content),
                "type": memory.memory_type,
                "importance": memory.importance_score,
                "tags": memory.tags,
                "patient_specific": memory.patient_id is not None
            })
        
        return formatted_memories
    
    async def store_task_memory(
        self,
        task_id: str,
        task_description: str,
        result: Any,
        evaluation: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Store memory of a completed task"""
        
        # Calculate importance based on evaluation
        importance = self._calculate_task_importance(evaluation)
        
        # Extract patient ID if present
        patient_id = context.get('patient_id')
        
        # Determine privacy level
        privacy_level = "high" if patient_id else "low"
        
        # Create comprehensive memory content
        memory_content = {
            "task_id": task_id,
            "description": task_description,
            "result": result,
            "evaluation": evaluation,
            "context": {k: v for k, v in context.items() if k != 'protected_data'},  # Exclude sensitive data
            "lessons_learned": self._extract_lessons_learned(result, evaluation)
        }
        
        # Store as episodic memory
        await self.store_memory(
            content=memory_content,
            memory_type="episodic",
            importance_score=importance,
            tags=self._generate_memory_tags(task_description, result),
            patient_id=patient_id,
            privacy_level=privacy_level
        )
        
        # If this task generated new knowledge, store as semantic memory too
        if evaluation.get('novel_insights'):
            semantic_content = {
                "insights": evaluation['novel_insights'],
                "task_context": task_description,
                "validation": evaluation.get('confidence', 0.5)
            }
            
            await self.store_memory(
                content=semantic_content,
                memory_type="semantic",
                importance_score=importance * 1.2,  # Boost importance for insights
                tags=["insight", "knowledge"] + self._generate_memory_tags(task_description, result),
                privacy_level="low"  # General knowledge is typically not sensitive
            )
    
    async def _store_embedding(self, memory_id: str, content: str):
        """Generate and store embedding for similarity search"""
        
        # For now, create a simple hash-based embedding
        # In production, use actual embeddings from OpenAI or similar
        
        # Simple bag-of-words embedding for demonstration
        words = content.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create a simple fixed-size vector
        common_words = ['patient', 'medical', 'diagnosis', 'treatment', 'symptom', 
                       'disease', 'health', 'clinical', 'therapy', 'medicine',
                       'hospital', 'doctor', 'nurse', 'surgery', 'medication']
        
        embedding = np.array([word_counts.get(word, 0) for word in common_words])
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding)
                VALUES (?, ?)
            """, (memory_id, embedding.tobytes()))
    
    async def _rank_by_similarity(
        self,
        query_text: str,
        memories: List[MemoryItem]
    ) -> List[MemoryItem]:
        """Rank memories by similarity to query text"""
        
        # Generate query embedding
        words = query_text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        common_words = ['patient', 'medical', 'diagnosis', 'treatment', 'symptom', 
                       'disease', 'health', 'clinical', 'therapy', 'medicine',
                       'hospital', 'doctor', 'nurse', 'surgery', 'medication']
        
        query_embedding = np.array([word_counts.get(word, 0) for word in common_words])
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Calculate similarities
        similarities = []
        with sqlite3.connect(self.db_path) as conn:
            for memory in memories:
                cursor = conn.execute(
                    "SELECT embedding FROM memory_embeddings WHERE memory_id = ?",
                    (memory.memory_id,)
                )
                row = cursor.fetchone()
                if row:
                    memory_embedding = np.frombuffer(row[0], dtype=np.float64)
                    similarity = np.dot(query_embedding, memory_embedding)
                    similarities.append((similarity, memory))
                else:
                    similarities.append((0.0, memory))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [memory for _, memory in similarities]
    
    async def _update_access_counts(self, memory_ids: List[str]):
        """Update access counts for retrieved memories"""
        
        with sqlite3.connect(self.db_path) as conn:
            for memory_id in memory_ids:
                conn.execute("""
                    UPDATE memories 
                    SET access_count = access_count + 1, 
                        last_accessed = ?
                    WHERE memory_id = ?
                """, (datetime.now().isoformat(), memory_id))
    
    async def _check_consolidation(self):
        """Check if memory consolidation is needed"""
        
        # Count short-term memories older than threshold
        threshold_time = datetime.now() - timedelta(hours=self.config["consolidation_threshold"])
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM memories 
                WHERE memory_type = 'short_term' AND created_at < ?
            """, (threshold_time.isoformat(),))
            
            old_short_term_count = cursor.fetchone()[0]
        
        if old_short_term_count > 0:
            await self._consolidate_memories()
    
    async def _consolidate_memories(self):
        """Consolidate old short-term memories to long-term"""
        
        threshold_time = datetime.now() - timedelta(hours=self.config["consolidation_threshold"])
        
        with sqlite3.connect(self.db_path) as conn:
            # Find short-term memories to consolidate
            cursor = conn.execute("""
                SELECT memory_id, importance_score, access_count 
                FROM memories 
                WHERE memory_type = 'short_term' AND created_at < ?
                ORDER BY importance_score DESC, access_count DESC
            """, (threshold_time.isoformat(),))
            
            candidates = cursor.fetchall()
            
            # Move high-importance memories to long-term
            for memory_id, importance, access_count in candidates:
                if importance > 0.5 or access_count > 5:
                    conn.execute("""
                        UPDATE memories 
                        SET memory_type = 'long_term',
                            importance_score = importance_score * 1.1
                        WHERE memory_id = ?
                    """, (memory_id,))
                else:
                    # Delete low-importance memories
                    conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                    conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
        
        self.logger.info("Completed memory consolidation")
    
    def _calculate_task_importance(self, evaluation: Dict[str, Any]) -> float:
        """Calculate importance score for a task memory"""
        
        base_importance = 0.5
        
        # Boost importance for successful tasks
        if evaluation.get('success', False):
            base_importance += 0.2
        
        # Boost for high confidence
        confidence = evaluation.get('confidence', 0.5)
        base_importance += confidence * 0.2
        
        # Boost for novel insights
        if evaluation.get('novel_insights'):
            base_importance += 0.3
        
        # Boost for high performance score
        performance = evaluation.get('performance_score', 0.5)
        base_importance += performance * 0.1
        
        return min(base_importance, 1.0)
    
    def _extract_lessons_learned(self, result: Any, evaluation: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from task execution"""
        
        lessons = []
        
        if not evaluation.get('success', False) and evaluation.get('feedback'):
            lessons.append(f"Failed task lesson: {evaluation['feedback']}")
        
        if evaluation.get('improvement_suggestions'):
            lessons.extend(evaluation['improvement_suggestions'])
        
        if evaluation.get('novel_insights'):
            lessons.extend([f"Insight: {insight}" for insight in evaluation['novel_insights']])
        
        return lessons
    
    def _generate_memory_tags(self, task_description: str, result: Any) -> List[str]:
        """Generate tags for memory indexing"""
        
        tags = []
        
        # Extract medical terms (simplified)
        medical_keywords = [
            'diagnosis', 'treatment', 'symptom', 'disease', 'patient',
            'clinical', 'medical', 'health', 'therapy', 'medication',
            'surgery', 'hospital', 'doctor', 'nurse', 'laboratory'
        ]
        
        description_lower = task_description.lower()
        for keyword in medical_keywords:
            if keyword in description_lower:
                tags.append(keyword)
        
        # Add result-based tags
        result_str = str(result).lower() if result else ""
        if 'positive' in result_str:
            tags.append('positive_result')
        if 'negative' in result_str:
            tags.append('negative_result')
        if 'uncertain' in result_str or 'unclear' in result_str:
            tags.append('uncertain_result')
        
        return list(set(tags))  # Remove duplicates
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Count by type
            cursor = conn.execute("""
                SELECT memory_type, COUNT(*), AVG(importance_score)
                FROM memories 
                GROUP BY memory_type
            """)
            
            type_stats = {}
            for memory_type, count, avg_importance in cursor.fetchall():
                type_stats[memory_type] = {
                    "count": count,
                    "avg_importance": avg_importance
                }
            
            # Total statistics
            cursor = conn.execute("SELECT COUNT(*), AVG(importance_score) FROM memories")
            total_count, avg_importance = cursor.fetchone()
            
            # Patient-specific statistics
            cursor = conn.execute("SELECT COUNT(DISTINCT patient_id) FROM memories WHERE patient_id IS NOT NULL")
            unique_patients = cursor.fetchone()[0]
        
        return {
            "total_memories": total_count,
            "average_importance": avg_importance,
            "memory_types": type_stats,
            "unique_patients": unique_patients,
            "database_path": str(self.db_path)
        }