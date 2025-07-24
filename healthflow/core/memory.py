import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from ..evaluation.evaluator import EvaluationResult

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages the persistent memory of the HealthFlow system.
    It stores experiences (task executions and their evaluations) for learning.
    """
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.experiences_path = memory_dir / "experiences.jsonl"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MemoryManager initialized. Storing experiences in {self.experiences_path}")

    async def add_experience(
        self,
        task_id: str,
        task_description: str,
        trace: List[Any],
        evaluation: EvaluationResult,
    ):
        """Saves a new experience to the memory log."""

        def format_trace(raw_trace: List[Any]) -> List[Dict[str, str]]:
            """Formats the trace into a serializable list of dicts."""
            formatted = []
            for msg in raw_trace:
                if hasattr(msg, 'role_name') and hasattr(msg, 'content'):
                    entry = {
                        "role": msg.role_name,
                        "content": msg.content
                    }
                    if hasattr(msg, 'meta_dict') and msg.meta_dict and 'tool_calls' in msg.meta_dict:
                        entry['tool_calls'] = str(msg.meta_dict['tool_calls'])
                    formatted.append(entry)
            return formatted

        experience = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "task_description": task_description,
            "evaluation": evaluation.to_dict(),
            "trace": format_trace(trace)
        }

        try:
            with self.experiences_path.open("a") as f:
                f.write(json.dumps(experience) + "\n")
            logger.info(f"Successfully stored experience for task {task_id}.")
        except IOError as e:
            logger.error(f"Failed to write experience to {self.experiences_path}: {e}")

    def store_experience(self, experience: Dict[str, Any]):
        """Backward compatibility method."""
        # This is a simple synchronous version for backward compatibility
        try:
            with self.experiences_path.open("a") as f:
                f.write(json.dumps(experience) + "\n")
            logger.info(f"Successfully stored experience for task {experience.get('task_id', 'unknown')}.")
        except IOError as e:
            logger.error(f"Failed to write experience to {self.experiences_path}: {e}")
