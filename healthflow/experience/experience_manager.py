import json
import aiofiles
from pathlib import Path
from typing import List
from loguru import logger
import re

from .experience_models import Experience

class ExperienceManager:
    """
    Manages the persistent storage and retrieval of structured experiences
    using a JSONL file. This forms the system's long-term, evolving memory.
    """
    def __init__(self, workspace_dir: str):
        self.experience_path = Path(workspace_dir) / "experience.jsonl"
        # Ensure the workspace directory exists
        self.experience_path.parent.mkdir(exist_ok=True)
        # Ensure the file exists
        if not self.experience_path.exists():
            self.experience_path.touch()
        logger.info(f"ExperienceManager initialized. Knowledge base at: {self.experience_path}")

    async def save_experiences(self, experiences: List[Experience]):
        """
        Appends a list of new Experience objects to the experience.jsonl file.
        Each experience is saved as a single JSON line.
        """
        if not experiences:
            return

        async with aiofiles.open(self.experience_path, mode='a', encoding='utf-8') as f:
            for exp in experiences:
                await f.write(exp.model_dump_json() + '\n')

        logger.info(f"Saved {len(experiences)} new experiences to the knowledge base.")

    async def retrieve_experiences(self, query: str, k: int = 5) -> List[Experience]:
        """
        Retrieves the top k most relevant experiences using a simple keyword matching algorithm.

        This is a basic RAG implementation. For production systems, this could be
        upgraded to use vector embeddings and a vector database for semantic search.

        Args:
            query: The user request string to find relevant experiences for.
            k: The maximum number of experiences to return.

        Returns:
            A list of the most relevant Experience objects.
        """
        if not self.experience_path.exists() or self.experience_path.stat().st_size == 0:
            return []

        all_experiences: List[Experience] = []
        async with aiofiles.open(self.experience_path, mode='r', encoding='utf-8') as f:
            async for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        all_experiences.append(Experience(**data))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Skipping corrupted line in experience.jsonl: {e}")

        if not all_experiences:
            return []

        # Simple keyword-based scoring
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        scored_experiences = []
        for exp in all_experiences:
            score = 0
            content_words = set(re.findall(r'\b\w+\b', (exp.content + " " + exp.category).lower()))
            matching_words = query_words.intersection(content_words)
            score = len(matching_words)

            if score > 0:
                scored_experiences.append((score, exp))

        # Sort by score in descending order
        scored_experiences.sort(key=lambda x: x[0], reverse=True)

        # Return the top k experiences (just the Experience object)
        top_k_experiences = [exp for score, exp in scored_experiences[:k]]

        logger.debug(f"Retrieved {len(top_k_experiences)} experiences for query: '{query}'")
        return top_k_experiences