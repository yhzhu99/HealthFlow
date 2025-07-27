import json
import aiofiles
from pathlib import Path
from typing import List
from loguru import logger
import re

from .experience_models import Experience
from ..core.llm_provider import LLMProvider, LLMMessage

class ExperienceManager:
    """
    Manages the persistent storage and retrieval of structured experiences
    using a JSONL file. This forms the system's long-term, evolving memory.
    """
    def __init__(self, experience_path: Path, llm_provider: LLMProvider = None):
        self.experience_path = experience_path
        self.llm_provider = llm_provider
        # Ensure the parent directory exists
        self.experience_path.parent.mkdir(parents=True, exist_ok=True)
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
        Retrieves the top k most relevant experiences using semantic search with LLM.
        Falls back to keyword matching if LLM is not available.

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
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        logger.warning(f"Skipping corrupted line in experience.jsonl: {e}")

        if not all_experiences:
            return []

        # Use semantic retrieval if LLM is available, otherwise fall back to keyword matching
        if self.llm_provider:
            try:
                return await self._semantic_retrieve(query, all_experiences, k)
            except Exception as e:
                logger.warning(f"Semantic retrieval failed, falling back to keyword matching: {e}")
                return await self._keyword_retrieve(query, all_experiences, k)
        else:
            return await self._keyword_retrieve(query, all_experiences, k)

    async def _semantic_retrieve(self, query: str, all_experiences: List[Experience], k: int) -> List[Experience]:
        """Use LLM to semantically match experiences to the query."""
        # Prepare experiences for LLM evaluation
        experiences_text = "\n\n".join([
            f"Experience {i+1}:\n- Type: {exp.type.value}\n- Category: {exp.category}\n- Content: {exp.content}"
            for i, exp in enumerate(all_experiences)
        ])
        
        system_prompt = """You are an expert at matching user queries to relevant past experiences in a healthcare AI system. Your task is to analyze a user query and rank the provided experiences by relevance.

Return a JSON object with a "rankings" array containing the indices (1-based) of the most relevant experiences, ordered from most to least relevant. Only include experiences that are actually relevant to the query."""

        user_prompt = f"""User Query: {query}

Available Experiences:
{experiences_text}

Please rank the experiences by relevance to the user query. Return only the indices of relevant experiences in order of decreasing relevance."""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        response = await self.llm_provider.generate(messages, json_mode=True)
        
        try:
            rankings_data = json.loads(response.content)
            rankings = rankings_data.get("rankings", [])
            
            # Convert 1-based indices to 0-based and get corresponding experiences
            relevant_experiences = []
            for idx in rankings[:k]:  # Limit to top k
                if 1 <= idx <= len(all_experiences):
                    relevant_experiences.append(all_experiences[idx - 1])
            
            logger.debug(f"Semantic retrieval found {len(relevant_experiences)} relevant experiences for query: '{query}'")
            return relevant_experiences
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse semantic retrieval response: {e}")
            # Fall back to keyword matching
            return await self._keyword_retrieve(query, all_experiences, k)

    async def _keyword_retrieve(self, query: str, all_experiences: List[Experience], k: int) -> List[Experience]:
        """Original keyword-based retrieval as fallback."""
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

        logger.debug(f"Keyword retrieval found {len(top_k_experiences)} experiences for query: '{query}'")
        return top_k_experiences