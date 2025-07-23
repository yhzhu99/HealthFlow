import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from healthflow.core.prompts import get_prompt_template
from healthflow.evaluation.evaluator import EvaluationResult

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages the persistent memory of the HealthFlow system, which is crucial
    for its self-evolving capabilities. It stores experiences, evaluation
    results, and evolving prompt templates.
    """
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.experiences_path = memory_dir / "experiences.jsonl"
        self.prompts_path = memory_dir / "prompts.json"
        self.prompts_db: Dict[str, Any] = {}

    async def initialize(self):
        """Loads the prompts database from disk."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if self.prompts_path.exists():
            with self.prompts_path.open("r") as f:
                self.prompts_db = json.load(f)
        else:
            # Initialize with default prompts
            for role in ["orchestrator", "expert", "analyst"]:
                self.prompts_db[role] = [{
                    "id": f"default_{role}_v1",
                    "version": 1,
                    "prompt": get_prompt_template(role),
                    "performance_scores": [],
                    "avg_score": 0.0,
                    "created_at": datetime.now().isoformat(),
                }]
            await self._save_prompts()
        logger.info("MemoryManager initialized.")

    async def _save_prompts(self):
        """Saves the current state of the prompts database to disk."""
        with self.prompts_path.open("w") as f:
            json.dump(self.prompts_db, f, indent=2)

    async def add_experience(
        self,
        task_id: str,
        task_description: str,
        trace: List[Dict[str, Any]],
        evaluation: EvaluationResult,
    ):
        """Saves a new experience to the memory log."""
        experience = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "task_description": task_description,
            "evaluation": evaluation.to_dict(),
            "trace": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "meta": msg.meta_dict
                }
                for msg in trace
            ]
        }
        with self.experiences_path.open("a") as f:
            f.write(json.dumps(experience) + "\n")

    def get_best_prompt(self, role: str) -> Tuple[str, str]:
        """
        Retrieves the best-performing prompt for a given agent role.
        Returns the prompt text and its ID.
        """
        if role not in self.prompts_db or not self.prompts_db[role]:
            return get_prompt_template(role), f"default_{role}_v1"

        prompts = self.prompts_db[role]
        best_prompt = max(prompts, key=lambda p: p.get("avg_score", 0.0))
        logger.info(f"Selected prompt '{best_prompt['id']}' for role '{role}' with avg score {best_prompt['avg_score']:.2f}")
        return best_prompt["prompt"], best_prompt["id"]

    async def update_prompt_performance(self, prompt_id: str, role: str, score: float):
        """Updates the performance score of a specific prompt."""
        if role not in self.prompts_db:
            return

        for prompt in self.prompts_db[role]:
            if prompt["id"] == prompt_id:
                scores = prompt.get("performance_scores", [])
                scores.append(score)
                prompt["performance_scores"] = scores
                prompt["avg_score"] = sum(scores) / len(scores)
                await self._save_prompts()
                break

    async def evolve_prompt(self, role: str, suggestion: str, score: float) -> str:
        """
        Creates a new, evolved version of a prompt for a role based on a suggestion.
        This is a simple evolution strategy. More complex strategies could involve
        an LLM rewriting the prompt.
        """
        if role not in self.prompts_db:
            return ""

        base_prompt_text, _ = self.get_best_prompt(role)

        new_prompt_text = (
            f"{base_prompt_text}\n\n"
            f"# EVOLUTION NOTE: Incorporate this feedback for better performance:\n"
            f"# {suggestion}"
        )

        version = len(self.prompts_db[role]) + 1
        new_prompt_id = f"{role}_v{version}"

        new_prompt = {
            "id": new_prompt_id,
            "version": version,
            "prompt": new_prompt_text,
            "performance_scores": [score], # Start with the score that triggered the evolution
            "avg_score": score,
            "created_at": datetime.now().isoformat(),
            "evolution_reason": suggestion,
        }

        self.prompts_db[role].append(new_prompt)
        await self._save_prompts()
        logger.info(f"Evolved new prompt '{new_prompt_id}' for role '{role}'.")
        return new_prompt_id