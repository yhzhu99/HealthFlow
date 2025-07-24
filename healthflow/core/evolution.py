import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

from .prompts import get_prompt as get_default_prompt

logger = logging.getLogger(__name__)

class EvolutionManager:
    """Manages the evolution of prompts and strategies for HealthFlow."""

    def __init__(self, evolution_dir: Path):
        self.evolution_dir = evolution_dir
        self.evolution_dir.mkdir(parents=True, exist_ok=True)

        self.prompts_file = self.evolution_dir / "prompts.json"
        self.strategies_file = self.evolution_dir / "strategies.json"

        self.prompts = self._load_json(self.prompts_file, self._get_default_prompts)
        self.strategies = self._load_json(self.strategies_file, self._get_default_strategies)

    def _load_json(self, file_path: Path, default_factory):
        if not file_path.exists():
            data = default_factory()
            self._save_json(file_path, data)
            return data
        try:
            with file_path.open('r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load {file_path}, using defaults. Error: {e}")
            return default_factory()

    def _save_json(self, file_path: Path, data: Dict):
        try:
            with file_path.open('w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Could not save to {file_path}. Error: {e}")

    def save_all(self):
        """Saves all evolution data to disk."""
        self._save_json(self.prompts_file, self.prompts)
        self._save_json(self.strategies_file, self.strategies)
        logger.info("Evolution data saved.")

    def save_evolution_data(self):
        """Alias for save_all for backward compatibility."""
        self.save_all()

    # --- Prompt Evolution ---

    def _get_default_prompts(self) -> Dict:
        prompts = {}
        for role in ["orchestrator", "expert", "analyst"]:
            prompts[role] = [{
                "id": f"{role}_v1",
                "content": get_default_prompt(role),
                "score": 5.0, # Lower initial score to encourage evolution
                "created_at": datetime.now().isoformat(),
                "feedback": "Initial prompt."
            }]
        return prompts

    def get_best_prompt(self, role: str) -> Tuple[str, float]:
        """Gets the content and score of the best prompt for a role."""
        if role not in self.prompts or not self.prompts[role]:
            return get_default_prompt(role), 5.0

        best_prompt_version = max(self.prompts[role], key=lambda p: p['score'])
        return best_prompt_version['content'], best_prompt_version['score']

    def get_best_prompt_with_id(self, role: str) -> Tuple[str, str, float]:
        """Gets the ID, content, and score of the best prompt."""
        if role not in self.prompts or not self.prompts[role]:
            return f"{role}_v1", get_default_prompt(role), 5.0

        best = max(self.prompts[role], key=lambda p: p['score'])
        return best['id'], best['content'], best['score']

    def add_prompt_version(self, role: str, new_content: str, score: float, feedback: str):
        """Adds a new evolved prompt version."""
        if role not in self.prompts:
            self.prompts[role] = []

        version_num = len(self.prompts[role]) + 1
        new_version = {
            "id": f"{role}_v{version_num}",
            "content": new_content,
            "score": score,
            "created_at": datetime.now().isoformat(),
            "feedback": feedback
        }
        self.prompts[role].append(new_version)
        self.save_all()
        logger.info(f"Added new prompt version {new_version['id']} for role '{role}' with score {score:.2f}")

    def get_prompt_status(self) -> Dict:
        """Gets a summary of prompt evolution status."""
        status = {}
        for role, versions in self.prompts.items():
            if versions:
                best_score = max(p['score'] for p in versions)
                status[role] = {"versions": len(versions), "best_score": best_score}
            else:
                status[role] = {"versions": 0, "best_score": 0.0}
        return status

    # --- Strategy Evolution ---

    def _get_default_strategies(self) -> Dict:
        return {
            "analyst_only": {"usage_count": 0, "success_count": 0, "success_rate": 0.0},
            "expert_only": {"usage_count": 0, "success_count": 0, "success_rate": 0.0},
            "expert_then_analyst": {"usage_count": 0, "success_count": 0, "success_rate": 0.0},
        }

    def record_strategy_usage(self, strategy_name: str):
        if strategy_name not in self.strategies:
            return
        self.strategies[strategy_name]["usage_count"] += 1
        self.save_all()

    def update_strategy_performance(self, strategy_name: str, success: bool):
        if strategy_name not in self.strategies:
            return
        stats = self.strategies[strategy_name]
        if success:
            stats["success_count"] += 1

        total = stats["usage_count"]
        if total > 0:
            stats["success_rate"] = stats["success_count"] / total
        self.save_all()

    def get_strategy_performance(self) -> Dict:
        return self.strategies
