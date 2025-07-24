import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import re

from camel.messages import BaseMessage

from healthflow.core.config import HealthFlowConfig
from healthflow.core.prompts import get_prompt

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Rich evaluation result with feedback for self-evolution."""
    evaluation_id: str
    task_id: str
    timestamp: datetime
    overall_success: bool
    overall_score: float
    scores: Dict[str, float]
    executive_summary: str
    improvement_suggestions: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class LLMTaskEvaluator:
    """
    Analyzes a task's execution trace to provide structured feedback for self-evolution.
    """
    def __init__(self, config: HealthFlowConfig):
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType
        self.config = config
        self.evaluator_llm = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=config.model_name,
            api_key=config.api_key,
            url=config.base_url
        )
        self.evaluation_prompt_template = get_prompt("evaluator")

    async def evaluate_task(
        self,
        task_id: str,
        task_description: str,
        conversation_trace: List[BaseMessage],
    ) -> EvaluationResult:
        """Evaluates a completed task by analyzing its conversation trace."""
        trace_str = self._format_trace(task_description, conversation_trace)
        full_prompt = f"Task:\n{task_description}\n\nTrace:\n{trace_str}\n\nEvaluation Instructions:\n{self.evaluation_prompt_template}"

        try:
            response = await self.evaluator_llm.run(full_prompt)
            eval_data = self._parse_evaluation_response(response)

            # Calculate weighted score
            scores = eval_data.get("scores", {})
            weights = {"success_accuracy": 3, "strategy_reasoning": 2, "tool_usage_agentic_skill": 2, "safety_clarity": 1}
            total_score = sum(scores.get(k, 0) * w for k, w in weights.items())
            total_weight = sum(weights.values())
            overall_score = (total_score / total_weight) if total_weight > 0 else 0.0

            return EvaluationResult(
                evaluation_id=str(uuid.uuid4()),
                task_id=task_id,
                timestamp=datetime.now(),
                overall_success=overall_score >= self.config.success_threshold,
                overall_score=overall_score,
                scores=scores,
                executive_summary=eval_data.get("executive_summary", "Evaluation failed to produce a summary."),
                improvement_suggestions=eval_data.get("improvement_suggestions", {}),
            )
        except Exception as e:
            logger.error(f"Error during task evaluation: {e}", exc_info=True)
            return self._create_fallback_evaluation(task_id, str(e))

    def _format_trace(self, task_desc: str, trace: List[BaseMessage]) -> str:
        formatted = ""
        for msg in trace:
            role = msg.role_name
            content = msg.content.strip()
            formatted += f"\n> {role}:\n{content}\n"
            if msg.meta_dict and 'tool_calls' in msg.meta_dict:
                formatted += f"  [TOOL CALLS]: {str(msg.meta_dict['tool_calls'])}\n"
        return formatted

    def _parse_evaluation_response(self, response_content: str) -> Dict[str, Any]:
        """Parses the JSON output from the evaluator LLM."""
        try:
            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            logger.warning("No JSON object found in evaluation response.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}\nResponse was:\n{response_content}")
            return {}

    def _create_fallback_evaluation(self, task_id: str, error: str) -> EvaluationResult:
        """Creates a default evaluation result when an error occurs."""
        return EvaluationResult(
            evaluation_id=str(uuid.uuid4()),
            task_id=task_id,
            timestamp=datetime.now(),
            overall_success=False,
            overall_score=0.0,
            scores={},
            executive_summary=f"Evaluation failed due to an error: {error}",
            improvement_suggestions={},
        )