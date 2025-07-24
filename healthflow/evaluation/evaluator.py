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

        # FIX: Check for critical failures BEFORE calling LLM evaluation
        has_critical_failures = self._has_critical_failures(trace_str, "")

        full_prompt = f"Task:\n{task_description}\n\nTrace:\n{trace_str}\n\nEvaluation Instructions:\n{self.evaluation_prompt_template}"

        # FIX: The model's run method expects a list of message dictionaries, not a raw string.
        # We wrap the prompt in the correct format.
        messages_for_llm = [{"role": "user", "content": full_prompt}]

        try:
            # FIX: The run method returns a response object, not a string. We must parse it.
            response_obj = self.evaluator_llm.run(messages_for_llm)

            response_content = ""
            if response_obj and hasattr(response_obj, 'choices') and response_obj.choices and hasattr(response_obj.choices[0], 'message'):
                response_content = response_obj.choices[0].message.content or ""

            eval_data = self._parse_evaluation_response(response_content)

            # FIX: Better handling of evaluation failures and critical errors
            if not eval_data or "scores" not in eval_data:
                logger.warning("Evaluation parsing failed or returned empty results")
                return self._create_fallback_evaluation(task_id, "Evaluation parsing failed", has_critical_failures)

            # Calculate weighted score
            scores = eval_data.get("scores", {})
            weights = {"success_accuracy": 3, "strategy_reasoning": 2, "tool_usage_agentic_skill": 2, "safety_clarity": 1}

            # FIX: Handle missing scores more gracefully
            total_score = 0.0
            total_weight = 0.0
            for key, weight in weights.items():
                if key in scores and isinstance(scores[key], (int, float)):
                    total_score += scores[key] * weight
                    total_weight += weight
                else:
                    logger.warning(f"Missing or invalid score for {key}")

            overall_score = (total_score / total_weight) if total_weight > 0 else 0.0

            # FIX: CRITICAL - If there were execution failures, cap the score significantly
            if has_critical_failures:
                overall_score = min(overall_score, 2.0)  # Cap at 2.0 if there were critical failures
                logger.info(f"Critical failures detected, capping score at {overall_score}")

            # FIX: More robust success determination
            # Consider a task successful only if it has a reasonable score AND no critical failures
            task_success = (
                overall_score >= self.config.success_threshold and
                not has_critical_failures
            )

            return EvaluationResult(
                evaluation_id=str(uuid.uuid4()),
                task_id=task_id,
                timestamp=datetime.now(),
                overall_success=task_success,
                overall_score=overall_score,
                scores=scores,
                executive_summary=eval_data.get("executive_summary", "Evaluation completed but summary is missing."),
                improvement_suggestions=eval_data.get("improvement_suggestions", {}),
            )
        except Exception as e:
            logger.error(f"Error during task evaluation: {e}", exc_info=True)
            return self._create_fallback_evaluation(task_id, str(e), has_critical_failures)

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
                parsed_data = json.loads(json_str)

                # FIX: Validate the parsed data structure
                if not isinstance(parsed_data, dict):
                    logger.warning("Evaluation response is not a dictionary")
                    return {}

                # Ensure required fields exist
                if "scores" not in parsed_data:
                    logger.warning("Evaluation response missing 'scores' field")
                    return {}

                return parsed_data
            logger.warning("No JSON object found in evaluation response.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}\nResponse was:\n{response_content}")
            return {}

    def _has_critical_failures(self, trace_str: str, eval_response: str) -> bool:
        """Check for critical failures in the execution trace."""
        failure_indicators = [
            "error:",
            "exception:",
            "traceback",
            "failed to",
            "execution failed",
            "timeout",
            "could not",
            "unable to",
            "can't be used in 'await' expression",  # FIX: Add specific error patterns
            "TypeError:",
            "ValueError:",
            "AttributeError:",
            "ImportError:",
            "Failed to get model response",
            "Error calling model"
        ]

        trace_lower = trace_str.lower()
        for indicator in failure_indicators:
            if indicator.lower() in trace_lower:
                logger.info(f"Critical failure detected with indicator: {indicator}")
                return True
        return False

    def _create_fallback_evaluation(self, task_id: str, error: str, has_critical_failures: bool = True) -> EvaluationResult:
        """Creates a default evaluation result when an error occurs."""
        # FIX: Score should reflect the actual failure state
        if has_critical_failures:
            fallback_score = 1.0  # Very low score for critical failures
            success = False
        else:
            fallback_score = 3.0  # Low score for evaluation failure but no execution failure
            success = False

        return EvaluationResult(
            evaluation_id=str(uuid.uuid4()),
            task_id=task_id,
            timestamp=datetime.now(),
            overall_success=success,
            overall_score=fallback_score,
            scores={"success_accuracy": fallback_score, "strategy_reasoning": fallback_score,
                   "tool_usage_agentic_skill": fallback_score, "safety_clarity": fallback_score},
            executive_summary=f"Task evaluation failed or execution had critical errors: {error}",
            improvement_suggestions={"prompt_templates": ["Fix system errors and improve error handling"],
                                   "tool_creation": [], "collaboration_strategy": []},
        )
