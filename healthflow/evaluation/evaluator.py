import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from camel.messages import BaseMessage

from healthflow.core.prompts import get_prompt_template
from healthflow.core.config import HealthFlowConfig
from healthflow.core.llm_provider import (LLMMessage, LLMProvider,
                                           create_llm_provider)

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
    The core of HealthFlow's self-improvement loop. It analyzes a task's
    execution trace and provides structured, multi-dimensional feedback.
    """

    def __init__(self, config: HealthFlowConfig):
        self.config = config
        self.llm_provider = create_llm_provider(
            api_key=config.api_key,
            base_url=config.base_url,
            model_name=config.model_name,
        )
        self.evaluation_prompt = get_prompt_template("evaluator")

    async def evaluate_task(
        self,
        task_id: str,
        task_description: str,
        conversation_trace: List[BaseMessage],
    ) -> EvaluationResult:
        """
        Evaluates a completed task by analyzing its conversation trace.
        """
        trace_str = self._format_trace(task_description, conversation_trace)
        prompt = f"TASK & TRACE:\n{trace_str}\n\nEVALUATION_REQUEST:\n{self.evaluation_prompt}"

        try:
            response = await self.llm_provider.generate(
                messages=[LLMMessage(role="user", content=prompt)],
                max_tokens=2048,
                temperature=0.1,
                timeout=self.config.evaluation_timeout,
            )
            eval_data = self._parse_evaluation_response(response.content)

            overall_score = eval_data.get("overall_score", 0.0)
            return EvaluationResult(
                evaluation_id=str(uuid.uuid4()),
                task_id=task_id,
                timestamp=datetime.now(),
                overall_success=overall_score >= self.config.success_threshold,
                overall_score=overall_score,
                scores=eval_data.get("scores", {}),
                executive_summary=eval_data.get("executive_summary", "Evaluation failed."),
                improvement_suggestions=eval_data.get("improvement_suggestions", {}),
            )
        except Exception as e:
            logger.error(f"Error during task evaluation: {e}", exc_info=True)
            return self._create_fallback_evaluation(task_id, str(e))

    def _format_trace(self, task_desc: str, trace: List[BaseMessage]) -> str:
        """Formats the conversation trace into a readable string for the evaluator."""
        formatted = f"User Query: {task_desc}\n\n--- Conversation Trace ---\n"
        for msg in trace:
            role = msg.role
            content = msg.content.strip()
            tool_calls = msg.meta_dict.get('tool_calls')

            formatted += f"\n[{role.upper()}]:\n{content}\n"
            if tool_calls:
                formatted += "[TOOL CALLS]:\n"
                for tc in tool_calls:
                    formatted += f"  - Function: {tc.function}\n"
                    formatted += f"    Args: {tc.params}\n"
        formatted += "\n--- End of Trace ---"
        return formatted

    def _parse_evaluation_response(self, response_content: str) -> Dict[str, Any]:
        """Parses the JSON output from the evaluator LLM."""
        try:
            # Clean the response to extract only the JSON part
            json_match = __import__("re").search(r"\{.*\}", response_content, __import__("re").DOTALL)
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