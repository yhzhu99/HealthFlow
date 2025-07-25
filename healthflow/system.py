import asyncio
import time
import uuid
from pathlib import Path
from loguru import logger
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from rich.live import Live
from rich.spinner import Spinner

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

from .core.config import HealthFlowConfig
from .core.llm_provider import create_llm_provider
from .agents.task_decomposer_agent import TaskDecomposerAgent
from .agents.evaluator_agent import EvaluatorAgent
from .agents.reflector_agent import ReflectorAgent
from .execution.claude_executor import ClaudeCodeExecutor
from .experience.experience_manager import ExperienceManager
from .experience.experience_models import Experience

class HealthFlowSystem:
    """
    The main orchestrator for the Plan -> Delegate -> Evaluate -> Reflect -> Evolve cycle.
    This class manages the entire lifecycle of a task from user request to completion.
    """
    def __init__(self, config: HealthFlowConfig):
        self.config = config
        self.llm_provider = create_llm_provider(config.llm)
        self.experience_manager = ExperienceManager(config.system.workspace_dir)

        # Initialize agents with the shared LLM provider
        self.decomposer = TaskDecomposerAgent(self.llm_provider)
        self.evaluator = EvaluatorAgent(self.llm_provider)
        self.reflector = ReflectorAgent(self.llm_provider)

        self.executor = ClaudeCodeExecutor()

        self.workspace_dir = Path(config.system.workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)

    async def run_task(self, user_request: str, live: Optional[Live] = None, spinner: Optional[Spinner] = None) -> dict:
        """
        Executes a single task through the full HealthFlow lifecycle.
        This is the main entry point for processing a user request.
        """
        task_id = str(uuid.uuid4())
        task_workspace = self.workspace_dir / task_id
        task_workspace.mkdir(exist_ok=True)
        logger.info(f"[{task_id}] New task started. Request: '{user_request}'")
        logger.info(f"[{task_id}] Workspace created at: {task_workspace}")

        start_time = time.time()

        if spinner and live: spinner.text = "[cyan]Retrieving experiences and triaging task...[/cyan]"
        retrieved_experiences = await self.experience_manager.retrieve_experiences(user_request, k=5)
        logger.info(f"[{task_id}] Retrieved {len(retrieved_experiences)} relevant experiences for triage.")

        triage_result = await self.decomposer.triage_task(user_request, retrieved_experiences)
        task_type = triage_result.get("task_type")

        if task_type == "simple_qa":
            result = await self._run_simple_qa_flow(task_id, task_workspace, user_request, triage_result, live, spinner)
        else:
            result = await self._run_code_execution_flow(task_id, task_workspace, user_request, triage_result, retrieved_experiences, live, spinner)

        execution_time = time.time() - start_time
        result["execution_time"] = execution_time
        result["workspace_path"] = str(task_workspace)

        return result

    async def _run_simple_qa_flow(self, task_id: str, task_workspace: Path, user_request: str, triage_result: Dict[str, Any], live: Optional[Live], spinner: Optional[Spinner]) -> Dict[str, Any]:
        """Handles the workflow for a simple question-answering task."""
        if spinner and live: spinner.text = "[cyan]Answering question...[/cyan]"
        answer = triage_result.get("answer", "I could not generate an answer for this question.")

        if spinner and live: spinner.text = "[cyan]Evaluating answer...[/cyan]"
        evaluation = await self.evaluator.evaluate_qa(user_request, answer)

        is_success = evaluation["score"] >= self.config.evaluation.success_threshold

        full_history = {
            "task_id": task_id,
            "user_request": user_request,
            "task_type": "simple_qa",
            "answer": answer,
            "evaluation": evaluation
        }

        if is_success:
            if spinner and live: spinner.text = "[cyan]Reflecting on answer...[/cyan]"
            new_experiences = await self.reflector.synthesize_qa_experience(user_request, answer, task_id)
            if new_experiences:
                await self.experience_manager.save_experiences(new_experiences)
                logger.info(f"[{task_id}] Saved {len(new_experiences)} new experiences from QA.")
            full_history["experiences"] = [exp.model_dump() for exp in new_experiences]

        final_summary = f"Answer: {answer}\n\nEvaluation: {evaluation.get('reasoning', 'N/A')}"

        history_path = task_workspace / "full_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(full_history, f, indent=2, cls=DateTimeEncoder)

        return {"success": is_success, "final_summary": final_summary}

    async def _run_code_execution_flow(self, task_id: str, task_workspace: Path, user_request: str, triage_result: Dict[str, Any], retrieved_experiences: List[Experience], live: Optional[Live], spinner: Optional[Spinner]) -> Dict[str, Any]:
        """Handles the iterative workflow for a task requiring code execution."""
        full_history = {"task_id": task_id, "user_request": user_request, "task_type": "code_execution", "attempts": []}
        is_success = False
        final_summary = "Task failed to complete within the allowed attempts."
        task_list_md = triage_result.get("plan", "# Fallback Plan\n\nNo plan was generated.")

        for attempt in range(self.config.system.max_retries + 1):
            attempt_num = attempt + 1
            if spinner and live: spinner.text = f"[cyan]Starting attempt {attempt_num}/{self.config.system.max_retries + 1}...[/cyan]"
            logger.info(f"[{task_id}] Starting attempt {attempt_num}/{self.config.system.max_retries + 1}")

            attempt_history = {"attempt": attempt_num, "feedback": None}

            if attempt > 0: # For retries, generate a new plan with feedback
                if spinner and live: spinner.text = "[cyan]Retrieving experiences and creating a new plan...[/cyan]"
                previous_feedback = full_history["attempts"][-1]["evaluation"]["feedback"]
                triage_with_feedback = await self.decomposer.triage_task(user_request, retrieved_experiences, previous_feedback)
                task_list_md = triage_with_feedback.get("plan", task_list_md) # Fallback to old plan

            task_list_path = task_workspace / f"task_list_v{attempt_num}.md"
            with open(task_list_path, "w", encoding="utf-8") as f: f.write(task_list_md)
            attempt_history["task_list"] = task_list_md
            logger.info(f"[{task_id}] Using task list for attempt {attempt_num}.")

            if spinner and live: spinner.text = "[cyan]Delegating to Claude Code for execution...[/cyan]"
            execution_result = await self.executor.execute(task_list_path, task_workspace)
            attempt_history["execution"] = execution_result
            logger.info(f"[{task_id}] Execution completed for attempt {attempt_num}. Success: {execution_result['success']}")

            if spinner and live: spinner.text = "[cyan]Evaluating the execution outcome...[/cyan]"
            evaluation = await self.evaluator.evaluate(user_request, task_list_md, execution_result["log"])
            attempt_history["evaluation"] = evaluation
            logger.info(f"[{task_id}] Evaluation completed. Score: {evaluation['score']}/10.")

            full_history["attempts"].append(attempt_history)

            if evaluation["score"] >= self.config.evaluation.success_threshold:
                logger.info(f"[{task_id}] Task succeeded on attempt {attempt_num}.")
                is_success = True
                final_summary = f"Task completed successfully. Evaluation: {evaluation.get('reasoning', 'N/A')}"
                break
            else:
                logger.warning(f"[{task_id}] Attempt {attempt_num} failed with score {evaluation['score']}. Retrying if possible.")
                final_summary = f"Task failed after {attempt_num} attempts. Final feedback: {evaluation['feedback']}"

        if is_success:
            if spinner and live: spinner.text = "[cyan]Reflecting on success and saving new experiences...[/cyan]"
            logger.info(f"[{task_id}] Reflecting on successful execution to generate new experiences.")
            new_experiences = await self.reflector.synthesize_experience(full_history)
            if new_experiences:
                await self.experience_manager.save_experiences(new_experiences)
                logger.info(f"[{task_id}] Saved {len(new_experiences)} new experiences to the knowledge base.")

        history_path = task_workspace / "full_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(full_history, f, indent=2, cls=DateTimeEncoder)

        return {"success": is_success, "final_summary": final_summary}
