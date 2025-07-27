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
from .agents.meta_agent import MetaAgent
from .agents.evaluator_agent import EvaluatorAgent
from .agents.reflector_agent import ReflectorAgent
from .execution.claude_executor import ClaudeCodeExecutor
from .experience.experience_manager import ExperienceManager
from .experience.experience_models import Experience

class HealthFlowSystem:
    """
    The main orchestrator for the unified Plan -> Delegate -> Evaluate -> Reflect -> Evolve cycle.
    This class manages the entire lifecycle of a task from user request to completion.
    """
    def __init__(self, config: HealthFlowConfig, experience_path: Path, shell: str):
        self.config = config
        self.llm_provider = create_llm_provider(config.llm)
        self.experience_manager = ExperienceManager(experience_path)

        # Initialize agents with the shared LLM provider
        self.meta_agent = MetaAgent(self.llm_provider)
        self.evaluator = EvaluatorAgent(self.llm_provider)
        self.reflector = ReflectorAgent(self.llm_provider)

        self.executor = ClaudeCodeExecutor(shell=shell)

        self.workspace_dir = Path(config.system.workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)

    async def run_task(self, user_request: str, live: Optional[Live] = None, spinner: Optional[Spinner] = None) -> dict:
        """
        Executes a single task through the full, unified HealthFlow lifecycle.
        This is the main entry point for processing any user request.
        """
        task_id = str(uuid.uuid4())
        task_workspace = self.workspace_dir / task_id
        task_workspace.mkdir(exist_ok=True)
        logger.info(f"[{task_id}] New task started. Request: '{user_request}'")
        logger.info(f"[{task_id}] Workspace created at: {task_workspace}")

        start_time = time.time()
        result = await self._run_unified_flow(task_id, task_workspace, user_request, live, spinner)
        execution_time = time.time() - start_time

        result["execution_time"] = execution_time
        result["workspace_path"] = str(task_workspace)
        return result

    async def _run_unified_flow(self, task_id: str, task_workspace: Path, user_request: str, live: Optional[Live], spinner: Optional[Spinner]) -> Dict[str, Any]:
        """Handles the iterative workflow for any task, from planning to reflection."""
        if spinner and live: spinner.text = "Retrieving relevant experiences..."
        retrieved_experiences = await self.experience_manager.retrieve_experiences(user_request, k=5)
        logger.info(f"[{task_id}] Retrieved {len(retrieved_experiences)} relevant experiences.")

        full_history = {"task_id": task_id, "user_request": user_request, "retrieved_experiences": [exp.model_dump() for exp in retrieved_experiences], "attempts": []}
        is_success = False
        final_summary = "Task failed to complete within the allowed attempts."
        final_answer = "No answer generated."

        for attempt in range(self.config.system.max_retries + 1):
            attempt_num = attempt + 1
            logger.info(f"[{task_id}] Starting attempt {attempt_num}/{self.config.system.max_retries + 1}")

            attempt_history = {"attempt": attempt_num}
            previous_feedback = full_history["attempts"][-1]["evaluation"]["feedback"] if attempt > 0 else None

            # 1. Plan
            if spinner and live: spinner.text = f"Attempt {attempt_num}: Generating plan..."
            task_list_md = await self.meta_agent.generate_plan(user_request, retrieved_experiences, previous_feedback)
            task_list_path = task_workspace / f"task_list_v{attempt_num}.md"
            with open(task_list_path, "w", encoding="utf-8") as f: f.write(task_list_md)
            attempt_history["task_list"] = task_list_md
            logger.info(f"[{task_id}] Plan generated for attempt {attempt_num}.")

            # 2. Delegate & Execute
            if spinner and live: spinner.text = f"Attempt {attempt_num}: Delegating to executor..."
            execution_result = await self.executor.execute(task_list_path, task_workspace)
            attempt_history["execution"] = execution_result
            logger.info(f"[{task_id}] Execution completed for attempt {attempt_num}. Success: {execution_result['success']}")

            # 3. Evaluate
            if spinner and live: spinner.text = f"Attempt {attempt_num}: Evaluating outcome..."
            evaluation = await self.evaluator.evaluate(user_request, task_list_md, execution_result["log"])
            attempt_history["evaluation"] = evaluation
            logger.info(f"[{task_id}] Evaluation completed. Score: {evaluation['score']}/10.")

            full_history["attempts"].append(attempt_history)

            if evaluation["score"] >= self.config.evaluation.success_threshold:
                logger.info(f"[{task_id}] Task succeeded on attempt {attempt_num}.")
                is_success = True
                final_answer = self._extract_answer_from_execution(execution_result["log"], user_request)
                final_summary = f"Task completed successfully. Evaluation: {evaluation.get('reasoning', 'N/A')}"
                break
            else:
                logger.warning(f"[{task_id}] Attempt {attempt_num} failed with score {evaluation['score']}. Retrying if possible.")
                final_summary = f"Task failed after {attempt_num} attempts. Final feedback: {evaluation['feedback']}"

        # 4. Reflect & Evolve
        if is_success:
            if spinner and live: spinner.text = "Reflecting on success and saving new experiences..."
            logger.info(f"[{task_id}] Reflecting on successful execution to generate new experiences.")
            new_experiences = await self.reflector.synthesize_experience(full_history)
            if new_experiences:
                await self.experience_manager.save_experiences(new_experiences)
                logger.info(f"[{task_id}] Saved {len(new_experiences)} new experiences to the knowledge base.")
            full_history["new_experiences"] = [exp.model_dump() for exp in new_experiences]


        history_path = task_workspace / "full_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(full_history, f, indent=2, cls=DateTimeEncoder)

        return {"success": is_success, "final_summary": final_summary, "answer": final_answer}

    def _extract_answer_from_execution(self, execution_log: str, user_request: str) -> str:
        """Extract the final answer from the execution log."""
        lines = execution_log.split('\n')

        # Priority 1: Simple echo output for simple questions
        # This handles cases like `echo "I am HealthFlow"`
        if len(lines) < 5: # Likely a very simple command
            for line in lines:
                if line.startswith("STDOUT: "):
                    content = line.replace("STDOUT: ", "").strip()
                    if content: return content # Return the first piece of stdout

        # Priority 2: Look for explicit answer indicators from the end of the log
        answer_indicators = [
            "final answer:", "answer:", "result:", "conclusion:",
            "output:", "solution:", "the answer is", "priority:"
        ]
        for i in range(len(lines) - 1, -1, -1):
            line_content = lines[i].replace("STDOUT: ", "").replace("STDERR: ", "").strip()
            line_lower = line_content.lower()
            if any(indicator in line_lower for indicator in answer_indicators):
                return line_content

        # Priority 3: Return the last meaningful line of stdout
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if line.startswith("STDOUT: "):
                content = line.replace("STDOUT: ", "").strip()
                if content and len(content) > 10:
                    return content

        # Fallback if no clear answer is found
        return f"Execution completed. Please review the log in the workspace for details. The original request was: '{user_request}'"