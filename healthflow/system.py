import asyncio
import time
import uuid
from pathlib import Path
from loguru import logger
import json
from typing import Optional

from rich.live import Live
from rich.spinner import Spinner

from .core.config import HealthFlowConfig
from .core.llm_provider import create_llm_provider
from .agents.task_decomposer_agent import TaskDecomposerAgent
from .agents.evaluator_agent import EvaluatorAgent
from .agents.reflector_agent import ReflectorAgent
from .execution.claude_executor import ClaudeCodeExecutor
from .experience.experience_manager import ExperienceManager

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

        # The execution engine for delegating tasks to Claude Code
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

        full_history = {
            "task_id": task_id,
            "user_request": user_request,
            "attempts": []
        }

        is_success = False
        final_summary = "Task failed to complete within the allowed attempts."

        for attempt in range(self.config.system.max_retries + 1):
            attempt_num = attempt + 1
            if spinner and live:
                spinner.text = f"[cyan]Starting attempt {attempt_num}/{self.config.system.max_retries + 1}...[/cyan]"
                live.update(spinner)

            logger.info(f"[{task_id}] Starting attempt {attempt_num}/{self.config.system.max_retries + 1}")

            attempt_history = {"attempt": attempt_num, "feedback": None}
            previous_feedback = full_history["attempts"][-1]["evaluation"]["feedback"] if attempt > 0 else None

            # 1. PLAN: Decompose task, informed by experience
            if spinner and live: spinner.text = "[cyan]Retrieving experiences and creating a plan...[/cyan]"
            retrieved_experiences = await self.experience_manager.retrieve_experiences(user_request, k=5)
            logger.info(f"[{task_id}] Retrieved {len(retrieved_experiences)} relevant experiences.")

            task_list_md = await self.decomposer.generate_task_list(user_request, retrieved_experiences, previous_feedback)
            task_list_path = task_workspace / f"task_list_v{attempt_num}.md"
            with open(task_list_path, "w", encoding="utf-8") as f:
                f.write(task_list_md)
            attempt_history["task_list"] = task_list_md
            logger.info(f"[{task_id}] Generated task list for attempt {attempt_num}.")

            # 2. DELEGATE: Execute the task list with Claude Code
            if spinner and live: spinner.text = "[cyan]Delegating to Claude Code for execution...[/cyan]"
            execution_result = await self.executor.execute(task_list_path, task_workspace)
            attempt_history["execution"] = execution_result
            logger.info(f"[{task_id}] Execution completed for attempt {attempt_num}. Success: {execution_result['success']}")

            # 3. EVALUATE: Assess the performance
            if spinner and live: spinner.text = "[cyan]Evaluating the execution outcome...[/cyan]"
            evaluation = await self.evaluator.evaluate(user_request, task_list_md, execution_result["log"])
            attempt_history["evaluation"] = evaluation
            logger.info(f"[{task_id}] Evaluation completed. Score: {evaluation['score']}/10. Reasoning: {evaluation['reasoning']}")

            full_history["attempts"].append(attempt_history)

            # Check for success
            if evaluation["score"] >= self.config.evaluation.success_threshold:
                logger.info(f"[{task_id}] Task succeeded on attempt {attempt_num}.")
                is_success = True
                final_summary = f"Task completed successfully. Evaluation: {evaluation.get('reasoning', 'N/A')}"
                break
            else:
                logger.warning(f"[{task_id}] Attempt {attempt_num} failed with score {evaluation['score']}. Retrying if possible.")
                final_summary = f"Task failed after {attempt_num} attempts. Final feedback: {evaluation['feedback']}"

        # 4. REFLECT: Synthesize experience if successful
        if is_success:
            if spinner and live: spinner.text = "[cyan]Reflecting on success and saving new experiences...[/cyan]"
            logger.info(f"[{task_id}] Reflecting on successful execution to generate new experiences.")
            new_experiences = await self.reflector.synthesize_experience(full_history)
            if new_experiences:
                await self.experience_manager.save_experiences(new_experiences)
                logger.info(f"[{task_id}] Saved {len(new_experiences)} new experiences to the knowledge base.")

        execution_time = time.time() - start_time

        # Save the full history for auditing and analysis
        history_path = task_workspace / "full_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            # Use a custom encoder to handle Pydantic models if they exist in history
            json.dump(full_history, f, indent=2)

        return {
            "success": is_success,
            "workspace_path": str(task_workspace),
            "execution_time": execution_time,
            "final_summary": final_summary,
        }