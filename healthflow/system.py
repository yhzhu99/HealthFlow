import asyncio
import time
import uuid
from pathlib import Path
from loguru import logger
import json

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
    """
    def __init__(self, config: HealthFlowConfig):
        self.config = config
        # The LLM provider is now initialized with the specific 'llm' sub-config
        self.llm_provider = create_llm_provider(config.llm)
        self.experience_manager = ExperienceManager(config.system.workspace_dir)

        self.decomposer = TaskDecomposerAgent(self.llm_provider)
        self.evaluator = EvaluatorAgent(self.llm_provider)
        self.reflector = ReflectorAgent(self.llm_provider)
        self.executor = ClaudeCodeExecutor()

        self.workspace_dir = Path(config.system.workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)

    async def run_task(self, user_request: str) -> dict:
        """
        Executes a single task through the full HealthFlow lifecycle.
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

        final_result = {}
        is_success = False

        for attempt in range(self.config.system.max_retries + 1):
            attempt_num = attempt + 1
            logger.info(f"[{task_id}] Starting attempt {attempt_num}/{self.config.system.max_retries + 1}")

            attempt_history = {"attempt": attempt_num, "feedback": None}

            # 1. PLAN: Decompose task, informed by experience
            retrieved_experiences = await self.experience_manager.retrieve_experiences(user_request, k=5)
            logger.info(f"[{task_id}] Retrieved {len(retrieved_experiences)} relevant experiences.")

            previous_feedback = full_history["attempts"][-1]["evaluation"]["feedback"] if attempt > 0 else None

            task_list_md = await self.decomposer.generate_task_list(
                user_request, retrieved_experiences, previous_feedback
            )
            task_list_path = task_workspace / f"task_list_v{attempt_num}.md"
            with open(task_list_path, "w", encoding="utf-8") as f:
                f.write(task_list_md)
            attempt_history["task_list"] = task_list_md
            logger.info(f"[{task_id}] Generated task list for attempt {attempt_num}.")

            # 2. DELEGATE: Execute the task list with Claude Code
            execution_result = await self.executor.execute(task_list_path, task_workspace)
            attempt_history["execution"] = execution_result
            logger.info(f"[{task_id}] Execution completed for attempt {attempt_num}. Success: {execution_result['success']}")

            # 3. EVALUATE: Assess the performance
            evaluation = await self.evaluator.evaluate(
                user_request=user_request,
                task_list=task_list_md,
                execution_log=execution_result["log"]
            )
            attempt_history["evaluation"] = evaluation
            logger.info(f"[{task_id}] Evaluation completed for attempt {attempt_num}. Score: {evaluation['score']}/10")

            full_history["attempts"].append(attempt_history)

            # Check for success
            if evaluation["score"] >= self.config.evaluation.success_threshold:
                logger.info(f"[{task_id}] Task succeeded on attempt {attempt_num}.")
                is_success = True
                final_result = {
                    "final_summary": f"Task completed successfully. The final output is in the workspace. Evaluation reasoning: {evaluation.get('reasoning', 'N/A')}",
                }
                break
            else:
                logger.warning(f"[{task_id}] Attempt {attempt_num} failed with score {evaluation['score']}. Retrying if possible.")
                final_result = {
                     "final_summary": f"Task failed after {attempt_num} attempts. Final evaluation feedback: {evaluation['feedback']}",
                }

        # 4. REFLECT: Synthesize experience if successful
        if is_success:
            logger.info(f"[{task_id}] Reflecting on successful execution to generate new experiences.")
            new_experiences = await self.reflector.synthesize_experience(full_history)
            if new_experiences:
                await self.experience_manager.save_experiences(new_experiences)
                logger.info(f"[{task_id}] Saved {len(new_experiences)} new experiences to the knowledge base.")

        execution_time = time.time() - start_time

        # Save the full history for auditing
        history_path = task_workspace / "full_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(full_history, f, indent=2)

        return {
            "success": is_success,
            "workspace_path": str(task_workspace),
            "execution_time": execution_time,
            **final_result
        }