from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttemptPaths:
    task_root: Path
    sandbox_dir: Path
    runtime_dir: Path
    attempt_number: int
    attempt_dir: Path
    summary_path: Path
    memory_dir: Path
    retrieval_context_path: Path
    retrieval_result_path: Path
    planner_dir: Path
    planner_input_messages_path: Path
    planner_output_raw_path: Path
    planner_output_parsed_path: Path
    planner_call_path: Path
    planner_repair_trace_path: Path
    planner_plan_markdown_path: Path
    executor_dir: Path
    executor_prompt_path: Path
    executor_command_path: Path
    executor_stdout_path: Path
    executor_stderr_path: Path
    executor_combined_log_path: Path
    executor_telemetry_path: Path
    executor_usage_path: Path
    executor_artifacts_index_path: Path
    evaluator_dir: Path
    evaluator_input_messages_path: Path
    evaluator_output_raw_path: Path
    evaluator_output_parsed_path: Path
    evaluator_call_path: Path
    evaluator_repair_trace_path: Path

    @classmethod
    def build(cls, task_root: Path, sandbox_dir: Path, runtime_dir: Path, attempt_number: int) -> "AttemptPaths":
        attempt_dir = runtime_dir / "attempts" / f"attempt_{attempt_number:03d}"
        memory_dir = attempt_dir / "memory"
        planner_dir = attempt_dir / "planner"
        executor_dir = attempt_dir / "executor"
        evaluator_dir = attempt_dir / "evaluator"
        return cls(
            task_root=task_root,
            sandbox_dir=sandbox_dir,
            runtime_dir=runtime_dir,
            attempt_number=attempt_number,
            attempt_dir=attempt_dir,
            summary_path=attempt_dir / "summary.json",
            memory_dir=memory_dir,
            retrieval_context_path=memory_dir / "retrieval_context.json",
            retrieval_result_path=memory_dir / "retrieval_result.json",
            planner_dir=planner_dir,
            planner_input_messages_path=planner_dir / "input_messages.json",
            planner_output_raw_path=planner_dir / "output_raw.txt",
            planner_output_parsed_path=planner_dir / "output_parsed.json",
            planner_call_path=planner_dir / "call.json",
            planner_repair_trace_path=planner_dir / "repair_trace.json",
            planner_plan_markdown_path=planner_dir / "plan.md",
            executor_dir=executor_dir,
            executor_prompt_path=executor_dir / "prompt.md",
            executor_command_path=executor_dir / "command.json",
            executor_stdout_path=executor_dir / "stdout.txt",
            executor_stderr_path=executor_dir / "stderr.txt",
            executor_combined_log_path=executor_dir / "combined.log",
            executor_telemetry_path=executor_dir / "telemetry.json",
            executor_usage_path=executor_dir / "usage.json",
            executor_artifacts_index_path=executor_dir / "artifacts_index.json",
            evaluator_dir=evaluator_dir,
            evaluator_input_messages_path=evaluator_dir / "input_messages.json",
            evaluator_output_raw_path=evaluator_dir / "output_raw.txt",
            evaluator_output_parsed_path=evaluator_dir / "output_parsed.json",
            evaluator_call_path=evaluator_dir / "call.json",
            evaluator_repair_trace_path=evaluator_dir / "repair_trace.json",
        )

    def ensure_dirs(self) -> None:
        for path in [
            self.attempt_dir,
            self.memory_dir,
            self.planner_dir,
            self.executor_dir,
            self.evaluator_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def relative_path(self, path: Path | str | None) -> str | None:
        if path is None:
            return None
        candidate = Path(path)
        try:
            return str(candidate.relative_to(self.task_root))
        except ValueError:
            return str(candidate)


@dataclass(frozen=True)
class TaskRuntimePaths:
    task_root: Path
    sandbox_dir: Path
    runtime_dir: Path
    attempts_dir: Path
    run_dir: Path
    reflection_dir: Path
    index_path: Path
    events_path: Path
    report_path: Path
    run_summary_path: Path
    run_trajectory_path: Path
    run_costs_path: Path
    final_evaluation_path: Path
    task_state_path: Path
    direct_response_path: Path
    reflection_input_path: Path
    reflection_output_raw_path: Path
    reflection_output_parsed_path: Path
    reflection_call_path: Path
    reflection_repair_trace_path: Path

    @classmethod
    def build(cls, task_root: Path) -> "TaskRuntimePaths":
        sandbox_dir = task_root / "sandbox"
        runtime_dir = task_root / "runtime"
        attempts_dir = runtime_dir / "attempts"
        run_dir = runtime_dir / "run"
        reflection_dir = runtime_dir / "reflection"
        return cls(
            task_root=task_root,
            sandbox_dir=sandbox_dir,
            runtime_dir=runtime_dir,
            attempts_dir=attempts_dir,
            run_dir=run_dir,
            reflection_dir=reflection_dir,
            index_path=runtime_dir / "index.json",
            events_path=runtime_dir / "events.jsonl",
            report_path=runtime_dir / "report.md",
            run_summary_path=run_dir / "summary.json",
            run_trajectory_path=run_dir / "trajectory.json",
            run_costs_path=run_dir / "costs.json",
            final_evaluation_path=run_dir / "final_evaluation.json",
            task_state_path=run_dir / "task_state.json",
            direct_response_path=run_dir / "direct_response.json",
            reflection_input_path=reflection_dir / "input.json",
            reflection_output_raw_path=reflection_dir / "output_raw.txt",
            reflection_output_parsed_path=reflection_dir / "output_parsed.json",
            reflection_call_path=reflection_dir / "call.json",
            reflection_repair_trace_path=reflection_dir / "repair_trace.json",
        )

    def ensure_base_dirs(self) -> None:
        for path in [
            self.task_root,
            self.sandbox_dir,
            self.runtime_dir,
            self.attempts_dir,
            self.run_dir,
            self.reflection_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def attempt(self, attempt_number: int) -> AttemptPaths:
        return AttemptPaths.build(
            task_root=self.task_root,
            sandbox_dir=self.sandbox_dir,
            runtime_dir=self.runtime_dir,
            attempt_number=attempt_number,
        )

    def relative_path(self, path: Path | str | None) -> str | None:
        if path is None:
            return None
        candidate = Path(path)
        try:
            return str(candidate.relative_to(self.task_root))
        except ValueError:
            return str(candidate)
