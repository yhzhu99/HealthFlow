import unittest
from pathlib import Path

from healthflow.core.config import BackendCLIConfig
from healthflow.execution.base import ExecutionContext
from healthflow.execution.cli_adapters import CLISubprocessExecutor, ClaudeCodeExecutor, PiExecutor
from healthflow.execution.factory import create_executor_adapter


class ExecutionFactoryTests(unittest.TestCase):
    def test_claude_code_uses_specialized_executor(self):
        executor = create_executor_adapter(
            "claude_code",
            BackendCLIConfig(binary="claude", args=["--print"]),
        )
        self.assertIsInstance(executor, ClaudeCodeExecutor)

    def test_unknown_backend_falls_back_to_generic_cli_executor(self):
        executor = create_executor_adapter(
            "external_backend",
            BackendCLIConfig(binary="external-agent", args=["--print"]),
        )
        self.assertIsInstance(executor, CLISubprocessExecutor)
        self.assertNotIsInstance(executor, ClaudeCodeExecutor)

    def test_pi_backend_uses_pi_executor(self):
        executor = create_executor_adapter(
            "pi",
            BackendCLIConfig(binary="pi"),
        )
        self.assertIsInstance(executor, PiExecutor)

    def test_shared_executor_prompt_contains_neutral_workspace_rules(self):
        context = ExecutionContext(
            user_request="Build a cohort table.",
            task_family="cohort_extraction",
            data_profile="No structured data profile provided.",
        )
        prompt = context.render_prompt()
        self.assertIn("Do not rely on repository-level executor-specific instruction files", prompt)
        self.assertIn("Save every artifact inside the current workspace", prompt)
        self.assertIn("Prefer Python and reproducible CLI workflows", prompt)
        self.assertIn("## Deliverable Guidance", prompt)
        self.assertIn("not fixed file-level contracts", prompt)

    def test_repo_root_does_not_depend_on_claude_md(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse((repo_root / "CLAUDE.md").exists())


if __name__ == "__main__":
    unittest.main()
