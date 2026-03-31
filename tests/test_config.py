import tempfile
import unittest
from pathlib import Path

from healthflow.core.config import get_config
from healthflow.execution.cli_adapters import CLISubprocessExecutor, HealthFlowAgentExecutor, PiExecutor
from healthflow.execution.factory import create_executor_adapter


class ConfigTests(unittest.TestCase):
    def test_executor_defaults_are_applied_when_backends_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "test")
            self.assertEqual(config.active_executor_name, "healthflow_agent")
            self.assertIn("healthflow_agent", config.executor.backends)
            self.assertIn("opencode", config.executor.backends)
            self.assertIn("claude_code", config.executor.backends)
            self.assertIn("pi", config.executor.backends)
            self.assertEqual(config.system.workspace_dir, "workspace/tasks")

    def test_named_pi_backend_uses_specialized_executor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"

[executor]
active_backend = "pi"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "test")
            executor = create_executor_adapter(config.active_executor_name, config.active_executor)
            self.assertIsInstance(executor, PiExecutor)

    def test_custom_configured_backend_uses_generic_executor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"

[executor]
active_backend = "custom_runner"

[executor.backends.custom_runner]
binary = "custom-agent"
args = ["--print"]
prompt_mode = "append"
timeout_seconds = 30
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "test")
            executor = create_executor_adapter(config.active_executor_name, config.active_executor)
            self.assertIsInstance(executor, CLISubprocessExecutor)
            self.assertNotIsInstance(executor, HealthFlowAgentExecutor)

    def test_llm_roles_can_override_default_reasoning_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.default]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "planner-model"

[llm.judge]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "judge-model"

[llm_roles]
evaluator = "judge"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "default")
            self.assertEqual(config.llm_config_for_role("planner").model_name, "planner-model")
            self.assertEqual(config.llm_config_for_role("evaluator").model_name, "judge-model")


if __name__ == "__main__":
    unittest.main()
