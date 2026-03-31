import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from healthflow.core.config import get_config
from healthflow.core.config import SystemConfig
from healthflow.execution.cli_adapters import CLISubprocessExecutor, ClaudeCodeExecutor, CodexExecutor, PiExecutor
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
            self.assertEqual(config.active_executor_name, "opencode")
            self.assertIn("opencode", config.executor.backends)
            self.assertIn("claude_code", config.executor.backends)
            self.assertIn("codex", config.executor.backends)
            self.assertIn("pi", config.executor.backends)
            self.assertEqual(config.system.max_attempts, 3)
            self.assertEqual(config.system.workspace_dir, "workspace/tasks")
            self.assertEqual(config.memory.write_policy, "append")

    def test_default_backend_uses_opencode_executor(self):
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
            executor = create_executor_adapter(config.active_executor_name, config.active_executor)
            self.assertIsInstance(executor, CLISubprocessExecutor)
            self.assertNotIsInstance(executor, ClaudeCodeExecutor)

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

    def test_named_codex_backend_uses_specialized_executor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"

[executor]
active_backend = "codex"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "test")
            executor = create_executor_adapter(config.active_executor_name, config.active_executor)
            self.assertIsInstance(executor, CodexExecutor)
            self.assertEqual(config.active_executor.binary, "codex")
            self.assertEqual(config.active_executor.prompt_mode, "stdin")

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
            self.assertNotIsInstance(executor, ClaudeCodeExecutor)

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

    def test_llm_api_key_can_be_loaded_from_env_variable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm."openai/gpt-5.2"]
api_key_env = "ZENMUX_API_KEY"
base_url = "https://zenmux.ai/api/v1"
model_name = "openai/gpt-5.2"
""".strip(),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"ZENMUX_API_KEY": "env-key"}, clear=False):
                config = get_config(config_path, "openai/gpt-5.2")

            self.assertEqual(config.llm.api_key, "env-key")
            self.assertEqual(config.llm.api_key_env, "ZENMUX_API_KEY")
            self.assertEqual(config.llm.model_name, "openai/gpt-5.2")

    def test_inline_api_key_takes_precedence_over_api_key_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "inline-key"
api_key_env = "ZENMUX_API_KEY"
base_url = "https://example.com/v1"
model_name = "model"
""".strip(),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"ZENMUX_API_KEY": "env-key"}, clear=False):
                config = get_config(config_path, "test")

            self.assertEqual(config.llm.api_key, "inline-key")

    def test_missing_env_backed_api_key_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key_env = "ZENMUX_API_KEY"
base_url = "https://example.com/v1"
model_name = "model"
""".strip(),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(
                    ValueError,
                    "ZENMUX_API_KEY",
                ):
                    get_config(config_path, "test")

    def test_legacy_memory_mode_maps_to_write_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"

[memory]
mode = "frozen_train"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "test")
            self.assertEqual(config.memory.write_policy, "freeze")

    def test_removed_memory_policy_keys_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"

[memory]
write_policy = "append"
strategy_k = 3
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "strategy_k"):
                get_config(config_path, "test")

    def test_removed_policy_sections_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"

[verification]
require_verifier_pass = true
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"\[verification\]"):
                get_config(config_path, "test")

    def test_system_config_rejects_zero_max_attempts(self):
        with self.assertRaises(ValidationError):
            SystemConfig(max_attempts=0)


if __name__ == "__main__":
    unittest.main()
