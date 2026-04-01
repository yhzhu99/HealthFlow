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
            self.assertEqual(config.environment.python_version, "3.12")
            self.assertEqual(config.environment.package_manager, "uv")
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
            self.assertEqual(config.active_executor.args, ["run", "--variant", "high", "--format", "json"])
            self.assertEqual(config.active_executor.prompt_mode, "append")
            self.assertEqual(config.active_executor.output_mode, "json_events")
            self.assertEqual(config.active_executor.model, "model")

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
            self.assertEqual(config.active_executor.model, "openai/gpt-5.4")
            self.assertIn('model_reasoning_effort="high"', config.active_executor.arg_templates)
            self.assertIn('model_reasoning_summary="detailed"', config.active_executor.arg_templates)

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

    def test_executor_model_defaults_to_active_llm_model_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.default]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "openai/gpt-5.4"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "default")

            self.assertEqual(config.active_executor.model, "openai/gpt-5.4")

    def test_codex_default_model_is_pinned_to_gpt_5_4(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.default]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "deepseek/deepseek-v3.2"

[executor]
active_backend = "codex"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "default")

            self.assertEqual(config.active_executor.model, "openai/gpt-5.4")
            self.assertFalse(config.active_executor.inherit_active_llm)

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

    def test_unknown_llm_role_target_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.default]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "planner-model"

[llm_roles]
reflector = "missing-model"
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"llm_roles\.reflector='missing-model'"):
                get_config(config_path, "default")

    def test_executor_inherits_executor_model_name_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.deepseek]
api_key = "key"
base_url = "https://api.deepseek.com"
model_name = "deepseek-chat"
executor_model_name = "deepseek/deepseek-chat"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "deepseek")

            self.assertEqual(config.active_executor.model, "deepseek/deepseek-chat")

    def test_environment_defaults_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"

[environment]
python_version = "3.12.2"
package_manager = "uv"
install_command = "uv add --dev"
run_prefix = "uv run"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path, "test")
            self.assertEqual(config.environment.python_version, "3.12.2")
            self.assertEqual(config.environment.install_command, "uv add --dev")

    def test_legacy_tools_config_raises_migration_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            legacy_config = "\n".join(
                [
                    "[llm.test]",
                    'api_key = "key"',
                    'base_url = "https://example.com/v1"',
                    'model_name = "model"',
                    "",
                    "[too" "ls.legacy_bridge]",
                    'surface = "' + "m" + "cp" + '"',
                    'description = "Legacy bridge"',
                    'invocation_hint = "connector-defined"',
                ]
            )
            config_path.write_text(legacy_config, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"Legacy tools configuration"):
                get_config(config_path, "test")

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

    def test_repo_config_example_parses_with_env_backed_models_and_llm_roles(self):
        config_path = Path(__file__).resolve().parents[1] / "config.toml"
        with patch.dict(
            "os.environ",
            {
                "ZENMUX_API_KEY": "zenmux-key",
                "DEEPSEEK_API_KEY": "deepseek-key",
            },
            clear=False,
        ):
            config = get_config(config_path, "deepseek/deepseek-v3.2")

        self.assertEqual(config.active_llm_name, "deepseek/deepseek-v3.2")
        self.assertEqual(config.llm_config_for_role("planner").model_name, "deepseek-chat")
        self.assertEqual(config.llm_config_for_role("evaluator").model_name, "openai/gpt-5.4")
        self.assertEqual(config.llm_config_for_role("reflector").model_name, "google/gemini-3-flash-preview")

    def test_system_shell_raises_migration_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"

[system]
shell = "/usr/bin/zsh"
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"system\.shell"):
                get_config(config_path, "test")

    def test_unknown_memory_keys_raise_clear_error(self):
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

    def test_unknown_policy_sections_raise_clear_error(self):
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
