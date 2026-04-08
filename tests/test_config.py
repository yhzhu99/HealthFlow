import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger
from pydantic import ValidationError

from healthflow.core.config import get_config
from healthflow.core.config import setup_logging
from healthflow.core.config import SystemConfig
from healthflow.execution.cli_adapters import CLISubprocessExecutor, ClaudeCodeExecutor, CodexExecutor, PiExecutor
from healthflow.execution.factory import create_executor_adapter


class ConfigTests(unittest.TestCase):
    def _get_config(
        self,
        config_path: Path,
        *,
        planner_llm: str = "test",
        evaluator_llm: str | None = None,
        reflector_llm: str | None = None,
        executor_llm: str | None = None,
        active_executor: str | None = None,
    ):
        return get_config(
            config_path,
            planner_llm=planner_llm,
            evaluator_llm=evaluator_llm or planner_llm,
            reflector_llm=reflector_llm or planner_llm,
            executor_llm=executor_llm or planner_llm,
            active_executor=active_executor,
        )

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
            config = self._get_config(config_path)
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

    def test_setup_logging_resolves_relative_log_file_under_workspace_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                f"""
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"

[runtime]
planner_llm = "test"
evaluator_llm = "test"
reflector_llm = "test"
executor_llm = "test"

[system]
workspace_dir = "{workspace_root / 'tasks'}"

[logging]
log_file = "healthflow.log"
""".strip(),
                encoding="utf-8",
            )
            config = self._get_config(config_path)

            setup_logging(config)
            logger.info("workspace scoped log line")
            logger.complete()
            logger.remove()

            self.assertTrue((workspace_root / "healthflow.log").exists())
            self.assertFalse((Path(tmpdir) / "healthflow.log").exists())

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
            config = self._get_config(config_path)
            executor = create_executor_adapter(config.active_executor_name, config.active_executor)
            self.assertIsInstance(executor, CLISubprocessExecutor)
            self.assertNotIsInstance(executor, ClaudeCodeExecutor)
            self.assertEqual(config.active_executor.args, ["run", "--variant", "$reasoning_effort", "--format", "json"])
            self.assertEqual(config.active_executor.reasoning_effort, "high")
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
            config = self._get_config(config_path)
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
            config = self._get_config(config_path)
            executor = create_executor_adapter(config.active_executor_name, config.active_executor)
            self.assertIsInstance(executor, CodexExecutor)
            self.assertEqual(config.active_executor.binary, "codex")
            self.assertEqual(config.active_executor.prompt_mode, "stdin")
            self.assertEqual(config.active_executor.model, "openai/gpt-5.4")
            self.assertEqual(config.active_executor.reasoning_effort, "high")
            self.assertIn('model_reasoning_effort="$reasoning_effort"', config.active_executor.arg_templates)
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
            config = self._get_config(config_path)
            executor = create_executor_adapter(config.active_executor_name, config.active_executor)
            self.assertIsInstance(executor, CLISubprocessExecutor)
            self.assertNotIsInstance(executor, ClaudeCodeExecutor)

    def test_executor_model_defaults_to_executor_llm_model_name(self):
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
            config = self._get_config(config_path, planner_llm="default")

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
            config = self._get_config(config_path, planner_llm="default")

            self.assertEqual(config.active_executor.model, "openai/gpt-5.4")
            self.assertFalse(config.active_executor.inherit_executor_llm)

    def test_runtime_section_can_target_specific_roles(self):
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

[runtime]
planner_llm = "default"
evaluator_llm = "judge"
reflector_llm = "default"
executor_llm = "default"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(config_path)
            self.assertEqual(config.llm_config_for_role("planner").model_name, "planner-model")
            self.assertEqual(config.llm_config_for_role("evaluator").model_name, "judge-model")

    def test_cli_runtime_overrides_take_precedence_over_runtime_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.default]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "planner-model"

[llm.alt]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "alt-model"

[runtime]
planner_llm = "default"
evaluator_llm = "default"
reflector_llm = "default"
executor_llm = "default"
""".strip(),
                encoding="utf-8",
            )
            config = get_config(
                config_path,
                planner_llm="alt",
                evaluator_llm="default",
                reflector_llm="default",
                executor_llm="default",
            )
            self.assertEqual(config.planner_llm_name, "alt")
            self.assertEqual(config.planner_llm.model_name, "alt-model")

    def test_missing_runtime_selection_raises_clear_error(self):
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
            with self.assertRaisesRegex(ValueError, r"Missing runtime LLM selections"):
                get_config(config_path)

    def test_legacy_llm_roles_section_raises_migration_error(self):
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
            with self.assertRaisesRegex(ValueError, r"\[llm_roles\]"):
                self._get_config(config_path, planner_llm="default")

    def test_executor_inherits_executor_model_name_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.deepseek]
api_key = "key"
base_url = "https://api.deepseek.com"
model_name = "deepseek-chat"
executor_model_name = "deepseek-chat"
executor_provider = "deepseek"
executor_provider_base_url = "https://api.deepseek.com/anthropic"
executor_provider_api = "anthropic-messages"
executor_provider_api_key_env = "DEEPSEEK_API_KEY"
""".strip(),
                encoding="utf-8",
            )
            config = self._get_config(config_path, planner_llm="deepseek")

            self.assertEqual(config.active_executor.model, "deepseek-chat")
            self.assertEqual(config.active_executor.provider, "deepseek")
            self.assertEqual(config.active_executor.provider_base_url, "https://api.deepseek.com/anthropic")
            self.assertEqual(config.active_executor.provider_api, "anthropic-messages")
            self.assertEqual(config.active_executor.provider_api_key_env, "DEEPSEEK_API_KEY")

    def test_claude_code_backend_inherits_executor_provider_settings_from_executor_llm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.deepseek]
api_key = "key"
base_url = "https://api.deepseek.com"
model_name = "deepseek-chat"
executor_model_name = "deepseek-chat"
executor_provider = "deepseek"
executor_provider_base_url = "https://api.deepseek.com/anthropic"
executor_provider_api = "anthropic-messages"
executor_provider_api_key_env = "DEEPSEEK_API_KEY"

[executor]
active_backend = "claude_code"
""".strip(),
                encoding="utf-8",
            )
            config = self._get_config(config_path, planner_llm="deepseek")

            self.assertEqual(config.active_executor.model, "deepseek-chat")
            self.assertEqual(config.active_executor.provider, "deepseek")
            self.assertEqual(config.active_executor.provider_base_url, "https://api.deepseek.com/anthropic")
            self.assertEqual(config.active_executor.provider_api, "anthropic-messages")
            self.assertEqual(config.active_executor.provider_api_key_env, "DEEPSEEK_API_KEY")

    def test_legacy_inherit_active_llm_key_raises_migration_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.test]
api_key = "key"
base_url = "https://example.com/v1"
model_name = "model"

[executor.backends.opencode]
binary = "opencode"
args = ["run"]
inherit_active_llm = false
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"inherit_active_llm"):
                self._get_config(config_path)

    def test_deepseek_executor_model_name_rejects_provider_prefixed_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm.deepseek]
api_key = "key"
base_url = "https://api.deepseek.com"
model_name = "deepseek-chat"
executor_provider = "deepseek"
executor_model_name = "deepseek/deepseek-chat"
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "bare model id"):
                self._get_config(config_path, planner_llm="deepseek")

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
            config = self._get_config(config_path)
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
                self._get_config(config_path)

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
                config = self._get_config(config_path, planner_llm="openai/gpt-5.2")

            self.assertEqual(config.planner_llm.api_key, "env-key")
            self.assertEqual(config.planner_llm.api_key_env, "ZENMUX_API_KEY")
            self.assertEqual(config.planner_llm.model_name, "openai/gpt-5.2")
            self.assertIsNone(config.planner_llm.reasoning_effort)

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
                config = self._get_config(config_path)

            self.assertEqual(config.planner_llm.api_key, "inline-key")

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
                with self.assertRaisesRegex(ValueError, "ZENMUX_API_KEY"):
                    self._get_config(config_path)

    def test_repo_config_example_parses_with_env_backed_models_and_runtime_roles(self):
        config_path = Path(__file__).resolve().parents[1] / "config.toml"
        with patch.dict(
            "os.environ",
            {
                "ZENMUX_API_KEY": "zenmux-key",
                "DEEPSEEK_API_KEY": "deepseek-key",
            },
            clear=False,
        ):
            config = get_config(config_path)

        self.assertEqual(config.planner_llm_name, "deepseek/deepseek-chat")
        self.assertEqual(config.evaluator_llm_name, "openai/gpt-5.4")
        self.assertEqual(config.reflector_llm_name, "google/gemini-3-flash-preview")
        self.assertEqual(config.executor_llm_name, "deepseek/deepseek-chat")
        self.assertIsNone(config.planner_llm.reasoning_effort)
        self.assertEqual(config.evaluator_llm.reasoning_effort, "high")
        self.assertEqual(config.reflector_llm.reasoning_effort, "high")
        self.assertIsNone(config.executor_llm.reasoning_effort)
        self.assertEqual(config.active_executor.provider, "deepseek")
        self.assertEqual(config.active_executor.model, "deepseek-chat")
        self.assertEqual(config.active_executor.reasoning_effort, "high")

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
                self._get_config(config_path)

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
                self._get_config(config_path)

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
                self._get_config(config_path)

    def test_system_config_rejects_zero_max_attempts(self):
        with self.assertRaises(ValidationError):
            SystemConfig(max_attempts=0)


if __name__ == "__main__":
    unittest.main()
