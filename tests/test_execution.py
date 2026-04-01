import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthflow.core.contracts import ExecutionPlan
from healthflow.core.config import BackendCLIConfig, EnvironmentConfig, default_executor_backends, get_config
from healthflow.execution.base import ExecutionContext
from healthflow.execution.cli_adapters import (
    CLISubprocessExecutor,
    ClaudeCodeExecutor,
    CodexExecutor,
    OpenCodeExecutor,
    PiExecutor,
)
from healthflow.execution.factory import create_executor_adapter
from healthflow.execution.opencode_parser import parse_opencode_json_events


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

    def test_codex_backend_uses_specialized_executor(self):
        executor = create_executor_adapter(
            "codex",
            BackendCLIConfig(binary="codex", args=["exec"], prompt_mode="stdin"),
        )
        self.assertIsInstance(executor, CodexExecutor)

    def test_pi_backend_uses_pi_executor(self):
        executor = create_executor_adapter(
            "pi",
            BackendCLIConfig(binary="pi"),
        )
        self.assertIsInstance(executor, PiExecutor)

    def test_opencode_backend_uses_specialized_executor(self):
        executor = create_executor_adapter(
            "opencode",
            BackendCLIConfig(binary="opencode", args=["run"]),
        )
        self.assertIsInstance(executor, OpenCodeExecutor)

    def test_opencode_command_uses_provider_prefixed_model_in_text_mode(self):
        executor = OpenCodeExecutor(
            "opencode",
            BackendCLIConfig(
                binary="opencode",
                args=["run"],
                provider="zenmux",
                model="openai/gpt-5.4",
                model_flag="-m",
                model_template="$provider/$model",
                output_mode="text",
            ),
        )

        command = executor._build_command("say hi")

        self.assertEqual(command, ["opencode", "run", "-m", "zenmux/openai/gpt-5.4", "say hi"])

    def test_opencode_uses_builtin_deepseek_provider_without_double_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[llm."deepseek/deepseek-v3.2"]
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
            config = get_config(
                config_path,
                planner_llm="deepseek/deepseek-v3.2",
                evaluator_llm="deepseek/deepseek-v3.2",
                reflector_llm="deepseek/deepseek-v3.2",
                executor_llm="deepseek/deepseek-v3.2",
                active_executor="opencode",
            )
            executor = create_executor_adapter(config.active_executor_name, config.active_executor)

        command = executor._build_command("say hi")

        self.assertEqual(
            command,
            ["opencode", "run", "--variant", "high", "--format", "json", "-m", "deepseek/deepseek-chat", "say hi"],
        )

    def test_appended_prompt_is_redacted_from_saved_command(self):
        executor = OpenCodeExecutor(
            "opencode",
            BackendCLIConfig(
                binary="opencode",
                args=["run", "--format", "json"],
                provider="zenmux",
                model="openai/gpt-5.4",
                model_flag="-m",
                model_template="$provider/$model",
                output_mode="json_events",
            ),
        )

        command = executor._build_command("say hi")
        redacted = executor._redacted_command(command, "say hi")

        self.assertEqual(
            redacted,
            ["opencode", "run", "--format", "json", "-m", "zenmux/openai/gpt-5.4", "<prompt omitted>"],
        )

    def test_codex_command_renders_provider_override_templates(self):
        executor = CodexExecutor(
            "codex",
            BackendCLIConfig(
                binary="codex",
                args=["exec", "--skip-git-repo-check"],
                arg_templates=[
                    "-c",
                    'model_provider="$provider"',
                    "-c",
                    'model_providers.$provider={name="ZenMux", base_url="$provider_base_url", env_key="$provider_api_key_env", wire_api="responses"}',
                ],
                provider="zenmux",
                provider_base_url="https://zenmux.ai/api/v1",
                provider_api_key_env="ZENMUX_API_KEY",
                model="openai/gpt-5.4",
                model_flag="-m",
                prompt_mode="stdin",
            ),
        )

        command = executor._build_command("ignored")

        self.assertEqual(
            command,
            [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "-c",
                'model_provider="zenmux"',
                "-c",
                'model_providers.zenmux={name="ZenMux", base_url="https://zenmux.ai/api/v1", env_key="ZENMUX_API_KEY", wire_api="responses"}',
                "-m",
                "openai/gpt-5.4",
            ],
        )

    def test_executor_environment_expands_required_variables(self):
        executor = ClaudeCodeExecutor(
            "claude_code",
            BackendCLIConfig(
                binary="claude",
                env={"ANTHROPIC_API_KEY": "${ZENMUX_API_KEY}"},
            ),
        )

        with patch.dict("os.environ", {"ZENMUX_API_KEY": "env-key"}, clear=False):
            environment = executor._build_environment(Path.cwd())

        self.assertEqual(environment["ANTHROPIC_API_KEY"], "env-key")

    def test_executor_environment_raises_when_required_variable_is_missing(self):
        executor = ClaudeCodeExecutor(
            "claude_code",
            BackendCLIConfig(
                binary="claude",
                env={"ANTHROPIC_API_KEY": "${ZENMUX_API_KEY}"},
            ),
        )

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ZENMUX_API_KEY"):
                executor._build_environment(Path.cwd())

    def test_claude_code_environment_uses_resolved_provider_settings(self):
        executor = ClaudeCodeExecutor(
            "claude_code",
            BackendCLIConfig(
                binary="claude",
                env={"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
                provider="deepseek",
                provider_base_url="https://api.deepseek.com/anthropic",
                provider_api="anthropic-messages",
                provider_api_key_env="DEEPSEEK_API_KEY",
                model="deepseek-chat",
                model_flag="--model",
            ),
        )

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "deepseek-key"}, clear=False):
            environment = executor._build_environment(Path.cwd())

        self.assertEqual(environment["ANTHROPIC_BASE_URL"], "https://api.deepseek.com/anthropic")
        self.assertEqual(environment["ANTHROPIC_API_KEY"], "deepseek-key")
        self.assertEqual(environment["ANTHROPIC_AUTH_TOKEN"], "deepseek-key")
        self.assertEqual(environment["ANTHROPIC_MODEL"], "deepseek-chat")
        self.assertEqual(environment["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "deepseek-chat")
        self.assertEqual(environment["ANTHROPIC_SMALL_FAST_MODEL"], "deepseek-chat")
        self.assertEqual(environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")

    def test_default_backend_environment_includes_project_venv_bin(self):
        for backend_name, backend_config in default_executor_backends().items():
            resolved_config = backend_config
            if backend_name == "pi" and backend_config.model is None:
                resolved_config = backend_config.model_copy(update={"model": "openai/gpt-5.4"})
            executor = create_executor_adapter(backend_name, resolved_config)
            with patch.dict(
                "os.environ",
                {"HOME": "/tmp/demo", "PATH": "/usr/bin:/bin", "ZENMUX_API_KEY": "demo-key"},
                clear=True,
            ):
                environment = executor._build_environment(Path.cwd())

            path_entries = environment["PATH"].split(":")
            self.assertEqual(path_entries[0], str(Path.cwd() / ".venv" / "bin"))
            self.assertTrue(shutil.which("sh", path=environment["PATH"]))

    def test_pi_executor_writes_runtime_models_json(self):
        executor = PiExecutor(
            "pi",
            BackendCLIConfig(
                binary="pi",
                args=["--print"],
                provider="zenmux",
                provider_flag="--provider",
                provider_base_url="https://zenmux.ai/api/v1",
                provider_api="openai-completions",
                provider_api_key_env="ZENMUX_API_KEY",
                model="openai/gpt-5.4",
                model_flag="--model",
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir)
            environment = executor._build_environment(working_dir)
            agent_dir = Path(environment["PI_CODING_AGENT_DIR"])
            models_json = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))

        self.assertEqual(agent_dir, working_dir / ".healthflow_pi_agent")
        self.assertEqual(models_json["providers"]["zenmux"]["baseUrl"], "https://zenmux.ai/api/v1")
        self.assertEqual(models_json["providers"]["zenmux"]["api"], "openai-completions")
        self.assertEqual(models_json["providers"]["zenmux"]["apiKey"], "ZENMUX_API_KEY")
        self.assertEqual(models_json["providers"]["zenmux"]["models"][0]["id"], "openai/gpt-5.4")

    def test_pi_executor_writes_anthropic_runtime_models_json_for_deepseek(self):
        executor = PiExecutor(
            "pi",
            BackendCLIConfig(
                binary="pi",
                args=["--print"],
                provider="deepseek",
                provider_flag="--provider",
                provider_base_url="https://api.deepseek.com/anthropic",
                provider_api="anthropic-messages",
                provider_api_key_env="DEEPSEEK_API_KEY",
                model="deepseek-chat",
                model_flag="--model",
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir)
            environment = executor._build_environment(working_dir)
            agent_dir = Path(environment["PI_CODING_AGENT_DIR"])
            models_json = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))

        self.assertEqual(models_json["providers"]["deepseek"]["baseUrl"], "https://api.deepseek.com/anthropic")
        self.assertEqual(models_json["providers"]["deepseek"]["api"], "anthropic-messages")
        self.assertEqual(models_json["providers"]["deepseek"]["apiKey"], "DEEPSEEK_API_KEY")
        self.assertEqual(models_json["providers"]["deepseek"]["models"][0]["id"], "deepseek-chat")

    def test_shared_executor_prompt_contains_neutral_workspace_rules(self):
        context = ExecutionContext(
            user_request="Build a cohort table.",
            plan=ExecutionPlan(
                objective="Build a cohort table.",
                assumptions_to_check=["Confirm what files are available in the workspace."],
                recommended_steps=["Inspect the workspace.", "Create the cohort artifact.", "Summarize the result."],
                recommended_workflows=["Use reproducible Python scripts.", "Persist a cohort artifact."],
                avoidances=["Do not write outside the workspace."],
                success_signals=["A cohort artifact exists in the workspace."],
                executor_brief="Prefer a small reproducible script if the logic is non-trivial.",
            ),
            execution_environment=EnvironmentConfig(),
            workflow_recommendations=["Prefer workspace-local Python entrypoints via `uv run`."],
        )
        prompt = context.render_prompt()
        self.assertIn("Do not rely on repository-level executor-specific instruction files", prompt)
        self.assertIn("Save every artifact inside the current workspace", prompt)
        self.assertIn("CodeAct-style executor", prompt)
        self.assertIn("Your public assistant identity is always HealthFlow.", prompt)
        self.assertIn("Do not present MetaAgent, the evaluator, the reflector, or the executor backend name", prompt)
        self.assertIn("answer as HealthFlow", prompt)
        self.assertIn("## Execution Environment", prompt)
        self.assertIn("HealthFlow does not manage MCP servers", prompt)
        self.assertIn("## EHR Safeguards", prompt)
        self.assertIn("## Workflow Recommendations", prompt)
        self.assertIn("## Workflow Memory", prompt)
        self.assertIn("Keep execution narration out of the final user-facing reply", prompt)

    def test_shared_executor_prompt_adds_report_guidance_without_requiring_executor_report_file(self):
        context = ExecutionContext(
            user_request="Build a cohort table.",
            plan=ExecutionPlan(
                objective="Build a cohort table.",
                assumptions_to_check=["Confirm what files are available in the workspace."],
                recommended_steps=["Inspect the workspace.", "Create the cohort artifact.", "Summarize the result."],
                recommended_workflows=["Prefer reproducible Python scripts."],
                avoidances=["Do not write outside the workspace."],
                success_signals=["A cohort artifact exists in the workspace."],
                executor_brief="Prefer a small reproducible script if the logic is non-trivial.",
            ),
            execution_environment=EnvironmentConfig(),
            workflow_recommendations=["Prefer workspace-local Python entrypoints via `uv run`."],
            report_requested=True,
        )

        prompt = context.render_prompt()

        self.assertIn("system-generated report", prompt)
        self.assertIn("key artifacts you produced", prompt)
        self.assertNotIn("final_report.md", prompt)

    def test_repo_root_does_not_depend_on_claude_md(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse((repo_root / "CLAUDE.md").exists())


class OpenCodeParserTests(unittest.TestCase):
    def test_parse_opencode_json_events_extracts_cost_tokens_and_tool_timing(self):
        stdout = "\n".join(
            [
                '{"type":"step_start","timestamp":1774979807302,"sessionID":"ses_demo","part":{"type":"step-start"}}',
                '{"type":"tool_use","timestamp":1774979807336,"sessionID":"ses_demo","part":{"type":"tool","tool":"read","state":{"status":"completed","input":{"filePath":"demo.txt"},"output":"hello","time":{"start":1774979807304,"end":1774979807326}}}}',
                '{"type":"step_finish","timestamp":1774979807344,"sessionID":"ses_demo","part":{"type":"step-finish","reason":"tool-calls","cost":0.03171,"tokens":{"total":10612,"input":10537,"output":11,"reasoning":0,"cache":{"read":64,"write":0}}}}',
                '{"type":"step_start","timestamp":1774979816658,"sessionID":"ses_demo","part":{"type":"step-start"}}',
                '{"type":"text","timestamp":1774979816662,"sessionID":"ses_demo","part":{"type":"text","text":"hello","time":{"start":1774979816660,"end":1774979816660}}}',
                '{"type":"step_finish","timestamp":1774979816666,"sessionID":"ses_demo","part":{"type":"step-finish","reason":"stop","cost":0.0003,"tokens":{"total":10658,"input":97,"output":1,"reasoning":0,"cache":{"read":10560,"write":0}}}}',
            ]
        )

        parsed = parse_opencode_json_events(stdout)

        self.assertIn("TOOL[read] status=completed", parsed.log)
        self.assertIn("STDOUT: hello", parsed.log)
        self.assertEqual(parsed.usage["estimated_cost_usd"], 0.03201)
        self.assertEqual(parsed.usage["input_tokens"], 10634)
        self.assertEqual(parsed.usage["output_tokens"], 12)
        self.assertEqual(parsed.usage["total_tokens"], 21270)
        self.assertEqual(parsed.usage["cache_read_tokens"], 10624)
        self.assertEqual(parsed.usage["tool_call_count"], 1)
        self.assertEqual(parsed.usage["step_count"], 2)
        self.assertAlmostEqual(parsed.usage["tool_time_seconds"], 0.022, places=4)
        self.assertEqual(parsed.telemetry["session_id"], "ses_demo")
        self.assertEqual(parsed.telemetry["tool_names"], ["read"])
        self.assertEqual(parsed.telemetry["step_reasons"], {"tool-calls": 1, "stop": 1})
        self.assertEqual(len(parsed.telemetry["steps"]), 2)

    def test_parse_opencode_json_events_falls_back_to_raw_stdout_when_unparsed_lines_exist(self):
        parsed = parse_opencode_json_events("not-json\nstill-not-json")

        self.assertEqual(parsed.telemetry["event_count"], 0)
        self.assertEqual(parsed.telemetry["parse_error_count"], 2)
        self.assertIn("STDOUT: not-json", parsed.log)
        self.assertEqual(parsed.usage["estimated_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
