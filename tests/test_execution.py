import unittest
from pathlib import Path

from healthflow.core.config import BackendCLIConfig
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
            BackendCLIConfig(binary="opencode", args=["run", "--format", "json"]),
        )
        self.assertIsInstance(executor, OpenCodeExecutor)

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
