import asyncio
import io
import unittest
from contextlib import suppress

from prompt_toolkit.document import Document
from rich.console import Console

from healthflow.interactive_cli import InteractiveShell


class _FakePromptSession:
    def __init__(self):
        self.calls = []

    async def prompt_async(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise EOFError


class _UnsupportedEscapeMonitor:
    def supported(self) -> bool:
        return False

    async def watch(self) -> None:
        return None


class InteractiveShellTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.console = Console(file=self.output, force_terminal=False, color_system=None)
        self.system_counter = 0
        self.prompt_counter = 0

        def _system_factory():
            self.system_counter += 1
            return {"session": self.system_counter}

        def _prompt_factory():
            self.prompt_counter += 1
            return _FakePromptSession()

        async def _task_runner(system, task):
            return {"success": task != "fail", "task": task, "system": system}

        self.shell = InteractiveShell(
            console=self.console,
            system_factory=_system_factory,
            task_runner=_task_runner,
            prompt_session_factory=_prompt_factory,
            escape_monitor_factory=lambda on_escape: _UnsupportedEscapeMonitor(),
        )

    async def test_new_command_recreates_system_and_prompt_session(self):
        self.assertEqual(self.system_counter, 1)
        self.assertEqual(self.prompt_counter, 1)
        self.assertEqual(self.shell.session_index, 1)

        should_exit = self.shell._handle_command("/new")

        self.assertFalse(should_exit)
        self.assertEqual(self.system_counter, 2)
        self.assertEqual(self.prompt_counter, 2)
        self.assertEqual(self.shell.session_index, 2)

    async def test_clear_command_redraws_without_recreating_system(self):
        should_exit = self.shell._handle_command("/clear")

        self.assertFalse(should_exit)
        self.assertEqual(self.system_counter, 1)
        self.assertEqual(self.prompt_counter, 1)

    async def test_unknown_command_reports_help_hint(self):
        should_exit = self.shell._handle_command("/missing")

        self.assertFalse(should_exit)
        self.assertIn("Try /help.", self.output.getvalue())

    async def test_exit_alias_returns_true(self):
        should_exit = await self.shell._handle_input("quit")

        self.assertTrue(should_exit)

    async def test_failed_task_does_not_exit_shell(self):
        should_exit = await self.shell._handle_input("fail")

        self.assertFalse(should_exit)

    def test_command_completion_prefix_requires_strict_line_start_slash(self):
        self.assertEqual(InteractiveShell.command_completion_prefix("/"), "/")
        self.assertEqual(InteractiveShell.command_completion_prefix("/he"), "/he")
        self.assertIsNone(InteractiveShell.command_completion_prefix(" /"))
        self.assertIsNone(InteractiveShell.command_completion_prefix("hello /"))
        self.assertIsNone(InteractiveShell.command_completion_prefix("/help now"))

    def test_completer_only_suggests_commands_from_line_start(self):
        leading_slash = [
            completion.text
            for completion in self.shell._completer.get_completions(Document(text="/"), None)
        ]
        inline_text = [
            completion.text
            for completion in self.shell._completer.get_completions(Document(text="hello /"), None)
        ]
        leading_space = [
            completion.text
            for completion in self.shell._completer.get_completions(Document(text=" /"), None)
        ]

        self.assertEqual(leading_slash, ["/help", "/clear", "/new", "/exit"])
        self.assertEqual(inline_text, [])
        self.assertEqual(leading_space, [])

    async def test_double_escape_cancels_active_run(self):
        active_task = asyncio.create_task(asyncio.sleep(30))
        self.shell._active_run_task = active_task

        first_press = await self.shell.handle_escape_press(now=10.0)
        second_press = await self.shell.handle_escape_press(now=11.0)

        self.assertFalse(first_press)
        self.assertTrue(second_press)
        with suppress(asyncio.CancelledError):
            await active_task
        self.assertTrue(active_task.cancelled())

    async def test_escape_timeout_requires_rearming(self):
        active_task = asyncio.create_task(asyncio.sleep(30))
        self.shell._active_run_task = active_task

        first_press = await self.shell.handle_escape_press(now=10.0)
        second_press = await self.shell.handle_escape_press(now=13.5)

        self.assertFalse(first_press)
        self.assertFalse(second_press)
        self.assertFalse(active_task.cancelled())
        active_task.cancel()
        with suppress(asyncio.CancelledError):
            await active_task


if __name__ == "__main__":
    unittest.main()
