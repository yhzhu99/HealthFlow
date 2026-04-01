from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.panel import Panel

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - unsupported platforms.
    termios = None
    tty = None

ShellTaskRunner = Callable[[Any, str], Awaitable[dict[str, Any]]]
SystemFactory = Callable[[], Any]
PromptSessionFactory = Callable[[], Any]


class EscapeMonitor:
    def __init__(
        self,
        *,
        on_escape: Callable[[], Awaitable[bool] | bool],
        stdin: Any | None = None,
    ):
        self._on_escape = on_escape
        self._stdin = stdin or sys.stdin

    def supported(self) -> bool:
        if termios is None or tty is None:
            return False
        stream = self._stdin
        if stream is None or not hasattr(stream, "isatty") or not stream.isatty():
            return False
        try:
            stream.fileno()
        except (AttributeError, OSError, ValueError):
            return False
        return True

    async def watch(self) -> None:
        if not self.supported():
            return

        loop = asyncio.get_running_loop()
        fd = self._stdin.fileno()
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        previous = termios.tcgetattr(fd)

        def _on_readable() -> None:
            try:
                data = os.read(fd, 32)
            except OSError:
                return
            if data:
                queue.put_nowait(data)

        tty.setcbreak(fd)
        loop.add_reader(fd, _on_readable)
        try:
            while True:
                data = await queue.get()
                if b"\x1b" not in data:
                    continue
                should_stop = self._on_escape()
                if inspect.isawaitable(should_stop):
                    should_stop = await should_stop
                if should_stop:
                    return
        finally:
            loop.remove_reader(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)


class InteractiveShell:
    COMMANDS: dict[str, str] = {
        "/help": "Show commands and keyboard hints.",
        "/clear": "Clear the terminal and redraw the session banner.",
        "/new": "Start a fresh local session and preserve long-term memory.",
        "/exit": "Exit interactive mode.",
    }
    EXIT_ALIASES = {"exit", "quit"}
    INTERRUPT_WINDOW_SECONDS = 2.0

    def __init__(
        self,
        *,
        console: Console,
        system_factory: SystemFactory,
        task_runner: ShellTaskRunner,
        prompt_session_factory: PromptSessionFactory | None = None,
        escape_monitor_factory: Callable[[Callable[[], Awaitable[bool] | bool]], EscapeMonitor | None] | None = None,
        time_source: Callable[[], float] | None = None,
        verbose: bool = False,
    ):
        self.console = console
        self._system_factory = system_factory
        self._task_runner = task_runner
        self._prompt_session_factory = prompt_session_factory or self._default_prompt_session_factory
        self._escape_monitor_factory = escape_monitor_factory or self._default_escape_monitor_factory
        self._time_source = time_source or time.monotonic
        self._completer = WordCompleter(sorted(self.COMMANDS), ignore_case=True, sentence=False)
        self._verbose = verbose

        self._session_index = 0
        self._prompt_session = None
        self._system = None
        self._active_run_task: asyncio.Task[dict[str, Any]] | None = None
        self._interrupt_armed_at: float | None = None
        self._escape_supported = False
        self._start_new_session(clear_screen=False)

    @property
    def system(self) -> Any:
        return self._system

    @property
    def session_index(self) -> int:
        return self._session_index

    async def run(self) -> None:
        self._render_banner()
        while True:
            try:
                text = await self._read_prompt()
            except EOFError:
                self.console.print("[yellow]Exiting interactive mode.[/yellow]")
                return
            except KeyboardInterrupt:
                self.console.print("[yellow]Use /exit to leave interactive mode.[/yellow]")
                continue

            if text is None:
                continue
            user_input = text.strip()
            if not user_input:
                continue
            if await self._handle_input(user_input):
                return

    async def handle_escape_press(self, *, now: float | None = None) -> bool:
        if self._active_run_task is None or self._active_run_task.done():
            return False

        current_time = now if now is not None else self._time_source()
        if self._interrupt_armed_at is not None:
            if current_time - self._interrupt_armed_at <= self.INTERRUPT_WINDOW_SECONDS:
                self._interrupt_armed_at = None
                self.console.print("[yellow]Interrupting current run...[/yellow]")
                self._active_run_task.cancel()
                return True
            self._interrupt_armed_at = None

        self._interrupt_armed_at = current_time
        self.console.print("[yellow]Press ESC again within 2 seconds to cancel the current run.[/yellow]")
        return False

    async def _handle_input(self, user_input: str) -> bool:
        lowered = user_input.lower()
        if lowered in self.EXIT_ALIASES:
            self.console.print("[yellow]Exiting interactive mode.[/yellow]")
            return True
        if user_input.startswith("/"):
            return self._handle_command(user_input)
        await self._run_task(user_input)
        return False

    def _handle_command(self, user_input: str) -> bool:
        parts = user_input.split()
        command = parts[0].lower()
        args = parts[1:]

        if command not in self.COMMANDS:
            self.console.print(f"[red]Unknown command:[/red] {command}. Try [bold]/help[/bold].")
            return False
        if args:
            self.console.print(f"[red]{command} does not accept arguments.[/red]")
            return False
        if command == "/help":
            self._render_help()
            return False
        if command == "/clear":
            self.console.clear()
            self._render_banner()
            return False
        if command == "/new":
            self._start_new_session(clear_screen=True)
            self.console.print("[yellow]Started a new session. Long-term memory was preserved.[/yellow]")
            self._render_banner()
            return False
        if command == "/exit":
            self.console.print("[yellow]Exiting interactive mode.[/yellow]")
            return True
        return False

    async def _run_task(self, user_input: str) -> dict[str, Any]:
        self._interrupt_armed_at = None
        hint = "Press ESC twice to interrupt the current run." if self._escape_supported else "Use Ctrl+C in this terminal if you need to stop the shell."
        self.console.print(f"[dim]{hint}[/dim]")
        self._active_run_task = asyncio.create_task(self._task_runner(self._system, user_input))
        escape_task: asyncio.Task[None] | None = None
        escape_monitor = self._escape_monitor_factory(self.handle_escape_press)
        if escape_monitor is not None and escape_monitor.supported():
            self._escape_supported = True
            escape_task = asyncio.create_task(escape_monitor.watch())
        else:
            self._escape_supported = False

        try:
            return await self._active_run_task
        finally:
            self._interrupt_armed_at = None
            self._active_run_task = None
            if escape_task is not None:
                escape_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await escape_task

    async def _read_prompt(self) -> str:
        with patch_stdout():
            return await self._prompt_session.prompt_async(
                "HealthFlow > ",
                completer=self._completer,
                complete_while_typing=True,
                bottom_toolbar=self._bottom_toolbar,
                reserve_space_for_menu=4,
            )

    def _render_banner(self) -> None:
        subtitle = f"Session {self._session_index}"
        interrupt_hint = "ESC ESC interrupt" if self._escape_supported else "ESC ESC interrupt (TTY only)"
        body = "\n".join(
            [
                "Type a task or a slash command.",
                "",
                "Commands: /help  /clear  /new  /exit",
                f"Keys: Tab complete  {interrupt_hint}",
            ]
        )
        self.console.print(
            Panel(
                body,
                title="[bold green]HealthFlow Interactive Mode[/bold green]",
                subtitle=subtitle,
                border_style="green",
            )
        )

    def _render_help(self) -> None:
        command_lines = [f"[bold]{name}[/bold]  {description}" for name, description in self.COMMANDS.items()]
        command_lines.append("[bold]exit[/bold] / [bold]quit[/bold]  Exit interactive mode.")
        command_lines.append("Keyboard: `Tab` completes commands, `ESC ESC` interrupts the current run.")
        self.console.print(Panel("\n".join(command_lines), title="[bold cyan]Interactive Commands[/bold cyan]", border_style="cyan"))

    def _bottom_toolbar(self) -> str:
        return "Type a task or /help. Tab completes commands."

    def _default_prompt_session_factory(self) -> PromptSession[Any]:
        return PromptSession(history=InMemoryHistory())

    def _default_escape_monitor_factory(
        self,
        on_escape: Callable[[], Awaitable[bool] | bool],
    ) -> EscapeMonitor:
        return EscapeMonitor(on_escape=on_escape)

    def _start_new_session(self, *, clear_screen: bool) -> None:
        if clear_screen:
            self.console.clear()
        self._system = self._system_factory()
        self._prompt_session = self._prompt_session_factory()
        self._session_index += 1
        self._active_run_task = None
        self._interrupt_armed_at = None
        escape_monitor = self._escape_monitor_factory(self.handle_escape_press)
        self._escape_supported = escape_monitor is not None and escape_monitor.supported()
