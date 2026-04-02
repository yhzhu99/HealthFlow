#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from healthflow.core.config import default_executor_backends, get_config
from healthflow.execution import create_executor_adapter
from healthflow.system import HealthFlowSystem


DEFAULT_BACKENDS = ("claude_code", "codex", "opencode", "pi")
DEFAULT_SCENARIOS = ("oneehr", "tooluniverse")
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"
TOOLUNIVERSE_SERVER_NAME = "healthflow-tu"
PROJECT_VENV_BIN = REPO_ROOT / ".venv" / "bin"
ONEEHR_BIN = PROJECT_VENV_BIN / "oneehr"
TU_BIN = PROJECT_VENV_BIN / "tu"
TOOLUNIVERSE_BIN = PROJECT_VENV_BIN / "tooluniverse"


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def combined_output(self) -> str:
        if self.stdout and self.stderr:
            return f"{self.stdout}\n{self.stderr}".strip()
        return self.stdout or self.stderr


@dataclass
class ValidationCheck:
    category: str
    name: str
    status: str
    summary: str
    backend: str | None = None
    scenario: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _merge_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    if overrides:
        environment.update(overrides)
    return environment


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 120,
    artifact_dir: Path | None = None,
    name: str | None = None,
) -> CommandResult:
    artifact_stem = name or "_".join(part.replace("/", "_") for part in command[:2]) or "command"
    effective_env = _merge_env(env)
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=effective_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        result = CommandResult(
            command=list(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        result = CommandResult(
            command=list(command),
            returncode=-1,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace"),
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace"),
            timed_out=True,
        )

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_text(artifact_dir / f"{artifact_stem}.command.txt", shlex.join(result.command))
        _write_text(artifact_dir / f"{artifact_stem}.stdout.txt", result.stdout)
        _write_text(artifact_dir / f"{artifact_stem}.stderr.txt", result.stderr)
        _write_json(
            artifact_dir / f"{artifact_stem}.result.json",
            {
                "command": result.command,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
            },
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CLI tool exposure and MCP wiring for HealthFlow executor backends.",
    )
    parser.add_argument("--config", default="config.toml", help="HealthFlow config path.")
    parser.add_argument(
        "--workspace-root",
        default="workspace/tool_validation",
        help="Root directory for validation artifacts.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=list(DEFAULT_BACKENDS),
        help="Executor backends to validate.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        choices=list(DEFAULT_SCENARIOS),
        help="Live HealthFlow scenarios to run.",
    )
    parser.add_argument(
        "--executor-timeout-seconds",
        type=int,
        default=600,
        help="Timeout to apply to each HealthFlow executor run.",
    )
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip static CLI availability checks.",
    )
    parser.add_argument(
        "--skip-healthflow",
        action="store_true",
        help="Skip live HealthFlow smoke runs.",
    )
    parser.add_argument(
        "--skip-mcp",
        action="store_true",
        help="Skip direct MCP configuration checks.",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run only static CLI checks.",
    )
    parser.add_argument(
        "--strict-skips",
        action="store_true",
        help="Exit non-zero if any validation step is skipped.",
    )
    return parser


def _make_output_dir(workspace_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = workspace_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _scenario_request(name: str) -> tuple[str, dict[str, bytes] | None]:
    if name == "oneehr":
        request = textwrap.dedent(
            """
            The uploaded CSV is a tiny EHR cohort for mortality prediction.
            Verify the project-local OneEHR CLI surfaced by HealthFlow for this run.

            Requirements:
            1. Use shell commands, not memory.
            2. Run `uv run oneehr --help`.
            3. Save the raw stdout exactly as `oneehr_help.txt`.
            4. Save `tool_validation.json` with this shape:
               {
                 "tool": "oneehr",
                 "commands": [{"command": "uv run oneehr --help", "exit_code": 0}],
                 "observed_subcommands": ["preprocess", "train", "..."]
               }
            5. Save `tool_validation.md` with a short summary.
            6. Final answer must reference `oneehr_help.txt`, `tool_validation.json`, and `tool_validation.md`.

            Do not install anything. Do not use the network.
            """
        ).strip()
        uploaded_files = {
            "patients.csv": (
                "subject_id,mortality,label,age,event_time,discharge_time\n"
                "1,0,0,65,2020-01-01,2020-01-03\n"
                "2,1,1,71,2020-01-02,2020-01-04\n"
            ).encode("utf-8")
        }
        return request, uploaded_files

    if name == "tooluniverse":
        request = textwrap.dedent(
            """
            Use ToolUniverse for a biomedical tool lookup smoke test.
            Verify the project-local ToolUniverse CLI surfaced by HealthFlow for this run.

            Requirements:
            1. Use shell commands, not memory.
            2. Run `uv run tu --help` and save the raw stdout exactly as `tu_help.txt`.
            3. Run `uv run tu status` and save the raw stdout exactly as `tu_status.txt`.
            4. Save `tool_validation.json` with this shape:
               {
                 "tool": "tooluniverse",
                 "commands": [
                   {"command": "uv run tu --help", "exit_code": 0},
                   {"command": "uv run tu status", "exit_code": 0}
                 ],
                 "observed_commands": ["list", "find", "run", "serve", "..."],
                 "status_summary": "..."
               }
            5. Save `tool_validation.md` with a short summary.
            6. Final answer must reference `tu_help.txt`, `tu_status.txt`, `tool_validation.json`, and `tool_validation.md`.

            Do not install anything. Do not use the network.
            """
        ).strip()
        return request, None

    raise ValueError(f"Unsupported scenario: {name}")


def _validate_oneehr_workspace(task_workspace: Path) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    artifacts: dict[str, str] = {}

    help_path = task_workspace / "oneehr_help.txt"
    json_path = task_workspace / "tool_validation.json"
    md_path = task_workspace / "tool_validation.md"
    for path in (help_path, json_path, md_path):
        artifacts[path.name] = str(path)
        if not path.exists():
            errors.append(f"Missing expected artifact: {path.name}")

    if help_path.exists():
        help_text = _read_text(help_path)
        for token in ("OneEHR CLI", "preprocess", "train", "test", "analyze", "plot", "convert"):
            if token not in help_text:
                errors.append(f"`oneehr_help.txt` is missing expected token: {token}")

    if json_path.exists():
        try:
            payload = json.loads(_read_text(json_path))
        except json.JSONDecodeError as exc:
            errors.append(f"`tool_validation.json` is not valid JSON: {exc}")
        else:
            if payload.get("tool") != "oneehr":
                errors.append("`tool_validation.json` did not record tool='oneehr'.")
            commands = payload.get("commands") or []
            if not any(
                item.get("command") == "uv run oneehr --help" and int(item.get("exit_code", 1)) == 0
                for item in commands
                if isinstance(item, dict)
            ):
                errors.append("`tool_validation.json` does not contain a successful `uv run oneehr --help` record.")
            observed = {str(item) for item in payload.get("observed_subcommands") or []}
            if not {"preprocess", "train", "test"}.issubset(observed):
                errors.append("`tool_validation.json` does not list the expected OneEHR subcommands.")

    if md_path.exists() and not _read_text(md_path).strip():
        errors.append("`tool_validation.md` is empty.")

    return errors, artifacts


def _validate_tooluniverse_workspace(task_workspace: Path) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    artifacts: dict[str, str] = {}

    help_path = task_workspace / "tu_help.txt"
    status_path = task_workspace / "tu_status.txt"
    json_path = task_workspace / "tool_validation.json"
    md_path = task_workspace / "tool_validation.md"
    for path in (help_path, status_path, json_path, md_path):
        artifacts[path.name] = str(path)
        if not path.exists():
            errors.append(f"Missing expected artifact: {path.name}")

    if help_path.exists():
        help_text = _read_text(help_path)
        for token in ("ToolUniverse CLI", "find", "run", "status", "serve"):
            if token not in help_text:
                errors.append(f"`tu_help.txt` is missing expected token: {token}")

    if status_path.exists():
        status_text = _read_text(status_path)
        for token in ("tools loaded:", "categories:", "workspace:"):
            if token not in status_text:
                errors.append(f"`tu_status.txt` is missing expected token: {token}")

    if json_path.exists():
        try:
            payload = json.loads(_read_text(json_path))
        except json.JSONDecodeError as exc:
            errors.append(f"`tool_validation.json` is not valid JSON: {exc}")
        else:
            tool_name = str(payload.get("tool") or "").lower()
            if tool_name not in {"tooluniverse", "tu"}:
                errors.append("`tool_validation.json` did not record tool='tooluniverse'.")
            commands = payload.get("commands") or []
            expected_commands = {
                "uv run tu --help": False,
                "uv run tu status": False,
            }
            for item in commands:
                if not isinstance(item, dict):
                    continue
                command = item.get("command")
                exit_code = int(item.get("exit_code", 1))
                if command in expected_commands and exit_code == 0:
                    expected_commands[command] = True
            for command, found in expected_commands.items():
                if not found:
                    errors.append(f"`tool_validation.json` does not contain a successful `{command}` record.")
            observed = {str(item) for item in payload.get("observed_commands") or []}
            if not {"find", "run", "serve", "status"}.issubset(observed):
                errors.append("`tool_validation.json` does not list the expected ToolUniverse commands.")
            if not str(payload.get("status_summary") or "").strip():
                errors.append("`tool_validation.json` is missing `status_summary`.")

    if md_path.exists() and not _read_text(md_path).strip():
        errors.append("`tool_validation.md` is empty.")

    return errors, artifacts


def _validate_healthflow_workspace(scenario: str, task_workspace: Path) -> tuple[list[str], dict[str, str]]:
    if scenario == "oneehr":
        return _validate_oneehr_workspace(task_workspace)
    if scenario == "tooluniverse":
        return _validate_tooluniverse_workspace(task_workspace)
    raise ValueError(f"Unsupported scenario: {scenario}")


def _load_healthflow_config(config_path: Path, backend: str, workspace_dir: Path, timeout_seconds: int):
    config = get_config(config_path, active_executor=backend)
    backends = dict(config.executor.backends)
    backends[backend] = backends[backend].model_copy(update={"timeout_seconds": timeout_seconds})
    return config.model_copy(
        update={
            "active_executor_name": backend,
            "system": config.system.model_copy(update={"max_attempts": 1, "workspace_dir": str(workspace_dir)}),
            "memory": config.memory.model_copy(update={"write_policy": "freeze"}),
            "executor": config.executor.model_copy(update={"active_backend": backend, "backends": backends}),
        }
    )


def _static_cli_checks(output_dir: Path, backends: Sequence[str]) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    static_dir = output_dir / "static"

    backend_paths = {backend: shutil.which(default_executor_backends()[backend].binary) for backend in backends}
    missing_backends = sorted(name for name, path in backend_paths.items() if path is None)
    checks.append(
        ValidationCheck(
            category="static_cli",
            name="backend_binaries",
            status=STATUS_PASS if not missing_backends else STATUS_FAIL,
            summary=(
                "All selected executor binaries are on PATH."
                if not missing_backends
                else "Missing executor binaries: " + ", ".join(missing_backends)
            ),
            details=backend_paths,
        )
    )

    project_binaries = {
        "oneehr": str(ONEEHR_BIN) if ONEEHR_BIN.exists() else None,
        "tu": str(TU_BIN) if TU_BIN.exists() else None,
        "tooluniverse": str(TOOLUNIVERSE_BIN) if TOOLUNIVERSE_BIN.exists() else None,
    }
    missing_project = sorted(name for name, path in project_binaries.items() if path is None)
    checks.append(
        ValidationCheck(
            category="static_cli",
            name="project_cli_binaries",
            status=STATUS_PASS if not missing_project else STATUS_FAIL,
            summary=(
                "Project-local CLI binaries exist in .venv/bin."
                if not missing_project
                else "Missing project-local binaries: " + ", ".join(missing_project)
            ),
            details=project_binaries,
        )
    )

    command_specs = {
        "oneehr_help": ["uv", "run", "oneehr", "--help"],
        "tu_help": ["uv", "run", "tu", "--help"],
        "tooluniverse_help": ["uv", "run", "tooluniverse", "--help"],
    }
    for name, command in command_specs.items():
        result = _run_command(command, artifact_dir=static_dir, name=name)
        checks.append(
            ValidationCheck(
                category="static_cli",
                name=name,
                status=STATUS_PASS if result.ok else STATUS_FAIL,
                summary=(
                    f"`{shlex.join(command)}` completed successfully."
                    if result.ok
                    else f"`{shlex.join(command)}` failed with return code {result.returncode}."
                ),
                details={
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                },
                artifacts={
                    "stdout": str(static_dir / f"{name}.stdout.txt"),
                    "stderr": str(static_dir / f"{name}.stderr.txt"),
                },
            )
        )

    env_details: dict[str, dict[str, str | bool | None]] = {}
    for backend in backends:
        backend_config = default_executor_backends()[backend]
        if backend == "pi" and backend_config.model is None:
            backend_config = backend_config.model_copy(update={"model": "openai/gpt-5.4"})
        executor = create_executor_adapter(backend, backend_config)
        environment = executor._build_environment(output_dir / "env_probe")
        env_details[backend] = {
            "path_first_entry": environment["PATH"].split(os.pathsep)[0] if environment.get("PATH") else None,
            "oneehr_visible": bool(shutil.which("oneehr", path=environment.get("PATH"))),
            "tu_visible": bool(shutil.which("tu", path=environment.get("PATH"))),
            "tooluniverse_visible": bool(shutil.which("tooluniverse", path=environment.get("PATH"))),
        }

    env_failures = [
        backend
        for backend, details in env_details.items()
        if not (details["oneehr_visible"] and details["tu_visible"] and details["tooluniverse_visible"])
    ]
    checks.append(
        ValidationCheck(
            category="static_cli",
            name="executor_environment_path_exposure",
            status=STATUS_PASS if not env_failures else STATUS_FAIL,
            summary=(
                "All selected executor environments resolve oneehr, tu, and tooluniverse via PATH."
                if not env_failures
                else "Some executor environments do not expose project-local CLI tools: " + ", ".join(env_failures)
            ),
            details=env_details,
        )
    )

    return checks


async def _run_healthflow_scenario(
    *,
    output_dir: Path,
    config_path: Path,
    backend: str,
    scenario: str,
    timeout_seconds: int,
) -> ValidationCheck:
    scenario_dir = output_dir / "healthflow" / backend / scenario
    task_root = scenario_dir / "tasks"
    request, uploaded_files = _scenario_request(scenario)
    _write_text(scenario_dir / "request.txt", request)

    binary_name = default_executor_backends()[backend].binary
    if shutil.which(binary_name) is None:
        return ValidationCheck(
            category="healthflow_live",
            name="live_smoke",
            status=STATUS_SKIP,
            backend=backend,
            scenario=scenario,
            summary=f"Skipped live HealthFlow smoke because `{binary_name}` is not on PATH.",
        )

    try:
        config = _load_healthflow_config(config_path, backend, task_root, timeout_seconds)
    except Exception as exc:
        return ValidationCheck(
            category="healthflow_live",
            name="live_smoke",
            status=STATUS_SKIP,
            backend=backend,
            scenario=scenario,
            summary=f"Skipped live HealthFlow smoke because config could not be loaded: {exc}",
            details={"traceback": traceback.format_exc()},
        )

    experience_path = scenario_dir / "memory" / "experience.jsonl"
    system = HealthFlowSystem(config=config, experience_path=experience_path)

    try:
        result = await system.run_task(
            request,
            uploaded_files=uploaded_files,
            report_requested=False,
        )
    except Exception as exc:
        return ValidationCheck(
            category="healthflow_live",
            name="live_smoke",
            status=STATUS_FAIL,
            backend=backend,
            scenario=scenario,
            summary=f"HealthFlow live smoke crashed: {exc}",
            details={"traceback": traceback.format_exc()},
        )

    task_workspace = Path(result["workspace_path"])
    runtime_index = json.loads(_read_text(Path(result["runtime_index_path"])))
    run_summary = json.loads(_read_text(Path(result["run_summary_path"])))
    prompt_path = Path(result["last_executor_prompt_path"]) if result.get("last_executor_prompt_path") else None
    prompt_text = _read_text(prompt_path) if prompt_path and prompt_path.exists() else ""

    expected_tool_marker = "oneehr" if scenario == "oneehr" else "tooluniverse"
    expected_prompt_marker = "uv run oneehr" if scenario == "oneehr" else "uv run tu"
    errors: list[str] = []

    result_tools = " ".join(str(item) for item in result.get("available_project_cli_tools", [])).lower()
    summary_tools = " ".join(str(item) for item in run_summary.get("available_project_cli_tools", [])).lower()
    index_tools = " ".join(str(item) for item in runtime_index.get("workflow_recommendations", [])).lower()
    if expected_tool_marker not in result_tools:
        errors.append("Returned result did not include the expected project CLI tool contract.")
    if expected_tool_marker not in summary_tools and expected_tool_marker not in index_tools:
        errors.append("Runtime summaries did not include the expected project CLI tool contract.")
    if "## Available Project CLI Tools" not in prompt_text:
        errors.append("Executor prompt is missing the Available Project CLI Tools section.")
    if expected_prompt_marker not in prompt_text:
        errors.append("Executor prompt does not mention the expected preferred command.")

    workspace_errors, workspace_artifacts = _validate_healthflow_workspace(scenario, task_workspace)
    errors.extend(workspace_errors)

    artifacts = {
        "workspace_path": str(task_workspace),
        "runtime_index_path": str(Path(result["runtime_index_path"])),
        "run_summary_path": str(Path(result["run_summary_path"])),
        "log_path": str(Path(result["last_executor_log_path"])) if result.get("last_executor_log_path") else "",
        "prompt_path": str(prompt_path) if prompt_path else "",
    }
    artifacts.update(workspace_artifacts)

    status = STATUS_PASS if not errors else STATUS_FAIL
    summary = (
        f"HealthFlow live smoke passed for {backend}/{scenario}."
        if status == STATUS_PASS
        else f"HealthFlow live smoke failed for {backend}/{scenario}."
    )
    return ValidationCheck(
        category="healthflow_live",
        name="live_smoke",
        status=status,
        backend=backend,
        scenario=scenario,
        summary=summary,
        details={
            "success": result.get("success"),
            "cancelled": result.get("cancelled"),
            "evaluation_status": result.get("evaluation_status"),
            "errors": errors,
        },
        artifacts=artifacts,
    )


def _prepare_codex_mcp_home(base_dir: Path) -> dict[str, str]:
    home_dir = base_dir / "home"
    codex_dir = home_dir / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    source_config = Path.home() / ".codex" / "config.toml"
    target_config = codex_dir / "config.toml"
    if source_config.exists():
        shutil.copy2(source_config, target_config)
    else:
        _write_text(target_config, "")
    return {"HOME": str(home_dir)}


def _prepare_claude_mcp_home(base_dir: Path) -> dict[str, str]:
    home_dir = base_dir / "home"
    claude_dir = home_dir / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    source_settings = Path.home() / ".claude" / "settings.json"
    target_settings = claude_dir / "settings.json"
    if source_settings.exists():
        shutil.copy2(source_settings, target_settings)
    return {"HOME": str(home_dir)}


def _prepare_opencode_mcp_home(base_dir: Path) -> tuple[dict[str, str], Path]:
    home_dir = base_dir / "home"
    config_root = base_dir / "config"
    config_dir = config_root / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {"$schema": "https://opencode.ai/config.json"}
    source_config = Path.home() / ".config" / "opencode" / "opencode.jsonc"
    if source_config.exists():
        try:
            config = json.loads(_read_text(source_config))
        except json.JSONDecodeError:
            pass

    config["mcp"] = {
        TOOLUNIVERSE_SERVER_NAME: {
            "type": "local",
            "command": [str(TOOLUNIVERSE_BIN), "--workspace", str(base_dir / "tu-workspace")],
            "enabled": True,
        }
    }
    target_config = config_dir / "opencode.jsonc"
    _write_text(target_config, json.dumps(config, indent=2))
    return {"HOME": str(home_dir), "XDG_CONFIG_HOME": str(config_root)}, target_config


def _run_direct_mcp_check(output_dir: Path, backend: str) -> ValidationCheck:
    category = "direct_mcp"
    mcp_dir = output_dir / "mcp" / backend
    binary = default_executor_backends()[backend].binary
    if backend == "pi":
        return ValidationCheck(
            category=category,
            name="mcp_smoke",
            status=STATUS_SKIP,
            backend=backend,
            summary="Skipped direct MCP smoke because pi exposes CLI tools only in this setup.",
        )
    if shutil.which(binary) is None:
        return ValidationCheck(
            category=category,
            name="mcp_smoke",
            status=STATUS_SKIP,
            backend=backend,
            summary=f"Skipped direct MCP smoke because `{binary}` is not on PATH.",
        )
    if not TOOLUNIVERSE_BIN.exists():
        return ValidationCheck(
            category=category,
            name="mcp_smoke",
            status=STATUS_SKIP,
            backend=backend,
            summary="Skipped direct MCP smoke because the ToolUniverse MCP server binary is missing.",
        )

    artifacts: dict[str, str] = {}
    details: dict[str, Any] = {}
    errors: list[str] = []

    if backend == "codex":
        env = _prepare_codex_mcp_home(mcp_dir)
        add_result = _run_command(
            [
                "codex",
                "mcp",
                "add",
                TOOLUNIVERSE_SERVER_NAME,
                "--",
                str(TOOLUNIVERSE_BIN),
                "--workspace",
                str(mcp_dir / "tu-workspace"),
            ],
            env=env,
            artifact_dir=mcp_dir,
            name="codex_mcp_add",
        )
        list_result = _run_command(["codex", "mcp", "list"], env=env, artifact_dir=mcp_dir, name="codex_mcp_list")
        get_result = _run_command(
            ["codex", "mcp", "get", TOOLUNIVERSE_SERVER_NAME],
            env=env,
            artifact_dir=mcp_dir,
            name="codex_mcp_get",
        )
        details["commands"] = {
            "add": add_result.returncode,
            "list": list_result.returncode,
            "get": get_result.returncode,
        }
        if not add_result.ok:
            errors.append("`codex mcp add` failed.")
        if not list_result.ok or TOOLUNIVERSE_SERVER_NAME not in list_result.combined_output:
            errors.append("`codex mcp list` did not show the ToolUniverse server.")
        if not get_result.ok or str(TOOLUNIVERSE_BIN) not in get_result.combined_output:
            errors.append("`codex mcp get` did not show the expected ToolUniverse command.")
        artifacts["codex_home"] = env["HOME"]

    elif backend == "claude_code":
        env = _prepare_claude_mcp_home(mcp_dir)
        add_result = _run_command(
            [
                "claude",
                "mcp",
                "add",
                "-s",
                "user",
                TOOLUNIVERSE_SERVER_NAME,
                "--",
                str(TOOLUNIVERSE_BIN),
                "--workspace",
                str(mcp_dir / "tu-workspace"),
            ],
            env=env,
            artifact_dir=mcp_dir,
            name="claude_mcp_add",
        )
        list_result = _run_command(["claude", "mcp", "list"], env=env, artifact_dir=mcp_dir, name="claude_mcp_list")
        get_result = _run_command(
            ["claude", "mcp", "get", TOOLUNIVERSE_SERVER_NAME],
            env=env,
            artifact_dir=mcp_dir,
            name="claude_mcp_get",
        )
        details["commands"] = {
            "add": add_result.returncode,
            "list": list_result.returncode,
            "get": get_result.returncode,
        }
        if not add_result.ok:
            errors.append("`claude mcp add` failed.")
        if not list_result.ok or TOOLUNIVERSE_SERVER_NAME not in list_result.combined_output:
            errors.append("`claude mcp list` did not show the ToolUniverse server.")
        if not get_result.ok or str(TOOLUNIVERSE_BIN) not in get_result.combined_output:
            errors.append("`claude mcp get` did not show the expected ToolUniverse command.")
        artifacts["claude_home"] = env["HOME"]

    elif backend == "opencode":
        env, config_path = _prepare_opencode_mcp_home(mcp_dir)
        list_result = _run_command(
            ["opencode", "mcp", "list"],
            env=env,
            artifact_dir=mcp_dir,
            name="opencode_mcp_list",
        )
        details["commands"] = {"list": list_result.returncode}
        if not list_result.ok or TOOLUNIVERSE_SERVER_NAME not in list_result.combined_output:
            errors.append("`opencode mcp list` did not show the ToolUniverse server.")
        if str(TOOLUNIVERSE_BIN) not in list_result.combined_output:
            errors.append("`opencode mcp list` did not show the expected ToolUniverse command.")
        artifacts["opencode_config_path"] = str(config_path)
        artifacts["opencode_home"] = env["HOME"]

    else:
        raise ValueError(f"Unsupported backend for direct MCP check: {backend}")

    status = STATUS_PASS if not errors else STATUS_FAIL
    summary = (
        f"Direct MCP smoke passed for {backend}."
        if status == STATUS_PASS
        else f"Direct MCP smoke failed for {backend}."
    )
    for path in sorted(mcp_dir.glob("*")):
        if path.is_file():
            artifacts[path.name] = str(path)
    return ValidationCheck(
        category=category,
        name="mcp_smoke",
        status=status,
        backend=backend,
        summary=summary,
        details={**details, "errors": errors},
        artifacts=artifacts,
    )


def _render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Executor Tool Validation",
        "",
        f"- Generated At: `{report['generated_at']}`",
        f"- Output Dir: `{report['output_dir']}`",
        f"- Config: `{report['config_path']}`",
        "",
        "## Summary",
        "",
        f"- Pass: `{report['summary']['pass']}`",
        f"- Fail: `{report['summary']['fail']}`",
        f"- Skip: `{report['summary']['skip']}`",
        "",
        "## Checks",
        "",
    ]
    for item in report["checks"]:
        scope = " / ".join(part for part in [item.get("category"), item.get("backend"), item.get("scenario"), item.get("name")] if part)
        lines.append(f"- `{item['status']}` {scope}: {item['summary']}")
        errors = item.get("details", {}).get("errors") or []
        for error in errors:
            lines.append(f"  - {error}")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.static_only:
        args.skip_healthflow = True
        args.skip_mcp = True

    config_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    workspace_root = (REPO_ROOT / args.workspace_root).resolve()
    output_dir = _make_output_dir(workspace_root)
    checks: list[ValidationCheck] = []

    if not args.skip_static:
        checks.extend(_static_cli_checks(output_dir, args.backends))

    if not args.skip_healthflow:
        for backend in args.backends:
            for scenario in args.scenarios:
                checks.append(
                    asyncio.run(
                        _run_healthflow_scenario(
                            output_dir=output_dir,
                            config_path=config_path,
                            backend=backend,
                            scenario=scenario,
                            timeout_seconds=args.executor_timeout_seconds,
                        )
                    )
                )

    if not args.skip_mcp:
        for backend in args.backends:
            checks.append(_run_direct_mcp_check(output_dir, backend))

    summary = {
        STATUS_PASS: sum(1 for item in checks if item.status == STATUS_PASS),
        STATUS_FAIL: sum(1 for item in checks if item.status == STATUS_FAIL),
        STATUS_SKIP: sum(1 for item in checks if item.status == STATUS_SKIP),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "backends": list(args.backends),
        "scenarios": list(args.scenarios),
        "summary": {
            "pass": summary[STATUS_PASS],
            "fail": summary[STATUS_FAIL],
            "skip": summary[STATUS_SKIP],
        },
        "checks": [asdict(item) for item in checks],
    }
    _write_json(output_dir / "validation_report.json", report)
    _write_text(output_dir / "validation_report.md", _render_markdown_report(report))

    print(json.dumps(report["summary"], indent=2))
    print(f"validation_report.json: {output_dir / 'validation_report.json'}")
    print(f"validation_report.md: {output_dir / 'validation_report.md'}")

    if summary[STATUS_FAIL] > 0:
        return 1
    if args.strict_skips and summary[STATUS_SKIP] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
