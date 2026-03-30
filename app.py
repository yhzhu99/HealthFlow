from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

import toml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from healthflow.core.config import get_config, setup_logging
from healthflow.system import HealthFlowSystem


PROJECT_ROOT = Path(__file__).parent.resolve()
WEB_ROOT = PROJECT_ROOT / "web"
CONFIG_PATH = PROJECT_ROOT / "config.toml"
DEFAULT_EXPERIENCE_PATH = PROJECT_ROOT / "workspace" / "experience.jsonl"


TaskRunner = Callable[[str, str, str | None, dict[str, bytes]], Awaitable[dict[str, Any]]]


def load_config_data(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if config_path.exists():
        return toml.load(config_path)
    return {}


def get_option_lists(config_data: dict[str, Any]) -> tuple[list[str], list[str], str]:
    llm_options = sorted(list(config_data.get("llm", {}).keys()))
    executor_options = list(config_data.get("executor", {}).get("backends", {}).keys()) or [
        "healthflow_agent",
        "claude_code",
        "opencode",
    ]
    default_executor = config_data.get("executor", {}).get("active_backend", "healthflow_agent")
    return llm_options, executor_options, default_executor


def initialize_system_from_file(
    active_llm_name: str,
    active_executor_name: str | None,
    config_path: Path = CONFIG_PATH,
    experience_path: Path = DEFAULT_EXPERIENCE_PATH,
) -> HealthFlowSystem:
    config = get_config(config_path, active_llm_name, active_executor_name)
    setup_logging(config)
    return HealthFlowSystem(config=config, experience_path=experience_path)


async def default_task_runner(
    task: str,
    active_llm: str,
    active_executor: str | None,
    uploaded_files: dict[str, bytes],
) -> dict[str, Any]:
    system = initialize_system_from_file(active_llm, active_executor)
    return await system.run_task(task, uploaded_files=uploaded_files)


def create_app(task_runner: TaskRunner | None = None, project_root: Path = PROJECT_ROOT) -> FastAPI:
    runner = task_runner or default_task_runner
    web_root = project_root / "web"
    config_path = project_root / "config.toml"

    app = FastAPI(
        title="HealthFlow Web API",
        description="Vue 3 frontend + Python HealthFlow backend.",
        version="0.1.0",
    )

    if web_root.exists():
        app.mount("/static", StaticFiles(directory=web_root), name="static")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/options")
    async def options() -> dict[str, Any]:
        config_data = load_config_data(config_path)
        llm_options, executor_options, default_executor = get_option_lists(config_data)
        return {
            "llm_options": llm_options,
            "executor_options": executor_options,
            "default_executor": default_executor,
        }

    @app.post("/api/run")
    async def run_task(
        task: str = Form(...),
        active_llm: str = Form(...),
        active_executor: str | None = Form(None),
        files: list[UploadFile] | None = File(default=None),
    ) -> dict[str, Any]:
        uploaded_files: dict[str, bytes] = {}
        for upload in files or []:
            uploaded_files[upload.filename] = await upload.read()
        return await runner(task, active_llm, active_executor, uploaded_files)

    @app.get("/api/artifact")
    async def artifact(path: str) -> dict[str, Any]:
        resolved = (project_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Artifact path must stay within the HealthFlow project root.") from exc

        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail=f"Artifact not found: {resolved}")

        if resolved.suffix == ".json":
            try:
                return {
                    "path": str(resolved),
                    "kind": "json",
                    "content": json.loads(resolved.read_text(encoding="utf-8")),
                }
            except json.JSONDecodeError:
                pass

        return {
            "path": str(resolved),
            "kind": "text",
            "content": resolved.read_text(encoding="utf-8", errors="ignore"),
        }

    @app.get("/{full_path:path}")
    async def index(full_path: str) -> FileResponse:
        index_file = web_root / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend assets are missing.")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
