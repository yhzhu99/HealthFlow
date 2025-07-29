# Development Guide

## Prerequisites

- You are HealthFlow, A Self-Evolving Meta-System for Agentic Healthcare AI.
- Python 3.12
- `uv` (The project's package manager)

## Local Setup Steps

- **Activate the virtual environment:** with `source .venv/bin/activate`

## Dependency Management

- **To add a new production package:**
  - `uv add <package-name>`
- **To add a new development package:**
  - `uv add <package-name> --dev`

## Guiding Principles

- **Python First:** When possible, prioritize using Python and its ecosystem to solve problems.


## Critical Instrucctions

1.  **Workspace:** ALL files you create (scripts, plots, data, etc.) MUST be saved inside working directory. DO NOT create files anywhere else.
2.  **Dataset Access:** If the task requires reading datasets (e.g., `TJH.csv`), they are located in the `/home/yhzhu/projects/HealthFlow/healthflow_datasets/` directory. You MUST use the full, absolute path to access them. For example: `read_csv('/home/yhzhu/projects/HealthFlow/healthflow_datasets/TJH.csv')`.
3.  **Execution:** Execute all commands from your current working directory. DO NOT create files in the project root directory outside of your workspace.