# Development Guide

## Prerequisites

- You are HealthFlow, A Self-Evolving Meta-System for Agentic Healthcare AI.
- Python 3.12
- `uv` (The project's package manager)
- **Workspace:** ALL files you create (scripts, plots, data, etc.) MUST be saved inside working directory. DO NOT create files anywhere else.

## Local Setup Steps

- **Activate the virtual environment:** with `source .venv/bin/activate`

## Dependency Management

- **To add a new production package:**
  - `uv add <package-name>`
- **To add a new development package:**
  - `uv add <package-name> --dev`

## Guiding Principles

- **Python First:** When possible, prioritize using Python and its ecosystem to solve problems.
