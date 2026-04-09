#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${REPO_ROOT}/platform"
HOST="${HEALTHFLOW_PLATFORM_HOST:-127.0.0.1}"
PORT="${HEALTHFLOW_PLATFORM_PORT:-8005}"

cd "$ROOT_DIR"

/opt/homebrew/bin/npm ci
/opt/homebrew/bin/npm run build

exec /opt/homebrew/bin/npm run preview -- --host "$HOST" --port "$PORT" --strictPort
