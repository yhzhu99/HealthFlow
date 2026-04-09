#!/bin/bash
set -euo pipefail

ROOT_DIR="/path/to/repo/platform"
HOST="${HEALTHFLOW_PLATFORM_HOST:-127.0.0.1}"
PORT="${HEALTHFLOW_PLATFORM_PORT:-8005}"

cd "$ROOT_DIR"

/opt/homebrew/bin/npm ci
/opt/homebrew/bin/npm run build

exec /opt/homebrew/bin/npm run preview -- --host "$HOST" --port "$PORT" --strictPort
