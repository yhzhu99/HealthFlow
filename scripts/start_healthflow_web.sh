#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${HEALTHFLOW_ENV_FILE:-$HOME/.config/healthflow/healthflow-web.env}"
CONFIG_PATH="${HEALTHFLOW_CONFIG_PATH:-$ROOT_DIR/config.toml}"
SERVER_NAME="${HEALTHFLOW_SERVER_NAME:-127.0.0.1}"
SERVER_PORT="${HEALTHFLOW_SERVER_PORT:-7860}"
GRADIO_ROOT_PATH="${GRADIO_ROOT_PATH:-/app}"
export GRADIO_ROOT_PATH

if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

if [ "${HEALTHFLOW_CLEAR_PROXY:-1}" = "1" ]; then
  unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
fi

if [ -z "${ZENMUX_API_KEY:-}" ]; then
  echo "ZENMUX_API_KEY is not set. Populate $ENV_FILE before starting HealthFlow." >&2
  exit 1
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "DEEPSEEK_API_KEY is not set. Populate $ENV_FILE before starting HealthFlow." >&2
  exit 1
fi

cd "$ROOT_DIR"

exec /opt/homebrew/bin/uv run --extra web healthflow web \
  --config "$CONFIG_PATH" \
  --server-name "$SERVER_NAME" \
  --server-port "$SERVER_PORT" \
  --root-path "$GRADIO_ROOT_PATH"
