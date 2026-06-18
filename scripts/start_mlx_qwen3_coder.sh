#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8101}"

if ! command -v mlx_lm.server >/dev/null 2>&1; then
  echo "mlx_lm.server not found. Install with: uv tool install mlx-lm" >&2
  exit 1
fi

exec mlx_lm.server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}"
