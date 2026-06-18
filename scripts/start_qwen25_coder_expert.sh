#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec ./work/tools/llama-b9700/llama-server \
  -m work/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -fa auto \
  -c 4096 \
  -t 8 \
  --host 127.0.0.1 \
  --port 8101

