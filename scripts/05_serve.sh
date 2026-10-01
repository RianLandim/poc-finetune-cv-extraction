#!/usr/bin/env bash
# Stage 5 -- serve one GGUF (plus its mmproj for vision) with full GPU offload.
#
# Only ONE model at a time (base and tuned never run concurrently). CTX is per slot: the
# KV pool is CTX * PARALLEL, so every request gets the whole window.
set -euo pipefail

MODEL="${1:?usage: 05_serve.sh <model.gguf> [mmproj.gguf] [port] [ctx] [parallel]}"
MMPROJ="${2:-}"
PORT="${3:-8080}"
CTX="${4:-6144}"
PARALLEL="${5:-1}"
LLAMA_DIR="${LLAMA_DIR:-vendor/llama.cpp}"

[ -f "$MODEL" ] || { echo "FATAL: $MODEL not found" >&2; exit 1; }
EXTRA=()
if [ -n "$MMPROJ" ]; then
  [ -f "$MMPROJ" ] || { echo "FATAL: $MMPROJ not found" >&2; exit 1; }
  EXTRA=(--mmproj "$MMPROJ")
fi

exec "$LLAMA_DIR/build/bin/llama-server" \
  -m "$MODEL" "${EXTRA[@]}" \
  -ngl 99 \
  -c "$((CTX * PARALLEL))" \
  -np "$PARALLEL" \
  -fa on \
  --host 127.0.0.1 \
  --port "$PORT"
