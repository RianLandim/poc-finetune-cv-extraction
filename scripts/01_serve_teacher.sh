#!/usr/bin/env bash
# Stage 1 helper -- download and serve the prose teacher named in configs/data.yaml.
#
# Text-only (no mmproj), several slots so generate workers are served in parallel.
# Never run it alongside training: it holds ~4GB of VRAM.
set -euo pipefail

DATA_CFG="${1:-configs/data.yaml}"
LLAMA_DIR="${LLAMA_DIR:-vendor/llama.cpp}"
read -r REPO FILE PARALLEL CTX PORT < <(uv run python - "$DATA_CFG" <<'PY'
import sys
from urllib.parse import urlparse
sys.path.insert(0, "src")
from cvx.config import load_data_config
t = load_data_config(sys.argv[1]).generate.teacher
print(t.hf_repo, t.hf_file, t.parallel, t.ctx, urlparse(t.base_url).port or 8080)
PY
)

DEST="outputs/teacher"
if [ ! -f "$DEST/$FILE" ]; then
  echo ">> downloading $REPO/$FILE into $DEST"
  uv run hf download "$REPO" "$FILE" --local-dir "$DEST"
fi

exec "$LLAMA_DIR/build/bin/llama-server" \
  -m "$DEST/$FILE" \
  -ngl 99 \
  -c "$CTX" \
  -np "$PARALLEL" \
  -fa on \
  --host 127.0.0.1 \
  --port "$PORT"
