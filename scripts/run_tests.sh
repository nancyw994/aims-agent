#!/usr/bin/env bash
# Run pytest with the project venv when present (avoids conda vs .venv mismatch).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi
exec "$PY" -m pytest "$@"
