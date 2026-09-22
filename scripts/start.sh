#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ARGS=(-m studio_cli app)
if [[ -n "${STUDIO_PORT:-}" ]]; then
    ARGS+=(--port "$STUDIO_PORT")
fi
exec "$ROOT/.venv/bin/python" "${ARGS[@]}"
