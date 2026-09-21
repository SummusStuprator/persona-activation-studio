#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${STUDIO_PORT:-8899}"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m streamlit run studio_app.py --server.address=127.0.0.1 --server.port="$PORT" --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false
