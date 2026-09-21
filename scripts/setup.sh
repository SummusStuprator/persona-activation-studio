#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-core}"
BOOTSTRAP="${PYTHON:-python3}"
"$BOOTSTRAP" - <<'PY'
import sys
if sys.version_info < (3,11) or sys.version_info >= (3,14):
    raise SystemExit("Persona Activation Studio requires Python 3.11, 3.12, or 3.13")
print("Bootstrap Python:", sys.executable, sys.version.split()[0])
PY
if [[ ! -d "$ROOT/.venv" ]]; then "$BOOTSTRAP" -m venv "$ROOT/.venv"; fi
PYBIN="$ROOT/.venv/bin/python"
"$PYBIN" -m pip install --upgrade pip setuptools wheel
case "$PROFILE" in
  core) EXTRA="" ;;
  scrape) EXTRA="[scrape]" ;;
  persona) EXTRA="[persona]" ;;
  train) EXTRA="[train]" ;;
  all) EXTRA="[all]" ;;
  *) echo "Usage: setup.sh [core|scrape|persona|train|all]"; exit 2 ;;
esac
"$PYBIN" -m pip install -e "$ROOT$EXTRA"
"$PYBIN" -m studio_cli init
echo "Studio is ready. Run scripts/start.sh"
