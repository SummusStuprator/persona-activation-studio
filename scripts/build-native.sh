#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${STUDIO_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3)"; fi
args=(--jobs "${JOBS:-2}" --architectures "${CUDA_ARCHITECTURES:-native}")
if [[ "${CUDA:-OFF}" == ON ]]; then args+=(--cuda); fi
if [[ -n "${CUDA_ROOT:-}" ]]; then args+=(--cuda-root "$CUDA_ROOT"); fi
cd "$ROOT"
exec "$PY" -m studio_native "${args[@]}"
