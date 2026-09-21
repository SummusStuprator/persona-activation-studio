#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/.vendor/llama.cpp"
BUILD="$ROOT/native/build"
RUNTIME="$ROOT/native/runtime"
COMMIT="c0bc8591e8815c63cb01dd3f051a8b0df02501c9"
CUDA="${CUDA:-OFF}"
JOBS="${JOBS:-}"
if [[ -z "$JOBS" ]]; then
  N="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
  JOBS=$(( N / 2 )); (( JOBS < 1 )) && JOBS=1; (( JOBS > 4 )) && JOBS=4
fi
if [[ ! -d "$VENDOR/.git" ]]; then
  mkdir -p "$VENDOR"
  git -C "$VENDOR" init
  git -C "$VENDOR" remote add origin https://github.com/ggerganov/llama.cpp.git
fi
git -C "$VENDOR" fetch --depth 1 origin "$COMMIT"
git -C "$VENDOR" checkout --detach FETCH_HEAD
cmake -S "$ROOT/native" -B "$BUILD" -DLLAMA_CPP_DIR="$VENDOR" -DGGML_CUDA="$CUDA" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target activation_bridge --parallel "$JOBS"
mkdir -p "$RUNTIME"
find "$BUILD" -type f \( -name '*.so' -o -name '*.so.*' -o -name '*.dylib' \) -exec cp -f {} "$RUNTIME/" \;
test -n "$(find "$RUNTIME" -maxdepth 1 -type f -name '*activation_bridge*' -print -quit)" || { echo 'activation bridge missing'; exit 1; }
echo "Native runtime copied to $RUNTIME (parallel jobs: $JOBS)"
