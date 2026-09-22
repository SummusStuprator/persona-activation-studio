#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/.vendor/llama.cpp"
CUDA="${CUDA:-OFF}"
flavor=cpu; [[ "$CUDA" == ON ]] && flavor=cuda
BUILD="$ROOT/native/build-$flavor"
JOBS="${JOBS:-2}"
COMMIT="c0bc8591e8815c63cb01dd3f051a8b0df02501c9"
if [[ ! -d "$VENDOR/.git" ]]; then
  mkdir -p "$VENDOR"
  git -C "$VENDOR" init
  git -C "$VENDOR" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if ! git -C "$VENDOR" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
  git -C "$VENDOR" fetch --depth 1 origin "$COMMIT"
fi
[[ -z "$(git -C "$VENDOR" status --porcelain)" ]] || { echo 'Preserve vendor edits first.'; exit 1; }
git -C "$VENDOR" checkout --quiet --detach "$COMMIT"
targets=(activation_bridge ggml-cpu)
options=(-S "$ROOT/native" -B "$BUILD" -DLLAMA_CPP_DIR="$VENDOR" -DGGML_CUDA="$CUDA" -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release)
deploy=()
if [[ "$CUDA" == ON ]]; then
  targets+=(ggml-cuda); deploy+=(--cuda)
  options+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-native}")
fi
cmake "${options[@]}"
cmake --build "$BUILD" --config Release --target "${targets[@]}" --parallel "$JOBS"
"$ROOT/.venv/bin/python" "$ROOT/native_deploy.py" "$BUILD" "${deploy[@]}"
