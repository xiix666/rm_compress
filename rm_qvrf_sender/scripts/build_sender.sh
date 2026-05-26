#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT}/build/onboard"
ENABLE_TRT="${ENABLE_TRT:-0}"
MVS_ROOT="${MVS_ROOT:-}"

args=(-S "${ROOT}/onboard" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release)
if [[ "${ENABLE_TRT}" == "1" ]]; then
  args+=(-DRMCOMPRESS_ENABLE_TENSORRT=ON)
fi
if [[ -n "${MVS_ROOT}" ]]; then
  args+=(-DMVS_ROOT="${MVS_ROOT}")
fi

cmake "${args[@]}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

mkdir -p "${ROOT}/bin"
cp -a "${BUILD_DIR}/rm_compress_cli" "${ROOT}/bin/"
cp -a "${BUILD_DIR}/librmcompress.so" "${ROOT}/bin/"
if [[ -x "${BUILD_DIR}/rm_camera_capture" ]]; then
  cp -a "${BUILD_DIR}/rm_camera_capture" "${ROOT}/bin/"
else
  echo "WARN: rm_camera_capture was not built. Set MVS_ROOT to Hikrobot MVS SDK root and rebuild."
fi
