#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/models/engines"
DEVICE="0"
SIZE="448"
BUILD_GS="1"
BUILD_SR="1"
BUILD_FUSED_SR="0"
SR_PRECISION="fp16"
GS_ONNX=""
SR_ONNX=""
FUSED_SR_ONNX=""
RLFN_MODEL="${ROOT}/models/rlfn_s_x2.pth"

usage() {
  cat <<EOF
Usage: $0 [options]

Build receiver-side TensorRT engines from ONNX only.

Options:
  --device N          CUDA device id (default: ${DEVICE})
  --size N            Fixed codec size, e.g. 192 or 448 (default: ${SIZE})
  --out-dir DIR       Output directory (default: models/engines)
  --gs-onnx PATH      QVRF g_s ONNX (default: models/msssim_g_s.onnx)
                      The model may be dynamic; this script builds a fixed profile.
  --sr-onnx PATH      RLFN ONNX (default: models/rlfn_s_x2_<size>.onnx)
  --fused-sr-onnx PATH
                      Fused QVRF g_s + RLFN ONNX (default: models/qvrf_gs_rlfn_x2_<size>.onnx)
  --rlfn-model PATH   RLFN checkpoint used if --sr-onnx is missing
  --gs-only           Build only QVRF receiver g_s
  --sr-only           Build only RLFN SR
  --fused-sr-only     Build only fused QVRF g_s + RLFN SR
  --with-fused-sr     Also build fused QVRF g_s + RLFN SR
  --sr-fp32           Build SR without FP16 tactics
  --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --gs-onnx) GS_ONNX="$2"; shift 2 ;;
    --sr-onnx) SR_ONNX="$2"; shift 2 ;;
    --fused-sr-onnx) FUSED_SR_ONNX="$2"; shift 2 ;;
    --rlfn-model) RLFN_MODEL="$2"; shift 2 ;;
    --gs-only) BUILD_GS="1"; BUILD_SR="0"; shift ;;
    --sr-only) BUILD_GS="0"; BUILD_SR="1"; shift ;;
    --fused-sr-only) BUILD_GS="0"; BUILD_SR="0"; BUILD_FUSED_SR="1"; shift ;;
    --with-fused-sr) BUILD_FUSED_SR="1"; shift ;;
    --sr-fp32) SR_PRECISION="fp32"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${SIZE}" -le 0 || $((SIZE % 16)) -ne 0 ]]; then
  echo "--size must be a positive multiple of 16" >&2
  exit 1
fi
LATENT_SIZE="$((SIZE / 16))"
SR_SIZE="$((SIZE * 2))"
if [[ -z "${GS_ONNX}" ]]; then GS_ONNX="${ROOT}/models/msssim_g_s.onnx"; fi
if [[ -z "${SR_ONNX}" ]]; then SR_ONNX="${ROOT}/models/rlfn_s_x2_${SIZE}.onnx"; fi
if [[ -z "${FUSED_SR_ONNX}" ]]; then FUSED_SR_ONNX="${ROOT}/models/qvrf_gs_rlfn_x2_${SIZE}.onnx"; fi

if [[ "${OUT_DIR}" != /* ]]; then OUT_DIR="${ROOT}/${OUT_DIR}"; fi
if [[ "${GS_ONNX}" != /* ]]; then GS_ONNX="${ROOT}/${GS_ONNX}"; fi
if [[ "${SR_ONNX}" != /* ]]; then SR_ONNX="${ROOT}/${SR_ONNX}"; fi
if [[ "${FUSED_SR_ONNX}" != /* ]]; then FUSED_SR_ONNX="${ROOT}/${FUSED_SR_ONNX}"; fi
if [[ "${RLFN_MODEL}" != /* ]]; then RLFN_MODEL="${ROOT}/${RLFN_MODEL}"; fi
mkdir -p "${OUT_DIR}"

command -v trtexec >/dev/null || { echo "trtexec not found" >&2; exit 1; }

read -r TRT_VERSION GPU_SAFE CC <<<"$(python3 - "${DEVICE}" <<'PY'
import re
import sys

import tensorrt as trt
from cuda.bindings import runtime as cudart

device = int(sys.argv[1])
status, count = cudart.cudaGetDeviceCount()
if status != cudart.cudaError_t.cudaSuccess or device >= count:
    raise SystemExit(f"CUDA device {device} unavailable")
status, props = cudart.cudaGetDeviceProperties(device)
if status != cudart.cudaError_t.cudaSuccess:
    raise SystemExit("cudaGetDeviceProperties failed")
name = props.name.decode("utf-8", errors="replace")
safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
print(trt.__version__.replace(".", "_"), safe, f"sm{props.major}{props.minor}")
PY
)"

if [[ "${BUILD_SR}" == "1" && ! -f "${SR_ONNX}" ]]; then
  python3 "${ROOT}/scripts/export_rlfn_to_onnx.py" \
    --rlfn-model "${RLFN_MODEL}" \
    --size "${SIZE}" \
    --output "${SR_ONNX}"
fi

if [[ "${BUILD_FUSED_SR}" == "1" && ! -f "${FUSED_SR_ONNX}" ]]; then
  uv run --directory "${ROOT}/client" python "${ROOT}/scripts/export_qvrf_rlfn_fused_to_onnx.py" \
    --rlfn-model "${RLFN_MODEL}" \
    --size "${SIZE}" \
    --fp32-output \
    --output "${FUSED_SR_ONNX}"
fi

if [[ "${BUILD_GS}" == "1" ]]; then
  [[ -f "${GS_ONNX}" ]] || { echo "Missing QVRF g_s ONNX: ${GS_ONNX}" >&2; exit 1; }
  GS_ENGINE="${OUT_DIR}/msssim_g_s_${SIZE}_fp32_fixed_trt${TRT_VERSION}_${GPU_SAFE}_${CC}.engine"
  trtexec \
    --onnx="${GS_ONNX}" \
    --saveEngine="${GS_ENGINE}" \
    --device="${DEVICE}" \
    --minShapes=y_hat:1x192x${LATENT_SIZE}x${LATENT_SIZE} \
    --optShapes=y_hat:1x192x${LATENT_SIZE}x${LATENT_SIZE} \
    --maxShapes=y_hat:1x192x${LATENT_SIZE}x${LATENT_SIZE} \
    --shapes=y_hat:1x192x${LATENT_SIZE}x${LATENT_SIZE} \
    --inputIOFormats=fp32:chw \
    --outputIOFormats=fp32:chw \
    --noTF32 \
    --skipInference
  echo "QVRF g_s engine: ${GS_ENGINE}"
fi

if [[ "${BUILD_SR}" == "1" ]]; then
  [[ -f "${SR_ONNX}" ]] || { echo "Missing RLFN ONNX: ${SR_ONNX}" >&2; exit 1; }
  SR_ENGINE="${OUT_DIR}/rlfn_s_x2_${SIZE}x${SR_SIZE}_${SR_PRECISION}_iofp32_fixed_trt${TRT_VERSION}_${GPU_SAFE}_${CC}.engine"
  SR_PRECISION_ARGS=()
  if [[ "${SR_PRECISION}" == "fp16" ]]; then
    SR_PRECISION_ARGS+=(--fp16)
  fi
  trtexec \
    --onnx="${SR_ONNX}" \
    --saveEngine="${SR_ENGINE}" \
    --device="${DEVICE}" \
    "${SR_PRECISION_ARGS[@]}" \
    --inputIOFormats=fp32:chw \
    --outputIOFormats=fp32:chw \
    --skipInference
  echo "RLFN SR engine: ${SR_ENGINE}"
fi

if [[ "${BUILD_FUSED_SR}" == "1" ]]; then
  [[ -f "${FUSED_SR_ONNX}" ]] || { echo "Missing fused QVRF+RLFN ONNX: ${FUSED_SR_ONNX}" >&2; exit 1; }
  FUSED_ENGINE="${OUT_DIR}/qvrf_gs_rlfn_x2_${SIZE}x${SR_SIZE}_${SR_PRECISION}_iofp32_fixed_trt${TRT_VERSION}_${GPU_SAFE}_${CC}.engine"
  FUSED_PRECISION_ARGS=()
  if [[ "${SR_PRECISION}" == "fp16" ]]; then
    FUSED_PRECISION_ARGS+=(--fp16)
  fi
  trtexec \
    --onnx="${FUSED_SR_ONNX}" \
    --saveEngine="${FUSED_ENGINE}" \
    --device="${DEVICE}" \
    "${FUSED_PRECISION_ARGS[@]}" \
    --inputIOFormats=fp32:chw \
    --outputIOFormats=fp32:chw \
    --skipInference
  echo "Fused QVRF g_s + RLFN SR engine: ${FUSED_ENGINE}"
fi
