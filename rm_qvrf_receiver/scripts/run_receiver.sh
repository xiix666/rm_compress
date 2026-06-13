#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/config/receiver.env"

usage() {
  cat <<EOF
Usage: ./scripts/run_receiver.sh [options]

Options:
  --preset qvrf192x2x24|qvrf448x6x8
  --mqtt-host HOST
  --mqtt-port PORT
  --client-id ID
  --rx-backend cuda|cpu|openvino|auto
  --torch-device DEV
  --no-trt-fused          Use non-fused receiver path for diagnosis
  --debug-rx-chunks
EOF
}

USE_FUSED=1
USER_NO_FUSED=0
# PRESET=qvrf448x6x8
PRESET=qvrf192x2x24
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) PRESET="$2"; shift 2 ;;
    --mqtt-host) MQTT_HOST="$2"; shift 2 ;;
    --mqtt-port) MQTT_PORT="$2"; shift 2 ;;
    --client-id|--id) CLIENT_ID="$2"; shift 2 ;;
    --rx-backend) RX_BACKEND="$2"; shift 2 ;;
    --torch-device) TORCH_DEVICE="$2"; shift 2 ;;
    --no-trt-fused) USE_FUSED=0; USER_NO_FUSED=1; shift ;;
    --debug-rx-chunks) DEBUG_RX_CHUNKS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

case "${PRESET}" in
  qvrf192x1x24_lowlat|qvrf192x2x24)
    CODEC_PROFILE=codec192x4x12; CODEC_SIZE=192; DISPLAY_SIZE=384
    RX_FUSED_SR_TRT_ENGINE="${RX_FUSED_SR_TRT_ENGINE_192:-models/engines/qvrf_gs_rlfn_x2_192x384_fp16_iofp32_fixed_trt10_16_1_11_nvidia_geforce_rtx_4060_laptop_gpu_sm89.engine}"
    ;;
  qvrf448x6x8)
    CODEC_PROFILE=codec448x9x5; CODEC_SIZE=448; DISPLAY_SIZE=896
    RX_FUSED_SR_TRT_ENGINE="${RX_FUSED_SR_TRT_ENGINE_448:-${RX_FUSED_SR_TRT_ENGINE}}"
    ;;
  *)
    echo "Unknown preset: ${PRESET}" >&2
    usage
    exit 1
    ;;
esac

if [[ "${USER_NO_FUSED}" == "1" ]]; then
  SR_BACKEND=none
  SR_ENGINE=torch
else
  SR_BACKEND=rlfn
  SR_ENGINE=tensorrt
fi

if [[ "${RX_FUSED_SR_TRT_ENGINE}" != /* ]]; then
  RX_FUSED_SR_TRT_ENGINE="${ROOT}/${RX_FUSED_SR_TRT_ENGINE}"
fi
if [[ "${USE_FUSED}" == "1" && ! -f "${RX_FUSED_SR_TRT_ENGINE}" ]]; then
  echo "ERROR: missing fused TRT engine: ${RX_FUSED_SR_TRT_ENGINE}" >&2
  exit 1
fi
if [[ ! -x "${ROOT}/client/.venv/bin/python" ]]; then
  echo "ERROR: client/.venv missing. Run ./install.sh first." >&2
  exit 1
fi

echo "=== RM QVRF Receiver ==="
echo "preset=${PRESET} codec=${CODEC_SIZE} display=${DISPLAY_SIZE}"
echo "mqtt=${MQTT_HOST}:${MQTT_PORT} client_id=${CLIENT_ID}"
echo "backend=${RX_BACKEND} torch=${TORCH_DEVICE} fused=${USE_FUSED}"

ARGS=(
  --codec msssim_qvrf
  --codec-profile "${CODEC_PROFILE}"
  --codec-size "${CODEC_SIZE}"
  --display-size "${DISPLAY_SIZE}"
  --receive-mode mqtt
  # --receive-mode ipc
  # --ipc-host 127.0.0.1
  # --ipc-port 49031
  --mqtt-host "${MQTT_HOST}"
  --mqtt-port "${MQTT_PORT}"
  --client-id "${CLIENT_ID}"
  --rx-backend "${RX_BACKEND}"
  --torch-device "${TORCH_DEVICE}"
  --sr-backend "${SR_BACKEND}"
  --sr-scale "${SR_SCALE}"
)
if [[ "${SR_ENGINE}" == "tensorrt" ]]; then
  ARGS+=(--sr-engine tensorrt)
fi
if [[ "${USE_FUSED}" == "1" ]]; then
  ARGS+=(--rx-fused-sr-trt-engine "${RX_FUSED_SR_TRT_ENGINE}" --rx-fused-sr-trt-device "${RX_FUSED_SR_TRT_DEVICE}")
fi

cd "${ROOT}"

QT_ROOT="${ROOT}/client/.venv/lib/python3.11/site-packages/PyQt5/Qt5"
export LD_LIBRARY_PATH="${QT_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="${QT_ROOT}/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_ROOT}/plugins/platforms"

export PYTHONPATH="${ROOT}/client/src:${ROOT}/commu/src:${ROOT}/compress-ai-gray-minimal:${PYTHONPATH:-}"
export RM_STREAM_DEBUG_RX_CHUNKS="${DEBUG_RX_CHUNKS}"
export RM_STREAM_BACKEND="${RX_BACKEND}"
export RM_STREAM_TORCH_DEVICE="${TORCH_DEVICE}"
exec "${ROOT}/client/.venv/bin/python" -m rm_stream.gui "${ARGS[@]}"
