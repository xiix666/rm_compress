#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CODEC="mbt"
TRANSPORT="serial"
PRESET=""
FRAMES="0"
CAMERA_FRAMES="0"
CAMERA_INDEX="0"
ROI_SIZE="1080"
CAMERA_ROI_MODE="max-square"
CAMERA_FPS="24"
EXPOSURE_US="20000"
SHM_NAME="/rm_camera_frames"
FPS="24"
CODEC_PROFILE="legacy128x2x24"
CODEC_SIZE="128"
DISPLAY_SIZE="256"
DISPLAY_SIZE_SET="0"
CHUNKS_PER_FRAME="2"
FEC_DATA_CHUNKS="0"
CHUNK_RATE_HZ="48"
MAX_QUEUE_CHUNKS="16"
PREBUFFER_CHUNKS="4"
TAIL_FLUSH_CHUNKS="4"
CHUNK_ORDER="0312"
BUDGET="1600"
GAIN="0.8"
TX_DEVICE="GPU.0"
TX_GA_BACKEND="openvino"
TX_TRT_ENGINE=""
TX_TRT_DEVICE="0"
RX_BACKEND="cuda"
RX_GS_BACKEND=""
RX_GS_TRT_ENGINE=""
RX_GS_TRT_DEVICE="0"
RX_FUSED_SR_TRT_ENGINE=""
RX_FUSED_SR_TRT_DEVICE="0"
TORCH_DEVICE="cuda:0"
TX_TORCH_DEVICE=""
GUI_TORCH_DEVICE=""
MQTT_HOST="192.168.12.1"
MQTT_PORT="3333"
CLIENT_ID="1"
SERIAL_PORT="/dev/ttyUSB0"
BAUDRATE="921600"
IPC_HOST="127.0.0.1"
IPC_PORT="49031"
LOG_DIR="/tmp"
ENABLE_SR="0"
SR_BACKEND="none"
SR_ENGINE="torch"
SR_TRT_ENGINE=""
SR_TRT_DEVICE="0"
SR_SCALE="2"
REALESR_MODEL="${ROOT}/models/realesr-general-x4v3.pth"
RLFN_MODEL="${ROOT}/models/rlfn_s_x2.pth"
DEBUG_RX_CHUNKS="0"
SERIAL_WAIT="1"
QVRF_CPP_SENDER="0"

usage() {
  cat <<EOF
Usage: $0 [options]

Camera stream launcher.

Pipeline:
  serial:
    camera -> shm -> real sender -> serial 0x0310 -> MQTT -> GUI
  offline-debug:
    camera -> shm -> real sender -> localhost TCP IPC -> GUI

Options:
  --codec mbt|msssim_qvrf       Codec/sender family (default: ${CODEC})
  --transport serial|offline-debug
                                offline-debug = local TCP IPC
  --preset NAME                 mbt: legacy128x2x24, codec192x4x12, codec256x4x12,
                                codec256x6x8, codec320x8x6, codec448x9x5, codec512x7x6,
                                offline320x9x5
                                msssim_qvrf: qvrf192x1x24_lowlat, qvrf192x2x24,
                                qvrf320x6x8, qvrf320x4x12,
                                qvrf448x4x12, qvrf448x6x8
  --frames N                    Sender frames, 0=forever (default: ${FRAMES})
  --camera-frames N             Camera capture frames, 0=forever (default: ${CAMERA_FRAMES})
  --camera-index N              Hikvision device index (default: ${CAMERA_INDEX})
  --roi-size N                  Fixed camera center ROI size; also sets --camera-roi-mode fixed
  --camera-roi-mode MODE        fixed|max-square (default: ${CAMERA_ROI_MODE})
                                max-square = detect max resolution, center-crop shortest-side square
  --camera-fps FPS              Camera FPS (default: ${CAMERA_FPS})
  --exposure-us N               Camera exposure us (default: ${EXPOSURE_US})
  --fps FPS                     Sender FPS (default from preset)
  --codec-size N                Codec input size
  --display-size N              GUI display size
  --chunks-per-frame N          Fixed chunks per frame
  --fec-data-chunks N           MBT FEC data chunks, 0 disables
  --chunk-rate-hz HZ            Physical/logical chunk rate
  --max-queue-chunks N          Sender queue cap
  --prebuffer-chunks N          Sender chunks to buffer before TX starts
  --tail-flush-chunks N         Non-video flush packets after finite TX
  --chunk-order ORDER           MBT 4-chunk order, e.g. 0312
  --budget B                    MS-SSIM QVRF byte budget
  --gain G                      MS-SSIM QVRF gain
  --tx-device DEV               OpenVINO device for C++ MBT sender and C++ QVRF sender
  --tx-ga-backend openvino|tensorrt
                                C++ sender g_a backend. Default openvino preserves
                                --tx-device behavior. TensorRT is explicit only.
  --tx-trt-engine PATH          Existing TensorRT .engine for C++ sender g_a
  --tx-trt-device N             CUDA device id for TensorRT g_a (default: ${TX_TRT_DEVICE})
  --rx-backend auto|openvino|cuda|cpu
                                Receiver backend. h_a/h_s entropy-parameter
                                networks are hard-pinned to OpenVINO FP32 CPU;
                                g_s may use CUDA/OpenVINO/CPU.
  --rx-gs-backend cuda|openvino|tensorrt|cpu
                                Receiver QVRF g_s backend only. h_s remains
                                OpenVINO FP32 CPU and entropy/Gaussian remain CPU/host.
  --rx-gs-trt-engine PATH       Existing TensorRT .engine for receiver QVRF g_s
  --rx-gs-trt-device N          CUDA device id for receiver TensorRT g_s (default: ${RX_GS_TRT_DEVICE})
  --rx-fused-sr-trt-engine PATH Existing fused TensorRT .engine for QVRF g_s + RLFN SR
  --rx-fused-sr-trt-device N    CUDA device id for fused receiver SR (default: ${RX_FUSED_SR_TRT_DEVICE})
  --torch-device DEV            Torch device for both GUI and Python QVRF sender
  --tx-torch-device DEV         Torch device for Python QVRF sender only
  --gui-torch-device DEV        Torch device for GUI QVRF decoder only
  --mqtt-host HOST              Serial transport MQTT broker host
  --mqtt-port PORT              Serial transport MQTT broker port
  --client-id ID                GUI MQTT client id
  --serial-port PORT            Serial port
  --baudrate RATE               Serial baudrate
  --no-serial-wait              Do not wait/reconnect for serial
  --ipc-host HOST               Offline debug TCP host (default: ${IPC_HOST})
  --ipc-port PORT               Offline debug TCP port (default: ${IPC_PORT})
  --shm-name NAME               Shared memory ring name
  --enable-sr                   Compatibility shortcut for --sr-backend msa
  --sr-backend none|msa|realesr|rlfn|rlfn_trt
                                Receiver postprocess backend (default: ${SR_BACKEND})
  --sr-engine torch|tensorrt    Engine for --sr-backend rlfn (default: ${SR_ENGINE})
  --sr-trt-engine PATH          Existing TensorRT .engine for RLFN 448->896 SR
  --sr-trt-device N             CUDA device id for TensorRT SR (default: ${SR_TRT_DEVICE})
  --sr-scale N                  Receiver SR scale for explicit SR backends (default: ${SR_SCALE})
  --realesr-model PATH          realesr-general-x4v3.pth for --sr-backend realesr
  --rlfn-model PATH             rlfn_s_x2.pth for --sr-backend rlfn
  --debug-rx-chunks             Log every received chunk, for diagnosis only
  --qvrf-cpp-sender             Use the supported C++ QVRF sender. With
                                --rx-backend cuda,
                                hs_backend=1 QVRF decode uses the mixed safe path:
                                OpenVINO FP32 CPU h_s plus CPU entropy/Gaussian
                                and CUDA g_s.
                                Default QVRF remains Python/Torch.
  --qvrf-cpp-experiment         Compatibility alias for --qvrf-cpp-sender
  --log-dir DIR                 Log directory
  --help                        Show this help
EOF
}

apply_preset() {
  local name="$1"
  case "${name}" in
    legacy128x2x24) CODEC="mbt"; FPS="24"; CODEC_PROFILE="${name}"; CODEC_SIZE="128"; DISPLAY_SIZE="256"; CHUNKS_PER_FRAME="2"; FEC_DATA_CHUNKS="0"; CHUNK_RATE_HZ="48" ;;
    codec192x4x12) CODEC="mbt"; FPS="12"; CODEC_PROFILE="${name}"; CODEC_SIZE="192"; DISPLAY_SIZE="512"; CHUNKS_PER_FRAME="4"; FEC_DATA_CHUNKS="0"; CHUNK_RATE_HZ="48" ;;
    codec256x4x12) CODEC="mbt"; FPS="12"; CODEC_PROFILE="${name}"; CODEC_SIZE="256"; DISPLAY_SIZE="512"; CHUNKS_PER_FRAME="4"; FEC_DATA_CHUNKS="0"; CHUNK_RATE_HZ="48" ;;
    codec256x6x8) CODEC="mbt"; FPS="8"; CODEC_PROFILE="${name}"; CODEC_SIZE="256"; DISPLAY_SIZE="512"; CHUNKS_PER_FRAME="6"; FEC_DATA_CHUNKS="0"; CHUNK_RATE_HZ="48" ;;
    codec320x8x6) CODEC="mbt"; FPS="6"; CODEC_PROFILE="${name}"; CODEC_SIZE="320"; DISPLAY_SIZE="512"; CHUNKS_PER_FRAME="8"; FEC_DATA_CHUNKS="0"; CHUNK_RATE_HZ="48" ;;
    offline320x9x5) CODEC="mbt"; FPS="5"; CODEC_PROFILE="codec320x8x6"; CODEC_SIZE="320"; DISPLAY_SIZE="512"; CHUNKS_PER_FRAME="9"; FEC_DATA_CHUNKS="0"; CHUNK_RATE_HZ="45"; MAX_QUEUE_CHUNKS="18" ;;
    codec448x9x5) CODEC="mbt"; FPS="5"; CODEC_PROFILE="${name}"; CODEC_SIZE="448"; DISPLAY_SIZE="512"; CHUNKS_PER_FRAME="9"; FEC_DATA_CHUNKS="0"; CHUNK_RATE_HZ="49" ;;
    codec512x7x6) CODEC="mbt"; FPS="6"; CODEC_PROFILE="codec448x9x5"; CODEC_SIZE="512"; DISPLAY_SIZE="512"; CHUNKS_PER_FRAME="7"; FEC_DATA_CHUNKS="0"; CHUNK_RATE_HZ="48"; MAX_QUEUE_CHUNKS="21" ;;
    qvrf192x1x24_lowlat) CODEC="msssim_qvrf"; FPS="24"; CODEC_PROFILE="codec192x4x12"; CODEC_SIZE="192"; DISPLAY_SIZE="384"; CHUNKS_PER_FRAME="1"; BUDGET="280"; CHUNK_RATE_HZ="24"; CAMERA_ROI_MODE="fixed"; ROI_SIZE="640"; EXPOSURE_US="8000" ;;
    qvrf192x2x24) CODEC="msssim_qvrf"; FPS="24"; CODEC_PROFILE="codec192x4x12"; CODEC_SIZE="192"; DISPLAY_SIZE="384"; CHUNKS_PER_FRAME="2"; BUDGET="560"; CHUNK_RATE_HZ="48"; MAX_QUEUE_CHUNKS="16" ;;
    qvrf320x6x8) CODEC="msssim_qvrf"; FPS="8"; CODEC_PROFILE="codec320x8x6"; CODEC_SIZE="320"; DISPLAY_SIZE="640"; CHUNKS_PER_FRAME="6"; BUDGET="1600"; CHUNK_RATE_HZ="48" ;;
    qvrf320x4x12) CODEC="msssim_qvrf"; FPS="12"; CODEC_PROFILE="codec320x8x6"; CODEC_SIZE="320"; DISPLAY_SIZE="640"; CHUNKS_PER_FRAME="4"; BUDGET="1120"; CHUNK_RATE_HZ="48" ;;
    qvrf448x4x12) CODEC="msssim_qvrf"; FPS="12"; CODEC_PROFILE="codec448x9x5"; CODEC_SIZE="448"; DISPLAY_SIZE="640"; CHUNKS_PER_FRAME="4"; BUDGET="1120"; CHUNK_RATE_HZ="48" ;;
    qvrf448x6x8) CODEC="msssim_qvrf"; FPS="8"; CODEC_PROFILE="codec448x9x5"; CODEC_SIZE="448"; DISPLAY_SIZE="640"; CHUNKS_PER_FRAME="6"; BUDGET="1600"; CHUNK_RATE_HZ="48" ;;
    *) echo "Unknown preset: ${name}" >&2; usage; exit 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --codec) CODEC="$2"; shift 2 ;;
    --transport) TRANSPORT="$2"; shift 2 ;;
    --offline-debug) TRANSPORT="offline-debug"; shift ;;
    --preset|--mode) PRESET="$2"; apply_preset "$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --camera-frames) CAMERA_FRAMES="$2"; shift 2 ;;
    --camera-index) CAMERA_INDEX="$2"; shift 2 ;;
    --roi-size) ROI_SIZE="$2"; CAMERA_ROI_MODE="fixed"; shift 2 ;;
    --camera-roi-mode|--roi-mode) CAMERA_ROI_MODE="$2"; shift 2 ;;
    --auto-square-roi) CAMERA_ROI_MODE="max-square"; shift ;;
    --camera-fps) CAMERA_FPS="$2"; shift 2 ;;
    --exposure-us) EXPOSURE_US="$2"; shift 2 ;;
    --fps|--send-fps) FPS="$2"; shift 2 ;;
    --codec-profile) CODEC_PROFILE="$2"; shift 2 ;;
    --codec-size) CODEC_SIZE="$2"; shift 2 ;;
    --display-size) DISPLAY_SIZE="$2"; DISPLAY_SIZE_SET="1"; shift 2 ;;
    --chunks-per-frame) CHUNKS_PER_FRAME="$2"; shift 2 ;;
    --fec-data-chunks) FEC_DATA_CHUNKS="$2"; shift 2 ;;
    --chunk-rate-hz) CHUNK_RATE_HZ="$2"; shift 2 ;;
    --max-queue-chunks) MAX_QUEUE_CHUNKS="$2"; shift 2 ;;
    --prebuffer-chunks) PREBUFFER_CHUNKS="$2"; shift 2 ;;
    --tail-flush-chunks) TAIL_FLUSH_CHUNKS="$2"; shift 2 ;;
    --chunk-order) CHUNK_ORDER="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --gain) GAIN="$2"; shift 2 ;;
    --tx-device) TX_DEVICE="$2"; shift 2 ;;
    --tx-ga-backend) TX_GA_BACKEND="$2"; shift 2 ;;
    --tx-trt-engine) TX_TRT_ENGINE="$2"; shift 2 ;;
    --tx-trt-device) TX_TRT_DEVICE="$2"; shift 2 ;;
    --rx-backend) RX_BACKEND="$2"; shift 2 ;;
    --rx-gs-backend) RX_GS_BACKEND="$2"; shift 2 ;;
    --rx-gs-trt-engine) RX_GS_TRT_ENGINE="$2"; shift 2 ;;
    --rx-gs-trt-device) RX_GS_TRT_DEVICE="$2"; shift 2 ;;
    --rx-fused-sr-trt-engine) RX_FUSED_SR_TRT_ENGINE="$2"; shift 2 ;;
    --rx-fused-sr-trt-device) RX_FUSED_SR_TRT_DEVICE="$2"; shift 2 ;;
    --torch-device) TORCH_DEVICE="$2"; shift 2 ;;
    --tx-torch-device) TX_TORCH_DEVICE="$2"; shift 2 ;;
    --gui-torch-device|--rx-torch-device) GUI_TORCH_DEVICE="$2"; shift 2 ;;
    --mqtt-host) MQTT_HOST="$2"; shift 2 ;;
    --mqtt-port) MQTT_PORT="$2"; shift 2 ;;
    --client-id) CLIENT_ID="$2"; shift 2 ;;
    --serial-port) SERIAL_PORT="$2"; shift 2 ;;
    --baudrate) BAUDRATE="$2"; shift 2 ;;
    --no-serial-wait) SERIAL_WAIT="0"; shift ;;
    --serial-wait) SERIAL_WAIT="1"; shift ;;
    --ipc-host) IPC_HOST="$2"; shift 2 ;;
    --ipc-port) IPC_PORT="$2"; shift 2 ;;
    --shm-name) SHM_NAME="$2"; shift 2 ;;
    --enable-sr) ENABLE_SR="1"; SR_BACKEND="msa"; shift ;;
    --sr-backend) SR_BACKEND="$2"; shift 2 ;;
    --sr-engine) SR_ENGINE="$2"; shift 2 ;;
    --sr-trt-engine) SR_TRT_ENGINE="$2"; shift 2 ;;
    --sr-trt-device) SR_TRT_DEVICE="$2"; shift 2 ;;
    --sr-scale) SR_SCALE="$2"; shift 2 ;;
    --realesr-model) REALESR_MODEL="$2"; shift 2 ;;
    --rlfn-model) RLFN_MODEL="$2"; shift 2 ;;
    --debug-rx-chunks) DEBUG_RX_CHUNKS="1"; shift ;;
    --qvrf-cpp-sender|--qvrf-cpp-experiment) QVRF_CPP_SENDER="1"; shift ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

case "${CODEC}" in mbt|msssim_qvrf) ;; *) echo "Unknown codec: ${CODEC}" >&2; exit 1 ;; esac
case "${TRANSPORT}" in serial|offline-debug) ;; *) echo "Unknown transport: ${TRANSPORT}" >&2; exit 1 ;; esac
case "${CAMERA_ROI_MODE}" in fixed|max-square) ;; *) echo "Unknown camera ROI mode: ${CAMERA_ROI_MODE}" >&2; exit 1 ;; esac
case "${TX_GA_BACKEND}" in openvino|tensorrt) ;; *) echo "Unknown --tx-ga-backend: ${TX_GA_BACKEND}" >&2; exit 1 ;; esac
if [[ "${TRANSPORT}" == "offline-debug" && -z "${PRESET}" && "${CODEC}" == "mbt" ]]; then
  PRESET="offline320x9x5"
  apply_preset "${PRESET}"
fi
if [[ "${REALESR_MODEL}" != /* ]]; then
  REALESR_MODEL="${ROOT}/${REALESR_MODEL}"
fi
if [[ "${RLFN_MODEL}" != /* ]]; then
  RLFN_MODEL="${ROOT}/${RLFN_MODEL}"
fi
if [[ -n "${RX_GS_TRT_ENGINE}" && "${RX_GS_TRT_ENGINE}" != /* ]]; then
  RX_GS_TRT_ENGINE="${ROOT}/${RX_GS_TRT_ENGINE}"
fi
if [[ -n "${RX_FUSED_SR_TRT_ENGINE}" && "${RX_FUSED_SR_TRT_ENGINE}" != /* ]]; then
  RX_FUSED_SR_TRT_ENGINE="${ROOT}/${RX_FUSED_SR_TRT_ENGINE}"
fi
if [[ -n "${SR_TRT_ENGINE}" && "${SR_TRT_ENGINE}" != /* ]]; then
  SR_TRT_ENGINE="${ROOT}/${SR_TRT_ENGINE}"
fi
case "${RX_GS_BACKEND}" in ""|cuda|openvino|tensorrt|cpu) ;; *) echo "Unknown --rx-gs-backend: ${RX_GS_BACKEND}" >&2; exit 1 ;; esac
case "${SR_BACKEND}" in none|msa|realesr|rlfn|rlfn_trt) ;; *) echo "Unknown --sr-backend: ${SR_BACKEND}" >&2; exit 1 ;; esac
case "${SR_ENGINE}" in torch|tensorrt) ;; *) echo "Unknown --sr-engine: ${SR_ENGINE}" >&2; exit 1 ;; esac
if [[ "${SR_BACKEND}" == "rlfn_trt" ]]; then
  SR_BACKEND="rlfn"
  SR_ENGINE="tensorrt"
fi
if [[ "${ENABLE_SR}" == "1" && "${SR_BACKEND}" != "msa" ]]; then
  echo "--enable-sr is a compatibility shortcut for --sr-backend msa" >&2
  exit 1
fi
if [[ "${SR_ENGINE}" == "tensorrt" && "${SR_BACKEND}" != "rlfn" ]]; then
  echo "--sr-engine tensorrt is only valid with --sr-backend rlfn/rlfn_trt" >&2
  exit 1
fi
if [[ "${SR_ENGINE}" == "tensorrt" && "${CODEC_SIZE}" != "448" ]]; then
  echo "--sr-engine tensorrt fixed RLFN engine currently supports only 448->896" >&2
  exit 1
fi
if [[ "${SR_BACKEND}" == "msa" && "${CODEC_SIZE}" != "128" ]]; then
  echo "--sr-backend msa / --enable-sr is only valid for 128->256" >&2
  exit 1
fi
if [[ "${SR_BACKEND}" == "msa" && "${SR_SCALE}" != "2" ]]; then
  echo "--sr-backend msa supports only --sr-scale 2" >&2
  exit 1
fi
if [[ "${SR_BACKEND}" == "realesr" ]]; then
  if [[ "${SR_SCALE}" != "2" ]]; then
    echo "--sr-backend realesr currently supports only --sr-scale 2" >&2
    exit 1
  fi
  if [[ "${DISPLAY_SIZE_SET}" == "0" ]]; then
    DISPLAY_SIZE="$((CODEC_SIZE * SR_SCALE))"
  fi
  if [[ ! -f "${REALESR_MODEL}" ]]; then
    echo "Missing RealESR model: ${REALESR_MODEL}" >&2
    echo "Expected realesr-general-x4v3.pth; pass --realesr-model PATH." >&2
    exit 1
  fi
fi
if [[ "${SR_BACKEND}" == "rlfn" ]]; then
  if [[ "${SR_SCALE}" != "2" ]]; then
    echo "--sr-backend rlfn currently supports only --sr-scale 2" >&2
    exit 1
  fi
  if [[ "${DISPLAY_SIZE_SET}" == "0" ]]; then
    DISPLAY_SIZE="$((CODEC_SIZE * SR_SCALE))"
  fi
  if [[ ! -f "${RLFN_MODEL}" ]]; then
    echo "Missing RLFN model: ${RLFN_MODEL}" >&2
    echo "Expected rlfn_s_x2.pth; pass --rlfn-model PATH." >&2
    exit 1
  fi
fi
if [[ "${SR_ENGINE}" == "tensorrt" ]]; then
  if [[ -z "${SR_TRT_ENGINE}" && -z "${RX_FUSED_SR_TRT_ENGINE}" ]]; then
    echo "--sr-trt-engine is required when --sr-engine tensorrt" >&2
    exit 1
  fi
  if [[ -n "${SR_TRT_ENGINE}" && ! -f "${SR_TRT_ENGINE}" ]]; then
    echo "Missing RLFN TensorRT SR engine: ${SR_TRT_ENGINE}" >&2
    exit 1
  fi
fi
if [[ -n "${RX_FUSED_SR_TRT_ENGINE}" ]]; then
  if [[ "${CODEC}" != "msssim_qvrf" || "${CODEC_SIZE}" != "448" ]]; then
    echo "--rx-fused-sr-trt-engine currently requires --codec msssim_qvrf and codec size 448" >&2
    exit 1
  fi
  if [[ ! -f "${RX_FUSED_SR_TRT_ENGINE}" ]]; then
    echo "Missing fused QVRF g_s + RLFN TensorRT SR engine: ${RX_FUSED_SR_TRT_ENGINE}" >&2
    exit 1
  fi
fi
if [[ "${TX_GA_BACKEND}" == "tensorrt" && -z "${TX_TRT_ENGINE}" ]]; then
  echo "--tx-trt-engine is required when --tx-ga-backend tensorrt" >&2
  exit 1
fi
if [[ "${RX_GS_BACKEND}" == "tensorrt" ]]; then
  if [[ "${CODEC}" != "msssim_qvrf" ]]; then
    echo "--rx-gs-backend tensorrt is only valid with --codec msssim_qvrf" >&2
    exit 1
  fi
  if [[ "${CODEC_SIZE}" != "448" ]]; then
    echo "--rx-gs-backend tensorrt fixed engine currently supports only qvrf448x6x8 / codec size 448" >&2
    exit 1
  fi
  if [[ -z "${RX_GS_TRT_ENGINE}" ]]; then
    echo "--rx-gs-trt-engine is required when --rx-gs-backend tensorrt" >&2
    exit 1
  fi
  if [[ ! -f "${RX_GS_TRT_ENGINE}" ]]; then
    echo "Missing receiver QVRF g_s TensorRT engine: ${RX_GS_TRT_ENGINE}" >&2
    exit 1
  fi
fi
if [[ -z "${TX_TORCH_DEVICE}" ]]; then TX_TORCH_DEVICE="${TORCH_DEVICE}"; fi
if [[ -z "${GUI_TORCH_DEVICE}" ]]; then GUI_TORCH_DEVICE="${TORCH_DEVICE}"; fi
if [[ "${QVRF_CPP_SENDER}" == "1" && "${CODEC}" != "msssim_qvrf" ]]; then
  echo "--qvrf-cpp-sender is only valid with --codec msssim_qvrf" >&2
  exit 1
fi
if [[ "${CODEC}" == "msssim_qvrf" && "${QVRF_CPP_SENDER}" != "1" ]]; then
  if [[ "${TX_GA_BACKEND}" != "openvino" ]]; then
    echo "--tx-ga-backend only applies to the C++ sender. Default QVRF uses Python/Torch; use --qvrf-cpp-sender for C++ QVRF." >&2
    exit 1
  fi
  echo "Note: QVRF uses the Python/Torch sender and decoder; --tx-device/OpenVINO does not apply." >&2
elif [[ "${QVRF_CPP_SENDER}" == "1" ]]; then
  if [[ "${RX_GS_BACKEND}" == "tensorrt" ]]; then
    echo "Note: using C++ QVRF sender with g_a backend=${TX_GA_BACKEND}; hs_backend=1 GUI decode uses OpenVINO FP32 CPU h_s, CPU entropy/Gaussian, and TensorRT g_s on cuda:${RX_GS_TRT_DEVICE}. Default QVRF remains Python/Torch." >&2
  elif [[ "${RX_BACKEND}" == "cuda" && "${GUI_TORCH_DEVICE}" == cuda* ]]; then
    echo "Note: using C++ QVRF sender with g_a backend=${TX_GA_BACKEND}; hs_backend=1 GUI decode uses OpenVINO FP32 CPU h_s, CPU entropy/Gaussian, and CUDA g_s on ${GUI_TORCH_DEVICE}. Default QVRF remains Python/Torch." >&2
  else
    echo "Note: using C++ QVRF sender with g_a backend=${TX_GA_BACKEND}; hs_backend=1 GUI decode uses OpenVINO FP32 CPU h_s, CPU entropy/Gaussian, and CPU-safe QVRF synthesis. Default QVRF remains Python/Torch." >&2
  fi
fi

CAM_BIN="${ROOT}/onboard/build/rm_camera_capture"
CPP_TX_BIN="${ROOT}/onboard/build/rm_compress_cli"
QVRF_SENDER="${ROOT}/experiments/sender_msssim_qvrf_v2.py"
MODEL="${ROOT}/models/mbt_g_a.xml"
if [[ "${CODEC_SIZE}" != "128" ]]; then
  MODEL="${ROOT}/models/mbt_g_a_${CODEC_SIZE}.xml"
fi
_VENV_PY_VER=$(ls "${ROOT}/client/.venv/lib/" | grep '^python3\.' | head -1)
OV_LIBS="${ROOT}/client/.venv/lib/${_VENV_PY_VER}/site-packages/openvino/libs"
PYQT5_PLUGINS="${ROOT}/client/.venv/lib/${_VENV_PY_VER}/site-packages/PyQt5/Qt5/plugins"

RX_LOG="${LOG_DIR}/rm_stream_rx.log"
GUI_LOG="${LOG_DIR}/rm_stream_gui.log"
CAM_LOG="${LOG_DIR}/rm_stream_camera.log"
TX_LOG="${LOG_DIR}/rm_stream_sender.log"

if [[ ! -x "${CAM_BIN}" || ! -x "${CPP_TX_BIN}" ]]; then
  echo "Missing onboard binaries. Build first:" >&2
  echo "  cmake -S onboard -B onboard/build && cmake --build onboard/build -j\$(nproc)" >&2
  exit 1
fi
if [[ "${CODEC}" == "mbt" && ! -f "${MODEL}" ]]; then
  echo "Missing MBT model: ${MODEL}" >&2
  exit 1
fi
if [[ "${CODEC}" == "msssim_qvrf" && ! -f "${QVRF_SENDER}" ]]; then
  echo "Missing QVRF sender: ${QVRF_SENDER}" >&2
  exit 1
fi

CONFLICT_RE='[p]ython -m rm_stream.gui|[r]m_compress_cli|[r]m_camera_capture|[p]ython .*[s]ender_msssim_qvrf'
if pgrep -af "${CONFLICT_RE}" >/dev/null; then
  echo "Another stream process appears to be running:" >&2
  pgrep -af "${CONFLICT_RE}" >&2
  echo "Stop it first to avoid receiver, serial, or camera conflicts." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
rm -f "${RX_LOG}" "${GUI_LOG}" "${CAM_LOG}" "${TX_LOG}" "/dev/shm${SHM_NAME}" 2>/dev/null || true

cat <<EOF
Resolved stream profile:
  codec        : ${CODEC}
  transport    : ${TRANSPORT}
  preset       : ${PRESET:-custom}
  codec size   : ${CODEC_SIZE}x${CODEC_SIZE}
  display size : ${DISPLAY_SIZE}x${DISPLAY_SIZE}
  fps          : ${FPS}
  chunks/frame : ${CHUNKS_PER_FRAME}
  chunk rate   : ${CHUNK_RATE_HZ} Hz
  prebuffer    : ${PREBUFFER_CHUNKS} chunks
  tail flush   : ${TAIL_FLUSH_CHUNKS} chunks
  camera       : index=${CAMERA_INDEX}, roi_mode=${CAMERA_ROI_MODE}, roi=${ROI_SIZE}, fps=${CAMERA_FPS}, exposure=${EXPOSURE_US}us
  torch        : tx=${TX_TORCH_DEVICE}, gui=${GUI_TORCH_DEVICE}
  sender g_a   : backend=${TX_GA_BACKEND}, device=$([[ "${TX_GA_BACKEND}" == "tensorrt" ]] && echo "cuda:${TX_TRT_DEVICE}, engine=${TX_TRT_ENGINE}" || echo "${TX_DEVICE}")
  h_a/h_s      : OpenVINO FP32 CPU (hard requirement)
  entropy      : CPU/host CompressAI CDF/rANS contract
  rx g_s       : backend=${RX_GS_BACKEND:-derived-from-rx-backend}, device=$([[ "${RX_GS_BACKEND}" == "tensorrt" ]] && echo "cuda:${RX_GS_TRT_DEVICE}, engine=${RX_GS_TRT_ENGINE}" || echo "${GUI_TORCH_DEVICE:-cpu}")
  rx fused sr  : $([[ -n "${RX_FUSED_SR_TRT_ENGINE}" ]] && echo "device=cuda:${RX_FUSED_SR_TRT_DEVICE}, engine=${RX_FUSED_SR_TRT_ENGINE}" || echo "disabled")
  sr backend   : ${SR_BACKEND}, engine=${SR_ENGINE}, scale=${SR_SCALE}$([[ "${SR_ENGINE}" == "tensorrt" ]] && echo ", device=cuda:${SR_TRT_DEVICE}, engine=${SR_TRT_ENGINE}")
  qvrf cpp tx  : ${QVRF_CPP_SENDER}
  shm          : ${SHM_NAME}
  logs         : ${LOG_DIR}
EOF

GUI_PID=""
CAM_PID=""
TX_PID=""

kill_tree() {
  local sig="$1"
  local pid="$2"
  [[ -z "${pid}" ]] && return 0
  kill -0 "${pid}" 2>/dev/null || return 0
  local child
  while read -r child; do
    [[ -n "${child}" ]] && kill_tree "${sig}" "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill "-${sig}" "${pid}" 2>/dev/null || true
}

cleanup() {
  set +e
  echo
  echo "Stopping camera stream..."
  for pid in "${TX_PID}" "${CAM_PID}" "${GUI_PID}"; do
    kill_tree TERM "${pid}"
  done
  sleep 0.5
  for pid in "${TX_PID}" "${CAM_PID}" "${GUI_PID}"; do
    kill_tree KILL "${pid}"
    [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null
  done
  echo "Logs:"
  echo "  RX : ${RX_LOG}"
  echo "  GUI: ${GUI_LOG}"
  echo "  CAM: ${CAM_LOG}"
  echo "  TX : ${TX_LOG}"
}
trap cleanup EXIT INT TERM

echo "Starting GUI receiver..."
GUI_CMD=(
  uv run --directory "${ROOT}/client" python -m rm_stream.gui
  --codec "${CODEC}"
  --codec-profile "${CODEC_PROFILE}"
  --codec-size "${CODEC_SIZE}"
  --display-size "${DISPLAY_SIZE}"
  --rx-backend "${RX_BACKEND}"
)
if [[ "${TRANSPORT}" == "offline-debug" ]]; then
  GUI_CMD+=(--receive-mode ipc --ipc-host "${IPC_HOST}" --ipc-port "${IPC_PORT}")
else
  GUI_CMD+=(--receive-mode mqtt --mqtt-host "${MQTT_HOST}" --mqtt-port "${MQTT_PORT}" --client-id "${CLIENT_ID}")
fi
if [[ -n "${GUI_TORCH_DEVICE}" ]]; then
  GUI_CMD+=(--torch-device "${GUI_TORCH_DEVICE}")
fi
if [[ -n "${RX_GS_BACKEND}" ]]; then
  GUI_CMD+=(--rx-gs-backend "${RX_GS_BACKEND}")
fi
if [[ "${RX_GS_BACKEND}" == "tensorrt" ]]; then
  GUI_CMD+=(--rx-gs-trt-engine "${RX_GS_TRT_ENGINE}" --rx-gs-trt-device "${RX_GS_TRT_DEVICE}")
fi
if [[ "${ENABLE_SR}" == "1" ]]; then
  GUI_CMD+=(--enable-sr)
fi
GUI_CMD+=(--sr-backend "${SR_BACKEND}" --sr-scale "${SR_SCALE}")
if [[ "${SR_BACKEND}" == "realesr" ]]; then
  GUI_CMD+=(--realesr-model "${REALESR_MODEL}")
fi
if [[ "${SR_BACKEND}" == "rlfn" ]]; then
  GUI_CMD+=(--rlfn-model "${RLFN_MODEL}")
fi
if [[ "${SR_ENGINE}" == "tensorrt" ]]; then
  GUI_CMD+=(--sr-engine tensorrt --sr-trt-device "${SR_TRT_DEVICE}")
  [[ -n "${SR_TRT_ENGINE}" ]] && GUI_CMD+=(--sr-trt-engine "${SR_TRT_ENGINE}")
fi
if [[ -n "${RX_FUSED_SR_TRT_ENGINE}" ]]; then
  GUI_CMD+=(--rx-fused-sr-trt-engine "${RX_FUSED_SR_TRT_ENGINE}" --rx-fused-sr-trt-device "${RX_FUSED_SR_TRT_DEVICE}")
fi
(
  cd "${ROOT}"
  RM_STREAM_DEBUG_RX_LOG="${RX_LOG}" \
  RM_STREAM_DEBUG_RX_CHUNKS="${DEBUG_RX_CHUNKS}" \
  RM_STREAM_BACKEND="${RX_BACKEND}" \
  RM_STREAM_RX_GS_BACKEND="${RX_GS_BACKEND}" \
  RM_STREAM_RX_GS_TRT_ENGINE="${RX_GS_TRT_ENGINE}" \
  RM_STREAM_RX_GS_TRT_DEVICE="${RX_GS_TRT_DEVICE}" \
  RM_STREAM_RX_FUSED_SR_TRT_ENGINE="${RX_FUSED_SR_TRT_ENGINE}" \
  RM_STREAM_RX_FUSED_SR_TRT_DEVICE="${RX_FUSED_SR_TRT_DEVICE}" \
  RM_STREAM_SR_ENGINE="${SR_ENGINE}" \
  RM_STREAM_SR_TRT_ENGINE="${SR_TRT_ENGINE}" \
  RM_STREAM_SR_TRT_DEVICE="${SR_TRT_DEVICE}" \
  RM_STREAM_TORCH_DEVICE="${GUI_TORCH_DEVICE}" \
  QT_QPA_PLATFORM_PLUGIN_PATH="${PYQT5_PLUGINS}/platforms" \
  QT_PLUGIN_PATH="${PYQT5_PLUGINS}" \
  "${GUI_CMD[@]}"
) >"${GUI_LOG}" 2>&1 &
GUI_PID=$!
echo "  GUI pid=${GUI_PID}, log=${GUI_LOG}"
if [[ "${TRANSPORT}" == "offline-debug" ]]; then
  echo "Waiting for GUI IPC listener ${IPC_HOST}:${IPC_PORT}..."
  for i in $(seq 1 30); do
    if ! kill -0 "${GUI_PID}" 2>/dev/null; then
      echo "GUI exited before IPC listener became ready:" >&2
      tail -120 "${GUI_LOG}" >&2 || true
      exit 1
    fi
    if python3 - "${IPC_HOST}" "${IPC_PORT}" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.create_connection((host, port), timeout=0.2):
    pass
PY
    then
      echo "  GUI IPC ready"
      break
    fi
    sleep 0.5
    if [[ "${i}" == "30" ]]; then
      echo "Timed out waiting for GUI IPC listener ${IPC_HOST}:${IPC_PORT}" >&2
      tail -120 "${GUI_LOG}" >&2 || true
      exit 1
    fi
  done
else
  sleep 3
fi

echo "Starting Hikvision camera capture..."
CAM_CMD=(
  "${CAM_BIN}"
  --device-index "${CAMERA_INDEX}"
  --roi-mode "${CAMERA_ROI_MODE}"
  --fps "${CAMERA_FPS}"
  --exposure-us "${EXPOSURE_US}"
  --shm-name "${SHM_NAME}"
  --slots 4
)
if [[ "${CAMERA_ROI_MODE}" == "fixed" ]]; then
  CAM_CMD+=(--roi-size "${ROI_SIZE}")
fi
if [[ "${CAMERA_FRAMES}" != "0" ]]; then
  CAM_CMD+=(--frames "${CAMERA_FRAMES}")
fi
(
  attempt=0
  trap 'exit 0' TERM INT
  while true; do
    LD_LIBRARY_PATH="${OV_LIBS}:${ROOT}/onboard/build:${LD_LIBRARY_PATH:-}" "${CAM_CMD[@]}"
    rc=$?
    [[ "${CAMERA_FRAMES}" != "0" ]] && exit "${rc}"
    attempt=$((attempt + 1))
    echo "camera_capture exited rc=${rc}; restarting in 2s attempt=${attempt}" >&2
    sleep 2
  done
) >"${CAM_LOG}" 2>&1 &
CAM_PID=$!
echo "  CAM pid=${CAM_PID}, log=${CAM_LOG}"

echo "Waiting for camera shared memory..."
for i in $(seq 1 15); do
  [[ -e "/dev/shm${SHM_NAME}" ]] && break
  if ! kill -0 "${CAM_PID}" 2>/dev/null; then
    echo "Camera process exited during startup:" >&2
    tail -80 "${CAM_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ ! -e "/dev/shm${SHM_NAME}" ]]; then
  echo "Camera shared memory not found after 15s: /dev/shm${SHM_NAME}" >&2
  tail -80 "${CAM_LOG}" >&2 || true
  exit 1
fi

echo "Starting ${CODEC} sender..."
if [[ "${CODEC}" == "mbt" || "${QVRF_CPP_SENDER}" == "1" ]]; then
  CPP_CODEC="mbt"
  if [[ "${QVRF_CPP_SENDER}" == "1" ]]; then
    CPP_CODEC="msssim_qvrf"
  fi
  TX_CMD=(
    "${CPP_TX_BIN}"
    --shm-input
    --shm-name "${SHM_NAME}"
    --codec "${CPP_CODEC}"
    -d "${TX_DEVICE}"
    --tx-ga-backend "${TX_GA_BACKEND}"
    -m "${MODEL}"
    --fps "${FPS}"
    --codec-size "${CODEC_SIZE}"
    --chunks-per-frame "${CHUNKS_PER_FRAME}"
    --fec-data-chunks "${FEC_DATA_CHUNKS}"
    --prebuffer-chunks "${PREBUFFER_CHUNKS}"
    --tail-flush-chunks "${TAIL_FLUSH_CHUNKS}"
    --chunk-rate-hz "${CHUNK_RATE_HZ}"
    --max-queue-chunks "${MAX_QUEUE_CHUNKS}"
    --chunk-order "${CHUNK_ORDER}"
    --profile
  )
  if [[ "${QVRF_CPP_SENDER}" == "1" ]]; then
    TX_CMD+=(--qvrf-cpp-sender)
  fi
  if [[ "${TX_GA_BACKEND}" == "tensorrt" ]]; then
    TX_CMD+=(--tx-trt-engine "${TX_TRT_ENGINE}" --tx-trt-device "${TX_TRT_DEVICE}")
  fi
  if [[ "${TRANSPORT}" == "offline-debug" ]]; then
    TX_CMD+=(--ipc-host "${IPC_HOST}" --ipc-port "${IPC_PORT}")
  else
    TX_CMD+=(-p "${SERIAL_PORT}" -b "${BAUDRATE}" -r 1)
    [[ "${SERIAL_WAIT}" == "1" ]] && TX_CMD+=(--serial-wait)
  fi
  [[ "${FRAMES}" != "0" ]] && TX_CMD+=(-n "${FRAMES}")
  if [[ "${QVRF_CPP_SENDER}" == "1" ]]; then
    RM_QVRF_CPP_SENDER=1 \
      LD_LIBRARY_PATH="${OV_LIBS}:${ROOT}/onboard/build:${LD_LIBRARY_PATH:-}" \
      "${TX_CMD[@]}" >"${TX_LOG}" 2>&1 &
  else
    LD_LIBRARY_PATH="${OV_LIBS}:${ROOT}/onboard/build:${LD_LIBRARY_PATH:-}" \
      "${TX_CMD[@]}" >"${TX_LOG}" 2>&1 &
  fi
else
  TX_CMD=(
    uv run --directory "${ROOT}/client" python "${QVRF_SENDER}"
    --input shm
    --shm-name "${SHM_NAME}"
    --codec-size "${CODEC_SIZE}"
    --gain "${GAIN}"
    --budget "${BUDGET}"
    --chunks-per-frame "${CHUNKS_PER_FRAME}"
    --fps "${FPS}"
    --chunk-rate-hz "${CHUNK_RATE_HZ}"
    --max-queue-chunks "${MAX_QUEUE_CHUNKS}"
    --device "${TX_TORCH_DEVICE}"
  )
  [[ "${FRAMES}" != "0" ]] && TX_CMD+=(--frames "${FRAMES}")
  if [[ "${TRANSPORT}" == "offline-debug" ]]; then
    TX_CMD+=(--ipc-host "${IPC_HOST}" --ipc-port "${IPC_PORT}")
  else
    TX_CMD+=(--serial-port "${SERIAL_PORT}" --baudrate "${BAUDRATE}")
    [[ "${SERIAL_WAIT}" == "1" ]] && TX_CMD+=(--serial-wait)
  fi
  "${TX_CMD[@]}" >"${TX_LOG}" 2>&1 &
fi
TX_PID=$!
echo "  TX pid=${TX_PID}, log=${TX_LOG}"
echo
echo "Camera stream running. Press Ctrl+C to stop."
echo "  tail -f ${TX_LOG}"
echo "  tail -f ${RX_LOG}"

wait "${TX_PID}"
sleep 2
