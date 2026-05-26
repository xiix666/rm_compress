#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/config/sender.env"

PRESET="${1:-qvrf192x2x24}"
case "${PRESET}" in
  qvrf192x1x24_lowlat) FPS=24; CODEC_SIZE=192; CHUNKS_PER_FRAME=1; CHUNK_RATE_HZ=24 ;;
  qvrf192x2x24) FPS=24; CODEC_SIZE=192; CHUNKS_PER_FRAME=2; CHUNK_RATE_HZ=48 ;;
  qvrf448x6x8) FPS=8; CODEC_SIZE=448; CHUNKS_PER_FRAME=6; CHUNK_RATE_HZ=48 ;;
  *) echo "Unknown preset: ${PRESET}" >&2; exit 1 ;;
esac

OV_LIBS=""
if [[ -d "${ROOT}/.venv/lib" ]]; then
  pyver="$(ls "${ROOT}/.venv/lib" | grep '^python3\\.' | head -1 || true)"
  [[ -n "${pyver}" ]] && OV_LIBS="${ROOT}/.venv/lib/${pyver}/site-packages/openvino/libs"
fi
export LD_LIBRARY_PATH="${OV_LIBS}:${ROOT}/bin:${LD_LIBRARY_PATH:-}"
export RM_QVRF_CPP_SENDER=1
export RM_COMPRESS_ROOT="${ROOT}"

exec "${ROOT}/bin/rm_compress_cli" \
  --codec msssim_qvrf --qvrf-cpp-sender \
  -d "${TX_DEVICE}" \
  --tx-ga-backend openvino \
  --fps "${FPS}" \
  --codec-size "${CODEC_SIZE}" \
  --chunks-per-frame "${CHUNKS_PER_FRAME}" \
  --chunk-rate-hz "${CHUNK_RATE_HZ}" \
  --prebuffer-chunks "${PREBUFFER_CHUNKS}" \
  --tail-flush-chunks "${TAIL_FLUSH_CHUNKS}" \
  -n "${FRAMES:-120}" \
  --dry-run --profile
