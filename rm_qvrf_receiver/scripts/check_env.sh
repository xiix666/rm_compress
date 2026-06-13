#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

source "${ROOT}/config/receiver.env"

echo "=== RM QVRF Receiver Environment ==="
echo "Host: $(hostname)"
echo "Arch: $(uname -m)"
echo "Kernel: $(uname -r)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv,noheader || true
else
  echo "nvidia-smi: MISSING"
fi

echo
echo "Python probes:"
if [[ -x "${ROOT}/client/.venv/bin/python" ]]; then
  PYTHONPATH="${ROOT}/client/src:${ROOT}/commu/src:${ROOT}/compress-ai-gray-minimal:${PYTHONPATH:-}" \
  "${ROOT}/client/.venv/bin/python" - <<'PY'
import socket
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch probe failed:", repr(exc))
try:
    import openvino as ov
    print("openvino_devices:", ov.Core().available_devices)
except Exception as exc:
    print("openvino probe failed:", repr(exc))
try:
    import tensorrt as trt
    print("tensorrt:", trt.__version__)
except Exception as exc:
    print("tensorrt probe failed:", repr(exc))
PY
else
  echo "client/.venv missing; run ./install.sh"
fi

echo
echo "MQTT TCP probe ${MQTT_HOST}:${MQTT_PORT}:"
python3 - "${MQTT_HOST}" "${MQTT_PORT}" <<'PY'
import socket, sys
host=sys.argv[1]; port=int(sys.argv[2])
try:
    s=socket.create_connection((host, port), timeout=2)
    s.close()
    print("OK")
except Exception as exc:
    print("FAIL", repr(exc))
PY

echo
echo "Models and engines:"
for f in \
  models/msssim_g_s_fp32.xml \
  models/msssim_h_s_fp32.xml \
  "${RX_FUSED_SR_TRT_ENGINE}"; do
  [[ -e "${ROOT}/${f}" ]] && echo "OK ${f}" || echo "MISSING ${f}"
done
