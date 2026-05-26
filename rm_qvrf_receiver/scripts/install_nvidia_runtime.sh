#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/client/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "ERROR: client/.venv missing. Run ./install.sh first." >&2
  exit 1
fi

CUDA_TORCH_INDEX="${CUDA_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "Installing NVIDIA Python runtime packages."
echo "PyTorch index: ${CUDA_TORCH_INDEX}"
uv pip install --python "${PY}" --index-url "${CUDA_TORCH_INDEX}" torch torchvision
uv pip install --python "${PY}" "tensorrt>=10.0" "cuda-python" "cupy-cuda12x>=13.0"

echo
echo "NVIDIA runtime install complete."
echo "Run: ./scripts/check_env.sh"
