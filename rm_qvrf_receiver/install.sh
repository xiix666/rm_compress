#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "RM QVRF receiver installer"
echo "Root: ${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv install failed. Add ~/.local/bin to PATH and rerun."
  exit 1
fi

missing=0
for p in \
  client/src/rm_stream/gui/__main__.py \
  commu/src/rm_custom_client/proto/rm_custom_client_pb2.py \
  models/msssim_g_s_fp32.xml \
  models/msssim_h_s_fp32.xml \
  models/msssim_h_a_fp32.xml \
  models/msssim_cdfs.bin \
  models/rlfn_s_x2.pth; do
  if [[ ! -e "${ROOT}/${p}" ]]; then
    echo "ERROR: missing ${p}"
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

if [[ ! -x "${ROOT}/client/.venv/bin/python" ]]; then
  uv venv "${ROOT}/client/.venv" --python 3.11
fi
uv pip install --python "${ROOT}/client/.venv/bin/python" -e "${ROOT}/client"
uv pip install --python "${ROOT}/client/.venv/bin/python" -e "${ROOT}/commu"

echo
echo "Install complete."
echo "Next: ./scripts/check_env.sh"
