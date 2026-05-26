#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "RM QVRF sender installer"
echo "Root: ${ROOT}"

missing=0
for cmd in bash python3 cmake; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "WARN: missing command: ${cmd}"
    missing=1
  fi
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv install failed. Add ~/.local/bin to PATH and rerun."
  exit 1
fi

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  uv venv "${ROOT}/.venv" --python 3.11
fi
uv pip install --python "${ROOT}/.venv/bin/python" "openvino>=2024.0.0" "numpy" "opencv-python-headless>=4.8"

for f in \
  models/msssim_g_a_fp32.xml models/msssim_g_a_fp32.bin \
  models/msssim_h_a_fp32.xml models/msssim_h_a_fp32.bin \
  models/msssim_h_s_fp32.xml models/msssim_h_s_fp32.bin \
  models/msssim_cdfs.bin; do
  if [[ ! -f "${ROOT}/${f}" ]]; then
    echo "ERROR: missing ${f}"
    exit 1
  fi
done

arch="$(uname -m)"
if [[ "${arch}" == "x86_64" && -x "${ROOT}/bin/rm_compress_cli" && -x "${ROOT}/bin/rm_camera_capture" ]]; then
  echo "Using bundled x86_64 sender binaries."
else
  echo "Bundled binaries are unavailable for arch=${arch}; building from source."
  "${ROOT}/scripts/build_sender.sh"
fi

if [[ "${missing}" -ne 0 ]]; then
  echo "WARN: some optional build tools were missing. Re-run ./scripts/check_env.sh."
fi

echo
echo "Install complete."
echo "Next: ./scripts/check_env.sh"
