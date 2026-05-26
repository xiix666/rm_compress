#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/config/sender.env"

echo "=== RM QVRF Sender Environment ==="
echo "Host: $(hostname)"
echo "Arch: $(uname -m)"
echo "Kernel: $(uname -r)"

for cmd in python3 cmake; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "${cmd}: $(command -v "${cmd}")"
  else
    echo "${cmd}: MISSING"
  fi
done

echo
echo "Binaries:"
for f in bin/rm_compress_cli bin/rm_camera_capture bin/librmcompress.so; do
  if [[ -e "${ROOT}/${f}" ]]; then
    ls -l "${ROOT}/${f}"
  else
    echo "MISSING ${f}"
  fi
done

echo
echo "Serial:"
find /dev -maxdepth 1 \( -name 'ttyUSB*' -o -name 'ttyACM*' -o -name 'ttyTHS*' \) -ls 2>/dev/null || true
find /dev/serial/by-id -maxdepth 1 -type l -ls 2>/dev/null || true
if [[ "${SERIAL_PORT}" == "auto" ]]; then
  echo "Configured serial: auto (tries /dev/serial/by-id/*, /dev/ttyUSB*, /dev/ttyACM*)"
elif [[ -e "${SERIAL_PORT}" ]]; then
  echo "Configured serial exists: ${SERIAL_PORT}"
else
  echo "Configured serial missing: ${SERIAL_PORT}"
fi
echo "Groups: $(id -nG)"

echo
echo "OpenVINO devices:"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  "${ROOT}/.venv/bin/python" - <<'PY'
try:
    import openvino as ov
    core = ov.Core()
    print(core.available_devices)
except Exception as exc:
    print("OpenVINO probe failed:", repr(exc))
PY
else
  echo ".venv missing; run ./install.sh"
fi

echo
echo "GPU/CPU frequency:"
for card in /sys/class/drm/card*; do
  [[ -r "${card}/device/vendor" ]] || continue
  vendor="$(cat "${card}/device/vendor" 2>/dev/null || true)"
  if [[ "${vendor}" == "0x8086" ]]; then
    echo "Intel GPU ${card}: min=$(cat "${card}/gt_min_freq_mhz" 2>/dev/null || echo '?') max=$(cat "${card}/gt_max_freq_mhz" 2>/dev/null || echo '?') boost=$(cat "${card}/gt_boost_freq_mhz" 2>/dev/null || echo '?') cur=$(cat "${card}/gt_cur_freq_mhz" 2>/dev/null || echo '?')"
  fi
done
governors="$(for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do [[ -r "$f" ]] && cat "$f"; done | sort -u | tr '\n' ' ')"
echo "CPU governors: ${governors:-unknown}"

echo
echo "Model files:"
for f in msssim_g_a_fp32.xml msssim_h_a_fp32.xml msssim_h_s_fp32.xml msssim_cdfs.bin; do
  [[ -f "${ROOT}/models/${f}" ]] && echo "OK ${f}" || echo "MISSING ${f}"
done
