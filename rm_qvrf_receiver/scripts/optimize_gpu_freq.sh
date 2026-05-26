#!/usr/bin/env bash
# Lock Intel iGPU to max frequency for stable inference performance
# Requires sudo
set -euo pipefail

# Auto-detect Intel GPU (vendor 0x8086)
GPU_CARD=""
for card in /sys/class/drm/card*; do
    vendor=$(cat "$card/device/vendor" 2>/dev/null || echo "")
    if [ "$vendor" = "0x8086" ]; then
        GPU_CARD="$card"
        break
    fi
done

if [ -z "$GPU_CARD" ]; then
    echo "No Intel GPU found"
    exit 1
fi
echo "Found Intel GPU at $GPU_CARD"

MAX_FREQ=$(cat "$GPU_CARD/gt_max_freq_mhz" 2>/dev/null || echo "1500")
echo "Locking iGPU to ${MAX_FREQ} MHz..."
echo "$MAX_FREQ" | tee "$GPU_CARD/gt_min_freq_mhz" > /dev/null
echo "$MAX_FREQ" | tee "$GPU_CARD/gt_max_freq_mhz" > /dev/null
echo "$MAX_FREQ" | tee "$GPU_CARD/gt_boost_freq_mhz" > /dev/null
echo "iGPU locked to ${MAX_FREQ} MHz"

# Also set CPU governor to performance
if command -v cpupower &>/dev/null; then
    cpupower frequency-set -g performance 2>/dev/null && echo "CPU governor set to performance" || echo "CPU governor: cpupower failed"
else
    echo "cpupower not found, skipping CPU governor"
fi
