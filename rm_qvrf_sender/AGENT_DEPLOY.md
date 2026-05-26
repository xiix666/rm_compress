# Agent Deployment Guide: Sender

Goal: deploy sender on NUC or Jetson NX so it sends QVRF chunks over serial
0x0310. Follow this as an executable checklist.

## 1. Identify Platform

Run:

```bash
uname -m
lsb_release -a 2>/dev/null || cat /etc/os-release
find /dev -maxdepth 1 \( -name 'ttyUSB*' -o -name 'ttyACM*' -o -name 'ttyTHS*' \) -ls 2>/dev/null || true
```

Decision:

- `x86_64` NUC: prefer bundled `bin/rm_compress_cli` and `bin/rm_camera_capture`.
- `aarch64` Jetson/NX: run `./scripts/build_sender.sh` on target.
- If no serial device exists, stop and report hardware blocker.

## 2. Install

Python policy:

- Use `uv` only.
- Let `./install.sh` create and manage `./.venv`.
- Do not use global `pip`.
- If an extra Python helper dependency is required, install it with
  `uv pip install --python ./.venv/bin/python PACKAGE`.

```bash
cd rm_qvrf_sender
./install.sh
./scripts/check_env.sh
```

If x86_64 install reports missing binaries, rebuild:

```bash
./scripts/build_sender.sh
```

If Jetson/NX needs TensorRT sender g_a, build with:

```bash
ENABLE_TRT=1 ./scripts/build_sender.sh
```

TensorRT engines are not portable across machines. Do not assume an engine from
another laptop works on NX.

## 3. Camera SDK

`rm_camera_capture` requires Hikrobot MVS SDK. If the target does not build or
run the camera binary, find the SDK root containing `include/MvCameraControl.h`
and `lib/64/libMvCameraControl.so`, then rebuild:

```bash
MVS_ROOT=/path/to/MVS ./scripts/build_sender.sh
```

## 4. Frequency And Permissions

On Intel NUC:

```bash
sudo ./scripts/optimize_gpu_freq.sh
```

For serial:

```bash
sudo usermod -aG dialout "$USER"
```

If group membership changed, tell the user to log out and back in, or use a new
login shell before testing.

## 5. Functional Tests

Dry compression test:

```bash
FRAMES=60 ./scripts/run_dry_sender.sh qvrf192x2x24
```

Camera + serial short test:

```bash
./scripts/run_camera_sender.sh --preset qvrf192x2x24 --frames 60
```

448 short test:

```bash
./scripts/run_camera_sender.sh --preset qvrf448x6x8 --frames 60
```

Success criteria:

- camera starts and creates `/dev/shm/rm_camera_frames`
- serial opens at 921600; default `SERIAL_PORT=auto` should prefer
  `/dev/serial/by-id/*`, then `/dev/ttyUSB*`, then `/dev/ttyACM*`
- `Errors: 0`
- `Queue drops: 0 chunks`
- `TX underruns: 0 ticks`
- `Over budget: 0`

## 6. Production Commands

192 QVRF 24 FPS:

```bash
./scripts/run_camera_sender.sh --preset qvrf192x2x24 --frames 0
```

448 QVRF 8 FPS:

```bash
./scripts/run_camera_sender.sh --preset qvrf448x6x8 --frames 0
```

## 7. Failure Triage

- `serial permission denied`: user not in `dialout`, or another process owns the port.
- `auto opens the wrong adapter`: pass a stable `/dev/serial/by-id/...` path
  with `--serial-port`.
- `OpenVINO GPU.0 missing`: use `--tx-device CPU` only for diagnosis, or install GPU drivers.
- `rm_camera_capture target disabled`: MVS SDK path is wrong or missing.
- `TensorRT g_a engine failed`: rebuild engine on that exact sender or use OpenVINO g_a.
- receiver reports missing chunks but sender says no drops: diagnose the 0x0310/VTX/MQTT bridge.
