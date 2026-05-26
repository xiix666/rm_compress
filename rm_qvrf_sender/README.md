# RM QVRF Sender Distribution

This package deploys the sender side of the current RoboMaster low-bandwidth
video link:

camera -> shared memory -> C++ QVRF sender -> serial 0x0310

## Targets

- Intel NUC / x86_64 Ubuntu: bundled binaries can run directly after install.
- Jetson NX / aarch64 Ubuntu: build from source on the device with
  `./scripts/build_sender.sh`. TensorRT sender g_a is optional and requires a
  matching engine built on that Jetson.

## Install

Python helper tools in this package are managed only with `uv`. Do not create a
system Python environment or install dependencies with global `pip`.
`./install.sh` installs `uv` if needed, creates `./.venv`, and installs helper
dependencies into that local environment.

```bash
cd rm_qvrf_sender
./install.sh
./scripts/check_env.sh
```

For Intel NUC timing, lock CPU/iGPU frequency:

```bash
sudo ./scripts/optimize_gpu_freq.sh
```

Make sure the user can open the serial device:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing groups.

## Serial Port Selection

The default sender configuration uses:

```bash
SERIAL_PORT=auto
BAUDRATE=921600
```

With `auto`, the C++ sender scans serial devices every time it opens or
reconnects. It tries `/dev/serial/by-id/*` first, then `/dev/ttyUSB*`, then
`/dev/ttyACM*`. This handles unplug/replug cases where Linux renames the device
from `/dev/ttyUSB0` to `/dev/ttyUSB1`.

If there are multiple USB-UART adapters, pass a stable by-id path explicitly:

```bash
./scripts/run_camera_sender.sh --serial-port /dev/serial/by-id/usb-... --preset qvrf192x2x24
```

The baudrate is not auto-detected because this sender is a one-way 0x0310
transmitter with no UART handshake or reply. Keep `BAUDRATE=921600` unless the
bridge is intentionally configured otherwise.

## Run 192 QVRF 24 FPS

```bash
./scripts/run_camera_sender.sh \
  --preset qvrf192x2x24 \
  --frames 0 \
  --tx-device GPU.0
```

## Run 448 QVRF 8 FPS

```bash
./scripts/run_camera_sender.sh \
  --preset qvrf448x6x8 \
  --frames 0 \
  --tx-device GPU.0
```

## TensorRT g_a Sender

Only use this when the TensorRT engine was built for the sender machine:

```bash
./scripts/run_camera_sender.sh \
  --preset qvrf448x6x8 \
  --tx-ga-backend tensorrt \
  --tx-trt-engine models/engines/msssim_g_a_448_fp32_fixed.engine \
  --tx-trt-device 0
```

The checked-in TensorRT engine is machine-specific. If it fails to load, rebuild
or fall back to `--tx-ga-backend openvino`.

## Expected Healthy Sender Output

Look for:

- `Serial port /dev/serial/by-id/... opened at 921600 baud, 8N1` or another
  scanned serial path
- `QVRF C++: sender=enabled`
- `TX underruns: 0 ticks`
- `Queue drops: 0 chunks`
- `Over budget: 0`

For `qvrf192x2x24`, target is 24 FPS and 48 serial chunks/s. For `qvrf448x6x8`,
target is 8 FPS and 48 serial chunks/s.

## Important Rules

- Use the package-local uv environment: `./.venv/bin/python`.
- Do not run `pip install` globally on the sender; use `uv pip install --python ./.venv/bin/python ...` if an agent needs an extra helper package.
- `h_a/h_s` must stay OpenVINO FP32 CPU. Do not quantize or move them to GPU.
- Serial 0x0310 frames must carry exactly 300 bytes. The sender already does this.
- Use `--prebuffer-chunks 2` as the current low-latency default.
- Do not run multiple programs on the same serial port.
- On Jetson/NX, build locally; do not use x86_64 binaries.
