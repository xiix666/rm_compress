# Agent Deployment Guide: Receiver

Goal: deploy GUI receiver on an NVIDIA laptop and receive serial-forwarded
MQTT `CustomByteBlock` video from the sender.

## 1. Identify Hardware

Run:

```bash
uname -m
nvidia-smi
python3 --version
```

Decision:

- If no NVIDIA GPU or no driver, report blocker for TRT/CUDA path.
- If NVIDIA exists but TensorRT engine fails, use `--no-trt-fused` or rebuild engines.

## 2. Install

Python policy:

- Use `uv` only.
- Let `./install.sh` create and manage `./client/.venv`.
- Do not use global `pip`.
- Install CUDA/TensorRT Python runtime packages with
  `./scripts/install_nvidia_runtime.sh`.
- If an extra package is needed, install it with
  `uv pip install --python ./client/.venv/bin/python PACKAGE`.

```bash
cd rm_qvrf_receiver
./install.sh
./scripts/check_env.sh
```

Verify:

- `torch cuda: True`
- `cuda_device` is the expected NVIDIA GPU
- MQTT TCP probe is `OK`
- `models/msssim_h_s_fp32.xml` exists

If CUDA/TensorRT Python packages are missing:

```bash
./scripts/install_nvidia_runtime.sh
./scripts/check_env.sh
```

## 3. Probe MQTT Client ID

Do not assume client id `1`. Probe:

```bash
./client/.venv/bin/python ./scripts/probe_mqtt_client_ids.py \
  --host 192.168.12.1 \
  --port 3333 \
  --ids 1,2,3,101,102,103,104,105
```

Use the ID with `Success`. If none succeeds, the official client/robot state is
not ready; stop and report.

## 4. Start Receiver

448 fused TRT:

```bash
./scripts/run_receiver.sh --preset qvrf448x6x8 --client-id <SUCCESS_ID>
```

192 24 FPS:

```bash
./scripts/run_receiver.sh --preset qvrf192x2x24 --client-id <SUCCESS_ID>
```

Both presets use a bundled fused TensorRT `QVRF g_s + RLFN x2` engine by
default. Append `--no-trt-fused` only for diagnosis.

## 5. Verify Runtime

Inspect GUI and logs. Healthy signals:

- MQTT connects
- `Assembled` increases
- `frame_ready` increases
- no `decode_error`, `bad magic`, `NaN`, or `non-finite`
- receiver prints fused TRT engine loaded

If `MQTT connect FAILED rc=133`, client id is wrong or already occupied.

## 6. Triage

- No MQTT connection: verify host/port, official client, client id.
- MQTT raw increases but no valid chunks: check sender is sending R1V1 video chunks, not stale retained messages.
- Assembled does not increase: transport/chunk loss, not decoder.
- Assembled increases but frame_ready does not: decoder/model/backend issue.
- TensorRT engine load error: rebuild engine on this laptop or use `--no-trt-fused`.
- CUDA unavailable: install CUDA PyTorch matching the driver.
