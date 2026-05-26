# RM QVRF Receiver Distribution

This package deploys the receiver side of the current link:

MQTT `CustomByteBlock` -> QVRF decode -> TensorRT fused QVRF g_s + RLFN x2 -> PyQt GUI

## Target

Ubuntu laptop with NVIDIA GPU. The bundled fused TensorRT engines are for:

- input codec size 192, output 384
- input codec size 448, output 896
- RTX 4060 Laptop GPU / SM89 class engine as built in the lab

TensorRT engines are machine-specific. If it cannot load, rebuild on the target
or run with `--no-trt-fused`.

Rebuild the 192 fused engine on a target NVIDIA laptop:

```bash
./scripts/build_receiver_trt_engines.sh --size 192 --fused-sr-only \
  --fused-sr-onnx models/qvrf_gs_rlfn_x2_192.onnx
```

Rebuild the 448 fused engine:

```bash
./scripts/build_receiver_trt_engines.sh --size 448 --fused-sr-only \
  --fused-sr-onnx models/qvrf_gs_rlfn_x2_448.onnx
```

## Install

Python in this receiver package is managed only with `uv`. Do not install
receiver dependencies into system Python. `./install.sh` installs `uv` if
needed, creates `client/.venv`, and installs `client/` plus `commu/` as editable
packages into that environment.

```bash
cd rm_qvrf_receiver
./install.sh
./scripts/check_env.sh
```

If PyTorch installed as CPU-only, install a CUDA-enabled build compatible with
the target driver, then rerun `./scripts/check_env.sh`:

```bash
./scripts/install_nvidia_runtime.sh
./scripts/check_env.sh
```

## Find The Current MQTT Client ID

The RoboMaster broker only accepts the robot ID currently connected by the
official client. Probe before running:

```bash
./client/.venv/bin/python ./scripts/probe_mqtt_client_ids.py \
  --host 192.168.12.1 \
  --port 3333 \
  --ids 1,2,3,101,102,103,104,105
```

Use the ID that prints `Success`. In the latest tested setup, that was `103`.

## Run 448 QVRF + Fused TRT GUI

```bash
./scripts/run_receiver.sh \
  --preset qvrf448x6x8 \
  --mqtt-host 192.168.12.1 \
  --mqtt-port 3333 \
  --client-id 103 \
  --rx-backend cuda \
  --torch-device cuda:0
```

## Run 192 QVRF 24 FPS GUI

192 uses the bundled fused TensorRT `QVRF g_s + RLFN x2` engine by default:

```bash
./scripts/run_receiver.sh \
  --preset qvrf192x2x24 \
  --mqtt-host 192.168.12.1 \
  --mqtt-port 3333 \
  --client-id 103 \
  --rx-backend cuda \
  --torch-device cuda:0
```

To diagnose without fused TensorRT, append `--no-trt-fused`. That path uses CUDA
`g_s` and no learned SR.

## Expected Healthy Receiver Output

Look for:

- `MQTT connect OK`
- `QVRF receiver contract: hs_backend=1 -> h_s=OpenVINO FP32 CPU`
- `TensorRT QVRF g_s + RLFN SR loaded`
- increasing `frame_ready`
- `decode_errors=0` in logs

The GUI Debug panel should show `Assembled` increasing and `MQTT Disc` stable.

## Important Rules

- Use the package-local uv environment: `./client/.venv/bin/python`.
- Do not use global `pip`; use `uv pip install --python ./client/.venv/bin/python ...`.
- CUDA/TensorRT Python runtime packages should be installed with `./scripts/install_nvidia_runtime.sh`.
- Do not run two clients with the same MQTT `client-id`.
- `h_s` must remain OpenVINO FP32 CPU for C++ sender bitstreams.
- OpenVINO `GPU.0` is usually Intel iGPU, not NVIDIA. Use `--rx-backend cuda`.
- Bundled fused TRT engines cover 192 -> 384 and 448 -> 896 on the lab RTX 4060 Laptop. Rebuild them on other NVIDIA GPUs, including 50-series laptops.
