"""Entry point for the custom client GUI.

Usage:
    uv run python -m rm_stream.gui
    uv run python -m rm_stream.gui --mbt-checkpoint PATH --sr-checkpoint PATH
    uv run python -m rm_stream.gui --mqtt-host 192.168.12.1 --mqtt-port 3333
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]  # compress_and_transmit/
_LEGACY = _ROOT / "compress-ai-gray-minimal"

# Must match sys.path setup from other scripts (test_m4_e2e_full.py, etc.)
sys.path.insert(0, str(_ROOT / "client" / "src"))
sys.path.insert(0, str(_LEGACY))
sys.path.insert(0, str(_ROOT / "commu" / "src"))  # for rm_custom_client.proto

DEFAULT_MBT_CKPT = str(_LEGACY / "mbt2018-mean-1-e522738d.pth.tar")
DEFAULT_SR_CKPT = str(_LEGACY / "checkpoints" / "expA_baseline_ch64_n4_15ep" / "e2e_sr_best.pth.tar")
DEFAULT_REALESR_MODEL = str(_ROOT / "models" / "realesr-general-x4v3.pth")
DEFAULT_RLFN_MODEL = str(_ROOT / "models" / "rlfn_s_x2.pth")


def main():
    parser = argparse.ArgumentParser(description="RM Custom Client — PyQt5 Ground Station")
    parser.add_argument("--mbt-checkpoint", default=DEFAULT_MBT_CKPT,
                        help="Path to MBT2018-mean checkpoint")
    parser.add_argument("--sr-checkpoint", default=DEFAULT_SR_CKPT,
                        help="Path to MSA-SR checkpoint")
    parser.add_argument("--mqtt-host", default="192.168.12.1",
                        help="MQTT broker host (default: 192.168.12.1)")
    parser.add_argument("--mqtt-port", type=int, default=3333,
                        help="MQTT broker port (default: 3333)")
    parser.add_argument("--client-id", default="1",
                        help="MQTT client ID (default: 1). The RoboMaster broker may only accept specific numeric IDs.")
    parser.add_argument("--enable-sr", action="store_true",
                        help="Compatibility shortcut for --sr-backend msa")
    parser.add_argument("--sr-backend", choices=("none", "msa", "realesr", "rlfn", "rlfn_trt"), default="none",
                        help="Receiver postprocess backend: none=bilinear, msa=MSA-SR, realesr=RealESRGAN, rlfn=RLFN_s_x2")
    parser.add_argument("--sr-engine", choices=("torch", "tensorrt"), default=os.environ.get("RM_STREAM_SR_ENGINE", "torch"),
                        help="Engine for --sr-backend rlfn (default: torch)")
    parser.add_argument("--sr-trt-engine", default=os.environ.get("RM_STREAM_SR_TRT_ENGINE", ""),
                        help="TensorRT engine for --sr-backend rlfn --sr-engine tensorrt")
    parser.add_argument("--sr-trt-device", type=int, default=int(os.environ.get("RM_STREAM_SR_TRT_DEVICE", "0")),
                        help="CUDA device id for TensorRT SR")
    parser.add_argument("--sr-scale", type=int, default=2,
                        help="Receiver SR output scale for explicit SR backends (default: 2)")
    parser.add_argument("--realesr-model", default=DEFAULT_REALESR_MODEL,
                        help="Path to realesr-general-x4v3.pth for --sr-backend realesr")
    parser.add_argument("--rlfn-model", default=DEFAULT_RLFN_MODEL,
                        help="Path to rlfn_s_x2.pth for --sr-backend rlfn")
    parser.add_argument("--codec-profile", choices=("legacy128x2x24", "codec192x4x12", "codec256x4x12", "codec256x6x8", "codec320x8x6", "codec448x9x5"),
                        default="legacy128x2x24",
                        help="Receiver display profile (default: legacy128x2x24)")
    parser.add_argument("--codec-size", type=int, default=0,
                        help="Override MBT codec size (default from --codec-profile)")
    parser.add_argument("--display-size", type=int, default=0,
                        help="Override GUI interpolation output size (default from --codec-profile)")
    parser.add_argument("--codec", choices=("mbt", "msssim_qvrf"), default="mbt",
                        help="Codec to use: mbt (default) or msssim_qvrf (MS-SSIM with QVRF gain)")
    parser.add_argument("--msssim-gain", type=float, default=0.0,
                        help="QVRF gain for MS-SSIM codec (0=auto from bitstream)")
    parser.add_argument("--rx-backend", choices=("auto", "openvino", "cuda", "cpu"),
                        default=os.environ.get("RM_STREAM_BACKEND", "auto"),
                        help="Inference backend: auto/openvino use OpenVINO Intel GPU when available; cuda uses PyTorch CUDA for NVIDIA")
    parser.add_argument("--rx-gs-backend", choices=("cuda", "openvino", "tensorrt", "cpu"),
                        default=os.environ.get("RM_STREAM_RX_GS_BACKEND", ""),
                        help="Receiver QVRF g_s backend. h_s remains OpenVINO FP32 CPU for hs_backend=1.")
    parser.add_argument("--rx-gs-trt-engine", default=os.environ.get("RM_STREAM_RX_GS_TRT_ENGINE", ""),
                        help="TensorRT engine for receiver QVRF g_s")
    parser.add_argument("--rx-gs-trt-device", type=int, default=int(os.environ.get("RM_STREAM_RX_GS_TRT_DEVICE", "0")),
                        help="CUDA device id for receiver QVRF TensorRT g_s")
    parser.add_argument("--rx-fused-sr-trt-engine", default=os.environ.get("RM_STREAM_RX_FUSED_SR_TRT_ENGINE", ""),
                        help="TensorRT engine for fused receiver QVRF g_s + RLFN SR")
    parser.add_argument("--rx-fused-sr-trt-device", type=int, default=int(os.environ.get("RM_STREAM_RX_FUSED_SR_TRT_DEVICE", "0")),
                        help="CUDA device id for fused receiver QVRF g_s + RLFN SR")
    parser.add_argument("--torch-device", default=os.environ.get("RM_STREAM_TORCH_DEVICE", ""),
                        help="Torch device for --rx-backend cuda/cpu, e.g. cuda:0")
    parser.add_argument("--receive-mode", choices=("mqtt", "ipc"), default="mqtt",
                        help="Receiver input: mqtt for serial/MQTT link, ipc for local TCP")
    parser.add_argument("--offline-debug", action="store_true",
                        help="Shortcut for --receive-mode ipc")
    parser.add_argument("--ipc-host", default="127.0.0.1",
                        help="IPC TCP listen host for --receive-mode ipc")
    parser.add_argument("--ipc-port", type=int, default=49031,
                        help="IPC TCP listen port for --receive-mode ipc")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available OpenVINO and Torch devices, then exit")
    args = parser.parse_args()
    if args.offline_debug:
        args.receive_mode = "ipc"
    if args.enable_sr:
        if args.sr_backend != "none" and args.sr_backend != "msa":
            print("ERROR: --enable-sr is a compatibility shortcut for --sr-backend msa and cannot be combined with another --sr-backend")
            return 1
        args.sr_backend = "msa"
    if args.sr_backend == "rlfn_trt":
        args.sr_backend = "rlfn"
        args.sr_engine = "tensorrt"

    os.environ["RM_STREAM_BACKEND"] = args.rx_backend
    if args.rx_gs_backend:
        os.environ["RM_STREAM_RX_GS_BACKEND"] = args.rx_gs_backend
    if args.rx_gs_trt_engine:
        os.environ["RM_STREAM_RX_GS_TRT_ENGINE"] = args.rx_gs_trt_engine
    os.environ["RM_STREAM_RX_GS_TRT_DEVICE"] = str(args.rx_gs_trt_device)
    if args.rx_fused_sr_trt_engine:
        os.environ["RM_STREAM_RX_FUSED_SR_TRT_ENGINE"] = args.rx_fused_sr_trt_engine
    os.environ["RM_STREAM_RX_FUSED_SR_TRT_DEVICE"] = str(args.rx_fused_sr_trt_device)
    os.environ["RM_STREAM_SR_ENGINE"] = args.sr_engine
    if args.sr_trt_engine:
        os.environ["RM_STREAM_SR_TRT_ENGINE"] = args.sr_trt_engine
    os.environ["RM_STREAM_SR_TRT_DEVICE"] = str(args.sr_trt_device)
    if args.torch_device:
        os.environ["RM_STREAM_TORCH_DEVICE"] = args.torch_device

    profile_defaults = {
        "legacy128x2x24": (128, 256),
        "codec192x4x12": (192, 512),
        "codec256x4x12": (256, 512),
        "codec256x6x8": (256, 512),
        "codec320x8x6": (320, 512),
        "codec448x9x5": (448, 512),
    }
    codec_size, display_size = profile_defaults[args.codec_profile]
    if args.codec_size:
        codec_size = args.codec_size
    if args.display_size:
        display_size = args.display_size
    elif args.sr_backend in ("realesr", "rlfn"):
        display_size = codec_size * args.sr_scale
    if args.sr_backend == "msa" and codec_size != 128:
        print("ERROR: --enable-sr currently supports only 128->256. Use interpolation for non-128 codec profiles.")
        return 1
    if args.sr_backend == "msa" and args.sr_scale != 2:
        print("ERROR: MSA-SR supports only --sr-scale 2.")
        return 1
    if args.sr_backend == "realesr" and args.sr_scale != 2:
        print("ERROR: RealESR receiver backend currently supports only --sr-scale 2.")
        return 1
    if args.sr_backend == "rlfn" and args.sr_scale != 2:
        print("ERROR: RLFN receiver backend currently supports only --sr-scale 2.")
        return 1
    if args.sr_engine == "tensorrt" and args.sr_backend != "rlfn":
        print("ERROR: --sr-engine tensorrt is only valid with --sr-backend rlfn/rlfn_trt.")
        return 1
    if args.sr_engine == "tensorrt" and codec_size not in (192, 448):
        print("ERROR: RLFN TensorRT SR fixed engines currently support only 192->384 and 448->896.")
        return 1
    if args.rx_gs_backend == "tensorrt" and args.codec != "msssim_qvrf":
        print("ERROR: --rx-gs-backend tensorrt currently applies only to --codec msssim_qvrf.")
        return 1
    if args.rx_gs_backend == "tensorrt" and codec_size not in (192, 448):
        print("ERROR: QVRF TensorRT g_s fixed engines currently support only codec sizes 192 and 448.")
        return 1
    if args.rx_fused_sr_trt_engine and (args.codec != "msssim_qvrf" or codec_size not in (192, 448)):
        print("ERROR: fused QVRF+RLFN TensorRT SR currently requires --codec msssim_qvrf and codec size 192 or 448.")
        return 1

    from rm_stream.inference_engine import detect_runtime_devices

    device_info = detect_runtime_devices()
    print("=== RM Stream Devices ===")
    print(f"Backend: {args.rx_backend}")
    print(f"OpenVINO devices: {device_info['openvino']['devices']}")
    cuda_devices = device_info["torch"]["cuda_devices"]
    print(f"Torch CUDA: {cuda_devices if cuda_devices else 'not available'}")
    if args.list_devices:
        return 0

    # Validate checkpoints
    if not Path(args.mbt_checkpoint).exists():
        print(f"ERROR: MBT checkpoint not found: {args.mbt_checkpoint}")
        return 1
    if args.sr_backend == "msa" and not Path(args.sr_checkpoint).exists():
        print(f"ERROR: SR checkpoint not found: {args.sr_checkpoint}")
        return 1
    if args.sr_backend == "realesr" and not Path(args.realesr_model).exists():
        print(f"ERROR: RealESR model not found: {args.realesr_model}")
        return 1
    if args.sr_backend == "rlfn" and not Path(args.rlfn_model).exists():
        print(f"ERROR: RLFN model not found: {args.rlfn_model}")
        return 1
    if args.rx_gs_backend == "tensorrt" and not Path(args.rx_gs_trt_engine).exists():
        print(f"ERROR: receiver QVRF g_s TensorRT engine not found: {args.rx_gs_trt_engine}")
        return 1
    if args.rx_fused_sr_trt_engine and not Path(args.rx_fused_sr_trt_engine).exists():
        print(f"ERROR: fused QVRF+RLFN TensorRT SR engine not found: {args.rx_fused_sr_trt_engine}")
        return 1
    if args.sr_engine == "tensorrt" and not args.rx_fused_sr_trt_engine and not Path(args.sr_trt_engine).exists():
        print(f"ERROR: RLFN TensorRT SR engine not found: {args.sr_trt_engine}")
        return 1
    if args.sr_backend == "realesr":
        try:
            import torch
            from rm_stream.realesr_model import check_realesr_dependencies

            check_realesr_dependencies()
        except Exception as exc:
            print(f"ERROR: --sr-backend realesr requires the 'realesrgan' package and dependencies: {exc}")
            return 1
        requested_device = args.torch_device.strip()
        if not requested_device and args.rx_backend == "cuda":
            requested_device = "cuda:0"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print(f"ERROR: RealESR CUDA requested ({requested_device}) but torch.cuda is unavailable")
            return 1
    if args.sr_backend == "rlfn":
        import torch

        requested_device = args.torch_device.strip()
        if not requested_device and args.rx_backend == "cuda":
            requested_device = "cuda:0"
        if args.sr_engine != "tensorrt" and requested_device.startswith("cuda") and not torch.cuda.is_available():
            print(f"ERROR: RLFN CUDA requested ({requested_device}) but torch.cuda is unavailable")
            return 1
    if args.rx_gs_backend == "tensorrt":
        print(
            f"Receiver contract: hs_backend=1 -> h_s=OpenVINO FP32 CPU; entropy/Gaussian=CPU/host; "
            f"g_s_backend=tensorrt device=cuda:{args.rx_gs_trt_device} engine={args.rx_gs_trt_engine}",
            flush=True,
        )
    if args.sr_engine == "tensorrt":
        print(
            f"Receiver SR: sr_backend=tensorrt/rlfn_trt device=cuda:{args.sr_trt_device} "
            f"engine={args.sr_trt_engine}",
            flush=True,
        )
    if args.rx_fused_sr_trt_engine:
        print(
            f"Receiver fused SR: g_s+rlfn_trt device=cuda:{args.rx_fused_sr_trt_device} "
            f"engine={args.rx_fused_sr_trt_engine}",
            flush=True,
        )

    from rm_stream.gui.client_window import ClientWindow

    return ClientWindow.launch(
        mbt_ckpt=args.mbt_checkpoint,
        sr_ckpt=args.sr_checkpoint,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        client_id=args.client_id,
        enable_sr=args.enable_sr,
        sr_backend=args.sr_backend,
        sr_scale=args.sr_scale,
        realesr_model=args.realesr_model,
        rlfn_model=args.rlfn_model,
        sr_engine=args.sr_engine,
        sr_trt_engine=args.sr_trt_engine,
        sr_trt_device=args.sr_trt_device,
        rx_gs_backend=args.rx_gs_backend,
        rx_gs_trt_engine=args.rx_gs_trt_engine,
        rx_gs_trt_device=args.rx_gs_trt_device,
        rx_fused_sr_trt_engine=args.rx_fused_sr_trt_engine,
        rx_fused_sr_trt_device=args.rx_fused_sr_trt_device,
        codec_size=codec_size,
        display_size=display_size,
        codec=args.codec,
        msssim_gain=args.msssim_gain,
        receive_mode=args.receive_mode,
        ipc_host=args.ipc_host,
        ipc_port=args.ipc_port,
    )


if __name__ == "__main__":
    sys.exit(main())
