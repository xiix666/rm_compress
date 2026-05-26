#!/usr/bin/env python3
"""Export MS-SSIM variant of MBT2018-mean subgraphs to ONNX.

Loads pretrained mbt2018_mean(quality=1, metric="ms-ssim") from compressai.zoo
and exports g_a, h_a, and h_s sub-networks to ONNX.

These files are diagnostic/export assets. The active QVRF realtime path is
Torch-based (`experiments/sender_msssim_qvrf_v2.py` plus
`MsssimQvrfDecoder`) and does not consume these ONNX files directly.

If these exports are used for a future OpenVINO path, h_s must stay FP32 after
conversion to avoid CDF index corruption in the gaussian conditional.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "compress-ai-gray-minimal"))

import torch
from compressai.zoo import mbt2018_mean

OUTPUT_DIR = _ROOT / "models"


def load_msssim():
    """Load pretrained MS-SSIM MBT2018-mean from the CompressAI zoo."""
    print("Loading mbt2018_mean(quality=1, metric='ms-ssim', pretrained=True) from compressai.zoo ...")
    model = mbt2018_mean(quality=1, metric="ms-ssim", pretrained=True)
    model.eval()
    print("  Model loaded successfully.")
    return model


def export_g_a(model, output_dir: Path, size: int = 128):
    """Export g_a (analysis transform): (B, 3, S, S) -> (B, 192, S/16, S/16)."""
    g_a = model.g_a
    g_a.eval()
    dummy = torch.randn(1, 3, size, size)
    path = str(output_dir / "msssim_g_a.onnx")
    torch.onnx.export(
        g_a, dummy, path,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "batch", 2: "height", 3: "width"},
            "y": {0: "batch", 2: "latent_height", 3: "latent_width"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported g_a -> {path}")

    # Verify with onnxruntime
    import onnxruntime as ort
    session = ort.InferenceSession(path)
    out = session.run(None, {"x": dummy.numpy()})
    print(f"  g_a ONNX verify: input {tuple(dummy.shape)} -> output {out[0].shape}")

    # Bit-exact match: PyTorch vs ONNX
    with torch.no_grad():
        pt_out = g_a(dummy)
    diff = (pt_out.detach().cpu().numpy() - out[0])
    max_diff = abs(diff).max()
    print(f"  g_a PT vs ONNX max diff: {max_diff:.2e}  {'OK' if max_diff < 1e-5 else 'WARNING: diff > 1e-5!'}")
    print(f"  ONNX model size: {Path(path).stat().st_size / 1024:.1f} KB")


def export_h_a(model, output_dir: Path, size: int = 128):
    """Export h_a (hyper-analysis): (B, 192, S/16, S/16) -> (B, 128, S/64, S/64)."""
    h_a = model.h_a
    h_a.eval()
    y_size = size // 16
    dummy = torch.randn(1, 192, y_size, y_size)
    path = str(output_dir / "msssim_h_a.onnx")
    torch.onnx.export(
        h_a, dummy, path,
        input_names=["y"],
        output_names=["z"],
        dynamic_axes={
            "y": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "z": {0: "batch", 2: "hyper_height", 3: "hyper_width"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported h_a -> {path}")

    # Verify with onnxruntime
    import onnxruntime as ort
    session = ort.InferenceSession(path)
    out = session.run(None, {"y": dummy.numpy()})
    print(f"  h_a ONNX verify: input {tuple(dummy.shape)} -> output {out[0].shape}")

    # Bit-exact match: PyTorch vs ONNX
    with torch.no_grad():
        pt_out = h_a(dummy)
    diff = (pt_out.detach().cpu().numpy() - out[0])
    max_diff = abs(diff).max()
    print(f"  h_a PT vs ONNX max diff: {max_diff:.2e}  {'OK' if max_diff < 1e-5 else 'WARNING: diff > 1e-5!'}")
    print(f"  ONNX model size: {Path(path).stat().st_size / 1024:.1f} KB")


def export_h_s(model, output_dir: Path, size: int = 128):
    """Export h_s (hyper-synthesis): (B, 128, S/64, S/64) -> (B, 384, S/16, S/16).

    CRITICAL: The ONNX export is FP32 by default. The subsequent OpenVINO
    conversion MUST also use FP32 (--compress_to_fp16=False) to prevent CDF
    index corruption. FP16 differences of ~1e-5 push Gaussian conditional
    scales across quantization boundaries, producing wrong CDF table selection
    and garbage decoded output.
    """
    h_s = model.h_s
    h_s.eval()
    z_size = size // 64
    dummy = torch.randn(1, 128, z_size, z_size)
    path = str(output_dir / "msssim_h_s.onnx")
    torch.onnx.export(
        h_s, dummy, path,
        input_names=["z_hat"],
        output_names=["gaussian_params"],
        dynamic_axes={
            "z_hat": {0: "batch", 2: "hyper_height", 3: "hyper_width"},
            "gaussian_params": {0: "batch", 2: "latent_height", 3: "latent_width"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported h_s -> {path}")

    # Verify with onnxruntime
    import onnxruntime as ort
    session = ort.InferenceSession(path)
    out = session.run(None, {"z_hat": dummy.numpy()})
    print(f"  h_s ONNX verify: input {tuple(dummy.shape)} -> output {out[0].shape}")

    # Bit-exact match: PyTorch vs ONNX
    with torch.no_grad():
        pt_out = h_s(dummy)
    diff = (pt_out.detach().cpu().numpy() - out[0])
    max_diff = abs(diff).max()
    print(f"  h_s PT vs ONNX max diff: {max_diff:.2e}  {'OK' if max_diff < 1e-5 else 'WARNING: diff > 1e-5!'}")
    print(f"  ONNX model size: {Path(path).stat().st_size / 1024:.1f} KB")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_msssim()
    export_g_a(model, OUTPUT_DIR)
    export_h_a(model, OUTPUT_DIR)
    export_h_s(model, OUTPUT_DIR)
    print("\nAll MS-SSIM ONNX exports complete.")
    print("These exports are not wired into the current QVRF realtime path.")
    print("For future OpenVINO work, convert h_s with ovc --compress_to_fp16=False.")


if __name__ == "__main__":
    main()
