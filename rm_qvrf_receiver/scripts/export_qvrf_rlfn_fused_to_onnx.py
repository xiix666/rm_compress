#!/usr/bin/env python3
"""Export fused QVRF g_s + RLFN SR ONNX for receiver latency testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client" / "src"))
sys.path.insert(0, str(ROOT / "compress-ai-gray-minimal"))

from compressai.zoo import mbt2018_mean  # noqa: E402
from rm_stream.rlfn_model import DEFAULT_RLFN_MODEL, RlfnModel  # noqa: E402


class _QvrfRlfnFused(nn.Module):
    def __init__(self, rlfn_model_path: str | Path, output_uint8: bool = True) -> None:
        super().__init__()
        self.output_uint8 = output_uint8
        codec = mbt2018_mean(quality=1, metric="ms-ssim", pretrained=True).eval()
        codec.update(force=True)
        self.g_s = codec.g_s.eval()
        rlfn = RlfnModel(rlfn_model_path, scale=2, device="cpu")._model
        if rlfn is None:
            raise RuntimeError("RLFN torch model did not load")
        self.rlfn = rlfn.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        x = self.g_s(y_hat).clamp(0.0, 1.0) * 255.0
        sr = self.rlfn(x).clamp(0.0, 255.0)
        if self.output_uint8:
            return sr.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        return sr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlfn-model", default=str(DEFAULT_RLFN_MODEL))
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--output", default="")
    parser.add_argument("--fp32-output", action="store_true", help="Export NCHW FP32 output instead of NHWC uint8")
    args = parser.parse_args()

    model = _QvrfRlfnFused(args.rlfn_model, output_uint8=not args.fp32_output).eval()
    dummy = torch.zeros((1, 192, args.size // 16, args.size // 16), dtype=torch.float32)
    output = args.output or str(ROOT / "models" / f"qvrf_gs_rlfn_x2_{args.size}.onnx")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["y_hat"],
        output_names=["sr"],
        opset_version=17,
        dynamic_axes=None,
        external_data=True,
    )
    with torch.no_grad():
        out = model(dummy)
    print(f"Exported fused QVRF g_s + RLFN -> {out_path}")
    print(f"  input {tuple(dummy.shape)} -> output {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
