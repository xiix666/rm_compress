#!/usr/bin/env python3
"""Export RLFN_s_x2 receiver SR to fixed-shape ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client" / "src"))

from rm_stream.rlfn_model import DEFAULT_RLFN_MODEL, RlfnModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export fixed-shape RLFN_s_x2 ONNX")
    parser.add_argument("--rlfn-model", default=str(DEFAULT_RLFN_MODEL))
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--output", default=str(ROOT / "models" / "rlfn_s_x2_448.onnx"))
    args = parser.parse_args()

    wrapper = RlfnModel(args.rlfn_model, scale=2, device="cpu")
    model = wrapper._model
    if model is None:
        raise RuntimeError("RLFN torch model did not load")
    dummy = torch.zeros((1, 3, args.size, args.size), dtype=torch.float32)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["x"],
        output_names=["sr"],
        opset_version=17,
        dynamic_axes=None,
    )
    with torch.no_grad():
        out = model(dummy)
    print(f"Exported RLFN -> {out_path}")
    print(f"  input {tuple(dummy.shape)} -> output {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
