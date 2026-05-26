#!/usr/bin/env python3
"""TensorRT subprocess worker used when the GUI Python cannot load TensorRT."""

from __future__ import annotations

import argparse
import io
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client" / "src"))

from rm_stream.tensorrt_runner import _InProcessTensorRtRunner  # noqa: E402


def _parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in text.split("x") if v)


def _read_message() -> bytes | None:
    header = sys.stdin.buffer.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise RuntimeError("truncated sidecar request header")
    size = struct.unpack("<Q", header)[0]
    if size == 0:
        return None
    data = sys.stdin.buffer.read(size)
    if len(data) != size:
        raise RuntimeError("truncated sidecar request body")
    return data


def _write_message(data: bytes) -> None:
    sys.stdout.buffer.write(struct.pack("<Q", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="TensorRT NPY sidecar")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--output-shape", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--label", default="tensor")
    args = parser.parse_args()

    runner = _InProcessTensorRtRunner(
        args.engine,
        _parse_shape(args.input_shape),
        _parse_shape(args.output_shape),
        args.device,
        args.label,
    )
    _write_message(b"READY")
    while True:
        req = _read_message()
        if req is None:
            break
        arr = np.load(io.BytesIO(req), allow_pickle=False)
        out = runner.infer(arr)
        buf = io.BytesIO()
        np.save(buf, out, allow_pickle=False)
        _write_message(buf.getvalue())
    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
