"""MSA²-SR super-resolution model: 128×128 → 256×256."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rm_stream.inference_engine import get_engine, get_ov_backend

LOGGER = logging.getLogger(__name__)

# Default OpenVINO model paths (relative to project root)
_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

# Ensure legacy compress-ai-gray-minimal is on the import path
_LEGACY = Path(__file__).resolve().parents[3] / "compress-ai-gray-minimal"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))


class SrModel:
    """MSA²-SR ×2 RGB super-resolution.

    Loads the trained checkpoint, fuses RepSR blocks for inference.
    """

    def __init__(
        self,
        checkpoint_path: str,
        ov_hint: str = "LATENCY",
        ov_streams: int | str = "1",
    ) -> None:
        """Load MSA²-SR model.

        Args:
            checkpoint_path: Path to SR checkpoint.
            ov_hint: OpenVINO performance hint ("LATENCY" or "THROUGHPUT").
            ov_streams: OpenVINO NUM_STREAMS (1 for deterministic low-jitter).
        """
        self._engine = get_engine()
        self._device = self._engine.device()
        self._model = self._load_model(checkpoint_path)

        backend_mode = os.environ.get("RM_STREAM_BACKEND", "auto").strip().lower()

        # Try OpenVINO (prefer INT8 quantized model for speed).  CUDA mode
        # intentionally uses PyTorch CUDA because OpenVINO GPU is not NVIDIA CUDA.
        self._ov_model = None
        ov = get_ov_backend()
        use_openvino = ov.available and backend_mode not in ("cuda", "cpu")
        if use_openvino:
            # 1. Prefer INT8 quantized IR
            sr_int8_xml = _MODELS_DIR / "msa2_sr_int8.xml"
            sr_xml = _MODELS_DIR / "msa2_sr.xml"
            sr_onnx = _MODELS_DIR / "msa2_sr.onnx"

            if sr_int8_xml.exists():
                self._ov_model = ov.load_ir(str(sr_int8_xml), ov_hint, ov_streams)
                LOGGER.info("SrModel: using OpenVINO INT8 (quantized, hint=%s, streams=%s)", ov_hint, ov_streams)
            elif sr_xml.exists():
                self._ov_model = ov.load_ir(str(sr_xml), ov_hint, ov_streams)
            elif sr_onnx.exists():
                self._ov_model = ov.load_ir(str(sr_onnx), ov_hint, ov_streams)
        if self._ov_model is not None:
            LOGGER.info("SrModel: using OpenVINO (hint=%s, streams=%s)", ov_hint, ov_streams)
        elif backend_mode == "openvino":
            raise RuntimeError("RM_STREAM_BACKEND=openvino requested but OpenVINO is unavailable")
        elif backend_mode == "cuda" and self._device.type != "cuda":
            raise RuntimeError("RM_STREAM_BACKEND=cuda requested but torch.cuda is unavailable")
        elif self._device.type == "cuda":
            LOGGER.info("SrModel: using PyTorch CUDA (%s)", self._device)
        else:
            LOGGER.info("SrModel: using TorchCPU")

    def _load_model(self, checkpoint_path: str):
        ckpt = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )
        args = ckpt.get("args", {})

        from src.models.msa2_sr import Msa2SrRgbX2

        model = Msa2SrRgbX2(
            channels=args.get("channels", 64),
            n_blocks=args.get("n_blocks", 4),
            use_sharpen=args.get("use_sharpen", False),
            block_type=args.get("block_type", "rep_sr"),
            head_type=args.get("head_type", "artifact_aware"),
            use_freq_gate=args.get("use_freq_gate", False),
            use_eca=args.get("use_eca", False),
        )
        model.load_state_dict(ckpt["model"])
        model.switch_to_deploy()
        return self._engine.load(model)

    def _forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Run SR on tensor (1, 3, 128, 128) → (1, 3, 256, 256)."""
        if self._ov_model is not None:
            return self._forward_ov(x_hat)
        return self._model(x_hat).clamp(0, 1)

    def _forward_ov(self, x_hat: torch.Tensor) -> torch.Tensor:
        """OpenVINO-accelerated SR forward pass."""
        arr = x_hat.cpu().numpy()
        out = self._ov_model.infer(arr)[0]
        return torch.from_numpy(out).to(self._device).clamp(0, 1)

    def _super_resolve_ov_numpy(self, arr_rgb: np.ndarray) -> np.ndarray:
        """Fast OpenVINO path: uint8 HWC RGB -> uint8 HWC RGB without Torch hops."""
        arr = arr_rgb.astype(np.float32, copy=False) / 255.0
        nchw = np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis, ...])
        out = self._ov_model.infer(nchw)[0]
        out = np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)
        return (out * 255.0).round().astype(np.uint8)

    def super_resolve(self, arr_rgb: np.ndarray) -> np.ndarray:
        """Super-resolve numpy array (128, 128, 3) uint8 → (256, 256, 3) uint8."""
        if self._ov_model is not None:
            return self._super_resolve_ov_numpy(arr_rgb)

        arr = np.ascontiguousarray(arr_rgb.astype(np.float32) / 255.0)
        t = (
            torch.from_numpy(arr)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device)
        )
        with torch.inference_mode():
            out = self._forward(t)
        out = out.squeeze(0).detach().cpu().numpy()
        out = np.transpose(out, (1, 2, 0))
        return (out * 255.0).round().astype(np.uint8)
