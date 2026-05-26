"""RLFN_s_x2 receiver-side enhancement wrapper."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
DEFAULT_RLFN_MODEL = _MODELS_DIR / "rlfn_s_x2.pth"


def _conv_layer(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    padding = int((kernel_size - 1) / 2)
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def _pixelshuffle_block(
    in_channels: int,
    out_channels: int,
    upscale_factor: int = 2,
    kernel_size: int = 3,
) -> nn.Sequential:
    conv = _conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size)
    return nn.Sequential(conv, nn.PixelShuffle(upscale_factor))


class _ESA(nn.Module):
    def __init__(self, esa_channels: int, n_feats: int) -> None:
        super().__init__()
        f = esa_channels
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class _RLFB(nn.Module):
    def __init__(self, in_channels: int, esa_channels: int = 16) -> None:
        super().__init__()
        self.c1_r = _conv_layer(in_channels, in_channels, 3)
        self.c2_r = _conv_layer(in_channels, in_channels, 3)
        self.c3_r = _conv_layer(in_channels, in_channels, 3)
        self.c5 = _conv_layer(in_channels, in_channels, 1)
        self.esa = _ESA(esa_channels, in_channels)
        self.act = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.c1_r(x))
        out = self.act(self.c2_r(out))
        out = self.act(self.c3_r(out))
        out = out + x
        return self.esa(self.c5(out))


class _RLFNS(nn.Module):
    """RLFN_S network definition matching bytedance/RLFN model/rlfn_s.py."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        feature_channels: int = 48,
        upscale: int = 2,
    ) -> None:
        super().__init__()
        self.conv_1 = _conv_layer(in_channels, feature_channels, kernel_size=3)
        self.block_1 = _RLFB(feature_channels)
        self.block_2 = _RLFB(feature_channels)
        self.block_3 = _RLFB(feature_channels)
        self.block_4 = _RLFB(feature_channels)
        self.block_5 = _RLFB(feature_channels)
        self.block_6 = _RLFB(feature_channels)
        self.conv_2 = _conv_layer(feature_channels, feature_channels, kernel_size=3)
        self.upsampler = _pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_feature = self.conv_1(x)
        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)
        out_low_resolution = self.conv_2(out_b6) + out_feature
        return self.upsampler(out_low_resolution)


class RlfnModel:
    """RLFN_s_x2 RGB super-resolution.

    The upstream demo feeds RGB tensors in 0..255 float range and divides model
    output by 255 before saving. This wrapper follows that contract.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_RLFN_MODEL,
        scale: int = 2,
        device: str | None = None,
        warmup_size: int | None = None,
        warmup_iters: int = 3,
        engine: str = "torch",
        trt_engine_path: str | Path | None = None,
        trt_device: int = 0,
    ) -> None:
        if scale != 2:
            raise ValueError("RLFN receiver backend currently supports only --sr-scale 2")

        self.scale = scale
        self._fixed_input_size = int(warmup_size or 448)
        if self._fixed_input_size <= 0:
            raise ValueError("RLFN TensorRT fixed input size must be positive")
        self._fixed_output_size = self._fixed_input_size * self.scale
        self.model_path = Path(model_path)
        self._engine_kind = (engine or os.environ.get("RM_STREAM_SR_ENGINE", "torch")).strip().lower()
        self._trt = None
        self._logged_first_run = False
        if self._engine_kind not in ("torch", "tensorrt"):
            raise ValueError("RLFN --sr-engine must be torch or tensorrt")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"RLFN model not found: {self.model_path}. "
                "Expected rlfn_s_x2.pth; pass --rlfn-model PATH."
            )
        if self._engine_kind == "tensorrt":
            engine_path = str(trt_engine_path or os.environ.get("RM_STREAM_SR_TRT_ENGINE", "")).strip()
            if not engine_path:
                raise RuntimeError("--sr-engine tensorrt requires --sr-trt-engine")
            from rm_stream.tensorrt_runner import TensorRtRunner

            self.device = torch.device("cpu")
            self._model = None
            self._trt = TensorRtRunner(
                engine_path,
                expected_input_shape=(1, 3, self._fixed_input_size, self._fixed_input_size),
                expected_output_shape=(1, 3, self._fixed_output_size, self._fixed_output_size),
                device_id=int(os.environ.get("RM_STREAM_SR_TRT_DEVICE", str(trt_device)).strip() or 0),
                label="RLFN SR",
            )
            print(
                f"RLFN: sr_backend=tensorrt device=cuda:{self._trt.device_id} "
                f"engine={engine_path} input={self._fixed_input_size}x{self._fixed_input_size} "
                f"output={self._fixed_output_size}x{self._fixed_output_size}",
                flush=True,
            )
            if warmup_size is not None and warmup_size > 0 and warmup_iters > 0:
                self.warmup(warmup_size, warmup_iters)
            return

        requested = (device or "cpu").strip()
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"RLFN CUDA requested ({requested}) but torch.cuda is unavailable")
            self.device = torch.device(requested)
            device_name = torch.cuda.get_device_name(self.device)
            print(f"RLFN: using NVIDIA CUDA device={self.device} name={device_name}", flush=True)
            LOGGER.info("RLFN: using NVIDIA CUDA device=%s name=%s", self.device, device_name)
        elif requested == "cpu":
            self.device = torch.device("cpu")
            print("RLFN: using CPU", flush=True)
            LOGGER.info("RLFN: using CPU")
        else:
            raise ValueError("RLFN --torch-device must be cpu or cuda:N")

        state = torch.load(self.model_path, map_location="cpu")
        if not isinstance(state, (dict, OrderedDict)):
            raise RuntimeError(f"Unexpected RLFN checkpoint type: {type(state).__name__}")
        state_dict = state.get("params") or state.get("state_dict") or state.get("model") or state
        model = _RLFNS(upscale=scale)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self._model = model.to(self.device)
        if warmup_size is not None and warmup_size > 0 and warmup_iters > 0:
            self.warmup(warmup_size, warmup_iters)

    def warmup(self, size: int, iters: int = 3) -> None:
        """Run fixed-shape dummy inferences to absorb first-frame backend setup."""
        if size <= 0 or iters <= 0:
            return
        if self._trt is not None:
            dummy = np.zeros((1, 3, size, size), dtype=np.float32)
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(iters):
                self._trt.infer(dummy)
            t1.record()
            torch.cuda.synchronize(torch.device(f"cuda:{self._trt.device_id}"))
            elapsed_ms = t0.elapsed_time(t1)
            print(
                f"RLFN TensorRT: warmup size={size} iters={iters} device=cuda:{self._trt.device_id} "
                f"elapsed_ms={elapsed_ms:.1f}",
                flush=True,
            )
            return
        t0 = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        dummy = torch.zeros((1, 3, size, size), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            if t0 is not None and t1 is not None:
                t0.record()
            for _ in range(iters):
                self._model(dummy).clamp(0.0, 255.0)
            if t0 is not None and t1 is not None:
                t1.record()
                torch.cuda.synchronize(self.device)
                elapsed_ms = t0.elapsed_time(t1)
            else:
                elapsed_ms = 0.0
        print(
            f"RLFN: warmup size={size} iters={iters} device={self.device} "
            f"elapsed_ms={elapsed_ms:.1f}",
            flush=True,
        )
        LOGGER.info(
            "RLFN: warmup size=%d iters=%d device=%s elapsed_ms=%.1f",
            size,
            iters,
            self.device,
            elapsed_ms,
        )

    def enhance(self, arr_rgb: np.ndarray) -> np.ndarray:
        """Enhance uint8 HWC RGB and return uint8 HWC RGB."""
        if arr_rgb.ndim != 3 or arr_rgb.shape[2] != 3:
            raise ValueError(f"RLFN expects HWC RGB input, got shape {arr_rgb.shape}")
        if arr_rgb.dtype != np.uint8:
            raise ValueError(f"RLFN expects uint8 input, got {arr_rgb.dtype}")

        if self._trt is not None:
            tensor = self._trt.input_host_array()
            if tensor.shape != (1, 3, arr_rgb.shape[0], arr_rgb.shape[1]):
                raise RuntimeError(
                    f"RLFN TensorRT input shape mismatch: expected {tensor.shape}, got {arr_rgb.shape}"
                )
            tensor[0, 0, :, :] = arr_rgb[:, :, 0]
            tensor[0, 1, :, :] = arr_rgb[:, :, 1]
            tensor[0, 2, :, :] = arr_rgb[:, :, 2]
            out = self._trt.infer(tensor, copy_output=False)
            out_arr = out.squeeze(0).transpose(1, 2, 0)
            out_arr = np.clip(out_arr, 0.0, 255.0).round().astype(np.uint8)
            if not self._logged_first_run:
                self._logged_first_run = True
                print(
                    f"RLFN: enhanced frame on TensorRT device=cuda:{self._trt.device_id} "
                    f"engine={self._trt.engine_path} input_shape={arr_rgb.shape} output_shape={out_arr.shape}",
                    flush=True,
                )
            return np.ascontiguousarray(out_arr)

        arr = np.ascontiguousarray(arr_rgb.astype(np.float32))
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out = self._model(tensor).clamp(0.0, 255.0) / 255.0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        out_arr = out.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        out_arr = (out_arr * 255.0).round().astype(np.uint8)
        if not self._logged_first_run:
            self._logged_first_run = True
            if self.device.type == "cuda":
                device_name = torch.cuda.get_device_name(self.device)
                print(
                    f"RLFN: enhanced frame on NVIDIA CUDA device={self.device} "
                    f"name={device_name} input_shape={arr_rgb.shape} output_shape={out_arr.shape}",
                    flush=True,
                )
            else:
                print(
                    f"RLFN: enhanced frame on CPU input_shape={arr_rgb.shape} "
                    f"output_shape={out_arr.shape}",
                    flush=True,
                )
        return np.ascontiguousarray(out_arr)
