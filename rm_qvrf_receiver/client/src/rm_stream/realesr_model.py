"""RealESRGAN receiver-side enhancement wrapper."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
DEFAULT_REALESR_MODEL = _MODELS_DIR / "realesr-general-x4v3.pth"


def _install_torchvision_functional_tensor_compat() -> None:
    """Provide the old basicsr torchvision import path on newer torchvision."""
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return
    try:
        from torchvision.transforms import functional as F
    except Exception:
        return
    shim = types.ModuleType(module_name)
    for name in dir(F):
        if not name.startswith("__"):
            setattr(shim, name, getattr(F, name))
    sys.modules[module_name] = shim


def check_realesr_dependencies() -> None:
    """Raise a clear error if RealESR runtime imports are unavailable."""
    _install_torchvision_functional_tensor_compat()
    try:
        import realesrgan  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "RealESR backend requires the Python package 'realesrgan' and its "
            f"runtime dependencies. Import failed: {exc}"
        ) from exc


class RealEsrModel:
    """RealESRGAN x4 model used with configurable output scale.

    The target model is `realesr-general-x4v3`, whose network scale is x4.
    Passing `outscale=2` to RealESRGANer produces the required 192 -> 384
    receiver enhancement without changing codec reconstruction.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_REALESR_MODEL,
        scale: int = 2,
        device: str | None = None,
    ) -> None:
        if scale != 2:
            raise ValueError("RealESR receiver backend currently supports only --sr-scale 2")

        self.scale = scale
        self.model_path = Path(model_path)
        self._logged_first_run = False
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"RealESR model not found: {self.model_path}. "
                "Expected realesr-general-x4v3.pth; pass --realesr-model PATH."
            )

        _install_torchvision_functional_tensor_compat()
        try:
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            from realesrgan.utils import RealESRGANer
        except Exception as exc:
            raise RuntimeError(
                "RealESR backend requires the Python package 'realesrgan' and its "
                f"runtime dependencies. Import failed: {exc}"
            ) from exc

        requested = (device or "cpu").strip()
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"RealESR CUDA requested ({requested}) but torch.cuda is unavailable")
            self.device = torch.device(requested)
            device_name = torch.cuda.get_device_name(self.device)
            print(f"RealESR: using NVIDIA CUDA device={self.device} name={device_name}", flush=True)
            LOGGER.info("RealESR: using NVIDIA CUDA device=%s name=%s", self.device, device_name)
        elif requested == "cpu":
            self.device = torch.device("cpu")
            print("RealESR: using CPU", flush=True)
            LOGGER.info("RealESR: using CPU")
        else:
            raise ValueError("RealESR --torch-device must be cpu or cuda:N")

        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        self._upsampler = RealESRGANer(
            scale=4,
            model_path=str(self.model_path),
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=self.device.type == "cuda",
            device=self.device,
        )

    def enhance(self, arr_rgb: np.ndarray) -> np.ndarray:
        """Enhance uint8 HWC RGB and return uint8 HWC RGB."""
        if arr_rgb.ndim != 3 or arr_rgb.shape[2] != 3:
            raise ValueError(f"RealESR expects HWC RGB input, got shape {arr_rgb.shape}")
        if arr_rgb.dtype != np.uint8:
            raise ValueError(f"RealESR expects uint8 input, got {arr_rgb.dtype}")

        bgr = cv2.cvtColor(np.ascontiguousarray(arr_rgb), cv2.COLOR_RGB2BGR)
        out_bgr, _ = self._upsampler.enhance(bgr, outscale=self.scale)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        if not self._logged_first_run:
            self._logged_first_run = True
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                device_name = torch.cuda.get_device_name(self.device)
                print(
                    f"RealESR: enhanced frame on NVIDIA CUDA device={self.device} "
                    f"name={device_name} input_shape={arr_rgb.shape} output_shape={out_rgb.shape}",
                    flush=True,
                )
            else:
                print(
                    f"RealESR: enhanced frame on CPU input_shape={arr_rgb.shape} "
                    f"output_shape={out_rgb.shape}",
                    flush=True,
                )
        return np.ascontiguousarray(out_rgb)
