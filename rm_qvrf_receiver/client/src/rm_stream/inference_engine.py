"""Inference engine abstraction.

Default receiver path is OpenVINO on Intel iGPU.  NVIDIA support is provided
through PyTorch CUDA because OpenVINO's GPU plugin is not a NVIDIA CUDA backend.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def _env_backend() -> str:
    return os.environ.get("RM_STREAM_BACKEND", "auto").strip().lower()


def detect_runtime_devices() -> dict[str, Any]:
    """Return best-effort runtime device information for startup diagnostics."""
    info: dict[str, Any] = {
        "backend": _env_backend(),
        "openvino": {"available": False, "devices": []},
        "torch": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [],
        },
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(idx)
            except Exception:
                name = f"cuda:{idx}"
            info["torch"]["cuda_devices"].append({"id": f"cuda:{idx}", "name": name})

    try:
        import openvino as ov

        core = ov.Core()
        info["openvino"]["available"] = True
        info["openvino"]["devices"] = list(core.available_devices)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if result.returncode == 0:
            info["nvidia_smi"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        info["nvidia_smi"] = []
    return info


class TorchBackend:
    """Loads PyTorch models on CPU or CUDA."""

    def __init__(self) -> None:
        requested = os.environ.get("RM_STREAM_TORCH_DEVICE", "").strip()
        backend = _env_backend()
        if not requested and backend == "cuda":
            requested = "cuda:0"
        if requested.startswith("cuda") and torch.cuda.is_available():
            self._device = torch.device(requested)
        else:
            if requested.startswith("cuda") and not torch.cuda.is_available():
                LOGGER.warning("Requested %s but torch.cuda is not available; using CPU", requested)
            self._device = torch.device("cpu")
        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        LOGGER.info("Torch backend device=%s", self._device)

    def device(self) -> torch.device:
        return self._device

    def load(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(self._device)
        model.eval()
        return model

    def infer(self, model: torch.nn.Module, *inputs: Any) -> Any:
        with torch.no_grad():
            return model(*inputs)


class OpenVINOBackend:
    """OpenVINO inference backend for exported IR models."""

    def __init__(self) -> None:
        self._available = False
        self._device = "CPU"
        try:
            import openvino as ov
            self._core = ov.Core()
            # Enable model caching for faster startup
            self._core.set_property({"CACHE_DIR": str(Path(__file__).resolve().parents[3] / "models" / "cache")})
            self._available = True
            requested_device = os.environ.get("RM_STREAM_OV_DEVICE")
            # Prefer explicit env override, then Intel iGPU (GPU.0), then CPU.
            devices = self._core.available_devices
            if requested_device and not requested_device.lower().startswith("cuda"):
                self._device = requested_device
            elif "GPU.0" in devices:
                self._device = "GPU.0"
            elif "GPU" in devices:
                self._device = "GPU"
            LOGGER.info("OpenVINO backend available (device=%s)", self._device)
        except ImportError:
            LOGGER.warning("OpenVINO not installed, OpenVINOBackend disabled")

    @property
    def available(self) -> bool:
        return self._available

    def load_ir(
        self,
        model_path: str,
        performance_hint: str = "THROUGHPUT",
        num_streams: int | str = "2",
        device: str | None = None,
    ) -> "OpenVINOModel":
        """Load an OpenVINO IR model, with per-model GPU->CPU fallback.

        Args:
            model_path: Path to .xml or .onnx model.
            performance_hint: "LATENCY" (low jitter, 1 frame at a time) or
                              "THROUGHPUT" (max throughput, may batch).
            num_streams: Number of parallel execution streams.
                         1 = deterministic single-stream.  "2" = default.
        """
        if not self._available:
            raise RuntimeError("OpenVINO is not installed")
        import openvino as ov
        model = self._core.read_model(model_path)
        config = {
            "PERFORMANCE_HINT": performance_hint,
            "INFERENCE_PRECISION_HINT": "f16",
            "NUM_STREAMS": str(num_streams),
        }
        # Try preferred device, fall back to CPU
        preferred = device or self._device
        for dev in (preferred, "CPU"):
            try:
                compiled = self._core.compile_model(model, dev, config)
                return OpenVINOModel(compiled)
            except Exception:
                if dev == "CPU":
                    raise
        raise RuntimeError(f"Failed to compile {model_path}")


class OpenVINOModel:
    """Wraps a compiled OpenVINO model for inference."""

    def __init__(self, compiled_model) -> None:
        self._model = compiled_model
        self._ireq = compiled_model.create_infer_request()
        self._input_shapes: list[tuple] = []
        # Cache input shape from first inference for warmup reuse
        for inp in compiled_model.inputs:
            shape = tuple(inp.get_partial_shape().get_min_shape())
            self._input_shapes.append(shape)

    def infer(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Run inference with reused infer_request (avoids alloc overhead)."""
        result = self._ireq.infer(inputs)
        return list(result.values())

    def warmup(self, n: int = 50) -> None:
        """Pre-warm the model with n dummy inferences to stabilize timing.

        First inference triggers GPU shader compilation, memory allocation,
        and internal OpenVINO graph scheduling.  Warmup amortizes these costs.
        """
        dummy_inputs = []
        for shape in self._input_shapes:
            # Use ones; any data works for warmup
            dummy = np.ones(shape, dtype=np.float32)
            dummy_inputs.append(dummy)
        for _ in range(n):
            self.infer(*dummy_inputs)


_engine: TorchBackend | None = None
_ov_backend: OpenVINOBackend | None = None


def get_engine() -> TorchBackend:
    """Get the TorchBackend singleton (always available)."""
    global _engine
    if _engine is None:
        _engine = TorchBackend()
    return _engine


def get_ov_backend() -> OpenVINOBackend:
    """Get the OpenVINOBackend singleton (may be disabled if openvino not installed)."""
    global _ov_backend
    if _ov_backend is None:
        _ov_backend = OpenVINOBackend()
    return _ov_backend
