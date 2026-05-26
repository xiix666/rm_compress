"""MS-SSIM model with QVRF gain-based BPP control."""
import io
import hashlib
import os
import struct
import threading

import numpy as np
import torch
from compressai.zoo import mbt2018_mean

MSSSIM_MAGIC = b"MSVG"
MSSSIM_VERSION = 1
HS_BACKEND_TORCH = 0
HS_BACKEND_OPENVINO = 1
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_MODELS_DIR = os.path.join(_ROOT, "models")


def pack_msssim_qvrf(gain: float, strings, shape, hs_backend: int = HS_BACKEND_TORCH) -> bytes:
    if hs_backend != HS_BACKEND_TORCH:
        raise ValueError("QVRF runtime only supports hs_backend=0 (PyTorch h_s)")
    y_strings, z_strings = strings
    buf = io.BytesIO()
    buf.write(MSSSIM_MAGIC)
    buf.write(struct.pack("<B", MSSSIM_VERSION))
    buf.write(struct.pack("<f", gain))
    buf.write(struct.pack("<HH", shape[0], shape[1]))
    buf.write(struct.pack("<I", len(y_strings)))
    for s in y_strings:
        buf.write(struct.pack("<I", len(s)))
        buf.write(s)
    buf.write(struct.pack("<I", len(z_strings)))
    for s in z_strings:
        buf.write(struct.pack("<I", len(s)))
        buf.write(s)
    buf.write(struct.pack("<H", hs_backend))
    return buf.getvalue()


def unpack_msssim_qvrf(data: bytes):
    if data[:4] != MSSSIM_MAGIC:
        raise ValueError(f"Bad magic: {data[:4]}")
    version = struct.unpack("<B", data[4:5])[0]
    gain = struct.unpack("<f", data[5:9])[0]
    zh, zw = struct.unpack("<HH", data[9:13])
    ptr = 13
    y_count = struct.unpack("<I", data[ptr:ptr + 4])[0]
    ptr += 4
    y_strings = []
    for _ in range(y_count):
        slen = struct.unpack("<I", data[ptr:ptr + 4])[0]
        ptr += 4
        y_strings.append(data[ptr:ptr + slen])
        ptr += slen
    z_count = struct.unpack("<I", data[ptr:ptr + 4])[0]
    ptr += 4
    z_strings = []
    for _ in range(z_count):
        slen = struct.unpack("<I", data[ptr:ptr + 4])[0]
        ptr += 4
        z_strings.append(data[ptr:ptr + slen])
        ptr += slen
    hs_backend = 0
    if ptr + 2 <= len(data):
        hs_backend = struct.unpack("<H", data[ptr:ptr + 2])[0]
    return gain, [y_strings, z_strings], (zh, zw), hs_backend


def is_msssim_qvrf(data: bytes) -> bool:
    return len(data) >= 4 and data[:4] == MSSSIM_MAGIC


def _select_torch_device(device=None):
    requested = str(device or os.environ.get("RM_STREAM_TORCH_DEVICE", "")).strip()
    if not requested and os.environ.get("RM_STREAM_BACKEND", "").strip().lower() == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")
    if requested:
        return torch.device(requested)
    return torch.device("cpu")


class MsssimQvrfDecoder:
    """Torch MS-SSIM model decoder with QVRF gain support."""

    def __init__(
        self,
        device=None,
        codec_size: int = 448,
        sr_scale: int = 2,
        gs_backend: str | None = None,
        gs_engine: str = "",
        gs_trt_device: int = 0,
        fused_sr_engine: str = "",
        fused_sr_trt_device: int | None = None,
    ):
        if codec_size <= 0 or codec_size % 16 != 0:
            raise ValueError(f"QVRF codec_size must be a positive multiple of 16, got {codec_size}")
        if sr_scale != 2:
            raise ValueError("QVRF fused RLFN TensorRT currently supports only sr_scale=2")
        self.codec_size = int(codec_size)
        self._latent_size = self.codec_size // 16
        self._sr_scale = int(sr_scale)
        self._sr_size = self.codec_size * self._sr_scale
        self.device = _select_torch_device(device)
        self._cpu_model = mbt2018_mean(quality=1, metric="ms-ssim", pretrained=True)
        self._cpu_model = self._cpu_model.to(torch.device("cpu")).eval()
        self._cpu_model.update(force=True)
        self.model = self._cpu_model
        self._device_model = None
        self._ov_h_s = None
        self._ov_g_s = None
        self._ov_h_s_lock = threading.Lock()
        self._ov_h_s_cache = {}
        self._fast_indexes_enabled = os.environ.get("RM_STREAM_QVRF_FAST_INDEXES", "1") != "0"
        self._fast_indexes_checked = False
        self._fast_indexes_disabled = False
        self._scale_table_np = None
        self._cupy = None
        self._cupy_kernel = None
        self._cupy_out = None
        self._cupy_owner = None
        self._cupy_disabled = False
        self._ov_h_s_path = os.path.join(_MODELS_DIR, "msssim_h_s_fp32.xml")
        if os.path.exists(self._ov_h_s_path):
            self._ov_h_s = self._load_openvino_h_s(self._ov_h_s_path)
        self._trt_g_s = None
        self._trt_fused_sr = None
        self._gs_backend = (gs_backend or os.environ.get("RM_STREAM_RX_GS_BACKEND", "")).strip().lower()
        if not self._gs_backend:
            rx_backend = os.environ.get("RM_STREAM_BACKEND", "").strip().lower()
            self._gs_backend = "cuda" if rx_backend == "cuda" else "cpu"
        self._gs_engine = (gs_engine or os.environ.get("RM_STREAM_RX_GS_TRT_ENGINE", "")).strip()
        self._gs_trt_device = int(os.environ.get("RM_STREAM_RX_GS_TRT_DEVICE", str(gs_trt_device)).strip() or 0)
        self._fused_sr_engine = (fused_sr_engine or os.environ.get("RM_STREAM_RX_FUSED_SR_TRT_ENGINE", "")).strip()
        self._fused_sr_trt_device = int(
            os.environ.get(
                "RM_STREAM_RX_FUSED_SR_TRT_DEVICE",
                str(self._gs_trt_device if fused_sr_trt_device is None else fused_sr_trt_device),
            ).strip() or 0
        )
        if self._gs_backend not in ("cuda", "tensorrt", "cpu", "openvino"):
            raise ValueError("QVRF receiver g_s backend must be cuda, tensorrt, openvino, or cpu")
        if self._fused_sr_engine:
            from rm_stream.tensorrt_runner import TensorRtRunner

            self._trt_fused_sr = TensorRtRunner(
                self._fused_sr_engine,
                expected_input_shape=(1, 192, self._latent_size, self._latent_size),
                expected_output_shape=(1, 3, self._sr_size, self._sr_size),
                device_id=self._fused_sr_trt_device,
                label="QVRF g_s + RLFN SR",
            )
            print(
                f"QVRF receiver contract: hs_backend=1 -> h_s=OpenVINO FP32 CPU; "
                f"entropy/Gaussian=CPU/host; fused_g_s_sr_backend=tensorrt "
                f"device=cuda:{self._fused_sr_trt_device} engine={self._fused_sr_engine}",
                flush=True,
            )
        elif self._gs_backend == "tensorrt":
            if not self._gs_engine:
                raise RuntimeError("--rx-gs-backend tensorrt requires --rx-gs-trt-engine")
            from rm_stream.tensorrt_runner import TensorRtRunner

            self._trt_g_s = TensorRtRunner(
                self._gs_engine,
                expected_input_shape=(1, 192, self._latent_size, self._latent_size),
                expected_output_shape=(1, 3, self.codec_size, self.codec_size),
                device_id=self._gs_trt_device,
                label="QVRF g_s",
            )
            print(
                f"QVRF receiver contract: hs_backend=1 -> h_s=OpenVINO FP32 CPU; "
                f"entropy/Gaussian=CPU/host; g_s_backend=tensorrt device=cuda:{self._gs_trt_device} "
                f"engine={self._gs_engine}",
                flush=True,
            )
        elif self._gs_backend == "openvino":
            g_s_path = os.path.join(_MODELS_DIR, "msssim_g_s_fp32.xml")
            if not os.path.exists(g_s_path):
                raise RuntimeError(f"QVRF OpenVINO g_s requested but model is missing: {g_s_path}")
            self._ov_g_s = self._load_openvino_g_s(g_s_path)
            print(
                f"QVRF receiver contract: hs_backend=1 -> h_s=OpenVINO FP32 CPU; "
                f"entropy/Gaussian=CPU/host; g_s_backend=openvino device=CPU model={g_s_path}",
                flush=True,
            )
        else:
            print(
                f"QVRF receiver contract: hs_backend=1 -> h_s=OpenVINO FP32 CPU; "
                f"entropy/Gaussian=CPU/host; g_s_backend={self._gs_backend}",
                flush=True,
            )
        self._cuda_g_s_experiment = os.environ.get("RM_STREAM_QVRF_CUDA_GS_EXPERIMENT", "") == "1"
        if self._gs_backend == "cuda":
            self._cuda_g_s_experiment = True
        if self.device.type == "cuda" and self._cuda_g_s_experiment:
            self._warmup_cuda_g_s()

    def _build_indexes(self, gaussian_conditional, scales: torch.Tensor) -> torch.Tensor:
        """Fast CPU equivalent of CompressAI GaussianConditional.build_indexes().

        CompressAI loops over the 63 scale-table boundaries in PyTorch. For the
        receiver hs_backend=1 path, scales already live on CPU and NumPy
        searchsorted avoids that Python/Torch loop while preserving the same
        lower-bound table selection.
        """
        if (
            not self._fast_indexes_enabled
            or self._fast_indexes_disabled
            or scales.device.type != "cpu"
        ):
            return gaussian_conditional.build_indexes(scales)

        try:
            scale_table = self._scale_table_np
            if scale_table is None:
                scale_table = gaussian_conditional.scale_table.detach().cpu().numpy().astype(np.float32, copy=True)
                self._scale_table_np = scale_table

            scales_np = scales.detach().numpy()
            bounded = np.maximum(scales_np, scale_table[0])
            indexes_np = np.searchsorted(scale_table, bounded, side="left").astype(np.int32, copy=False)
            np.minimum(indexes_np, len(scale_table) - 1, out=indexes_np)
            indexes = torch.from_numpy(indexes_np)

            if not self._fast_indexes_checked:
                reference = gaussian_conditional.build_indexes(scales)
                if not torch.equal(indexes, reference):
                    self._fast_indexes_disabled = True
                    import sys as _sys
                    print(
                        "QVRF fast build_indexes disabled: NumPy searchsorted did not match CompressAI",
                        file=_sys.stderr,
                        flush=True,
                    )
                    return reference
                self._fast_indexes_checked = True
                print("QVRF fast build_indexes enabled: numpy searchsorted", flush=True)
            return indexes
        except Exception as exc:
            self._fast_indexes_disabled = True
            import sys as _sys
            print(f"QVRF fast build_indexes disabled: {exc}", file=_sys.stderr, flush=True)
            return gaussian_conditional.build_indexes(scales)

    def _torch_decode_model(self):
        if self.device.type == "cpu":
            return self._cpu_model
        if self._device_model is None:
            self._device_model = mbt2018_mean(quality=1, metric="ms-ssim", pretrained=True)
            self._device_model = self._device_model.to(self.device).eval()
            self._device_model.update(force=True)
        return self._device_model

    def _warmup_cuda_g_s(self):
        model = self._torch_decode_model()
        with torch.no_grad():
            for spatial in (8, 12, 20, 28):
                if spatial > self._latent_size:
                    continue
                y = torch.zeros(1, 192, spatial, spatial, device=self.device)
                model.g_s(y).clamp(0, 1)
            torch.cuda.synchronize(self.device)

    def _fused_sr_to_rgb(self, y_hat: torch.Tensor) -> np.ndarray:
        y_arr = y_hat.detach().cpu().numpy()
        if self._trt_fused_sr is None:
            raise RuntimeError("fused SR TensorRT runner is not initialized")
        if not self._cupy_disabled:
            try:
                return self._fused_sr_to_rgb_cupy(y_arr)
            except Exception as exc:
                self._cupy_disabled = True
                import sys as _sys
                print(f"QVRF fused SR: CuPy postprocess disabled, falling back to NumPy: {exc}", file=_sys.stderr, flush=True)
        sr_nchw = self._trt_fused_sr.infer(y_arr, copy_output=False)
        return sr_nchw[0].transpose(1, 2, 0).round().astype("uint8")

    def _fused_sr_to_rgb_cupy(self, y_arr: np.ndarray) -> np.ndarray:
        cp = self._cupy
        if cp is None:
            import cupy as cp

            self._cupy = cp
            self._cupy_kernel = cp.RawKernel(
                r'''
                extern "C" __global__
                void nchw_float_to_hwc_u8(const float* src, unsigned char* dst, int h, int w) {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    int n = h * w * 3;
                    if (i >= n) return;
                    int c = i % 3;
                    int pix = i / 3;
                    int y = pix / w;
                    int x = pix - y * w;
                    float v = src[c * h * w + y * w + x];
                    v = fminf(255.0f, fmaxf(0.0f, nearbyintf(v)));
                    dst[i] = (unsigned char)v;
                }
                ''',
                "nchw_float_to_hwc_u8",
            )
            self._cupy_out = cp.empty((self._sr_size, self._sr_size, 3), dtype=cp.uint8)
            self._cupy_owner = self

        self._trt_fused_sr.infer_device(y_arr)
        h, w = self._sr_size, self._sr_size
        nbytes = self._trt_fused_sr.output_nbytes
        owner = self._cupy_owner or self
        mem = cp.cuda.UnownedMemory(self._trt_fused_sr.output_device_ptr, nbytes, owner)
        ptr = cp.cuda.MemoryPointer(mem, 0)
        src = cp.ndarray((1, 3, h, w), dtype=cp.float32, memptr=ptr)
        stream = cp.cuda.ExternalStream(self._trt_fused_sr.stream_ptr)
        with stream:
            total = h * w * 3
            block = 256
            grid = ((total + block - 1) // block,)
            self._cupy_kernel(grid, (block,), (src, self._cupy_out, h, w))
            out = cp.asnumpy(self._cupy_out, stream=stream)
        stream.synchronize()
        return out

    def _load_openvino_h_s(self, model_path: str):
        import openvino as ov

        core = ov.Core()
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": ov.Type.f32,
            "NUM_STREAMS": "1",
        }
        # HARD RULE: hs_backend=1 QVRF h_s must be OpenVINO FP32 CPU.
        # Do not use iGPU/GPU/CUDA for h_s; tiny backend differences corrupt
        # Gaussian indexes/means and can make y_string decode diverge.
        compiled = core.compile_model(model_path, "CPU", config)
        return compiled.create_infer_request()

    def _load_openvino_g_s(self, model_path: str):
        import openvino as ov

        core = ov.Core()
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": ov.Type.f32,
            "NUM_STREAMS": "1",
        }
        compiled = core.compile_model(model_path, "CPU", config)
        return compiled.create_infer_request()

    def _h_s_params(self, z_hat: torch.Tensor, hs_backend: int):
        if hs_backend == HS_BACKEND_TORCH:
            return self._torch_decode_model().h_s(z_hat)
        if hs_backend == HS_BACKEND_OPENVINO:
            if self._ov_h_s is None:
                raise RuntimeError(
                    "QVRF bitstream requires OpenVINO FP32 h_s (hs_backend=1), "
                    f"but {self._ov_h_s_path} was not loaded"
                )
            z_np = z_hat.detach().cpu().numpy()
            digest = hashlib.blake2b(z_np.tobytes(), digest_size=16).digest()
            cache_key = (tuple(z_hat.shape), str(z_np.dtype), digest)
            cache = getattr(self, "_ov_h_s_cache", {})
            gaussian_params = cache.get(cache_key)
            if gaussian_params is None:
                lock = getattr(self, "_ov_h_s_lock", threading.Lock())
                with lock:
                    gp = self._ov_h_s.infer([z_np])
                gaussian_params = list(gp.values())[0].copy()
                if len(cache) >= 256:
                    cache.clear()
                cache[cache_key] = gaussian_params
                self._ov_h_s_cache = cache
            return torch.from_numpy(gaussian_params.copy())
        raise RuntimeError(f"unsupported QVRF hs_backend={hs_backend}")

    def decode_frame(self, bitstream: bytes):
        import time as _time
        gain, strings, shape, hs_backend = unpack_msssim_qvrf(bitstream)
        y_strings, z_strings = strings

        _t0 = _time.perf_counter()
        rgb = None
        with torch.no_grad():
            if hs_backend == HS_BACKEND_OPENVINO:
                entropy_model = self._cpu_model
                cpu = torch.device("cpu")
                scale = torch.tensor(gain, device=cpu)
                rescale = 1.0 / scale

                z_hat = entropy_model.entropy_bottleneck.decompress(z_strings, shape)
                z_hat = z_hat.to(cpu)
                _t1 = _time.perf_counter()
                gp = self._h_s_params(z_hat, hs_backend).to(cpu)
                _t2 = _time.perf_counter()
                scales, means = gp.chunk(2, 1)

                indexes = self._build_indexes(entropy_model.gaussian_conditional, scales * scale)
                _t3 = _time.perf_counter()
                y_hat = entropy_model.gaussian_conditional.decompress(
                    y_strings, indexes, means=means * scale) * rescale
                _t4 = _time.perf_counter()
                if self._trt_fused_sr is not None:
                    rgb = self._fused_sr_to_rgb(y_hat)
                    _t5 = _time.perf_counter()
                elif self._trt_g_s is not None:
                    y_arr = y_hat.detach().cpu().numpy()
                    x_nchw = self._trt_g_s.infer(y_arr)
                    _t5 = _time.perf_counter()
                    np.clip(x_nchw, 0, 1, out=x_nchw)
                    rgb = (x_nchw[0].transpose(1, 2, 0) * 255).astype("uint8")
                elif self._ov_g_s is not None:
                    x_nchw = list(self._ov_g_s.infer([y_hat.detach().cpu().numpy()]).values())[0]
                    x_hat = torch.from_numpy(x_nchw).clamp(0, 1)
                    _t5 = _time.perf_counter()
                elif self.device.type == "cuda" and self._cuda_g_s_experiment:
                    synthesis_model = self._torch_decode_model()
                    x_hat = synthesis_model.g_s(y_hat.to(self.device)).clamp(0, 1)
                    _t5 = _time.perf_counter()
                else:
                    x_hat = entropy_model.g_s(y_hat).clamp(0, 1)
                    _t5 = _time.perf_counter()
            else:
                model = self._torch_decode_model()
                scale = torch.tensor(gain, device=self.device)
                rescale = 1.0 / scale

                z_hat = model.entropy_bottleneck.decompress(z_strings, shape)
                _t1 = _time.perf_counter()
                gp = self._h_s_params(z_hat, hs_backend)
                _t2 = _time.perf_counter()
                scales, means = gp.chunk(2, 1)

                indexes = self._build_indexes(model.gaussian_conditional, scales * scale)
                _t3 = _time.perf_counter()
                y_hat = model.gaussian_conditional.decompress(
                    y_strings, indexes, means=means * scale) * rescale
                _t4 = _time.perf_counter()
                if self._trt_fused_sr is not None:
                    rgb = self._fused_sr_to_rgb(y_hat)
                    _t5 = _time.perf_counter()
                elif self._trt_g_s is not None:
                    y_arr = y_hat.detach().cpu().numpy()
                    x_nchw = self._trt_g_s.infer(y_arr)
                    _t5 = _time.perf_counter()
                    np.clip(x_nchw, 0, 1, out=x_nchw)
                    rgb = (x_nchw[0].transpose(1, 2, 0) * 255).astype("uint8")
                elif self._ov_g_s is not None:
                    x_nchw = list(self._ov_g_s.infer([y_hat.detach().cpu().numpy()]).values())[0]
                    x_hat = torch.from_numpy(x_nchw).clamp(0, 1)
                    _t5 = _time.perf_counter()
                else:
                    x_hat = model.g_s(y_hat).clamp(0, 1)
                    _t5 = _time.perf_counter()

        if rgb is None:
            x_np = x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if not np.isfinite(x_np).all():
                raise RuntimeError("QVRF decode produced non-finite x_hat")
            rgb = (x_np * 255).astype("uint8")

        _decode_count = getattr(self, "_decode_count", 0) + 1
        self._decode_count = _decode_count
        if _decode_count % 20 == 0:
            import sys as _sys
            print(
                f"[QVRF decode profile #{_decode_count}] "
                f"eb_decomp={(_t1-_t0)*1000:.1f}ms "
                f"h_s={(_t2-_t1)*1000:.1f}ms "
                f"build_idx={(_t3-_t2)*1000:.1f}ms "
                f"rans_dec={(_t4-_t3)*1000:.1f}ms "
                f"g_s={(_t5-_t4)*1000:.1f}ms "
                f"total={(_t5-_t0)*1000:.1f}ms",
                file=_sys.stderr, flush=True,
            )
        return rgb


def encode_msssim_qvrf(x, model, gain=0.8):
    scale = torch.tensor(gain, device=x.device)
    rescale = 1.0 / scale

    with torch.no_grad():
        y = model.g_a(x)
        z = model.h_a(y)
        z_strings = model.entropy_bottleneck.compress(z)
        z_hat = model.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gp = model.h_s(z_hat)
        scales_hat, means_hat = gp.chunk(2, 1)
        indexes = model.gaussian_conditional.build_indexes(scales_hat * scale)
        y_strings = model.gaussian_conditional.compress(
            y * scale, indexes, means=means_hat * scale)

        y_hat = model.gaussian_conditional.decompress(
            y_strings, indexes, means=means_hat * scale) * rescale
        x_hat = model.g_s(y_hat).clamp(0, 1)

    total_bytes = sum(len(s) for s in y_strings) + sum(len(s) for s in z_strings)
    strings = [y_strings, z_strings]
    shape = z.size()[-2:]
    return strings, shape, x_hat, total_bytes


def encode_msssim_qvrf_fast(x, model, gain=0.8):
    scale = torch.tensor(gain, device=x.device)
    with torch.no_grad():
        y = model.g_a(x)
        z = model.h_a(y)
        z_strings = model.entropy_bottleneck.compress(z)
        z_hat = model.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gp = model.h_s(z_hat)
        scales_hat, means_hat = gp.chunk(2, 1)
        indexes = model.gaussian_conditional.build_indexes(scales_hat * scale)
        y_strings = model.gaussian_conditional.compress(
            y * scale, indexes, means=means_hat * scale)
    total_bytes = sum(len(s) for s in y_strings) + sum(len(s) for s in z_strings)
    return [y_strings, z_strings], z.size()[-2:], total_bytes


def encode_with_budget(x, model, gain=0.8, budget=1040, min_gain=0.3, fast=False):
    encode_fn = encode_msssim_qvrf_fast if fast else encode_msssim_qvrf
    for attempt in range(5):
        if fast:
            strings, shape, bs = encode_fn(x, model, gain)
            x_hat = None
        else:
            strings, shape, x_hat, bs = encode_fn(x, model, gain)
        packed = pack_msssim_qvrf(gain, strings, shape, hs_backend=HS_BACKEND_TORCH)
        if len(packed) <= budget or gain <= min_gain:
            return packed, x_hat, gain, len(packed)
        gain = max(min_gain, gain * (budget / len(packed)) * 0.95)
    return packed, x_hat, gain, len(packed)
