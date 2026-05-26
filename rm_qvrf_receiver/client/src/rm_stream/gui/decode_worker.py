"""Decode worker — runs codec decode plus receiver-side enhancement."""

import math
import os
import queue
import threading
import time
from collections import deque

import cv2

from PyQt5.QtCore import QObject, pyqtSignal

from rm_stream.codec_decoder import MbtDecoder
from rm_stream.protocol import parse_over_budget_marker
from rm_stream.realesr_model import DEFAULT_REALESR_MODEL, RealEsrModel
from rm_stream.rlfn_model import DEFAULT_RLFN_MODEL, RlfnModel
from rm_stream.sr_model import SrModel

try:
    from rm_stream.codecs.qvrf import MsssimQvrfDecoder
    _HAS_MSSSIM_QVRF = True
    _MSSSIM_QVRF_IMPORT_ERROR = None
except ImportError as exc:
    _HAS_MSSSIM_QVRF = False
    _MSSSIM_QVRF_IMPORT_ERROR = exc


class DecodeWorker(QObject):
    """QObject that decodes frames and super-resolves them in a worker thread.

    Move this object to a QThread via worker.moveToThread(thread), then
    call start() to load models.  Enqueue bitstreams via enqueue() from
    any thread — the internal processing loop fires every 10 ms, drains
    the queue, and emits frame_ready for each completed frame.
    """

    frame_ready = pyqtSignal(object, dict)
    frame_skipped = pyqtSignal(dict)
    decode_error = pyqtSignal(str)

    def __init__(
        self,
        mbt_checkpoint_path,
        sr_checkpoint_path,
        enable_sr: bool = False,
        sr_backend: str = "none",
        sr_scale: int = 2,
        realesr_model_path: str = "",
        rlfn_model_path: str = "",
        sr_engine: str = "torch",
        sr_trt_engine: str = "",
        sr_trt_device: int = 0,
        rx_gs_backend: str = "",
        rx_gs_trt_engine: str = "",
        rx_gs_trt_device: int = 0,
        rx_fused_sr_trt_engine: str = "",
        rx_fused_sr_trt_device: int = 0,
        codec_size: int = 128,
        display_size: int = 256,
        codec: str = "mbt",
        msssim_qvrf_gain: float = 0.0,
    ):
        super().__init__()
        self._mbt_path = mbt_checkpoint_path
        self._sr_path = sr_checkpoint_path
        self._sr_backend = "msa" if enable_sr else sr_backend
        self._sr_scale = sr_scale
        self._realesr_model_path = realesr_model_path
        self._rlfn_model_path = rlfn_model_path
        self._sr_engine = sr_engine
        self._sr_trt_engine = sr_trt_engine
        self._sr_trt_device = sr_trt_device
        self._rx_gs_backend = rx_gs_backend
        self._rx_gs_trt_engine = rx_gs_trt_engine
        self._rx_gs_trt_device = rx_gs_trt_device
        self._rx_fused_sr_trt_engine = rx_fused_sr_trt_engine
        self._rx_fused_sr_trt_device = rx_fused_sr_trt_device
        self._codec_size = codec_size
        self._display_size = display_size
        self._codec = codec
        self._msssim_gain = msssim_qvrf_gain
        self._decoder = None
        self._sr = None
        queue_depth = 1
        self._decode_queue = queue.Queue(maxsize=queue_depth)
        self._sr_queue = queue.Queue(maxsize=queue_depth)
        self._stop_event = threading.Event()
        self._decode_thread = None
        self._sr_thread = None
        self._decode_queue_drops = 0
        self._sr_queue_drops = 0
        self._stats_lock = threading.Lock()
        self._codec_ms_window = deque(maxlen=120)
        self._sr_ms_window = deque(maxlen=120)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Load models (heavy) and kick off decode/SR pipeline threads."""
        try:
            if self._codec == "msssim_qvrf":
                if not _HAS_MSSSIM_QVRF:
                    reason = f": {_MSSSIM_QVRF_IMPORT_ERROR}" if _MSSSIM_QVRF_IMPORT_ERROR else ""
                    raise RuntimeError(f"msssim_qvrf codec requested but rm_stream.codecs.qvrf is unavailable{reason}")
                torch_device = os.environ.get("RM_STREAM_TORCH_DEVICE", "").strip() or None
                self._decoder = MsssimQvrfDecoder(
                    device=torch_device,
                    codec_size=self._codec_size,
                    sr_scale=self._sr_scale,
                    gs_backend=self._rx_gs_backend,
                    gs_engine=self._rx_gs_trt_engine,
                    gs_trt_device=self._rx_gs_trt_device,
                    fused_sr_engine=self._rx_fused_sr_trt_engine,
                    fused_sr_trt_device=self._rx_fused_sr_trt_device,
                )
            else:
                self._decoder = MbtDecoder(self._mbt_path, codec_size=self._codec_size)
            if self._sr_backend == "msa":
                self._sr = SrModel(self._sr_path)
            elif self._sr_backend == "realesr":
                self._sr = RealEsrModel(
                    model_path=self._realesr_model_path or DEFAULT_REALESR_MODEL,
                    scale=self._sr_scale,
                    device=self._torch_sr_device(),
                )
            elif self._sr_backend == "rlfn":
                if not self._rx_fused_sr_trt_engine:
                    self._sr = RlfnModel(
                        model_path=self._rlfn_model_path or DEFAULT_RLFN_MODEL,
                        scale=self._sr_scale,
                        device=self._torch_sr_device(),
                        warmup_size=self._codec_size,
                        engine=self._sr_engine,
                        trt_engine_path=self._sr_trt_engine,
                        trt_device=self._sr_trt_device,
                    )
            elif self._sr_backend != "none":
                raise RuntimeError(f"unknown SR backend: {self._sr_backend}")
            print(
                f"DecodeWorker: sr_backend={self._sr_backend} sr_engine={self._sr_engine} sr_scale={self._sr_scale} "
                f"codec_size={self._codec_size} display_size={self._display_size}",
                flush=True,
            )
        except Exception as exc:
            self.decode_error.emit(f"model load fail: {exc}")
            self._stop_event.set()
            return
        self._stop_event.clear()
        self._decode_thread = threading.Thread(target=self._decode_loop, name="rm-decode", daemon=True)
        self._sr_thread = threading.Thread(target=self._sr_loop, name="rm-sr", daemon=True)
        self._decode_thread.start()
        self._sr_thread.start()

    def stop(self):
        """Stop pipeline threads."""
        self._stop_event.set()
        self._put_latest(self._decode_queue, None)
        self._put_latest(self._sr_queue, None)
        for thread in (self._decode_thread, self._sr_thread):
            if thread is not None:
                thread.join(timeout=1.0)
        self._decode_thread = None
        self._sr_thread = None

    def enqueue(self, frame_id, bitstream):
        """Thread-safe: add a bitstream to the decode queue."""
        self._put_latest(self._decode_queue, (frame_id, bitstream))

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _put_latest(self, q: queue.Queue, item):
        """Bound queues by dropping oldest frames instead of accumulating latency."""
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass
        self._drop_pending(q)
        try:
            q.put_nowait(item)
        except queue.Full:
            self._drop_pending(q)
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _drop_pending(self, q: queue.Queue) -> int:
        """Drop all frames currently waiting in a pipeline queue."""
        drops = 0
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
            drops += 1
        if drops:
            if q is self._decode_queue:
                self._decode_queue_drops += drops
            elif q is self._sr_queue:
                self._sr_queue_drops += drops
        return drops

    def _record_timing(self, name: str, value_ms: float) -> dict:
        """Track recent timing windows for low-latency diagnostics."""
        with self._stats_lock:
            window = self._codec_ms_window if name == "codec" else self._sr_ms_window
            window.append(value_ms)
            values = list(window)
        if not values:
            return {}
        values.sort()
        idx90 = int(round(0.90 * (len(values) - 1)))
        idx99 = int(round(0.99 * (len(values) - 1)))
        prefix = f"{name}_win"
        return {
            f"{prefix}_n": len(values),
            f"{prefix}_avg_ms": sum(values) / len(values),
            f"{prefix}_p90_ms": values[idx90],
            f"{prefix}_p99_ms": values[idx99],
            f"{prefix}_max_ms": values[-1],
        }

    def _torch_sr_device(self) -> str:
        """Return the requested torch device for explicit torch SR backends."""
        torch_device = os.environ.get("RM_STREAM_TORCH_DEVICE", "").strip()
        if torch_device:
            return torch_device
        backend = os.environ.get("RM_STREAM_BACKEND", "auto").strip().lower()
        return "cuda:0" if backend == "cuda" else "cpu"

    def _decode_loop(self):
        """Decode bitstreams to codec-size RGB and feed the SR/upscale stage."""
        while not self._stop_event.is_set():
            try:
                item = self._decode_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self._put_latest(self._sr_queue, None)
                break

            frame_id, bitstream = item
            try:
                marker = parse_over_budget_marker(bitstream)
                if marker is not None:
                    self.frame_skipped.emit({
                        "frame_id": frame_id,
                        "over_budget_frames": 1,
                        "over_budget_last": marker.frame_id,
                        "over_budget_bytes": marker.packed_len,
                        "over_budget_max": marker.max_len,
                        "bitstream_bytes": marker.packed_len,
                        "chunks_needed": math.ceil(marker.max_len / 280),
                        "bpp": 0.0,
                    })
                    continue

                t0 = time.perf_counter()
                x_hat = self._decoder.decode_frame(bitstream)
                t1 = time.perf_counter()

                pixels = max(1, x_hat.shape[0] * x_hat.shape[1] * x_hat.shape[2])
                bpp = (len(bitstream) * 8) / pixels if len(bitstream) > 0 else 0
                stats = {
                    "frame_id": frame_id,
                    "bitstream_bytes": len(bitstream),
                    "chunks_needed": math.ceil(len(bitstream) / 280),
                    "bpp": bpp,
                    "codec_ms": (t1 - t0) * 1000,
                }
                if self._rx_fused_sr_trt_engine:
                    stats["sr_already_applied"] = True
                stats.update(self._record_timing("codec", stats["codec_ms"]))
                self._put_latest(self._sr_queue, (frame_id, x_hat, stats))
            except Exception as exc:
                self.decode_error.emit(f"decode fail f{frame_id}: {exc}")

    def _sr_loop(self):
        """Run SR/upscale as a separate stage so decode and SR can overlap."""
        while not self._stop_event.is_set():
            try:
                item = self._sr_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break

            frame_id, x_hat, stats = item
            try:
                if not self._sr_queue.empty():
                    self._sr_queue_drops += 1
                    continue

                t0 = time.perf_counter()
                if stats.get("sr_already_applied"):
                    rgb_out = x_hat
                elif self._sr_backend == "msa":
                    rgb_out = self._sr.super_resolve(x_hat)
                elif self._sr_backend in ("realesr", "rlfn"):
                    rgb_out = self._sr.enhance(x_hat)
                else:
                    rgb_out = cv2.resize(
                        x_hat,
                        (self._display_size, self._display_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                t1 = time.perf_counter()

                stats = dict(stats)
                stats["sr_ms"] = (t1 - t0) * 1000
                stats.update(self._record_timing("sr", stats["sr_ms"]))
                stats["decode_ms"] = stats["codec_ms"] + stats["sr_ms"]
                stats["sr_backend"] = "rlfn_trt" if self._sr_backend == "rlfn" and self._sr_engine == "tensorrt" else self._sr_backend
                stats["sr_scale"] = self._sr_scale
                stats["output_shape"] = tuple(int(v) for v in rgb_out.shape)
                stats["decode_queue"] = self._decode_queue.qsize()
                stats["sr_queue"] = self._sr_queue.qsize()
                stats["decode_queue_drops"] = self._decode_queue_drops
                stats["sr_queue_drops"] = self._sr_queue_drops
                self.frame_ready.emit(rgb_out, stats)
            except Exception as exc:
                self.decode_error.emit(f"sr fail f{frame_id}: {exc}")
