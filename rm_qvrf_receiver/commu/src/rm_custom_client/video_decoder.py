from __future__ import annotations

import io
import logging
import queue
import subprocess
import threading
import time
from collections import deque

from PIL import Image

LOGGER = logging.getLogger(__name__)


def _hevc_nal_types(data: bytes) -> set[int]:
    nal_types: set[int] = set()
    index = 0
    end = len(data) - 4
    while index <= end:
        start_len = 0
        if data[index:index + 4] == b"\x00\x00\x00\x01":
            start_len = 4
        elif data[index:index + 3] == b"\x00\x00\x01":
            start_len = 3
        if start_len:
            header = index + start_len
            if header < len(data):
                nal_types.add((data[header] >> 1) & 0x3F)
            index = header + 2
            continue
        index += 1
    return nal_types


def _has_hevc_random_access_point(nal_types: set[int]) -> bool:
    has_parameter_sets = {32, 33, 34}.issubset(nal_types)
    has_idr = any(16 <= nal_type <= 21 for nal_type in nal_types)
    return has_parameter_sets and has_idr


class HevcPreviewDecoder:
    def __init__(self, output_queue: queue.Queue[Image.Image], width: int = 960, height: int = 540) -> None:
        self.output_queue = output_queue
        self.width = width
        self.height = height
        self.input_queue: queue.Queue[bytes] = queue.Queue(maxsize=600)
        self._stop = threading.Event()
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self.frames_decoded = 0
        self.last_frame_at: float | None = None
        self.last_error = ""
        self.waiting_for_keyframe = True
        self._pending: deque[bytes] = deque(maxlen=180)
        self._pending_nal_types: set[int] = set()

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._stop.clear()
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-analyzeduration",
                "0",
                "-probesize",
                "2048",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-flags2",
                "showall",
                "-f",
                "hevc",
                "-i",
                "pipe:0",
                "-an",
                "-vf",
                f"scale={self.width}:{self.height}:flags=fast_bilinear",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._reader = threading.Thread(target=self._read_frames, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()
        if self._writer is None or not self._writer.is_alive():
            self._writer = threading.Thread(target=self._write_hevc, daemon=True)
            self._writer.start()

    def stop(self) -> None:
        self._stop.set()
        self._close_process()

    def _close_process(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        self._proc.terminate()
        self._proc = None

    def _restart_process(self) -> None:
        self._close_process()
        self.waiting_for_keyframe = True
        self._pending.clear()
        self._pending_nal_types.clear()
        if not self._stop.is_set():
            return

    def feed(self, hevc_data: bytes) -> None:
        if self.waiting_for_keyframe:
            self._pending.append(hevc_data)
            self._pending_nal_types.update(_hevc_nal_types(hevc_data))
            if not _has_hevc_random_access_point(self._pending_nal_types):
                return
            self.waiting_for_keyframe = False
            self.start()
            pending_frames = list(self._pending)
            start_index = 0
            for index, pending in enumerate(pending_frames):
                if 32 in _hevc_nal_types(pending):
                    start_index = index
                    break
            for pending in pending_frames[start_index:]:
                self._put_input(pending)
            self._pending.clear()
            self._pending_nal_types.clear()
            return

        if self._proc is None or self._proc.poll() is not None:
            self._restart_process()
            self.feed(hevc_data)
            return
        self._put_input(hevc_data)

    def _put_input(self, hevc_data: bytes) -> None:
        try:
            self.input_queue.put_nowait(hevc_data)
        except queue.Full:
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.input_queue.put_nowait(hevc_data)
            except queue.Full:
                pass

    def _read_frames(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        frame_size = self.width * self.height * 3
        while not self._stop.is_set():
            chunks = bytearray()
            while len(chunks) < frame_size and not self._stop.is_set():
                chunk = self._proc.stdout.read(frame_size - len(chunks))
                if not chunk:
                    break
                chunks.extend(chunk)
            if len(chunks) != frame_size:
                break
            try:
                image = Image.frombytes("RGB", (self.width, self.height), bytes(chunks))
            except Exception:
                continue
            while self.output_queue.full():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
            self.output_queue.put(image)
            self.frames_decoded += 1
            self.last_frame_at = time.time()

    def _read_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        while not self._stop.is_set():
            line = self._proc.stderr.readline()
            if not line:
                break
            self.last_error = line.decode("utf-8", "replace").strip()

    def _write_hevc(self) -> None:
        while not self._stop.is_set():
            try:
                hevc_data = self.input_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._proc is None or self._proc.poll() is not None:
                self.start()
            if self._proc is None or self._proc.stdin is None:
                continue
            try:
                self._proc.stdin.write(hevc_data)
                self._proc.stdin.flush()
            except BrokenPipeError:
                LOGGER.warning("ffmpeg preview pipe closed")
                self._close_process()
                self.waiting_for_keyframe = True
                self._pending_nal_types.clear()
            except OSError as exc:
                LOGGER.warning("ffmpeg preview write failed: %s", exc)
                self._close_process()
                self.waiting_for_keyframe = True
                self._pending_nal_types.clear()
