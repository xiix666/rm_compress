from __future__ import annotations

import logging
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger(__name__)


@dataclass
class VideoStats:
    packets: int = 0
    bytes: int = 0
    frames_completed: int = 0
    frames_dropped: int = 0
    first_packet_at: float | None = None
    last_packet_at: float | None = None
    last_addr: tuple[str, int] | None = None


@dataclass
class _FrameBuffer:
    frame_no: int
    total_bytes: int
    chunks: dict[int, bytes] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add(self, frag_no: int, payload: bytes) -> None:
        self.chunks[frag_no] = payload
        self.updated_at = time.time()

    def assembled(self) -> bytes:
        return b"".join(self.chunks[index] for index in sorted(self.chunks))


class VideoReceiver(threading.Thread):
    def __init__(
        self,
        bind_host: str,
        port: int,
        output: Path | None,
        preview: bool = False,
        endian: str = "little",
        frame_timeout_sec: float = 0.5,
        frame_callback: Callable[[bytes], None] | None = None,
        receive_buffer_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        super().__init__(name="video-receiver", daemon=True)
        self.bind_host = bind_host
        self.port = port
        self.output = output
        self.preview = preview
        self.endian = endian
        self.frame_timeout_sec = frame_timeout_sec
        self.frame_callback = frame_callback
        self.receive_buffer_bytes = receive_buffer_bytes
        self.stats = VideoStats()
        self._stopping = threading.Event()
        self._frames: dict[int, _FrameBuffer] = {}
        self._file = None
        self._ffplay: subprocess.Popen[bytes] | None = None
        self._last_file_flush = 0.0

    def stop(self) -> None:
        self._stopping.set()

    def run(self) -> None:
        if self.output is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.output.open("ab")
            LOGGER.info("append HEVC stream to %s", self.output)

        if self.preview:
            self._ffplay = subprocess.Popen(
                [
                    "ffplay",
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-fflags",
                    "nobuffer",
                    "-flags",
                    "low_delay",
                    "-f",
                    "hevc",
                    "-i",
                    "pipe:0",
                ],
                stdin=subprocess.PIPE,
            )
            LOGGER.info("started ffplay preview")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.receive_buffer_bytes)
        sock.bind((self.bind_host, self.port))
        sock.settimeout(0.2)
        LOGGER.info("listening UDP HEVC on %s:%s", self.bind_host, self.port)

        try:
            while not self._stopping.is_set():
                try:
                    packet, addr = sock.recvfrom(65535)
                except socket.timeout:
                    self._drop_stale_frames()
                    continue

                self._handle_packet(packet, addr)
                self._drop_stale_frames()
        finally:
            sock.close()
            if self._file is not None:
                self._file.flush()
                self._file.close()
            if self._ffplay is not None:
                if self._ffplay.stdin is not None:
                    self._ffplay.stdin.close()
                self._ffplay.terminate()

    def _handle_packet(self, packet: bytes, addr: tuple[str, int]) -> None:
        now = time.time()
        self.stats.packets += 1
        self.stats.bytes += len(packet)
        self.stats.first_packet_at = self.stats.first_packet_at or now
        self.stats.last_packet_at = now
        self.stats.last_addr = addr

        if len(packet) < 8:
            LOGGER.warning("short UDP packet from %s: %d bytes", addr, len(packet))
            return

        frame_no = int.from_bytes(packet[0:2], self.endian)
        frag_no = int.from_bytes(packet[2:4], self.endian)
        total_bytes = int.from_bytes(packet[4:8], self.endian)
        payload = packet[8:]

        if total_bytes <= 0 or total_bytes > 20 * 1024 * 1024:
            LOGGER.warning(
                "invalid video header addr=%s frame=%d frag=%d total=%d len=%d",
                addr,
                frame_no,
                frag_no,
                total_bytes,
                len(packet),
            )
            return

        frame = self._frames.get(frame_no)
        if frame is None or frame.total_bytes != total_bytes:
            frame = _FrameBuffer(frame_no=frame_no, total_bytes=total_bytes)
            self._frames[frame_no] = frame
        frame.add(frag_no, payload)

        data = frame.assembled()
        if len(data) >= total_bytes:
            complete = data[:total_bytes]
            self._frames.pop(frame_no, None)
            self.stats.frames_completed += 1
            LOGGER.debug(
                "video frame=%d fragments=%d bytes=%d from=%s",
                frame_no,
                len(frame.chunks),
                len(complete),
                addr,
            )
            self._write_hevc(complete)

    def _write_hevc(self, data: bytes) -> None:
        if self._file is not None:
            self._file.write(data)
            now = time.time()
            if now - self._last_file_flush >= 1.0:
                self._file.flush()
                self._last_file_flush = now
        if self.frame_callback is not None:
            self.frame_callback(data)
        if self._ffplay is not None and self._ffplay.stdin is not None:
            try:
                self._ffplay.stdin.write(data)
                self._ffplay.stdin.flush()
            except BrokenPipeError:
                LOGGER.warning("ffplay pipe closed")
                self._ffplay = None

    def _drop_stale_frames(self) -> None:
        now = time.time()
        stale = [
            frame_no
            for frame_no, frame in self._frames.items()
            if now - frame.updated_at > self.frame_timeout_sec
        ]
        for frame_no in stale:
            frame = self._frames.pop(frame_no)
            self.stats.frames_dropped += 1
            LOGGER.debug(
                "drop stale frame=%d fragments=%d total=%d",
                frame.frame_no,
                len(frame.chunks),
                frame.total_bytes,
            )
