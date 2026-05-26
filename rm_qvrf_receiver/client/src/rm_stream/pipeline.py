"""Async pipeline with bounded queues and latest-frame semantics.

M0: source -> compress -> transport (pass-through) -> decode -> SR -> display
M2: producer half: source -> compress -> MQTT publish
    consumer half: MQTT callback -> decode -> SR -> display
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from rm_stream.display import Display
from rm_stream.codec_decoder import MbtDecoder
from rm_stream.sr_model import SrModel
from rm_stream.protocol import pack_frame
from rm_stream.frame_assembler import FrameAssembler

# Ensure legacy compress-ai-gray-minimal is on the path for sr_model imports
_LEGACY = Path(__file__).resolve().parents[3] / "compress-ai-gray-minimal"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

LOGGER = logging.getLogger(__name__)
QUEUE_SIZE = 2


class AsyncPipeline:
    """5-stage async pipeline: source -> compress -> transport -> decode -> SR -> display."""

    def __init__(
        self,
        video_path: str,
        mbt_checkpoint: str,
        sr_checkpoint: str,
        max_frames: int = 0,
    ) -> None:
        self._video_path = video_path
        self._mbt_checkpoint = mbt_checkpoint
        self._sr_checkpoint = sr_checkpoint
        self._max_frames = max_frames

        self._compress_q: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(QUEUE_SIZE)
        self._transport_q: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(QUEUE_SIZE)
        self._decode_q: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(QUEUE_SIZE)
        self._sr_q: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(QUEUE_SIZE)
        self._display_q: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(QUEUE_SIZE)

        self._display = Display()
        self._running = False
        self._stats: dict[str, int] = dict(
            frames_read=0, frames_compressed=0, frames_decoded=0,
            frames_sr=0, frames_displayed=0, frames_dropped=0,
        )

    async def run(self) -> dict[str, int]:
        self._running = True
        tasks = [
            asyncio.create_task(self._source_task()),
            asyncio.create_task(self._compress_task()),
            asyncio.create_task(self._transport_task()),
            asyncio.create_task(self._decode_task()),
            asyncio.create_task(self._sr_task()),
            asyncio.create_task(self._display_task()),
        ]
        await asyncio.gather(*tasks)
        self._display.close()
        return self._stats

    async def _source_task(self) -> None:
        """Read frames from video file, push to compress queue."""
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._video_path}")

        frame_id = 0
        loop = asyncio.get_running_loop()
        while self._running:
            if self._max_frames and frame_id >= self._max_frames:
                break

            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                break

            await self._enqueue(self._compress_q, (frame_id, frame))
            self._stats["frames_read"] += 1
            frame_id += 1

        cap.release()
        self._running = False

    async def _compress_task(self) -> None:
        decoder = MbtDecoder(self._mbt_checkpoint)

        while self._running:
            item = await self._dequeue(self._compress_q)
            if item is None:
                continue
            frame_id, frame = item

            bitstream = decoder.compress_numpy(frame)
            chunks = pack_frame(frame_id, bitstream)
            assembler = FrameAssembler(timeout_sec=5.0)
            assembled: bytes | None = None
            for c in chunks:
                assembled = assembler.add_chunk(c)
            if assembled is not None:
                await self._enqueue(self._transport_q, (frame_id, assembled))
                self._stats["frames_compressed"] += 1
            else:
                self._stats["frames_dropped"] += 1

    async def _transport_task(self) -> None:
        """In M0: pass-through. In M2: MQTT send/receive goes here."""
        while self._running:
            item = await self._dequeue(self._transport_q)
            if item is None:
                continue
            frame_id, bitstream = item
            await self._enqueue(self._decode_q, (frame_id, bitstream))

    async def _decode_task(self) -> None:
        decoder = MbtDecoder(self._mbt_checkpoint)

        while self._running:
            item = await self._dequeue(self._decode_q)
            if item is None:
                continue
            frame_id, bitstream = item

            x_hat = decoder.decode_frame(bitstream)
            await self._enqueue(self._sr_q, (frame_id, x_hat))
            self._stats["frames_decoded"] += 1

    async def _sr_task(self) -> None:
        sr = SrModel(self._sr_checkpoint)

        while self._running:
            item = await self._dequeue(self._sr_q)
            if item is None:
                continue
            frame_id, x_hat = item

            out = sr.super_resolve(x_hat)
            await self._enqueue(self._display_q, (frame_id, out))
            self._stats["frames_sr"] += 1

    async def _display_task(self) -> None:
        """Display frames in OpenCV window."""
        while self._running:
            item = await self._dequeue(self._display_q)
            if item is None:
                continue
            frame_id, frame = item

            keep_going = self._display.show(frame)
            self._stats["frames_displayed"] += 1
            if not keep_going:
                self._running = False

    async def _enqueue(self, q: asyncio.Queue, item) -> None:
        """Put item in queue. If full, drop oldest and put new."""
        if q.full():
            try:
                q.get_nowait()
                self._stats["frames_dropped"] += 1
            except asyncio.QueueEmpty:
                pass
        await q.put(item)

    async def _dequeue(self, q: asyncio.Queue):
        """Get item from queue. Returns None if pipeline stopped."""
        try:
            return await asyncio.wait_for(q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            if not self._running:
                return None
            return None


# ---------------------------------------------------------------------------
# M2 pipeline -- replaces pass-through transport with MQTT
# ---------------------------------------------------------------------------
class AsyncPipelineM2:
    """M2 async pipeline with MQTT transport.

    Supports two modes:
      - Producer: source -> compress -> MQTT publish_frame()
      - Consumer: MQTT callback -> decode -> SR -> display

    Each mode runs in its own process, connected by the MQTT broker.

    Producer usage::

        from rm_stream.mqtt_transport import MqttTransport
        transport = MqttTransport(broker_host="192.168.1.100", broker_port=1883)
        pipeline = AsyncPipelineM2.producer(
            video_path="video.mp4",
            mbt_checkpoint="...",
            transport=transport,
        )
        await pipeline.run()

    Consumer usage::

        from rm_stream.mqtt_transport import MqttTransport
        transport = MqttTransport(broker_host="192.168.1.100", broker_port=1883)
        pipeline = AsyncPipelineM2.consumer(
            mbt_checkpoint="...",
            sr_checkpoint="...",
            transport=transport,
        )
        await pipeline.run()
    """

    @staticmethod
    def producer(
        video_path: str,
        mbt_checkpoint: str,
        transport,  # MqttTransport
        max_frames: int = 0,
    ) -> "AsyncPipelineM2":
        """Create a producer pipeline: source -> compress -> MQTT publish.

        Parameters
        ----------
        video_path:
            Path to input video file.
        mbt_checkpoint:
            Path to MBT codec checkpoint.
        transport:
            MqttTransport instance (connected to broker).
        max_frames:
            Maximum frames to process (0 = unlimited).
        """
        return AsyncPipelineM2(
            mode="producer",
            video_path=video_path,
            mbt_checkpoint=mbt_checkpoint,
            sr_checkpoint="",
            transport=transport,
            max_frames=max_frames,
        )

    @staticmethod
    def consumer(
        mbt_checkpoint: str,
        sr_checkpoint: str,
        transport,  # MqttTransport
        max_frames: int = 0,
    ) -> "AsyncPipelineM2":
        """Create a consumer pipeline: MQTT receive -> decode -> SR -> display.

        Parameters
        ----------
        mbt_checkpoint:
            Path to MBT codec checkpoint.
        sr_checkpoint:
            Path to SR model checkpoint.
        transport:
            MqttTransport instance (connected to broker).
        max_frames:
            Maximum frames to display (0 = unlimited).
        """
        return AsyncPipelineM2(
            mode="consumer",
            video_path="",
            mbt_checkpoint=mbt_checkpoint,
            sr_checkpoint=sr_checkpoint,
            transport=transport,
            max_frames=max_frames,
        )

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        mode: str,
        video_path: str,
        mbt_checkpoint: str,
        sr_checkpoint: str,
        transport,  # MqttTransport
        max_frames: int = 0,
    ) -> None:
        if mode not in ("producer", "consumer"):
            raise ValueError(f"mode must be 'producer' or 'consumer', got {mode!r}")

        self._mode = mode
        self._video_path = video_path
        self._mbt_checkpoint = mbt_checkpoint
        self._sr_checkpoint = sr_checkpoint
        self._transport = transport
        self._max_frames = max_frames

        # Queues (producer side)
        self._compress_q: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(QUEUE_SIZE)
        self._transport_q: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(QUEUE_SIZE)

        # Queues (consumer side)
        self._decode_q: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(QUEUE_SIZE)
        self._sr_q: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(QUEUE_SIZE)
        self._display_q: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(QUEUE_SIZE)

        self._display = Display() if mode == "consumer" else None
        self._running = False
        self._frame_count = 0
        self._stats: dict[str, int] = dict(
            frames_read=0, frames_compressed=0, frames_sent=0,
            frames_decoded=0, frames_sr=0, frames_displayed=0,
            frames_dropped=0, frames_received=0,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    async def run(self) -> dict[str, int]:
        self._running = True

        if self._mode == "producer":
            await self._run_producer()
        else:
            await self._run_consumer()

        if self._display is not None:
            self._display.close()
        return self._stats

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------
    async def _run_producer(self) -> None:
        """Run producer pipeline: source -> compress -> MQTT publish."""
        tasks = [
            asyncio.create_task(self._source_task()),
            asyncio.create_task(self._compress_task()),
            asyncio.create_task(self._mqtt_publish_task()),
        ]
        await asyncio.gather(*tasks)

    async def _source_task(self) -> None:
        """Read frames from video file, push to compress queue."""
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._video_path}")

        frame_id = 0
        loop = asyncio.get_running_loop()
        while self._running:
            if self._max_frames and frame_id >= self._max_frames:
                break

            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                break

            await self._enqueue(self._compress_q, (frame_id, frame))
            self._stats["frames_read"] += 1
            frame_id += 1

        cap.release()
        self._running = False

    async def _compress_task(self) -> None:
        decoder = MbtDecoder(self._mbt_checkpoint)

        while self._running:
            item = await self._dequeue(self._compress_q)
            if item is None:
                continue
            frame_id, frame = item

            bitstream = decoder.compress_numpy(frame)
            await self._enqueue(self._transport_q, (frame_id, bitstream))
            self._stats["frames_compressed"] += 1

    async def _mqtt_publish_task(self) -> None:
        """Drain transport queue and publish frames via MQTT."""
        loop = asyncio.get_running_loop()
        while self._running:
            item = await self._dequeue(self._transport_q)
            if item is None:
                continue
            frame_id, bitstream = item

            # publish_frame is synchronous but brief; run in executor
            # to avoid blocking the event loop
            await loop.run_in_executor(
                None, self._transport.publish_frame, frame_id, bitstream
            )
            self._stats["frames_sent"] += 1

        # Stop the transport after all frames are sent
        self._transport.stop()
        LOGGER.info("Producer transport stopped")

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------
    async def _run_consumer(self) -> None:
        """Run consumer pipeline: MQTT receive -> decode -> SR -> display."""
        # Install MQTT callback that pushes received frames into decode_q.
        loop = asyncio.get_running_loop()

        def _on_frame(frame_id: int, bitstream: bytes) -> None:
            """Called from MQTT thread. Push frame into asyncio decode queue."""
            asyncio.run_coroutine_threadsafe(
                self._decode_q.put((frame_id, bitstream)), loop
            )
            self._stats["frames_received"] += 1

        # start_receiving is synchronous and starts a paho background thread.
        # It must be called from the main thread (paho constraint).
        self._transport.start_receiving(_on_frame)

        tasks = [
            asyncio.create_task(self._decode_task()),
            asyncio.create_task(self._sr_task()),
            asyncio.create_task(self._display_task()),
        ]
        await asyncio.gather(*tasks)

        self._transport.stop()
        LOGGER.info("Consumer transport stopped")

    async def _decode_task(self) -> None:
        decoder = MbtDecoder(self._mbt_checkpoint)

        while self._running:
            item = await self._dequeue(self._decode_q)
            if item is None:
                continue
            frame_id, bitstream = item

            x_hat = decoder.decode_frame(bitstream)
            await self._enqueue(self._sr_q, (frame_id, x_hat))
            self._stats["frames_decoded"] += 1

    async def _sr_task(self) -> None:
        sr = SrModel(self._sr_checkpoint)

        while self._running:
            item = await self._dequeue(self._sr_q)
            if item is None:
                continue
            frame_id, x_hat = item

            out = sr.super_resolve(x_hat)
            await self._enqueue(self._display_q, (frame_id, out))
            self._stats["frames_sr"] += 1

    async def _display_task(self) -> None:
        """Display frames in OpenCV window."""
        assert self._display is not None
        while self._running:
            item = await self._dequeue(self._display_q)
            if item is None:
                continue
            frame_id, frame = item

            keep_going = self._display.show(frame)
            self._stats["frames_displayed"] += 1
            self._frame_count += 1

            # Stop when we've displayed enough frames
            if self._max_frames and self._frame_count >= self._max_frames:
                self._running = False

            if not keep_going:
                self._running = False

    # ------------------------------------------------------------------
    # Queue helpers (shared with M0)
    # ------------------------------------------------------------------
    async def _enqueue(self, q: asyncio.Queue, item) -> None:
        """Put item in queue. If full, drop oldest and put new."""
        if q.full():
            try:
                q.get_nowait()
                self._stats["frames_dropped"] += 1
            except asyncio.QueueEmpty:
                pass
        await q.put(item)

    async def _dequeue(self, q: asyncio.Queue):
        """Get item from queue. Returns None if pipeline stopped."""
        try:
            return await asyncio.wait_for(q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            if not self._running:
                return None
            return None
