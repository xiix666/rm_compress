"""Frame assembler: collects chunks, reassembles frames, drops incomplete."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from rm_stream.protocol import (
    FLAG_FIXED4_FEC_TAIL,
    FLAG_SESSION_MASK,
    HEADER_LEN,
    MAX_PAYLOAD,
    parse_chunk_header,
    ChunkHeader,
)


@dataclass
class _PendingFrame:
    frame_id: int
    chunk_count: int
    flags: int = 0
    chunks: dict[int, bytes] = field(default_factory=dict)
    padded_chunks: dict[int, bytes] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def is_complete(self) -> bool:
        return len(self.chunks) == self.chunk_count


class FrameAssembler:
    """Collects 300B chunks and reassembles complete frames.

    Drops incomplete frames after ``timeout_sec``.
    """

    def __init__(self, timeout_sec: float = 0.12, require_start_frame: bool = False) -> None:
        self._timeout = timeout_sec
        self._pending: dict[tuple[int, int], _PendingFrame] = {}
        self._recent_completed: set[tuple[int, int]] = set()
        self._recent_completed_order: deque[tuple[int, int]] = deque()
        self._require_start_frame = require_start_frame
        self._active_flags: int | None = None
        self.dropped_stale = 0
        self.dropped_duplicates = 0
        self.dropped_session = 0
        self.completed_frames = 0
        self.stale_missing_chunk0 = 0
        self.stale_missing_chunk1 = 0
        self.stale_missing_chunks: dict[int, int] = {}
        self.recovered_single_chunk = 0
        self.stale_last_frame_id: int | None = None
        self.stale_last_missing: tuple[int, ...] = ()

    def add_chunk(self, data: bytes) -> bytes | None:
        """Feed a 300B chunk. Returns assembled bitstream if frame is complete."""
        self._drop_stale()
        header = parse_chunk_header(data)
        if header is None:
            return None

        if self._require_start_frame:
            session_flags = header.flags & FLAG_SESSION_MASK
            if self._active_flags is None:
                self._active_flags = session_flags
                self._pending.clear()
            elif session_flags != self._active_flags:
                self.dropped_session += len(self._pending)
                self._active_flags = session_flags
                self._pending.clear()

        key = (header.flags & FLAG_SESSION_MASK, header.frame_id)
        if key in self._recent_completed:
            return None

        pf = self._pending.get(key)
        if pf is None:
            pf = _PendingFrame(
                frame_id=header.frame_id,
                chunk_count=header.chunk_count,
                flags=header.flags,
            )
            self._pending[key] = pf

        if header.chunk_id in pf.chunks:
            self.dropped_duplicates += 1
            return None

        if header.chunk_count != pf.chunk_count:
            self._pending.pop(key, None)
            return None
        pf.flags |= header.flags & ~FLAG_SESSION_MASK

        payload = data[HEADER_LEN : HEADER_LEN + header.payload_len]
        pf.chunks[header.chunk_id] = payload
        pf.padded_chunks[header.chunk_id] = data[HEADER_LEN : HEADER_LEN + MAX_PAYLOAD]
        self._try_recover_fec(pf)

        if pf.is_complete() or self._has_complete_payload_prefix(pf):
            self._pending.pop(key)
            self._mark_completed(key)
            self.completed_frames += 1
            return self._assemble(pf)

        return None

    def poll_recovered(self) -> bytes | None:
        """Drop stale frames and return one recoverable complete bitstream, if any."""
        return self._drop_stale()

    def _assemble(self, pf: _PendingFrame) -> bytes:
        result = bytearray()
        for i in range(pf.chunk_count):
            if i not in pf.chunks:
                break
            result.extend(pf.chunks[i])
            if len(pf.chunks[i]) < MAX_PAYLOAD:
                break
        return bytes(result)

    def _has_complete_payload_prefix(self, pf: _PendingFrame) -> bool:
        """Return True once contiguous chunks from C0 contain the full payload.

        Fixed-N senders append zero-length placeholder chunks after the
        bitstream. If the last real payload chunk is shorter than MAX_PAYLOAD,
        any later chunks are known placeholders and are not required for decode.
        Full-size chunks still require the next chunk, so missing middle data is
        never treated as complete.
        """
        for chunk_id in range(pf.chunk_count):
            payload = pf.chunks.get(chunk_id)
            if payload is None:
                if (
                    pf.chunk_count == 4
                    and chunk_id == 3
                    and (pf.flags & FLAG_FIXED4_FEC_TAIL)
                ):
                    return True
                return False
            if len(payload) < MAX_PAYLOAD:
                return True
        return False

    def _try_recover_fec(self, pf: _PendingFrame) -> None:
        """Recover one missing full payload chunk using fixed4 XOR parity.

        This is only valid when the sender explicitly marks fixed4 FEC-tail
        mode. Plain 4-data-chunk mode must not reinterpret C3 as parity.
        """
        if pf.chunk_count != 4:
            return
        if not (pf.flags & FLAG_FIXED4_FEC_TAIL):
            return
        if 3 not in pf.chunks or len(pf.chunks[3]) != 0:
            return
        parity = pf.padded_chunks.get(3)
        if not parity or not any(parity):
            return
        missing = [i for i in range(3) if i not in pf.chunks]
        if len(missing) != 1:
            return
        missing_id = missing[0]
        recovered = bytearray(parity)
        for chunk_id in range(3):
            if chunk_id == missing_id:
                continue
            padded = pf.padded_chunks.get(chunk_id)
            if padded is None:
                return
            for i, value in enumerate(padded):
                recovered[i] ^= value
        pf.padded_chunks[missing_id] = bytes(recovered)
        pf.chunks[missing_id] = bytes(recovered)

    def _mark_completed(self, key: tuple[int, int]) -> None:
        self._recent_completed.add(key)
        self._recent_completed_order.append(key)
        while len(self._recent_completed_order) > 4096:
            old = self._recent_completed_order.popleft()
            self._recent_completed.discard(old)

    def _drop_stale(self) -> bytes | None:
        now = time.time()
        stale = [
            (key, pf)
            for key, pf in self._pending.items()
            if now - pf.created_at > self._timeout
        ]
        recovered: bytes | None = None
        for key, pf in stale:
            missing = tuple(i for i in range(pf.chunk_count) if i not in pf.chunks)
            if recovered is None and self._can_recover_prefix(pf):
                recovered = self._assemble(pf)
                self.recovered_single_chunk += 1
                self.completed_frames += 1
                self._mark_completed(key)
                self._pending.pop(key, None)
                continue
            if 0 in missing:
                self.stale_missing_chunk0 += 1
            if 1 in missing:
                self.stale_missing_chunk1 += 1
            for chunk_id in missing:
                self.stale_missing_chunks[chunk_id] = self.stale_missing_chunks.get(chunk_id, 0) + 1
            self.stale_last_frame_id = pf.frame_id
            self.stale_last_missing = missing
            self._pending.pop(key, None)
        self.dropped_stale += len(stale)
        return recovered

    def _can_recover_prefix(self, pf: _PendingFrame) -> bool:
        """Recover frames whose payload prefix is already complete.

        Fixed-N senders may emit zero-length placeholder chunks after the real
        payload. Once a contiguous prefix from C0 contains a short payload
        chunk, the bitstream is complete and missing tail placeholders are safe
        to ignore. This never decodes a partial full-size chunk.
        """
        return self._has_complete_payload_prefix(pf)

    @property
    def pending_frames(self) -> int:
        return len(self._pending)
