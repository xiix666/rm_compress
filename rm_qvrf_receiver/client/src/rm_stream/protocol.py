"""300-byte chunk protocol per HANDOFF_0310.md application packet format."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

MAGIC = b"R1V1"
VERSION = 1
HEADER_LEN = 20
MAX_PAYLOAD = 280
CHUNK_SIZE = 300
OVER_BUDGET_MAGIC = b"OVER"
OVER_BUDGET_VERSION = 1
FLAG_FIXED4_FEC_TAIL = 0x8000
FLAG_SESSION_MASK = 0x7FFF

# struct format: magic(4s) version(B) header_len(B) frame_id(I)
#                chunk_id(B) chunk_count(B) payload_len(H) payload_crc32(I) flags(H)
_FMT = "<4sBBIBBHIH"


@dataclass
class ChunkHeader:
    frame_id: int       # uint32 LE
    chunk_id: int       # 0-based index
    chunk_count: int    # total chunks
    payload_len: int    # actual payload bytes (max 280)
    payload_crc32: int  # CRC32 of payload bytes
    flags: int          # reserved, zero


@dataclass
class OverBudgetMarker:
    frame_id: int
    packed_len: int
    max_len: int
    beta: float


def pack_over_budget_marker(frame_id: int, packed_len: int, max_len: int, beta: float = 0.0) -> bytes:
    return struct.pack("<4sHIIIf", OVER_BUDGET_MAGIC, OVER_BUDGET_VERSION,
                       frame_id, packed_len, max_len, beta)


def parse_over_budget_marker(data: bytes) -> OverBudgetMarker | None:
    if len(data) < struct.calcsize("<4sHIIIf"):
        return None
    magic, version, frame_id, packed_len, max_len, beta = struct.unpack(
        "<4sHIIIf", data[:struct.calcsize("<4sHIIIf")]
    )
    if magic != OVER_BUDGET_MAGIC or version != OVER_BUDGET_VERSION:
        return None
    return OverBudgetMarker(frame_id=frame_id, packed_len=packed_len, max_len=max_len, beta=beta)


def pack_frame(frame_id: int, bitstream: bytes, flags: int = 0) -> list[bytes]:
    """Pack a bitstream into one or more 300-byte chunks."""
    chunks: list[bytes] = []
    total = len(bitstream)
    chunk_count = (total + MAX_PAYLOAD - 1) // MAX_PAYLOAD if total > 0 else 1
    flags &= FLAG_SESSION_MASK

    for chunk_id in range(chunk_count):
        start = chunk_id * MAX_PAYLOAD
        end = min(start + MAX_PAYLOAD, total)
        payload = bitstream[start:end]
        payload_len = len(payload)

        # Pad to MAX_PAYLOAD and CRC the full 280-byte payload area
        # (including zero-padding) for whole-chunk integrity.
        padded = payload + b"\x00" * (MAX_PAYLOAD - payload_len)
        payload_crc32 = zlib.crc32(padded) & 0xFFFFFFFF

        header = struct.pack(
            _FMT,
            MAGIC,         # 0..3
            VERSION,       # 4
            HEADER_LEN,    # 5
            frame_id,      # 6..9
            chunk_id,      # 10
            chunk_count,   # 11
            payload_len,   # 12..13
            payload_crc32, # 14..17
            flags & 0xFFFF,  # 18..19 stream/session flags
        )
        assert len(header) == HEADER_LEN

        chunk = header + padded
        assert len(chunk) == CHUNK_SIZE
        chunks.append(chunk)

    return chunks


def pack_frame_fixed2(frame_id: int, bitstream: bytes, flags: int = 0) -> list[bytes]:
    return pack_frame_fixed_n(frame_id, bitstream, 2, flags=flags)


def pack_frame_fixed_n(
    frame_id: int,
    bitstream: bytes,
    fixed_chunks: int,
    flags: int = 0,
    fec_data_chunks: int | None = None,
) -> list[bytes]:
    if fixed_chunks <= 0 or fixed_chunks > 255:
        raise ValueError("fixed_chunks must be in [1,255]")
    data_chunks = fixed_chunks
    if fec_data_chunks is not None and fec_data_chunks > 0:
        if fec_data_chunks >= fixed_chunks:
            raise ValueError("fec_data_chunks must be smaller than fixed_chunks")
        data_chunks = fec_data_chunks
    max_payload = MAX_PAYLOAD * data_chunks
    if len(bitstream) > max_payload:
        raise ValueError(f"fixed{fixed_chunks} frame payload exceeds {max_payload} bytes")
    chunks: list[bytes] = []
    padded_payloads: list[bytes] = []
    payload_lens: list[int] = []
    for chunk_id in range(fixed_chunks):
        if chunk_id < data_chunks:
            start = chunk_id * MAX_PAYLOAD
            payload = bitstream[start:min(start + MAX_PAYLOAD, len(bitstream))]
        else:
            payload = b""
        payload_len = len(payload)
        padded = payload + b"\x00" * (MAX_PAYLOAD - payload_len)
        padded_payloads.append(padded)
        payload_lens.append(payload_len)

    flags &= FLAG_SESSION_MASK
    if fixed_chunks == 4 and data_chunks == 3:
        flags |= FLAG_FIXED4_FEC_TAIL

    if fixed_chunks == 4 and payload_lens[3] == 0 and any(payload_lens[:3]):
        parity = bytearray(MAX_PAYLOAD)
        for padded in padded_payloads[:3]:
            for i, value in enumerate(padded):
                parity[i] ^= value
        padded_payloads[3] = bytes(parity)

    for chunk_id in range(fixed_chunks):
        payload_len = payload_lens[chunk_id]
        padded = padded_payloads[chunk_id]
        payload_crc32 = zlib.crc32(padded) & 0xFFFFFFFF
        header = struct.pack(
            _FMT,
            MAGIC,
            VERSION,
            HEADER_LEN,
            frame_id,
            chunk_id,
            fixed_chunks,
            payload_len,
            payload_crc32,
            flags & 0xFFFF,
        )
        chunks.append(header + padded)
    return chunks


def parse_chunk_header(data: bytes) -> ChunkHeader | None:
    """Parse a 300B chunk, return ChunkHeader or None on any error."""
    if len(data) != CHUNK_SIZE:
        return None
    if data[0:4] != MAGIC:
        return None

    fields = struct.unpack(_FMT, data[:HEADER_LEN])
    magic, version, header_len, frame_id, chunk_id, chunk_count, \
        payload_len, payload_crc32, flags = fields

    if version != VERSION:
        return None
    if header_len != HEADER_LEN:
        return None
    if chunk_id >= chunk_count:
        return None
    if payload_len > MAX_PAYLOAD:
        return None

    # Validate CRC32 over full 280-byte payload area (including zero-padding)
    payload_area = data[HEADER_LEN:HEADER_LEN + MAX_PAYLOAD]
    computed_crc = zlib.crc32(payload_area) & 0xFFFFFFFF
    if computed_crc != payload_crc32:
        return None

    return ChunkHeader(
        frame_id=frame_id,
        chunk_id=chunk_id,
        chunk_count=chunk_count,
        payload_len=payload_len,
        payload_crc32=payload_crc32,
        flags=flags,
    )


def assemble_frame(chunks: list[bytes]) -> bytes | None:
    """Reassemble a frame from its chunks. Returns None if incomplete/invalid."""
    if not chunks:
        return None

    # Parse all chunk headers, returning None on any parse failure
    headers: list[ChunkHeader] = []
    for c in chunks:
        h = parse_chunk_header(c)
        if h is None:
            return None
        headers.append(h)

    # All chunks must belong to the same frame, chunk_count must match
    frame_id = headers[0].frame_id
    expected_count = headers[0].chunk_count
    if len(headers) != expected_count:
        return None

    # Sort by chunk_id and verify completeness
    headers.sort(key=lambda h: h.chunk_id)
    for i, h in enumerate(headers):
        if h.frame_id != frame_id:
            return None
        if h.chunk_id != i:
            return None
        if h.chunk_count != expected_count:
            return None

    # Extract and concatenate payloads
    result = bytearray()
    for h in headers:
        for c in chunks:
            c_hdr = parse_chunk_header(c)
            if c_hdr is not None and c_hdr.chunk_id == h.chunk_id:
                payload = c[HEADER_LEN:HEADER_LEN + h.payload_len]
                result.extend(payload)
                break

    return bytes(result)
