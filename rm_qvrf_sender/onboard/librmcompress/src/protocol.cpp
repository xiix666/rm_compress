#include "rmcompress/protocol.h"

#include <cstring>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// CRC-32 (standard, matches Python zlib.crc32)
// poly=0xEDB88320 (reflected 0x04C11DB7), init=0xFFFFFFFF, xorout=0xFFFFFFFF
// ---------------------------------------------------------------------------
static uint32_t _CRC32_TABLE[256];

static void _make_crc32_table() {
    for (int i = 0; i < 256; i++) {
        uint32_t crc = static_cast<uint32_t>(i);
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320U;
            } else {
                crc >>= 1;
            }
        }
        _CRC32_TABLE[i] = crc;
    }
}

static uint32_t _crc32(const uint8_t* data, int len) {
    static bool inited = false;
    if (!inited) {
        _make_crc32_table();
        inited = true;
    }
    uint32_t crc = 0xFFFFFFFF;
    for (int i = 0; i < len; i++) {
        crc = _CRC32_TABLE[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

static void _write_u16le(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

static void _write_u32le(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

static void _write_f32le(std::vector<uint8_t>& buf, float v) {
    uint32_t raw;
    static_assert(sizeof(raw) == sizeof(v), "float must be 32-bit");
    std::memcpy(&raw, &v, sizeof(raw));
    _write_u32le(buf, raw);
}

// ---------------------------------------------------------------------------
// pack_frame — matches Python rm_stream.protocol.pack_frame()
// ---------------------------------------------------------------------------

std::vector<std::vector<uint8_t>> pack_frame(uint32_t frame_id, const uint8_t* bitstream,
                                             int len, uint16_t flags) {
    flags = static_cast<uint16_t>(flags & FLAG_SESSION_MASK);
    std::vector<std::vector<uint8_t>> chunks;
    int total = len;
    int chunk_count = (total > 0) ? ((total + MAX_PAYLOAD - 1) / MAX_PAYLOAD) : 1;

    for (int chunk_id = 0; chunk_id < chunk_count; chunk_id++) {
        int start = chunk_id * MAX_PAYLOAD;
        int end = std::min(start + MAX_PAYLOAD, total);
        int payload_len = end - start;

        // Build padded payload (280 bytes, zero-padded)
        std::vector<uint8_t> padded(MAX_PAYLOAD, 0);
        if (payload_len > 0) {
            std::memcpy(padded.data(), bitstream + start, payload_len);
        }

        // CRC32 over full 280-byte padded payload area (matches Python)
        uint32_t payload_crc32 = _crc32(padded.data(), MAX_PAYLOAD);

        // Build chunk: 20-byte header + 280-byte padded payload = 300 bytes
        std::vector<uint8_t> chunk(CHUNK_SIZE, 0);

        // Pack header (little-endian)
        // MAGIC "R1V1" = bytes [0x52, 0x31, 0x56, 0x31]
        chunk[0] = 'R';
        chunk[1] = '1';
        chunk[2] = 'V';
        chunk[3] = '1';
        // VERSION
        chunk[4] = 1;
        // HEADER_LEN
        chunk[5] = HEADER_LEN;
        // frame_id (uint32 LE)
        std::memcpy(chunk.data() + 6, &frame_id, 4);
        // chunk_id (uint8)
        chunk[10] = static_cast<uint8_t>(chunk_id);
        // chunk_count (uint8)
        chunk[11] = static_cast<uint8_t>(chunk_count);
        // payload_len (uint16 LE)
        chunk[12] = static_cast<uint8_t>(payload_len & 0xFF);
        chunk[13] = static_cast<uint8_t>((payload_len >> 8) & 0xFF);
        // payload_crc32 (uint32 LE)
        std::memcpy(chunk.data() + 14, &payload_crc32, 4);
        // flags/session id (uint16 LE)
        chunk[18] = static_cast<uint8_t>(flags & 0xFF);
        chunk[19] = static_cast<uint8_t>((flags >> 8) & 0xFF);

        // Copy padded payload
        std::memcpy(chunk.data() + HEADER_LEN, padded.data(), MAX_PAYLOAD);

        chunks.push_back(std::move(chunk));
    }

    return chunks;
}

std::vector<std::vector<uint8_t>> pack_frame_fixed2(
    uint32_t frame_id, const uint8_t* bitstream, int len, uint16_t flags) {
    return pack_frame_fixed_n(frame_id, bitstream, len, FIXED_CHUNKS_PER_FRAME, flags);
}

std::vector<std::vector<uint8_t>> pack_frame_fixed_n(
    uint32_t frame_id, const uint8_t* bitstream, int len,
    int fixed_chunks, uint16_t flags) {
    return pack_frame_fixed_n(frame_id, bitstream, len, fixed_chunks, 0, flags);
}

std::vector<std::vector<uint8_t>> pack_frame_fixed_n(
    uint32_t frame_id, const uint8_t* bitstream, int len,
    int fixed_chunks, int fec_data_chunks, uint16_t flags) {
    if (fixed_chunks <= 0 || fixed_chunks > 255) {
        throw std::runtime_error("pack_frame_fixed_n: invalid fixed chunk count");
    }
    flags = static_cast<uint16_t>(flags & FLAG_SESSION_MASK);
    int data_chunks = fixed_chunks;
    if (fec_data_chunks > 0) {
        if (fec_data_chunks >= fixed_chunks) {
            throw std::runtime_error("pack_frame_fixed_n: invalid FEC data chunk count");
        }
        data_chunks = fec_data_chunks;
    }
    if (fixed_chunks == 4 && data_chunks == 3) {
        flags = static_cast<uint16_t>(flags | FLAG_FIXED4_FEC_TAIL);
    }
    if (len > MAX_PAYLOAD * data_chunks) {
        throw std::runtime_error("pack_frame_fixed_n: payload exceeds fixed chunk budget");
    }

    std::vector<std::vector<uint8_t>> chunks;
    chunks.reserve(static_cast<size_t>(fixed_chunks));
    std::vector<std::vector<uint8_t>> padded_payloads;
    std::vector<int> payload_lens;
    padded_payloads.reserve(static_cast<size_t>(fixed_chunks));
    payload_lens.reserve(static_cast<size_t>(fixed_chunks));

    for (int chunk_id = 0; chunk_id < fixed_chunks; chunk_id++) {
        int payload_len = 0;
        int start = chunk_id * MAX_PAYLOAD;
        if (chunk_id < data_chunks) {
            int remaining = std::max(0, len - start);
            payload_len = std::min(MAX_PAYLOAD, remaining);
        }

        std::vector<uint8_t> padded(MAX_PAYLOAD, 0);
        if (payload_len > 0) {
            std::memcpy(padded.data(), bitstream + start, payload_len);
        }
        padded_payloads.push_back(std::move(padded));
        payload_lens.push_back(payload_len);
    }

    if (fixed_chunks == 4 && payload_lens[3] == 0 &&
        (payload_lens[0] > 0 || payload_lens[1] > 0 || payload_lens[2] > 0)) {
        std::fill(padded_payloads[3].begin(), padded_payloads[3].end(), 0);
        for (int chunk_id = 0; chunk_id < 3; ++chunk_id) {
            for (int i = 0; i < MAX_PAYLOAD; ++i) {
                padded_payloads[3][i] ^= padded_payloads[chunk_id][i];
            }
        }
    }

    for (int chunk_id = 0; chunk_id < fixed_chunks; chunk_id++) {
        int payload_len = payload_lens[chunk_id];
        const std::vector<uint8_t>& padded = padded_payloads[chunk_id];

        uint32_t payload_crc32 = _crc32(padded.data(), MAX_PAYLOAD);
        std::vector<uint8_t> chunk(CHUNK_SIZE, 0);
        chunk[0] = 'R';
        chunk[1] = '1';
        chunk[2] = 'V';
        chunk[3] = '1';
        chunk[4] = 1;
        chunk[5] = HEADER_LEN;
        std::memcpy(chunk.data() + 6, &frame_id, 4);
        chunk[10] = static_cast<uint8_t>(chunk_id);
        chunk[11] = static_cast<uint8_t>(fixed_chunks);
        chunk[12] = static_cast<uint8_t>(payload_len & 0xFF);
        chunk[13] = static_cast<uint8_t>((payload_len >> 8) & 0xFF);
        std::memcpy(chunk.data() + 14, &payload_crc32, 4);
        chunk[18] = static_cast<uint8_t>(flags & 0xFF);
        chunk[19] = static_cast<uint8_t>((flags >> 8) & 0xFF);
        std::memcpy(chunk.data() + HEADER_LEN, padded.data(), MAX_PAYLOAD);
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

std::vector<uint8_t> pack_over_budget_marker(
    uint32_t frame_id, int packed_len, int max_len, float beta) {
    std::vector<uint8_t> marker;
    marker.reserve(18);
    _write_u32le(marker, OVER_BUDGET_MAGIC);
    _write_u16le(marker, OVER_BUDGET_VERSION);
    _write_u32le(marker, frame_id);
    _write_u32le(marker, static_cast<uint32_t>(packed_len));
    _write_u32le(marker, static_cast<uint32_t>(max_len));
    _write_f32le(marker, beta);
    return marker;
}

// ---------------------------------------------------------------------------
// pack_strings — compact binary serialization matching Python _pack_strings()
// ---------------------------------------------------------------------------

std::vector<uint8_t> pack_strings(
    const std::vector<std::string>& y_strings,
    const std::vector<std::string>& z_strings,
    int z_h, int z_w, float beta, uint16_t hs_backend)
{
    std::vector<uint8_t> buf;

    // z spatial shape (2 uint16 LE)
    _write_u16le(buf, static_cast<uint16_t>(z_h));
    _write_u16le(buf, static_cast<uint16_t>(z_w));

    // y_strings: count + (len + data)*
    _write_u32le(buf, static_cast<uint32_t>(y_strings.size()));
    for (const auto& s : y_strings) {
        _write_u32le(buf, static_cast<uint32_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }

    // z_strings: count + (len + data)*
    _write_u32le(buf, static_cast<uint32_t>(z_strings.size()));
    for (const auto& s : z_strings) {
        _write_u32le(buf, static_cast<uint32_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }

    // beta-RC inverse scale. Python decoder treats missing trailer as beta=1.0.
    _write_f32le(buf, beta);
    _write_u16le(buf, hs_backend);

    return buf;
}

// ---------------------------------------------------------------------------
// pack_msssim_qvrf — matches Python msssim_qvrf_codec.pack_msssim_qvrf()
// ---------------------------------------------------------------------------

std::vector<uint8_t> pack_msssim_qvrf(
    const std::vector<std::string>& y_strings,
    const std::vector<std::string>& z_strings,
    int z_h, int z_w, float gain)
{
    std::vector<uint8_t> buf;

    // Header: magic "MSVG" (4B) + version (1B) + gain (4B f32 LE)
    //         + z_h (2B u16 LE) + z_w (2B u16 LE) = 13 bytes total
    _write_u32le(buf, MSSSIM_MAGIC);
    buf.push_back(MSSSIM_VERSION);
    _write_f32le(buf, gain);
    _write_u16le(buf, static_cast<uint16_t>(z_h));
    _write_u16le(buf, static_cast<uint16_t>(z_w));

    // y_strings: count + (len + data)*
    _write_u32le(buf, static_cast<uint32_t>(y_strings.size()));
    for (const auto& s : y_strings) {
        _write_u32le(buf, static_cast<uint32_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }

    // z_strings: count + (len + data)*
    _write_u32le(buf, static_cast<uint32_t>(z_strings.size()));
    for (const auto& s : z_strings) {
        _write_u32le(buf, static_cast<uint32_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }

    // Experimental C++/OpenVINO QVRF marker. The production Python GUI QVRF
    // decoder currently accepts only hs_backend=0 Torch bitstreams.
    _write_u16le(buf, static_cast<uint16_t>(1));

    return buf;
}
