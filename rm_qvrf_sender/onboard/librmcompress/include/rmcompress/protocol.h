#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <cstddef>

constexpr uint32_t MAGIC = 0x31563152;  // "R1V1" little-endian
constexpr int HEADER_LEN = 20;
constexpr int MAX_PAYLOAD = 280;
constexpr int CHUNK_SIZE = 300;
constexpr int FIXED_CHUNKS_PER_FRAME = 2;
constexpr uint16_t FLAG_FIXED4_FEC_TAIL = 0x8000;
constexpr uint16_t FLAG_SESSION_MASK = 0x7FFF;

// Experimental MS-SSIM QVRF bitstream constants. The active QVRF runtime is
// the Python/Torch sender/decoder path with hs_backend=0.
constexpr uint32_t MSSSIM_MAGIC = 0x4756534D;  // "MSVG" (written via _write_u32le, LE)
constexpr uint8_t  MSSSIM_VERSION = 1;

// Special payload emitted when compression cannot fit the strict 2-chunk budget.
// Receiver must keep displaying the previous frame and surface this in Debug.
constexpr uint32_t OVER_BUDGET_MAGIC = 0x5245564F;  // "OVER" little-endian
constexpr uint16_t OVER_BUDGET_VERSION = 1;

struct ChunkHeader {
    uint32_t frame_id;
    uint8_t  chunk_id;
    uint8_t  chunk_count;
    uint16_t payload_len;
    uint32_t payload_crc32;
    uint16_t flags;
};

// Pack bitstream into 300B chunks (matches Python protocol.pack_frame)
std::vector<std::vector<uint8_t>> pack_frame(uint32_t frame_id, const uint8_t* bitstream,
                                             int len, uint16_t flags = 0);

// Pack exactly fixed_chunks 300B chunks. Payloads longer than
// fixed_chunks * 280B are invalid. Empty tail chunks are zero-length
// placeholders.
std::vector<std::vector<uint8_t>> pack_frame_fixed_n(
    uint32_t frame_id, const uint8_t* bitstream, int len,
    int fixed_chunks, uint16_t flags = 0);

// Pack fixed_chunks physical chunks while reserving the final chunk(s) for FEC.
// fec_data_chunks is the number of chunks that may carry original bitstream
// bytes. For fixed_chunks=4 and fec_data_chunks=3, C3 carries C0..C2 XOR.
std::vector<std::vector<uint8_t>> pack_frame_fixed_n(
    uint32_t frame_id, const uint8_t* bitstream, int len,
    int fixed_chunks, int fec_data_chunks, uint16_t flags);

// Pack exactly two 300B chunks. Kept for legacy callers.
std::vector<std::vector<uint8_t>> pack_frame_fixed2(
    uint32_t frame_id, const uint8_t* bitstream, int len, uint16_t flags = 0);

std::vector<uint8_t> pack_over_budget_marker(
    uint32_t frame_id, int packed_len, int max_len, float beta);

// Pack rANS compressed strings into a bit-exact binary format matching
// Python codec_decoder._pack_strings().
// Format: z_shape[2B LE] + y_count[4B LE] + (y_len[4B LE] + y_data)*
//         + z_count[4B LE] + (z_len[4B LE] + z_data)*
//         + beta[4B f32 LE] + hs_backend[2B u16 LE]
std::vector<uint8_t> pack_strings(
    const std::vector<std::string>& y_strings,
    const std::vector<std::string>& z_strings,
    int z_h, int z_w, float beta = 1.0f, uint16_t hs_backend = 0);

// Pack C++ MS-SSIM QVRF bitstreams. These are marked hs_backend=1; the Python
// GUI decoder must use OpenVINO FP32 CPU h_s for this bitstream type.
// Format: "MSVG"[4B] + version[1B u8] + gain[4B f32 LE] + z_h[2B u16 LE]
//         + z_w[2B u16 LE] + y_count[4B u32 LE] + (y_len[4B u32 LE] + y_data)*
//         + z_count[4B u32 LE] + (z_len[4B u32 LE] + z_data)*
std::vector<uint8_t> pack_msssim_qvrf(
    const std::vector<std::string>& y_strings,
    const std::vector<std::string>& z_strings,
    int z_h, int z_w, float gain);
