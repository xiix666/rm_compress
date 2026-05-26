#include "rmcompress/entropy_coder.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace rmcompress {

// ============================================================================
// rANS64 core — matches CompressAI (based on ryg_rans by Fabian Giesen)
// ============================================================================

static constexpr int  kPrecision       = 16;  // 16-bit CDF precision
static constexpr int  kBypassPrecision = 4;   // 4 bits per bypass chunk
static constexpr int  kMaxBypassVal    = (1 << kBypassPrecision) - 1;  // 15

// rANS64 uses 64-bit state with 32-bit output words.
// RANS64_L is the lower bound for the state after renormalization.
static constexpr uint64_t kRansL = 1ull << 31;

using RansState = uint64_t;

// --- Encoder helpers (match CompressAI / ryg_rans rans64.h) ---

/// Initialize encoder state. Matches Rans64EncInit.
static inline void rans_enc_init(RansState* r) { *r = kRansL; }

/// Encode one symbol. Matches Rans64EncPut from ryg_rans rans64.h:
///   x' = ((x / freq) << scale_bits) + (x % freq) + start
/// Writes 32-bit output words to *--pptr when the state overflows.
static inline void rans_enc_put(RansState* r, uint32_t** pptr,
                                uint32_t start, uint32_t freq,
                                uint32_t scale_bits) {
    RansState x = *r;
    uint64_t x_max = ((kRansL >> scale_bits) << 32) * freq;
    if (x >= x_max) {
        *--(*pptr) = static_cast<uint32_t>(x);
        x >>= 32;
    }
    *r = ((x / freq) << scale_bits) + (x % freq) + start;
}

/// Flush encoder: write final state as two uint32 words.
/// Matches Rans64EncFlush from ryg_rans: writes bottom32 then top32 in reverse order.
/// Memory layout: [top32][bottom32], pptr at top32.
static inline void rans_enc_flush(RansState* r, uint32_t** pptr) {
    uint64_t x = *r;
    *--(*pptr) = static_cast<uint32_t>(x >> 32);  // top32 first (at higher addr)
    *--(*pptr) = static_cast<uint32_t>(x);        // bottom32   (at lower addr)
}

/// Write raw bypass bits. Matches CompressAI Rans64EncPutBits.
/// Uses threshold based on fixed 16-bit precision, not nbits.
static inline void rans_enc_put_bits(RansState* r, uint32_t** pptr,
                                     uint32_t val, uint32_t nbits) {
    // assert(nbits <= 16)
    // assert(val < (1u << nbits))
    RansState x = *r;
    // freq and x_max use the fixed 16-bit precision base, not nbits
    uint32_t freq = 1u << (16 - nbits);
    uint64_t x_max = ((kRansL >> 16) << 32) * freq;
    if (x >= x_max) {
        *--(*pptr) = static_cast<uint32_t>(x);
        x >>= 32;
    }
    *r = (x << nbits) | val;
}

// --- Decoder helpers (match CompressAI / ryg_rans rans64.h) ---

/// Initialize decoder from stream.
/// Matches Rans64DecInit from ryg_rans: memory [top32][bottom32], pptr at top32.
/// Reconstructs: x = top32 | (bottom32 << 32)
static inline void rans_dec_init(RansState* r, uint32_t** pptr) {
    uint64_t x;
    x  = static_cast<uint64_t>((*pptr)[0]) << 0;   // first word → low bits of x
    x |= static_cast<uint64_t>((*pptr)[1]) << 32;  // second word → high bits of x
    *pptr += 2;
    *r = x;
}

/// Get cumulative frequency from state (lower scale_bits).
/// Matches Rans64DecGet.
static inline uint32_t rans_dec_get(RansState* r, uint32_t scale_bits) {
    return static_cast<uint32_t>(*r) & ((1u << scale_bits) - 1);
}

/// Advance decoder state after decoding a symbol.
/// Matches Rans64DecAdvance.
static inline void rans_dec_advance(RansState* r, uint32_t** pptr,
                                     uint32_t start, uint32_t freq,
                                     uint32_t scale_bits) {
    uint64_t mask = (1u << scale_bits) - 1;
    uint64_t x = *r;
    x = freq * (x >> scale_bits) + (x & mask) - start;
    if (x < kRansL) {
        x = (x << 32) | **pptr;
        (*pptr)++;
    }
    *r = x;
}

/// Read raw bypass bits from the stream.
/// Matches CompressAI Rans64DecGetBits.
static inline uint32_t rans_dec_get_bits(RansState* r, uint32_t** pptr,
                                          uint32_t n_bits) {
    uint64_t x = *r;
    uint32_t val = x & ((1u << n_bits) - 1);
    x = x >> n_bits;
    if (x < kRansL) {
        x = (x << 32) | **pptr;
        (*pptr)++;
    }
    *r = x;
    return val;
}

// ============================================================================
// CDF binary format reader
// ============================================================================

// Magic and version for the .cdfs binary file
static constexpr uint32_t kCdfMagic   = 0x46444352;  // "RCDF"
static constexpr uint32_t kCdfVersion = 1;

#pragma pack(push, 1)
struct CdfFileHeader {
    uint32_t magic;
    uint32_t version;
    int32_t  precision;          // always 16
    int32_t  bottleneck_channels; // e.g. 128
    int32_t  bottleneck_max_len;  // max CDF row length
    int32_t  gaussian_tables;     // e.g. 64 (scale table entries)
    int32_t  gaussian_max_len;    // max CDF row length
    int32_t  scale_table_size;    // = gaussian_tables
    // Followed by:
    //   1. bottleneck_cdfs[128][bottleneck_max_len]   (int32 LE)
    //   2. bottleneck_offsets[128]                     (int32 LE)
    //   3. bottleneck_cdf_lens[128]                    (int32 LE)
    //   4. bottleneck_medians[128]                     (float32 LE)
    //   5. gaussian_cdfs[64][gaussian_max_len]         (int32 LE)
    //   6. gaussian_offsets[64]                        (int32 LE)
    //   7. gaussian_cdf_lens[64]                       (int32 LE)
    //   8. gaussian_scale_table[64]                    (float32 LE)
};
#pragma pack(pop)

bool EntropyCoder::load_cdfs(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size);
    ifs.read(reinterpret_cast<char*>(buf.data()), size);
    if (!ifs) return false;
    return load_cdfs(buf.data(), size);
}

bool EntropyCoder::load_cdfs(const uint8_t* data, size_t size) {
    if (size < sizeof(CdfFileHeader)) return false;

    const auto* hdr = reinterpret_cast<const CdfFileHeader*>(data);
    if (hdr->magic != kCdfMagic) return false;
    if (hdr->version != kCdfVersion) return false;

    int bc = hdr->bottleneck_channels;
    int bmax = hdr->bottleneck_max_len;
    int gc = hdr->gaussian_tables;
    int gmax = hdr->gaussian_max_len;
    int st_size = hdr->scale_table_size;

    const uint8_t* ptr = data + sizeof(CdfFileHeader);

    // Validate remaining size
    size_t needed = 0;
    needed += bc * bmax * sizeof(int32_t);   // bottleneck_cdfs
    needed += bc * sizeof(int32_t);          // bottleneck_offsets
    needed += bc * sizeof(int32_t);          // bottleneck_cdf_lens
    needed += bc * sizeof(float);            // bottleneck_medians
    needed += gc * gmax * sizeof(int32_t);   // gaussian_cdfs
    needed += gc * sizeof(int32_t);          // gaussian_offsets
    needed += gc * sizeof(int32_t);          // gaussian_cdf_lens
    needed += st_size * sizeof(float);       // gaussian_scale_table

    if (size - sizeof(CdfFileHeader) < needed) return false;

    // Read EntropyBottleneck CDFs
    _bottleneck_cdfs.resize(bc);
    _bottleneck_offsets.resize(bc);
    _bottleneck_cdf_lens.resize(bc);
    _bottleneck_medians.resize(bc);
    _bottleneck_channels = bc;

    for (int i = 0; i < bc; i++) {
        _bottleneck_cdfs[i].resize(bmax);
        std::memcpy(_bottleneck_cdfs[i].data(), ptr, bmax * sizeof(int32_t));
        ptr += bmax * sizeof(int32_t);
    }
    std::memcpy(_bottleneck_offsets.data(), ptr, bc * sizeof(int32_t));
    ptr += bc * sizeof(int32_t);
    std::memcpy(_bottleneck_cdf_lens.data(), ptr, bc * sizeof(int32_t));
    ptr += bc * sizeof(int32_t);
    std::memcpy(_bottleneck_medians.data(), ptr, bc * sizeof(float));
    ptr += bc * sizeof(float);

    // Read GaussianConditional CDFs
    _gaussian_cdfs.resize(gc);
    _gaussian_offsets.resize(gc);
    _gaussian_cdf_lens.resize(gc);
    _gaussian_scale_table.resize(st_size);

    for (int i = 0; i < gc; i++) {
        _gaussian_cdfs[i].resize(gmax);
        std::memcpy(_gaussian_cdfs[i].data(), ptr, gmax * sizeof(int32_t));
        ptr += gmax * sizeof(int32_t);
    }
    std::memcpy(_gaussian_offsets.data(), ptr, gc * sizeof(int32_t));
    ptr += gc * sizeof(int32_t);
    std::memcpy(_gaussian_cdf_lens.data(), ptr, gc * sizeof(int32_t));
    ptr += gc * sizeof(int32_t);
    std::memcpy(_gaussian_scale_table.data(), ptr, st_size * sizeof(float));
    ptr += st_size * sizeof(float);

    return true;
}

// ============================================================================
// Quantization helpers
// ============================================================================

std::vector<int32_t> EntropyCoder::_quantize_symbols(const float* data,
                                                       const float* means,
                                                       int count) {
    std::vector<int32_t> symbols(count);
    if (means != nullptr) {
        for (int i = 0; i < count; i++) {
            float v = data[i] - means[i];
            symbols[i] = static_cast<int32_t>(std::nearbyint(v));
        }
    } else {
        for (int i = 0; i < count; i++) {
            float v = data[i];
            symbols[i] = static_cast<int32_t>(std::nearbyint(v));
        }
    }
    return symbols;
}

std::vector<float> EntropyCoder::_dequantize_symbols(const int32_t* symbols,
                                                       const float* means,
                                                       int count) {
    std::vector<float> result(count);
    if (means != nullptr) {
        for (int i = 0; i < count; i++) {
            result[i] = static_cast<float>(symbols[i]) + means[i];
        }
    } else {
        for (int i = 0; i < count; i++) {
            result[i] = static_cast<float>(symbols[i]);
        }
    }
    return result;
}

// ============================================================================
// rANS encode / decode (matches CompressAI rans_interface.cpp)
// ============================================================================

/// Symbol entry for buffered encoding. Matches CompressAI RansSymbol.
struct RansSymbol {
    uint32_t start;   // cdf[value] for normal, or raw value for bypass
    uint32_t range;   // cdf[value+1] - cdf[value] for normal
    bool     bypass;  // true if this symbol encodes raw bits
};

std::string EntropyCoder::_rans_encode(
    const std::vector<int32_t>& symbols,
    const std::vector<int32_t>& indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdf_lengths,
    const std::vector<int32_t>& offsets) {

    const int count = static_cast<int>(symbols.size());
    std::vector<RansSymbol> syms;
    syms.reserve(count * 2);  // reserve extra for bypass overhead

    // --- Pass 1: build symbol buffer in forward order (matching CompressAI) ---
    for (int i = 0; i < count; i++) {
        const int32_t cdf_idx = indexes[i];
        const auto& cdf = cdfs[cdf_idx];
        const int32_t max_value = cdf_lengths[cdf_idx] - 2;

        int32_t value = symbols[i] - offsets[cdf_idx];

        uint32_t raw_val = 0;
        if (value < 0) {
            // Encode negative values as odd numbers: raw_val = -2*value - 1
            raw_val = static_cast<uint32_t>(-2 * value - 1);
            value = max_value;  // sentinel flag
        } else if (value >= max_value) {
            // Encode excess values as even numbers: raw_val = 2*(value - max_value)
            raw_val = static_cast<uint32_t>(2 * (value - max_value));
            value = max_value;  // sentinel flag
        }

        // Push the main symbol (normal or sentinel)
        syms.push_back({static_cast<uint32_t>(cdf[value]),
                        static_cast<uint32_t>(cdf[value + 1] - cdf[value]),
                        false});

        // Bypass coding: if we used the sentinel slot, encode raw_val
        if (value == max_value) {
            // Determine number of bypass chunks (4 bits each) needed
            int32_t n_bypass = 0;
            while ((raw_val >> (n_bypass * kBypassPrecision)) != 0) {
                ++n_bypass;
            }

            // Encode n_bypass using unary code with max_bypass_val=15 as continuation
            int32_t val = n_bypass;
            while (val >= kMaxBypassVal) {
                syms.push_back({static_cast<uint32_t>(kMaxBypassVal),
                                static_cast<uint32_t>(kMaxBypassVal + 1),
                                true});
                val -= kMaxBypassVal;
            }
            syms.push_back({static_cast<uint32_t>(val),
                            static_cast<uint32_t>(val + 1),
                            true});

            // Encode raw_val in 4-bit chunks, LSB first
            for (int32_t j = 0; j < n_bypass; ++j) {
                const uint32_t chunk =
                    (raw_val >> (j * kBypassPrecision)) & kMaxBypassVal;
                syms.push_back({chunk, chunk + 1, true});
            }
        }
    }

    // --- Pass 2: flush buffered symbols in reverse order (LIFO) ---
    // Allocate output buffer (generous sizing to avoid overflow)
    size_t max_words = syms.size() + 5;
    std::vector<uint32_t> out_buf(max_words, 0);
    uint32_t* ptr = out_buf.data() + max_words;  // write backwards

    RansState state;
    rans_enc_init(&state);

    // Process in reverse order (pop from back, matching CompressAI flush)
    while (!syms.empty()) {
        const RansSymbol sym = syms.back();
        syms.pop_back();

        if (!sym.bypass) {
            rans_enc_put(&state, &ptr, sym.start, sym.range, kPrecision);
        } else {
            rans_enc_put_bits(&state, &ptr, sym.start, kBypassPrecision);
        }
    }

    rans_enc_flush(&state, &ptr);

    // ptr now points to the first valid word
    size_t num_words = out_buf.data() + max_words - ptr;
    size_t num_bytes = num_words * sizeof(uint32_t);

    // Convert to byte string (LE order, as stored in uint32)
    return std::string(reinterpret_cast<const char*>(ptr), num_bytes);
}

std::vector<int32_t> EntropyCoder::_rans_decode(
    const std::string& data,
    const std::vector<int32_t>& indexes,
    const std::vector<std::vector<int32_t>>& cdfs,
    const std::vector<int32_t>& cdf_lengths,
    const std::vector<int32_t>& offsets) {

    const int count = static_cast<int>(indexes.size());
    if (count == 0) return {};

    // The data is a sequence of uint32 words (LE).
    // Pad if needed for safe reading (though data is always 4-byte aligned).
    size_t raw_bytes = data.size();
    size_t padded_bytes = (raw_bytes + 3) & ~3ull;
    std::vector<uint32_t> words(padded_bytes / sizeof(uint32_t) + 2, 0);
    std::memcpy(words.data(), data.data(), raw_bytes);

    uint32_t* ptr = words.data();
    RansState state;
    rans_dec_init(&state, &ptr);

    std::vector<int32_t> result;
    result.reserve(count);

    for (int i = 0; i < count; i++) {
        const int32_t cdf_idx = indexes[i];
        const auto& cdf = cdfs[cdf_idx];
        const int32_t max_value = cdf_lengths[cdf_idx] - 2;
        const int32_t offset = offsets[cdf_idx];

        // Decode cumulative frequency
        const uint32_t cum_freq = rans_dec_get(&state, kPrecision);

        // Find symbol s where cdf[s] <= cum_freq < cdf[s+1]
        // Using same algorithm as CompressAI: find first cdf > cum_freq, then s = it-1
        const int32_t* cdf_begin = cdf.data();
        const int32_t* cdf_end = cdf_begin + cdf_lengths[cdf_idx];
        const int32_t* it = cdf_begin;
        while (it < cdf_end && static_cast<uint32_t>(*it) <= cum_freq) {
            ++it;
        }
        int32_t s = static_cast<int32_t>(it - cdf_begin) - 1;
        if (s < 0) s = 0;  // safety clamp

        rans_dec_advance(&state, &ptr,
                         static_cast<uint32_t>(cdf[s]),
                         static_cast<uint32_t>(cdf[s + 1] - cdf[s]),
                         kPrecision);

        int32_t value = s;

        if (value == max_value) {
            // Bypass decoding mode (matching CompressAI exactly)
            int32_t val = static_cast<int32_t>(
                rans_dec_get_bits(&state, &ptr, kBypassPrecision));
            int32_t n_bypass = val;

            while (val == kMaxBypassVal) {
                val = static_cast<int32_t>(
                    rans_dec_get_bits(&state, &ptr, kBypassPrecision));
                n_bypass += val;
            }

            uint32_t raw_val = 0;
            for (int j = 0; j < n_bypass; ++j) {
                val = static_cast<int32_t>(
                    rans_dec_get_bits(&state, &ptr, kBypassPrecision));
                raw_val |= static_cast<uint32_t>(val) << (j * kBypassPrecision);
            }

            // Recover original value from raw_val encoding
            value = static_cast<int32_t>(raw_val >> 1);
            if (raw_val & 1) {
                // Odd -> negative value: raw_val = -2*value - 1
                value = -value - 1;
            } else {
                // Even -> excess: raw_val = 2*(value - max_value)
                value += max_value;
            }
        }

        result.push_back(value + offset);
    }

    return result;
}

// ============================================================================
// EntropyBottleneck: factorized prior (for z)
// ============================================================================

std::vector<std::string> EntropyCoder::compress_bottleneck(
    const float* z, int C, int H, int W) {

    int total = C * H * W;

    // Build per-channel indexes (each spatial position uses its channel's CDF)
    std::vector<int32_t> indexes(total);
    {
        int ch = 0, h = 0;
        for (int i = 0; i < total; i++) {
            indexes[i] = ch;
            h++;
            if (h == H * W) { ch++; h = 0; }
        }
    }

    // Build per-channel means (broadcast medians to spatial positions)
    std::vector<float> means(total);
    {
        int ch = 0, h = 0;
        for (int i = 0; i < total; i++) {
            means[i] = _bottleneck_medians[ch];
            h++;
            if (h == H * W) { ch++; h = 0; }
        }
    }

    // Quantize z to symbols
    auto symbols = _quantize_symbols(z, means.data(), total);

    // rANS encode
    std::string encoded = _rans_encode(symbols, indexes,
                                        _bottleneck_cdfs,
                                        _bottleneck_cdf_lens,
                                        _bottleneck_offsets);

    return {encoded};
}

std::vector<float> EntropyCoder::decompress_bottleneck(
    const std::vector<std::string>& strings, int C, int H, int W) {

    (void)C; // C is implied by _bottleneck_channels

    int total = _bottleneck_channels * H * W;

    // Build per-channel indexes
    std::vector<int32_t> indexes(total);
    {
        int ch = 0, h = 0;
        for (int i = 0; i < total; i++) {
            indexes[i] = ch;
            h++;
            if (h == H * W) { ch++; h = 0; }
        }
    }

    // Build per-channel means
    std::vector<float> means(total);
    {
        int ch = 0, h = 0;
        for (int i = 0; i < total; i++) {
            means[i] = _bottleneck_medians[ch];
            h++;
            if (h == H * W) { ch++; h = 0; }
        }
    }

    // Each batch element has its own compressed string
    int batch_count = static_cast<int>(strings.size());
    int syms_per_batch = total / batch_count;

    std::vector<int32_t> decoded;
    for (int b = 0; b < batch_count; b++) {
        const auto& s = strings[b];
        std::vector<int32_t> batch_indexes(
            indexes.begin() + b * syms_per_batch,
            indexes.begin() + (b + 1) * syms_per_batch);
        auto batch_decoded = _rans_decode(s, batch_indexes,
                                           _bottleneck_cdfs,
                                           _bottleneck_cdf_lens,
                                           _bottleneck_offsets);
        decoded.insert(decoded.end(), batch_decoded.begin(), batch_decoded.end());
    }

    return _dequantize_symbols(decoded.data(), means.data(),
                                static_cast<int>(decoded.size()));
}

// ============================================================================
// GaussianConditional: conditional Gaussian (for y)
// ============================================================================

std::vector<int32_t> EntropyCoder::build_indexes(const float* scales,
                                                    int C, int H, int W) {
    int total = C * H * W;
    int table_count = static_cast<int>(_gaussian_scale_table.size());
    // Python compressai clamps scales to lower_bound_scale (default 0.11)
    // before index lookup. Missing this clamp can select wrong CDF table
    // for very small scales (e.g., after QVRF gain << 1.0).
    const float scale_bound = 0.11f;

    std::vector<int32_t> indexes(total);
    for (int i = 0; i < total; i++) {
        float scale = std::max(scales[i], scale_bound);
        int32_t idx = table_count - 1;
        for (int j = 0; j < table_count - 1; j++) {
            if (scale <= _gaussian_scale_table[j]) {
                idx = j;
                break;
            }
        }
        indexes[i] = idx;
    }
    return indexes;
}

std::vector<std::string> EntropyCoder::compress_gaussian(
    const float* y, const int32_t* indexes,
    const float* means, int C, int H, int W) {

    int total = C * H * W;

    // Quantize y to symbols
    auto symbols = _quantize_symbols(y, means, total);

    std::vector<int32_t> idx_vec(indexes, indexes + total);

    // rANS encode
    std::string encoded = _rans_encode(symbols, idx_vec,
                                        _gaussian_cdfs,
                                        _gaussian_cdf_lens,
                                        _gaussian_offsets);
    return {encoded};
}

}  // namespace rmcompress
