#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rmcompress {

/// C++ port of the CompressAI entropy coder (rANS arithmetic coding).
///
/// Uses pre-computed CDF tables exported from the MBT model checkpoint.
/// Supports:
///   - EntropyBottleneck (factorized prior): compress + decompress z
///   - GaussianConditional: compress y (with scales + means)
///
/// Target: bit-exact output with CompressAI's Python `compressai.ans` module.
/// Uses the same 64-bit rANS algorithm (ryg_rans) with 16-bit precision.
class EntropyCoder {
public:
    EntropyCoder() = default;

    // ---- CDF table loading ----

    /// Load CDF tables from a binary file exported by `scripts/export_cdfs.py`.
    /// @return true on success.
    bool load_cdfs(const std::string& path);

    /// Load CDF tables from a memory buffer (for embedding in binary).
    bool load_cdfs(const uint8_t* data, size_t size);

    // ---- EntropyBottleneck (factorized prior, for z) ----

    /// Compress z latent tensor (channel-wise factorized coding).
    /// @param z   Pointer to float32 tensor, shape [C, H, W].
    /// @param C   Number of channels (e.g. 128 for MBT bottleneck).
    /// @param H   Height (spatial).
    /// @param W   Width (spatial).
    /// @return    One compressed byte-string per batch element (batch=1 for onboard).
    std::vector<std::string> compress_bottleneck(const float* z,
                                                  int C, int H, int W);

    /// Decompress z strings back to a latent tensor.
    /// Needed because h_s(z_hat) must run during the compression pipeline.
    /// @param strings  Compressed byte-strings (one per batch element).
    /// @param C        Number of channels.
    /// @param H        Spatial height.
    /// @param W        Spatial width.
    /// @return         Float32 tensor [C, H, W], dequantized.
    std::vector<float> decompress_bottleneck(
        const std::vector<std::string>& strings, int C, int H, int W);

    // ---- GaussianConditional (for y, conditioned on scales/means) ----

    /// Build CDF indexes from scale parameters.
    /// Matches CompressAI's `GaussianConditional.build_indexes(scales)`.
    /// @param scales  Pointer to float32 tensor, shape [C, H, W].
    /// @param C, H, W Tensor dimensions.
    /// @return        int32 indexes, same shape [C, H, W].
    std::vector<int32_t> build_indexes(const float* scales,
                                        int C, int H, int W);

    /// Quantize values exactly as GaussianConditional.compress() would before
    /// rANS coding. Diagnostic only; used by QVRF dry-run parity.
    std::vector<int32_t> quantize_symbols(const float* data,
                                          const float* means,
                                          int count) const {
        return _quantize_symbols(data, means, count);
    }

    /// Compress y latent tensor using Gaussian conditional coding.
    /// @param y        Pointer to float32 tensor, shape [C, H, W].
    /// @param indexes  CDF indexes from build_indexes(), same shape.
    /// @param means    Optional means from h_s, same shape (can be nullptr).
    /// @param C, H, W  Tensor dimensions.
    /// @return         One compressed byte-string per batch element.
    std::vector<std::string> compress_gaussian(const float* y,
                                                const int32_t* indexes,
                                                const float* means,
                                                int C, int H, int W);

    // ---- Accessors (for debugging) ----

    int bottleneck_channels() const { return _bottleneck_channels; }
    int gaussian_cdf_count() const { return static_cast<int>(_gaussian_cdfs.size()); }

    // ---- Testing / diagnostic wrappers (expose internal rANS for bit-exact verification) ----

    /// Direct rANS encode (public wrapper for testing).
    std::string rans_encode(const std::vector<int32_t>& symbols,
                            const std::vector<int32_t>& indexes,
                            const std::vector<std::vector<int32_t>>& cdfs,
                            const std::vector<int32_t>& cdf_lengths,
                            const std::vector<int32_t>& offsets) {
        return _rans_encode(symbols, indexes, cdfs, cdf_lengths, offsets);
    }

    /// Direct rANS decode (public wrapper for testing).
    std::vector<int32_t> rans_decode(const std::string& data,
                                      const std::vector<int32_t>& indexes,
                                      const std::vector<std::vector<int32_t>>& cdfs,
                                      const std::vector<int32_t>& cdf_lengths,
                                      const std::vector<int32_t>& offsets) {
        return _rans_decode(data, indexes, cdfs, cdf_lengths, offsets);
    }

private:
    // ---- CDF tables (loaded from binary) ----

    // EntropyBottleneck: one CDF row per channel
    // _bottleneck_cdfs[ch][k] = cumulative probability * 2^16
    std::vector<std::vector<int32_t>> _bottleneck_cdfs;     // [128][max_len]
    std::vector<int32_t>              _bottleneck_offsets;   // [128]
    std::vector<int32_t>              _bottleneck_cdf_lens;  // [128]
    std::vector<float>                _bottleneck_medians;   // [128]
    int _bottleneck_channels = 0;

    // GaussianConditional: one CDF row per scale table entry
    // _gaussian_cdfs[idx][k] = cumulative probability * 2^16
    std::vector<std::vector<int32_t>> _gaussian_cdfs;       // [64][max_len]
    std::vector<int32_t>              _gaussian_offsets;     // [64]
    std::vector<int32_t>              _gaussian_cdf_lens;    // [64]
    std::vector<float>                _gaussian_scale_table; // [64]

    // ---- Internal rANS encode/decode ----

    /// Core rANS encode: symbols + CDF indexes -> compressed bytes.
    /// @param symbols     Flat list of adjusted symbols (symbol - offset[index]).
    /// @param indexes     Which CDF table to use per symbol.
    /// @param cdfs        CDF tables.
    /// @param cdf_lengths Valid entry count per CDF table.
    /// @param offsets     Offset per CDF table (for bypass detection only).
    /// @return            Compressed byte string.
    std::string _rans_encode(const std::vector<int32_t>& symbols,
                             const std::vector<int32_t>& indexes,
                             const std::vector<std::vector<int32_t>>& cdfs,
                             const std::vector<int32_t>& cdf_lengths,
                             const std::vector<int32_t>& offsets);

    /// Core rANS decode: compressed bytes -> symbols.
    std::vector<int32_t> _rans_decode(const std::string& data,
                                       const std::vector<int32_t>& indexes,
                                       const std::vector<std::vector<int32_t>>& cdfs,
                                       const std::vector<int32_t>& cdf_lengths,
                                       const std::vector<int32_t>& offsets);

    /// Quantize float tensor to integer symbols (round to nearest).
    /// symbol = round(input - means)
    static std::vector<int32_t> _quantize_symbols(const float* data,
                                                    const float* means,
                                                    int count);

    /// Dequantize integer symbols back to float.
    /// output = symbols + means
    static std::vector<float> _dequantize_symbols(const int32_t* symbols,
                                                    const float* means,
                                                    int count);
};

}  // namespace rmcompress
