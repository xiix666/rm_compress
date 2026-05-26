#include "rmcompress/entropy_coder.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Helper: build a minimal synthetic CDF file in memory for self-contained tests
// ============================================================================

static bool build_test_cdfs(rmcompress::EntropyCoder& coder) {
    // Build an in-memory CDF binary matching the format from export_cdfs.py.
    // This gives us a small set of CDF tables (2 bottleneck + 2 gaussian) for
    // testing the rANS core without needing the real checkpoint.

    // Format: header + bottleneck data + gaussian data
    // Use 2 bottleneck channels, 2 gaussian tables, small CDF lengths.

    const int bc   = 2;    // bottleneck channels
    const int bmax = 8;    // max CDF row length
    const int gt   = 2;    // gaussian tables
    const int gmax = 8;    // max CDF row length
    const int st   = gt;   // scale_table_size

    // Total size
    size_t hdr_size = 40;  // 10 int32 fields
    size_t data_size = 0;
    data_size += bc * bmax * 4;   // bottleneck_cdfs
    data_size += bc * 4;          // bottleneck_offsets
    data_size += bc * 4;          // bottleneck_cdf_lens
    data_size += bc * 4;          // bottleneck_medians
    data_size += gt * gmax * 4;   // gaussian_cdfs
    data_size += gt * 4;          // gaussian_offsets
    data_size += gt * 4;          // gaussian_cdf_lens
    data_size += st * 4;          // gaussian_scale_table

    std::vector<uint8_t> buf(hdr_size + data_size, 0);
    auto* p = buf.data();

    // Header fields (all int32 LE or uint32 LE)
    auto w32 = [&p](uint32_t v) { memcpy(p, &v, 4); p += 4; };
    auto wf  = [&p](float v)    { memcpy(p, &v, 4); p += 4; };

    w32(0x46444352);  // magic "RCDF"
    w32(1);           // version
    w32(16);          // precision (int32)
    w32(bc);          // bottleneck_channels
    w32(bmax);        // bottleneck_max_len
    w32(gt);          // gaussian_tables
    w32(gmax);        // gaussian_max_len
    w32(st);          // scale_table_size

    // --- Bottleneck CDFs ---
    // Two rows, each with bmax=8 entries. Use simple equal-probability CDFs.
    // Precision: 65536 / 4 = 16384 per symbol (4 symbols per row)
    const int32_t kTotal = 65536;
    int32_t cdf_row0[] = {0, 16384, 32768, 49152, kTotal, 0, 0, 0};  // 5 valid
    int32_t cdf_row1[] = {0, 8192, 16384, 24576, 32768, 49152, kTotal, 0};  // 7 valid
    for (int i = 0; i < bmax; i++) w32(static_cast<uint32_t>(cdf_row0[i]));
    for (int i = 0; i < bmax; i++) w32(static_cast<uint32_t>(cdf_row1[i]));
    // Bottleneck offsets
    w32(0); w32(-2);
    // Bottleneck cdf_lens
    w32(5); w32(7);
    // Bottleneck medians
    wf(0.0f); wf(0.5f);

    // --- Gaussian CDFs ---
    // Two rows, simple CDFs
    int32_t g_cdf0[] = {0, 32768, kTotal, 0, 0, 0, 0, 0};  // 3 valid
    int32_t g_cdf1[] = {0, 8192, 16384, 24576, 32768, 49152, kTotal, 0};  // 7 valid
    for (int i = 0; i < gmax; i++) w32(static_cast<uint32_t>(g_cdf0[i]));
    for (int i = 0; i < gmax; i++) w32(static_cast<uint32_t>(g_cdf1[i]));
    // Gaussian offsets
    w32(0); w32(-1);
    // Gaussian cdf_lens
    w32(3); w32(7);
    // Gaussian scale_table
    wf(0.1f); wf(1.0f);

    return coder.load_cdfs(buf.data(), buf.size());
}

// ============================================================================
// Test 1: CDF loading
// ============================================================================
static bool test_load_cdfs() {
    printf("Test 1: CDF loading... ");
    rmcompress::EntropyCoder coder;
    if (!build_test_cdfs(coder)) {
        printf("FAIL (load_cdfs returned false)\n");
        return false;
    }
    if (coder.bottleneck_channels() != 2) {
        printf("FAIL (expected 2 channels, got %d)\n", coder.bottleneck_channels());
        return false;
    }
    if (coder.gaussian_cdf_count() != 2) {
        printf("FAIL (expected 2 gaussian tables, got %d)\n", coder.gaussian_cdf_count());
        return false;
    }
    printf("PASS\n");
    return true;
}

// ============================================================================
// Test 2: rANS encode/decode round-trip (synthetic symbols)
// ============================================================================
static bool test_rans_roundtrip() {
    printf("Test 2: rANS round-trip... ");

    rmcompress::EntropyCoder coder;
    if (!build_test_cdfs(coder)) {
        printf("FAIL (CDF loading)\n");
        return false;
    }

    // Bottleneck: 2 channels, 2x2 spatial = 8 values
    float z[] = {
        0.1f, 0.3f,   // ch0: [0,0], [0,1]
        0.8f, 0.2f,   // ch0: [1,0], [1,1]
        -0.5f, 0.4f,  // ch1: [0,0], [0,1]
        0.6f, -0.3f   // ch1: [1,0], [1,1]
    };

    auto strings = coder.compress_bottleneck(z, 2, 2, 2);
    if (strings.size() != 1) {
        printf("FAIL (expected 1 string, got %zu)\n", strings.size());
        return false;
    }

    auto z_hat = coder.decompress_bottleneck(strings, 2, 2, 2);
    if (z_hat.size() != 8) {
        printf("FAIL (expected 8 values, got %zu)\n", z_hat.size());
        return false;
    }

    // Check round-trip: all values should match quantize(dequantize) cycle.
    // EntropyBottleneck uses per-channel medians:
    //   symbol = round(z - median),  z_hat = symbol + median
    // So z_hat should equal round(z - median) + median for the channel.
    // Use round-to-nearest-even to match torch.round / std::nearbyint.
    auto manual_round = [](double v) -> double {
        return std::nearbyint(v);
    };
    float expected[8];
    expected[0] = static_cast<float>(manual_round(0.1 - 0.0) + 0.0);  // 0
    expected[1] = static_cast<float>(manual_round(0.3 - 0.0) + 0.0);  // 0
    expected[2] = static_cast<float>(manual_round(0.8 - 0.0) + 0.0);  // 1
    expected[3] = static_cast<float>(manual_round(0.2 - 0.0) + 0.0);  // 0
    expected[4] = static_cast<float>(manual_round(-0.5 - 0.5) + 0.5); // -0.5
    expected[5] = static_cast<float>(manual_round(0.4 - 0.5) + 0.5);  // 0.5
    expected[6] = static_cast<float>(manual_round(0.6 - 0.5) + 0.5);  // 0.5
    expected[7] = static_cast<float>(manual_round(-0.3 - 0.5) + 0.5); // -0.5

    for (int i = 0; i < 8; i++) {
        float diff = std::fabs(z_hat[i] - expected[i]);
        if (diff > 0.01f) {
            printf("FAIL at index %d: z=%f, z_hat=%f, expected=%f, diff=%f\n",
                   i, z[i], z_hat[i], expected[i], diff);
            return false;
        }
    }

    printf("PASS (compressed %zu bytes)\n", strings[0].size());
    return true;
}

// ============================================================================
// Test 3: Gaussian conditional compress
// ============================================================================
static bool test_gaussian_compress() {
    printf("Test 3: Gaussian conditional compress... ");

    rmcompress::EntropyCoder coder;
    if (!build_test_cdfs(coder)) {
        printf("FAIL (CDF loading)\n");
        return false;
    }

    // 2 channels, 2x2 spatial
    float y[]      = {1.0f, -2.0f, 3.0f, -1.0f, 0.5f, -0.5f, 1.5f, -1.5f};
    float scales[]  = {0.05f, 0.15f, 0.5f, 1.5f, 0.08f, 0.12f, 2.0f, 0.9f};
    float means[]   = {0.0f, 0.1f, -0.1f, 0.0f, 0.2f, 0.0f, -0.2f, 0.1f};

    auto indexes = coder.build_indexes(scales, 2, 2, 2);
    if (indexes.size() != 8) {
        printf("FAIL (indexes size)\n");
        return false;
    }

    // Verify indexes are within [0, 1] for our 2-table setup
    for (int i = 0; i < 8; i++) {
        if (indexes[i] < 0 || indexes[i] > 1) {
            printf("FAIL (index[%d]=%d out of range)\n", i, indexes[i]);
            return false;
        }
    }

    auto strings = coder.compress_gaussian(y, indexes.data(), means, 2, 2, 2);
    if (strings.size() != 1) {
        printf("FAIL (expected 1 string, got %zu)\n", strings.size());
        return false;
    }

    printf("PASS (compressed %zu bytes)\n", strings[0].size());
    return true;
}

// ============================================================================
// Test 4: build_indexes correctness
// ============================================================================
static bool test_build_indexes() {
    printf("Test 4: build_indexes correctness... ");

    rmcompress::EntropyCoder coder;
    if (!build_test_cdfs(coder)) {
        printf("FAIL (CDF loading)\n");
        return false;
    }

    // With scale_table = [0.1, 1.0]:
    //   scale <= 0.1  -> index 0
    //   scale <= 1.0  -> index 1
    //   scale >  1.0  -> index 1 (largest table)
    float scales[] = {0.05f, 0.1f, 0.5f, 1.0f, 2.0f};
    auto indexes = coder.build_indexes(scales, 5, 1, 1);

    int expected[] = {0, 0, 1, 1, 1};
    for (int i = 0; i < 5; i++) {
        if (indexes[i] != expected[i]) {
            printf("FAIL at %d: scale=%.3f gave index %d, expected %d\n",
                   i, scales[i], indexes[i], expected[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

// ============================================================================
// Test 5: Empty / edge cases
// ============================================================================
static bool test_edge_cases() {
    printf("Test 5: Edge cases... ");

    // Invalid path
    rmcompress::EntropyCoder coder;
    if (coder.load_cdfs("/nonexistent/path/to/cdfs.bin")) {
        printf("FAIL (load_cdfs should return false for nonexistent file)\n");
        return false;
    }

    // Empty buffer
    uint8_t empty = 0;
    if (coder.load_cdfs(&empty, 0)) {
        printf("FAIL (load_cdfs should return false for empty buffer)\n");
        return false;
    }

    printf("PASS\n");
    return true;
}

// ============================================================================
// Test 6: Determinism (same input -> same output)
// ============================================================================
static bool test_determinism() {
    printf("Test 6: Determinism... ");

    rmcompress::EntropyCoder coder;
    if (!build_test_cdfs(coder)) {
        printf("FAIL (CDF loading)\n");
        return false;
    }

    float z[] = {0.1f, 0.3f, 0.8f, 0.2f, -0.5f, 0.4f, 0.6f, -0.3f};

    auto s1 = coder.compress_bottleneck(z, 2, 2, 2);
    auto s2 = coder.compress_bottleneck(z, 2, 2, 2);

    if (s1[0] != s2[0]) {
        printf("FAIL (same input produced different compressed output)\n");
        return false;
    }

    auto z1 = coder.decompress_bottleneck(s1, 2, 2, 2);
    auto z2 = coder.decompress_bottleneck(s2, 2, 2, 2);

    for (size_t i = 0; i < z1.size(); i++) {
        if (std::fabs(z1[i] - z2[i]) > 1e-6f) {
            printf("FAIL (decompressed values differ)\n");
            return false;
        }
    }

    printf("PASS\n");
    return true;
}

// ============================================================================
// Test 7: Bit-exact encoding vs Python CompressAI reference
// ============================================================================
static bool test_bit_exact() {
    printf("Test 7: Bit-exact encoding vs Python CompressAI... ");

    // CDF tables matching Python test:
    // Table 0: [0, 16384, 32768, 49152, 65536] -> max_value = 3
    // Table 1: [0, 5000, 11000, 18000, 26000, 36000, 50000, 65536] -> max_value = 6
    std::vector<std::vector<int32_t>> cdfs = {
        {0, 16384, 32768, 49152, 65536},
        {0, 5000, 11000, 18000, 26000, 36000, 50000, 65536},
    };
    std::vector<int32_t> cdf_lengths = {5, 8};
    std::vector<int32_t> offsets = {0, -1};

    // Symbols and indexes (identical to Python reference test)
    std::vector<int32_t> symbols = {
        0, 0, 1, 0, 0, 0, -1, 0, 0, 3, 5, 1, 0, 0, 0, 0, 2, 0, 1, 0
    };
    std::vector<int32_t> indexes(20, 0);  // all use CDF table 0

    // Reference encoded bytes from CompressAI Python:
    //   symbols=[0,0,1,0,0,0,-1,0,0,3,5,1,0,0,0,0,2,0,1,0]
    //   all indexes=0, cdf table 0
    const uint8_t kExpected[] = {
        0x11, 0x00, 0x04, 0x2c, 0x01, 0x03, 0x00, 0x08, 0x41, 0xc0, 0x00, 0x01,
    };
    const size_t kExpectedLen = sizeof(kExpected);

    rmcompress::EntropyCoder coder;
    std::string encoded = coder.rans_encode(symbols, indexes, cdfs, cdf_lengths, offsets);

    if (encoded.size() != kExpectedLen) {
        printf("FAIL (size: %zu vs expected %zu)\n", encoded.size(), kExpectedLen);
        return false;
    }

    for (size_t i = 0; i < kExpectedLen; i++) {
        if (static_cast<uint8_t>(encoded[i]) != kExpected[i]) {
            printf("FAIL at byte %zu: got 0x%02x, expected 0x%02x\n",
                   i, static_cast<uint8_t>(encoded[i]), kExpected[i]);
            return false;
        }
    }

    // Also verify round-trip
    auto decoded = coder.rans_decode(encoded, indexes, cdfs, cdf_lengths, offsets);
    if (decoded.size() != symbols.size()) {
        printf("FAIL (decoded size %zu vs expected %zu)\n", decoded.size(), symbols.size());
        return false;
    }
    for (size_t i = 0; i < symbols.size(); i++) {
        if (decoded[i] != symbols[i]) {
            printf("FAIL at symbol %zu: decoded %d vs expected %d\n",
                   i, decoded[i], symbols[i]);
            return false;
        }
    }

    printf("PASS (12/12 bytes match, round-trip OK)\n");
    return true;
}

// ============================================================================
// Test 8: Bypass coding bit-exact (negative + large positive values)
// ============================================================================
static bool test_bypass_bit_exact() {
    printf("Test 8: Bypass coding bit-exact... ");

    // Same CDF tables as test 7
    std::vector<std::vector<int32_t>> cdfs = {
        {0, 16384, 32768, 49152, 65536},
        {0, 5000, 11000, 18000, 26000, 36000, 50000, 65536},
    };
    std::vector<int32_t> cdf_lengths = {5, 8};
    std::vector<int32_t> offsets = {0, -1};

    // Bypass test: values that trigger the bypass coding path
    // max_value for table 0 = 3, for table 1 = 6
    std::vector<int32_t> symbols = {-1, -2, -3, 3, 4, 5, 10, 100};
    std::vector<int32_t> indexes(8, 0);  // all use CDF table 0

    // Expected from Python: hex = 11f100330080000051d048c3142e0e33
    const uint8_t kExpected[] = {
        0x11, 0xf1, 0x00, 0x33, 0x00, 0x80, 0x00, 0x00, 0x51, 0xd0, 0x48, 0xc3,
        0x14, 0x2e, 0x0e, 0x33,
    };
    const size_t kExpectedLen = sizeof(kExpected);

    rmcompress::EntropyCoder coder;
    std::string encoded = coder.rans_encode(symbols, indexes, cdfs, cdf_lengths, offsets);

    if (encoded.size() != kExpectedLen) {
        printf("FAIL (size: %zu vs expected %zu)\n", encoded.size(), kExpectedLen);
        // Print hex for debugging
        printf("  got: ");
        for (size_t i = 0; i < encoded.size(); i++)
            printf("%02x", static_cast<uint8_t>(encoded[i]));
        printf("\n");
        return false;
    }

    for (size_t i = 0; i < kExpectedLen; i++) {
        if (static_cast<uint8_t>(encoded[i]) != kExpected[i]) {
            printf("FAIL at byte %zu: got 0x%02x, expected 0x%02x\n",
                   i, static_cast<uint8_t>(encoded[i]), kExpected[i]);
            // Print full hex for debugging
            printf("  got:      ");
            for (size_t j = 0; j < encoded.size(); j++)
                printf("%02x ", static_cast<uint8_t>(encoded[j]));
            printf("\n  expected: ");
            for (size_t j = 0; j < kExpectedLen; j++)
                printf("%02x ", kExpected[j]);
            printf("\n");
            return false;
        }
    }

    // Round-trip verification
    auto decoded = coder.rans_decode(encoded, indexes, cdfs, cdf_lengths, offsets);
    if (decoded.size() != symbols.size()) {
        printf("FAIL (decoded size %zu vs expected %zu)\n", decoded.size(), symbols.size());
        return false;
    }
    for (size_t i = 0; i < symbols.size(); i++) {
        if (decoded[i] != symbols[i]) {
            printf("FAIL at symbol %zu: decoded %d vs expected %d\n",
                   i, decoded[i], symbols[i]);
            return false;
        }
    }

    printf("PASS (%zu/%zu bytes match, bypass round-trip OK)\n",
           kExpectedLen, kExpectedLen);
    return true;
}

// ============================================================================
// Test 9: Mixed CDF tables bit-exact
// ============================================================================
static bool test_mixed_tables_bit_exact() {
    printf("Test 9: Mixed CDF tables bit-exact... ");

    // CDF tables
    std::vector<std::vector<int32_t>> cdfs = {
        {0, 16384, 32768, 49152, 65536},
        {0, 5000, 11000, 18000, 26000, 36000, 50000, 65536},
    };
    std::vector<int32_t> cdf_lengths = {5, 8};
    std::vector<int32_t> offsets = {0, -1};

    // Mixed tables: uses both CDF table 0 and table 1
    std::vector<int32_t> symbols = {
        0, 1, 2, -1, 3, 4, 5, 10, 0, 0, 0, -1, 50, 1, 2, 0
    };
    std::vector<int32_t> indexes = {
        0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0
    };

    // Expected encoded bytes from CompressAI Python
    const uint8_t kExpected[] = {
        0xa1, 0x13, 0xb5, 0xe1, 0xa7, 0xa3, 0x03, 0x00,
        0x10, 0x12, 0x4c, 0xc6, 0x11, 0xf2, 0xb3, 0x03,
    };
    const size_t kExpectedLen = sizeof(kExpected);

    rmcompress::EntropyCoder coder;
    std::string encoded = coder.rans_encode(symbols, indexes, cdfs, cdf_lengths, offsets);

    if (encoded.size() != kExpectedLen) {
        printf("FAIL (size: %zu vs expected %zu)\n", encoded.size(), kExpectedLen);
        printf("  got: ");
        for (size_t i = 0; i < encoded.size(); i++)
            printf("%02x", static_cast<uint8_t>(encoded[i]));
        printf("\n");
        return false;
    }

    for (size_t i = 0; i < kExpectedLen; i++) {
        if (static_cast<uint8_t>(encoded[i]) != kExpected[i]) {
            printf("FAIL at byte %zu: got 0x%02x, expected 0x%02x\n",
                   i, static_cast<uint8_t>(encoded[i]), kExpected[i]);
            printf("  got:      ");
            for (size_t j = 0; j < encoded.size(); j++)
                printf("%02x ", static_cast<uint8_t>(encoded[j]));
            printf("\n  expected: ");
            for (size_t j = 0; j < kExpectedLen; j++)
                printf("%02x ", kExpected[j]);
            printf("\n");
            return false;
        }
    }

    // Round-trip verification
    auto decoded = coder.rans_decode(encoded, indexes, cdfs, cdf_lengths, offsets);
    if (decoded.size() != symbols.size()) {
        printf("FAIL (decoded size mismatch)\n");
        return false;
    }
    for (size_t i = 0; i < symbols.size(); i++) {
        if (decoded[i] != symbols[i]) {
            printf("FAIL at symbol %zu: decoded %d vs expected %d\n",
                   i, decoded[i], symbols[i]);
            return false;
        }
    }

    printf("PASS (%zu/%zu bytes match, mixed-table round-trip OK)\n",
           kExpectedLen, kExpectedLen);
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("=== Entropy Coder Tests ===\n\n");

    bool ok = true;
    ok &= test_load_cdfs();
    ok &= test_rans_roundtrip();
    ok &= test_gaussian_compress();
    ok &= test_build_indexes();
    ok &= test_edge_cases();
    ok &= test_determinism();
    ok &= test_bit_exact();
    ok &= test_bypass_bit_exact();
    ok &= test_mixed_tables_bit_exact();

    printf("\n=== %s ===\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
