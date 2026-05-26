#pragma once
#include <cstdint>
#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rm_compressor_s rm_compressor_t;

typedef struct {
    const char* mbt_ir_path;    // path to mbt_g_a.xml
    const char* mbt_bin_path;   // path to mbt_g_a.bin (or nullptr if same stem)
    const char* ha_ir_path;     // path to mbt_h_a.xml (hyper-analysis)
    const char* hs_ir_path;     // path to mbt_h_s.xml (hyper-synthesis, FP32)
    int         width;          // input width (default 128)
    int         height;         // input height (default 128)
    int         codec_width;    // preprocessed codec width (default 128)
    int         codec_height;   // preprocessed codec height (default 128)
    const char* device;         // OpenVINO device (default "GPU.0")
    int         max_packed_bytes; // optional packed bitstream budget (default 560)
    int         codec;          // 0=MBT (default), 1=MS-SSIM_QVRF C++ sender
    const char* msssim_ga_path; // path to msssim_g_a.xml
    const char* msssim_ha_path; // path to msssim_h_a.xml
    const char* msssim_hs_path; // path to msssim_h_s_fp32.xml
    const char* msssim_cdf_path;// path to msssim_cdfs.bin
    const char* ga_backend;     // g_a backend: openvino (default) or tensorrt
    const char* trt_engine_path;// TensorRT g_a engine path when ga_backend=tensorrt
    int         trt_device;     // CUDA device id for TensorRT g_a (default 0)
} rm_compressor_config_t;

typedef struct {
    float preprocess_ms;
    float g_a_ms;
    float pass1_ms;
    float pass2_ms;
    float pass3_ms;
    float pack_ms;
    float total_ms;
    float beta;
    int   packed_bytes;
    int   entropy_bits;
    int   rc_passes;
    int   over_budget;
} rm_compress_stats_t;

/** Create compressor instance. Returns nullptr on failure. */
rm_compressor_t* rm_compressor_create(const rm_compressor_config_t* config);

/** Compress a BGR frame (width x height, uint8 interleaved) → bitstream.
 *  @param rgb_buf   Input RGB buffer (width*height*3 bytes).
 *  @param out_buf   Output buffer (caller must allocate, max 4096 bytes).
 *  @param out_len   [out] Actual bitstream length.
 *  @return 0 on success, -1 on error.
 */
int rm_compress_frame(rm_compressor_t* ctx,
                      const uint8_t* rgb_buf,
                      uint8_t* out_buf, int* out_len);

const rm_compress_stats_t* rm_compressor_last_stats(rm_compressor_t* ctx);

/** Destroy compressor instance. */
void rm_compressor_destroy(rm_compressor_t* ctx);

#ifdef __cplusplus
}
#endif
