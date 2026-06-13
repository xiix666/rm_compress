#include "rmcompress/rm_compress.h"
#include "rmcompress/ga_encoder.h"
#include "rmcompress/mbt_encoder.h"
#include "rmcompress/preprocess.h"
#include "rmcompress/entropy_coder.h"
#include "rmcompress/protocol.h"
#include <cstring>
#include <stdexcept>
#include <string>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iostream>
#include <cstdlib>

#ifndef RM_PROJECT_ROOT
#define RM_PROJECT_ROOT /home/wyh/compress_and_transmit
#endif
#define _STR_IMPL(x) #x
#define _STR(x) _STR_IMPL(x)
#define RM_PROJECT_ROOT_STR _STR(RM_PROJECT_ROOT)

struct rm_compressor_s {
    // ---- MBT codec ----
    rmcompress::GaEncoder encoder_g_a; // g_a: selectable OpenVINO or TensorRT backend
    MbtEncoder encoder_h_a;   // h_a: entropy parameters, OpenVINO FP32 CPU only
    MbtEncoder encoder_h_s;   // h_s: entropy parameters, OpenVINO FP32 CPU only
    rmcompress::EntropyCoder entropy;
    // ---- MS-SSIM QVRF codec ----
    rmcompress::GaEncoder encoder_msssim_g_a;
    MbtEncoder encoder_msssim_h_a;
    MbtEncoder encoder_msssim_h_s;
    rmcompress::EntropyCoder entropy_msssim;
    bool msssim_loaded = false;
    // ---- Common state ----
    rm_compressor_config_t config;
    bool initialized = false;
    int C_y = 192, H_y = 8, W_y = 8;
    int C_z = 128, H_z = 2, W_z = 2;
    int budget_bits = 4400;  // 550 bytes, fits in 2 chunks
    int max_packed_bytes = 560;  // 2 x 280B protocol payload chunks
    float next_beta = 1.30f;
    float next_gain = 0.80f;  // QVRF gain starting point
    std::string ga_backend = "openvino";
    std::string ga_device = "GPU.0";
    PreprocessRgbScratch preprocess_scratch;
    rm_compress_stats_t last_stats{};
};

extern "C" {

static std::string _model_dir(const std::string& path) {
    auto pos = path.rfind('/');
    if (pos != std::string::npos) return path.substr(0, pos + 1);
    return "";
}

// Prepend project root to relative paths so the binary can be run from any CWD.
static std::string _resolve_path(const char* path) {
    if (!path || !path[0]) return "";
    if (path[0] == '/') return std::string(path);
    const char* runtime_root = std::getenv("RM_COMPRESS_ROOT");
    if (runtime_root && runtime_root[0]) {
        return std::string(runtime_root) + "/" + std::string(path);
    }
    return std::string(RM_PROJECT_ROOT_STR) + "/" + std::string(path);
}

static bool _file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string _ga_backend(const rm_compressor_config_t* config) {
    if (!config->ga_backend || !config->ga_backend[0]) return "openvino";
    return std::string(config->ga_backend);
}

static std::string _codec_suffix(const rm_compressor_s* ctx) {
    int cw = ctx->config.codec_width > 0 ? ctx->config.codec_width : 128;
    int ch = ctx->config.codec_height > 0 ? ctx->config.codec_height : cw;
    if (cw == 128 && ch == 128) return "";
    if (cw == ch) return "_" + std::to_string(cw);
    return "_" + std::to_string(cw) + "x" + std::to_string(ch);
}

rm_compressor_t* rm_compressor_create(const rm_compressor_config_t* config) {
    auto* ctx = new rm_compressor_s();
    ctx->config = *config;
    int codec = config->codec;  // 0=MBT, 1=MS-SSIM_QVRF
    if (config->max_packed_bytes > 0) {
        ctx->max_packed_bytes = config->max_packed_bytes;
        if (codec == 1) {
            ctx->next_gain = 0.80f;
        } else {
            ctx->next_beta = 1.30f;
        }
    }
    ctx->budget_bits = std::max(1, (ctx->max_packed_bytes - 10) * 8);
    try {
        ctx->config.codec_width = config->codec_width > 0 ? config->codec_width : 128;
        ctx->config.codec_height = config->codec_height > 0 ? config->codec_height : ctx->config.codec_width;
        if (ctx->config.codec_width % 64 != 0 || ctx->config.codec_height % 64 != 0) {
            delete ctx; return nullptr;
        }
        ctx->H_y = ctx->config.codec_height / 16;
        ctx->W_y = ctx->config.codec_width / 16;
        ctx->H_z = ctx->config.codec_height / 64;
        ctx->W_z = ctx->config.codec_width / 64;

        std::string device = config->device ? config->device : "GPU.0";
        const std::string ga_backend = _ga_backend(config);
        if (ga_backend != "openvino" && ga_backend != "tensorrt") {
            throw std::runtime_error("Unsupported g_a backend: " + ga_backend);
        }
        if (ga_backend == "tensorrt" &&
            (!config->trt_engine_path || !config->trt_engine_path[0])) {
            throw std::runtime_error("TensorRT g_a backend requires trt_engine_path");
        }
        ctx->ga_backend = ga_backend;
        const std::string entropy_param_device = "CPU";
        std::string suffix = _codec_suffix(ctx);

        if (codec == 1) {
            // ---- MS-SSIM QVRF codec (dynamic shapes, no size suffix) ----
            const char* msssim_ga_raw = (config->msssim_ga_path && config->msssim_ga_path[0]) ? config->msssim_ga_path : nullptr;
            std::string msssim_base = msssim_ga_raw ? _model_dir(_resolve_path(msssim_ga_raw)) : _resolve_path("models/");
            std::string msssim_ga = msssim_ga_raw ? _resolve_path(msssim_ga_raw) : (msssim_base + "msssim_g_a_fp32.xml");
            if (!_file_exists(msssim_ga)) msssim_ga = msssim_base + "msssim_g_a.xml";
            // MS-SSIM g_a: OpenVINO uses FP32. FP16 IR is too loose for a sender parity gate.
            if (ga_backend == "tensorrt") {
                std::string engine = _resolve_path(config->trt_engine_path);
                int trt_device = config->trt_device >= 0 ? config->trt_device : 0;
                if (!ctx->encoder_msssim_g_a.load_tensorrt(
                        engine, trt_device, 3, ctx->config.codec_height, ctx->config.codec_width,
                        ctx->C_y, ctx->H_y, ctx->W_y)) { delete ctx; return nullptr; }
                // Enable CUDA preprocess: raw BGR frame → resize+normalize on GPU
                ctx->encoder_msssim_g_a.setup_cuda_preprocess(
                    ctx->config.height, ctx->config.width);
            } else if (!ctx->encoder_msssim_g_a.load_openvino_with_preprocess(
                           msssim_ga, device, true,
                           ctx->config.height, ctx->config.width,
                           ctx->config.codec_height, ctx->config.codec_width)) {
                delete ctx; return nullptr;
            }
            ctx->ga_device = ctx->encoder_msssim_g_a.device_name();

            const char* msssim_ha_raw = (config->msssim_ha_path && config->msssim_ha_path[0]) ? config->msssim_ha_path : nullptr;
            std::string msssim_ha = msssim_ha_raw ? _resolve_path(msssim_ha_raw) : (msssim_base + "msssim_h_a_fp32.xml");
            if (!_file_exists(msssim_ha)) msssim_ha = msssim_base + "msssim_h_a.xml";
            // HARD RULE: h_a and h_s define entropy/Gaussian parameters.
            // They must use the same OpenVINO FP32 CPU architecture on sender
            // and receiver. Do not move them to iGPU/GPU/CUDA.
            if (!ctx->encoder_msssim_h_a.load(msssim_ha, entropy_param_device, true)) { delete ctx; return nullptr; }

            const char* msssim_hs_raw = (config->msssim_hs_path && config->msssim_hs_path[0]) ? config->msssim_hs_path : nullptr;
            std::string msssim_hs = msssim_hs_raw ? _resolve_path(msssim_hs_raw) : (msssim_base + "msssim_h_s_fp32.xml");
            if (!_file_exists(msssim_hs)) msssim_hs = msssim_base + "msssim_h_s.xml";
            if (!ctx->encoder_msssim_h_s.load(msssim_hs, entropy_param_device, true)) { delete ctx; return nullptr; }

            const char* msssim_cdf_raw = (config->msssim_cdf_path && config->msssim_cdf_path[0]) ? config->msssim_cdf_path : nullptr;
            std::string msssim_cdf = msssim_cdf_raw ? _resolve_path(msssim_cdf_raw) : (msssim_base + "msssim_cdfs.bin");
            if (!ctx->entropy_msssim.load_cdfs(msssim_cdf)) { delete ctx; return nullptr; }

            ctx->msssim_loaded = true;
        } else {
            // ---- MBT codec (default) ----
            const char* mbt_ir_raw = (config->mbt_ir_path && config->mbt_ir_path[0]) ? config->mbt_ir_path : nullptr;
            std::string base = mbt_ir_raw ? _model_dir(_resolve_path(mbt_ir_raw)) : _resolve_path("models/");

            std::string ga_path = mbt_ir_raw ? _resolve_path(mbt_ir_raw) : (base + "mbt_g_a" + suffix + ".xml");
            if (ga_backend == "tensorrt") {
                std::string engine = _resolve_path(config->trt_engine_path);
                int trt_device = config->trt_device >= 0 ? config->trt_device : 0;
                if (!ctx->encoder_g_a.load_tensorrt(
                        engine, trt_device, 3, ctx->config.codec_height, ctx->config.codec_width,
                        ctx->C_y, ctx->H_y, ctx->W_y)) { delete ctx; return nullptr; }
                ctx->encoder_g_a.setup_cuda_preprocess(
                    ctx->config.height, ctx->config.width);
            } else if (!ctx->encoder_g_a.load_openvino_with_preprocess(
                           ga_path, device, false,
                           ctx->config.height, ctx->config.width,
                           ctx->config.codec_height, ctx->config.codec_width)) {
                delete ctx; return nullptr;
            }
            ctx->ga_device = ctx->encoder_g_a.device_name();

            const char* ha_ir_raw = (config->ha_ir_path && config->ha_ir_path[0]) ? config->ha_ir_path : nullptr;
            std::string ha_path = ha_ir_raw ? _resolve_path(ha_ir_raw) : (base + "mbt_h_a" + suffix + ".xml");
            // HARD RULE: h_a and h_s define entropy/Gaussian parameters.
            // They must use the same OpenVINO FP32 CPU architecture on sender
            // and receiver. g_a/g_s may use other devices, but h_a/h_s may not.
            if (!ctx->encoder_h_a.load(ha_path, entropy_param_device, true)) { delete ctx; return nullptr; }

            const char* hs_ir_raw = (config->hs_ir_path && config->hs_ir_path[0]) ? config->hs_ir_path : nullptr;
            std::string hs_path = hs_ir_raw ? _resolve_path(hs_ir_raw) : (base + "mbt_h_s" + suffix + "_fp32.xml");
            if (!_file_exists(hs_path)) hs_path = base + "mbt_h_s" + suffix + ".xml";
            if (!ctx->encoder_h_s.load(hs_path, entropy_param_device, true)) { delete ctx; return nullptr; }

            std::string cdf_path = base + "mbt_cdfs.bin";
            if (!ctx->entropy.load_cdfs(cdf_path)) { delete ctx; return nullptr; }
        }

        ctx->initialized = true;
        std::cerr << "rmcompress sender contract: codec=" << (codec == 1 ? "msssim_qvrf" : "mbt")
                  << " g_a_backend=" << ctx->ga_backend
                  << " g_a_device=" << ctx->ga_device
                  << " h_a/h_s=OpenVINO FP32 CPU entropy/Gaussian=CPU/host"
                  << std::endl;
        return ctx;
    } catch (const std::exception& e) {
        std::cerr << "rm_compressor_create failed: " << e.what() << std::endl;
        delete ctx; return nullptr;
    } catch (...) {
        std::cerr << "rm_compressor_create failed: unknown exception" << std::endl;
        delete ctx; return nullptr;
    }
}

// Count bits in encoded strings
static int _count_bits(const std::vector<std::string>& strings) {
    int total = 0;
    for (const auto& s : strings) total += (int)s.size() * 8;
    return total;
}

static constexpr float QVRF_MIN_GAIN = 0.05f;
static constexpr float QVRF_MAX_STABLE_GAIN = 1.00f;
static constexpr float QVRF_TARGET_FILL = 0.82f;

static float _clamp_float(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

static float _rc_persist_cap(const rm_compressor_s* ctx) {
    if (ctx->max_packed_bytes <= MAX_PAYLOAD) {
        return 3.2f;
    }
    if (ctx->max_packed_bytes >= MAX_PAYLOAD * 4) {
        return 4.5f;
    }
    return 2.6f;
}

static float _next_beta_after_frame(const rm_compressor_s* ctx,
                                    float start_beta,
                                    float frame_beta,
                                    int packed_bytes,
                                    bool over_budget) {
    const float cap = _rc_persist_cap(ctx);
    const float max_bytes = static_cast<float>(std::max(1, ctx->max_packed_bytes));
    const float fill = static_cast<float>(packed_bytes) / max_bytes;
    const float base = _clamp_float(start_beta, 1.0f, cap);
    float next = base;

    if (over_budget) {
        // A failed frame may need an emergency beta, but do not let that poison
        // future frames. Nudge the next starting point only slightly upward.
        next = base * 1.08f;
    } else if (fill < 0.72f) {
        // Clearly overcompressed: recover quality quickly.
        next = std::min(base, frame_beta) * 0.72f;
    } else if (fill < 0.82f) {
        next = std::min(base, frame_beta) * 0.84f;
    } else if (fill < 0.90f) {
        next = std::min(base, frame_beta) * 0.94f;
    } else if (fill > 0.98f) {
        next = base * 1.08f;
    } else if (fill > 0.94f) {
        next = base * 1.03f;
    } else {
        next = base * 0.98f;
    }

    return _clamp_float(next, 1.0f, cap);
}

// Entropy encode y with beta scaling applied before encoding
static void _entropy_encode_with_beta(
    rm_compressor_s* ctx, const float* y_data, float beta,
    std::vector<std::string>& y_strings_out,
    std::vector<std::string>& z_strings_out,
    int& z_h, int& z_w)
{
    int C_y = ctx->C_y, H_y = ctx->H_y, W_y = ctx->W_y;
    int C_z = ctx->C_z, H_z = ctx->H_z, W_z = ctx->W_z;

    // MBT: h_a takes RAW y (before beta scaling), matching Python model.h_a(y).
    auto z = ctx->encoder_h_a.encode(y_data, C_y, H_y, W_y);
    auto z_strings = ctx->entropy.compress_bottleneck(z.data(), C_z, H_z, W_z);
    auto z_hat = ctx->entropy.decompress_bottleneck(z_strings, C_z, H_z, W_z);
    auto gaussian_params = ctx->encoder_h_s.encode(z_hat.data());

    int sp = H_y * W_y, ss = C_y * sp;
    std::vector<float> scales(ss), means(ss);
    for (int i = 0; i < ss; i++) { scales[i] = gaussian_params[i]; means[i] = gaussian_params[i + ss]; }

    std::vector<float> y_scaled(ss);
    if (beta != 1.0f) {
        for (int i = 0; i < ss; i++) y_scaled[i] = y_data[i] / beta;
    } else {
        std::memcpy(y_scaled.data(), y_data, ss * sizeof(float));
    }

    auto indexes = ctx->entropy.build_indexes(scales.data(), C_y, H_y, W_y);
    y_strings_out = ctx->entropy.compress_gaussian(y_scaled.data(), indexes.data(), means.data(), C_y, H_y, W_y);
    z_strings_out = z_strings;
    z_h = H_z; z_w = W_z;
}

// QVRF gain rate-control: adapt next-frame gain based on current frame fill.
// Lower gain = more compression = smaller bitstream (opposite direction from beta).
static float _next_gain_after_frame(const rm_compressor_s* ctx,
                                     float start_gain,
                                     float frame_gain,
                                     int packed_bytes,
                                     bool over_budget) {
    const float max_bytes = static_cast<float>(std::max(1, ctx->max_packed_bytes));
    const float fill = static_cast<float>(packed_bytes) / max_bytes;
    const float base = _clamp_float(start_gain, QVRF_MIN_GAIN, QVRF_MAX_STABLE_GAIN);
    float next = base;

    if (over_budget) {
        // Over budget: reduce gain for more compression next frame.
        next = base * 0.85f;
    } else if (fill < 0.72f) {
        // Well under budget: increase gain for better quality.
        next = std::max(base, frame_gain) * 1.15f;
    } else if (fill < 0.82f) {
        next = std::max(base, frame_gain) * 1.08f;
    } else if (fill < 0.90f) {
        next = std::max(base, frame_gain) * 1.03f;
    } else if (fill > 0.98f) {
        next = base * 0.92f;
    } else if (fill > 0.94f) {
        next = base * 0.97f;
    } else {
        next = base;
    }

    return _clamp_float(next, QVRF_MIN_GAIN, QVRF_MAX_STABLE_GAIN);
}

static float _qvrf_retry_gain(const rm_compressor_s* ctx,
                              float gain,
                              int bits,
                              int packed_bytes) {
    const float by_bits = static_cast<float>(bits) /
        static_cast<float>(std::max(1, ctx->budget_bits));
    const float by_size = static_cast<float>(packed_bytes) /
        static_cast<float>(std::max(1, ctx->max_packed_bytes));
    const float ratio = std::max(by_bits, by_size);
    return _clamp_float(gain * QVRF_TARGET_FILL / std::max(1.0f, ratio),
                        QVRF_MIN_GAIN, QVRF_MAX_STABLE_GAIN);
}

// Entropy encode y with QVRF gain scaling.
// KEY DIFFERENCE from MBT _entropy_encode_with_beta:
//   y_scaled = y * gain  (MBT: y / beta)
//   scales *= gain       (MBT: scales NOT scaled)
// Build indexes from gain-modulated scales, then compress y_scaled.
static void _entropy_encode_qvrf(
    rm_compressor_s* ctx, const float* y_data, float gain,
    std::vector<std::string>& y_strings_out,
    std::vector<std::string>& z_strings_out,
    int& z_h, int& z_w)
{
    int C_y = ctx->C_y, H_y = ctx->H_y, W_y = ctx->W_y;
    int C_z = ctx->C_z, H_z = ctx->H_z, W_z = ctx->W_z;

    // Step 1: h_a takes RAW y (before gain scaling). Matches Python model.h_a(y).
    auto z = ctx->encoder_msssim_h_a.encode(y_data, C_y, H_y, W_y);
    auto z_strings = ctx->entropy_msssim.compress_bottleneck(z.data(), C_z, H_z, W_z);
    auto z_hat = ctx->entropy_msssim.decompress_bottleneck(z_strings, C_z, H_z, W_z);
    auto gaussian_params = ctx->encoder_msssim_h_s.encode(z_hat.data(), C_z, H_z, W_z);

    int sp = H_y * W_y, ss = C_y * sp;
    std::vector<float> scales(ss), means(ss);
    for (int i = 0; i < ss; i++) { scales[i] = gaussian_params[i]; means[i] = gaussian_params[i + ss]; }

    // Step 2: QVRF gain scaling — y, scales, and means ALL multiplied by gain.
    // Python: indexes = build_indexes(scales_hat * scale)
    //         compress(y * scale, indexes, means=means_hat * scale)
    std::vector<float> y_scaled(ss);
    if (gain != 1.0f) {
        for (int i = 0; i < ss; i++) {
            y_scaled[i] = y_data[i] * gain;
            scales[i] *= gain;
            means[i] *= gain;
        }
    } else {
        std::memcpy(y_scaled.data(), y_data, ss * sizeof(float));
    }

    auto indexes = ctx->entropy_msssim.build_indexes(scales.data(), C_y, H_y, W_y);
    y_strings_out = ctx->entropy_msssim.compress_gaussian(y_scaled.data(), indexes.data(), means.data(), C_y, H_y, W_y);
    z_strings_out = z_strings;
    z_h = H_z; z_w = W_z;
}

int rm_compress_frame(rm_compressor_t* ctx,
                      const uint8_t* rgb_buf,
                      uint8_t* out_buf, int* out_len) {
    if (!ctx || !ctx->initialized || !rgb_buf || !out_buf || !out_len)
        return -1;

    try {
        auto t_total0 = std::chrono::steady_clock::now();
        ctx->last_stats = {};
        int w = ctx->config.width > 0 ? ctx->config.width : 128;
        int h = ctx->config.height > 0 ? ctx->config.height : 128;

        auto t0 = std::chrono::steady_clock::now();
        int codec_w = ctx->config.codec_width > 0 ? ctx->config.codec_width : 128; //输入
        int codec_h = ctx->config.codec_height > 0 ? ctx->config.codec_height : codec_w;
        // Choose codec: route to MBT (beta-RC) or MS-SSIM QVRF (gain-RC)
        bool use_qvrf = (ctx->config.codec == 1 && ctx->msssim_loaded);
        // 1 图像预处理
        // When the g_a encoder has fused preprocess, skip CPU preprocess_rgb()
        // and pass the raw BGR frame directly to the iGPU.
        const bool qvrf_has_pp = use_qvrf && ctx->encoder_msssim_g_a.has_preprocess();
        const bool mbt_has_pp  = !use_qvrf && ctx->encoder_g_a.has_preprocess();

        // CPU preprocess only when needed (TRT path or PPP fallback)
        const std::vector<float>* rgb_float_ptr = nullptr;
        if (!qvrf_has_pp && !mbt_has_pp) {
            rgb_float_ptr = &preprocess_rgb(         
                rgb_buf, w, h, codec_w, codec_h, ctx->preprocess_scratch);
        }
        auto t1 = std::chrono::steady_clock::now();

        if (use_qvrf) {
            // ---- MS-SSIM QVRF path ----
            // 2 g_a 推理
            auto y = qvrf_has_pp
                ? ctx->encoder_msssim_g_a.encode_raw(rgb_buf, h, w)
                : ctx->encoder_msssim_g_a.encode(rgb_float_ptr->data(), 3, codec_h, codec_w);
            auto t2 = std::chrono::steady_clock::now();
            ctx->last_stats.preprocess_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();//预处理耗时
            ctx->last_stats.g_a_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();//g_a 推理耗时
            // 3 熵编码
            // Gain-RC: gain < 1 compresses latent; adapt from previous frame
            std::vector<std::string> y_str, z_str;
            int zh, zw;
            float start_gain = _clamp_float(ctx->next_gain, QVRF_MIN_GAIN, QVRF_MAX_STABLE_GAIN);
            float gain = start_gain;

            auto p0 = std::chrono::steady_clock::now();
            _entropy_encode_qvrf(ctx, y.data(), gain, y_str, z_str, zh, zw);
            auto p1 = std::chrono::steady_clock::now();
            ctx->last_stats.pass1_ms = std::chrono::duration<float, std::milli>(p1 - p0).count();
            ctx->last_stats.rc_passes = 1; //码率控制尝试次数
            int bits = _count_bits(y_str) + _count_bits(z_str);
            auto packed = pack_msssim_qvrf(y_str, z_str, zh, zw, gain);
            // 4 超预算或或打包的字节数超了
            // First retry: reduce gain to shrink bitstream
            if (bits > ctx->budget_bits || (int)packed.size() > ctx->max_packed_bytes) {
                gain = _qvrf_retry_gain(ctx, gain, bits, static_cast<int>(packed.size()));
                auto p20 = std::chrono::steady_clock::now();
                _entropy_encode_qvrf(ctx, y.data(), gain, y_str, z_str, zh, zw);
                auto p21 = std::chrono::steady_clock::now();
                ctx->last_stats.pass2_ms = std::chrono::duration<float, std::milli>(p21 - p20).count();
                ctx->last_stats.rc_passes = 2;
                bits = _count_bits(y_str) + _count_bits(z_str);
                packed = pack_msssim_qvrf(y_str, z_str, zh, zw, gain);
            }

            // Second retry: aggressive gain reduction
            if (bits > ctx->budget_bits || (int)packed.size() > ctx->max_packed_bytes) {
                gain = std::max(QVRF_MIN_GAIN, gain * 0.50f);
                auto p30 = std::chrono::steady_clock::now();
                _entropy_encode_qvrf(ctx, y.data(), gain, y_str, z_str, zh, zw);
                auto p31 = std::chrono::steady_clock::now();
                ctx->last_stats.pass3_ms = std::chrono::duration<float, std::milli>(p31 - p30).count();
                ctx->last_stats.rc_passes = 3;
                packed = pack_msssim_qvrf(y_str, z_str, zh, zw, gain);
            }

            if ((int)packed.size() > ctx->max_packed_bytes) {
                *out_len = (int)packed.size();
                ctx->last_stats.over_budget = 1;
                ctx->last_stats.beta = gain;  // stats.beta reused as gain for QVRF
                ctx->last_stats.packed_bytes = (int)packed.size();
                ctx->last_stats.entropy_bits = bits;
                ctx->last_stats.total_ms = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - t_total0).count();
                ctx->next_gain = _next_gain_after_frame(
                    ctx, start_gain, gain, static_cast<int>(packed.size()), true);
                return -2;  //代表这一帧字节数超了
            }

            if ((int)packed.size() > *out_len) return -1; //超了输出缓冲区容量
            auto tpack0 = std::chrono::steady_clock::now();
            std::memcpy(out_buf, packed.data(), packed.size());
            auto tpack1 = std::chrono::steady_clock::now();
            ctx->last_stats.pack_ms = std::chrono::duration<float, std::milli>(tpack1 - tpack0).count();
            *out_len = (int)packed.size();
            ctx->last_stats.beta = gain;  // stats.beta reused as gain for QVRF
            ctx->last_stats.packed_bytes = (int)packed.size();
            ctx->last_stats.entropy_bits = bits;
            ctx->last_stats.total_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_total0).count();
            ctx->next_gain = _next_gain_after_frame(
                ctx, start_gain, gain, static_cast<int>(packed.size()), false);
            if (ctx->last_stats.rc_passes > 1) {
                ctx->next_gain = std::min(ctx->next_gain, gain);
            }
            return 0;
        }

        // ---- MBT path (original) ----
        auto y = mbt_has_pp
            ? ctx->encoder_g_a.encode_raw(rgb_buf, h, w)
            : ctx->encoder_g_a.encode(rgb_float_ptr->data(), 3, codec_h, codec_w);
        auto t2 = std::chrono::steady_clock::now();
        ctx->last_stats.preprocess_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        ctx->last_stats.g_a_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();

        // Beta-RC: matches Python codec_decoder.py approach
        std::vector<std::string> y_str, z_str;
        int zh, zw;
        float start_beta = std::max(1.0f, ctx->next_beta);
        float beta = start_beta;

        auto p0 = std::chrono::steady_clock::now();
        _entropy_encode_with_beta(ctx, y.data(), beta, y_str, z_str, zh, zw);
        auto p1 = std::chrono::steady_clock::now();
        ctx->last_stats.pass1_ms = std::chrono::duration<float, std::milli>(p1 - p0).count();
        ctx->last_stats.rc_passes = 1;
        int bits = _count_bits(y_str) + _count_bits(z_str);
        auto packed = pack_strings(y_str, z_str, zh, zw, beta, 1);

        // RC must budget the final packed bitstream, not only entropy payload.
        // Length fields + beta/hs trailer add overhead; ignoring that allowed a
        // few 561-570B frames. Runtime TX now emits OVER instead of a third chunk,
        // but frequent misses still show up as visible frame holds.
        if (bits > ctx->budget_bits || (int)packed.size() > ctx->max_packed_bytes) {
            float by_bits = (float)bits / (float)ctx->budget_bits;
            float by_size = (float)packed.size() / (float)ctx->max_packed_bytes;
            beta = std::max(beta * 1.05f, std::max(by_bits, by_size) * 1.2f);
            auto p20 = std::chrono::steady_clock::now();
            _entropy_encode_with_beta(ctx, y.data(), beta, y_str, z_str, zh, zw);
            auto p21 = std::chrono::steady_clock::now();
            ctx->last_stats.pass2_ms = std::chrono::duration<float, std::milli>(p21 - p20).count();
            ctx->last_stats.rc_passes = 2;
            bits = _count_bits(y_str) + _count_bits(z_str);
            packed = pack_strings(y_str, z_str, zh, zw, beta, 1);
        }

        if (bits > ctx->budget_bits || (int)packed.size() > ctx->max_packed_bytes) {
            beta = beta * 1.7f;
            auto p30 = std::chrono::steady_clock::now();
            _entropy_encode_with_beta(ctx, y.data(), beta, y_str, z_str, zh, zw);
            auto p31 = std::chrono::steady_clock::now();
            ctx->last_stats.pass3_ms = std::chrono::duration<float, std::milli>(p31 - p30).count();
            ctx->last_stats.rc_passes = 3;
            packed = pack_strings(y_str, z_str, zh, zw, beta, 1);
        }

        if ((int)packed.size() > ctx->max_packed_bytes) {
            *out_len = (int)packed.size();
            ctx->last_stats.over_budget = 1;
            ctx->last_stats.beta = beta;
            ctx->last_stats.packed_bytes = (int)packed.size();
            ctx->last_stats.entropy_bits = bits;
            ctx->last_stats.total_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_total0).count();
            ctx->next_beta = _next_beta_after_frame(
                ctx, start_beta, beta, static_cast<int>(packed.size()), true);
            return -2;
        }

        if ((int)packed.size() > *out_len) return -1;
        auto tpack0 = std::chrono::steady_clock::now();
        std::memcpy(out_buf, packed.data(), packed.size());
        auto tpack1 = std::chrono::steady_clock::now();
        ctx->last_stats.pack_ms = std::chrono::duration<float, std::milli>(tpack1 - tpack0).count();
        *out_len = (int)packed.size();
        ctx->last_stats.beta = beta;
        ctx->last_stats.packed_bytes = (int)packed.size();
        ctx->last_stats.entropy_bits = bits;
        ctx->last_stats.total_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_total0).count();
        ctx->next_beta = _next_beta_after_frame(
            ctx, start_beta, beta, static_cast<int>(packed.size()), false);
        return 0;

    } catch (...) {
        return -1;
    }
}

const rm_compress_stats_t* rm_compressor_last_stats(rm_compressor_t* ctx) {
    if (!ctx) return nullptr;
    return &ctx->last_stats;
}

void rm_compressor_destroy(rm_compressor_t* ctx) { delete ctx; }

} // extern "C"
