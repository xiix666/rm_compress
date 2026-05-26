#pragma once

#include "rmcompress/mbt_encoder.h"

#include <memory>
#include <string>
#include <vector>

namespace rmcompress {

enum class GaBackend {
    OpenVINO = 0,
    TensorRT = 1,
};

class TensorRtGaEncoder {
public:
    TensorRtGaEncoder();
    ~TensorRtGaEncoder();

    bool load(const std::string& engine_path,
              int device_id,
              int input_c,
              int input_h,
              int input_w,
              int output_c,
              int output_h,
              int output_w);
    bool setup_cuda_preprocess(int src_h, int src_w);
    bool has_preprocess() const;
    std::vector<float> encode(const float* data, int C, int H, int W);
    std::vector<float> encode_raw(const uint8_t* bgr_host, int src_h, int src_w);

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

/// g_a-only encoder front end. h_a/h_s intentionally keep using MbtEncoder
/// directly so entropy-parameter placement cannot follow this backend switch.
class GaEncoder {
public:
    GaEncoder() = default;
    ~GaEncoder();

    bool load_openvino(const std::string& model_path,
                       const std::string& device,
                       bool f32_precision);

    /// Load OpenVINO with BGR uint8 NHWC preprocess fused via PrePostProcessor.
    /// Falls back to plain load_openvino() if PPP setup fails.
    bool load_openvino_with_preprocess(const std::string& model_path,
                                       const std::string& device,
                                       bool f32_precision,
                                       int src_h, int src_w,
                                       int dst_h, int dst_w);

    bool load_tensorrt(const std::string& engine_path,
                       int device_id,
                       int input_c,
                       int input_h,
                       int input_w,
                       int output_c,
                       int output_h,
                       int output_w);

    /// After load_tensorrt(), call this to enable CUDA preprocess for the TRT path.
    bool setup_cuda_preprocess(int src_h, int src_w);

    std::vector<float> encode(const float* data, int C = 0, int H = 0, int W = 0);
    std::vector<float> encode_raw(const uint8_t* bgr_data, int src_h, int src_w);
    bool has_preprocess() const;

    GaBackend backend() const { return _backend; }
    const std::string& backend_name() const { return _backend_name; }
    const std::string& device_name() const { return _device_name; }

private:
    GaBackend _backend = GaBackend::OpenVINO;
    std::string _backend_name = "openvino";
    std::string _device_name = "GPU.0";
    MbtEncoder _openvino;
    std::unique_ptr<TensorRtGaEncoder> _tensorrt;
};

}  // namespace rmcompress
