#pragma once

#include <openvino/openvino.hpp>

#include <string>
#include <vector>

/// MBT g_a encoder: runs the analysis transform via OpenVINO.
///
/// Input:  (1, 3, 128, 128) float32 NCHW, range [0, 1].
/// Output: y = (1, 192, 8, 8) float32 NCHW.
class MbtEncoder {
public:
    MbtEncoder() = default;
    ~MbtEncoder() = default;

    /// Load an OpenVINO IR model (.xml).
    /// @param model_path    Path to the .xml file (same-stem .bin expected alongside).
    /// @param device        OpenVINO device string, e.g. "GPU.0" or "CPU".
    /// @param f32_precision If true, use FP32 inference (required for h_s to avoid
    ///                       CDF index corruption). Default FP16 for g_a/g_s.
    /// @return true on success.
    bool load(const std::string& model_path, const std::string& device = "GPU.0",
              bool f32_precision = false);

    /// Run inference with explicit input shape.
    /// @param data  Pointer to float32 NCHW data (zero-copy, must be contiguous).
    /// @param C, H, W  Input spatial dimensions. Override dynamic dims in model shape.
    /// @return Output tensor data, contiguously laid out.
    std::vector<float> encode(const float* data, int C = 0, int H = 0, int W = 0);

    /// Expose the input element count from the compiled model.
    size_t input_size() const {
        auto ps = _compiled.input().get_partial_shape();
        size_t n = 1;
        for (auto& d : ps) n *= d.is_dynamic() ? 1 : d.get_length();
        return n;
    }

    /// Expose the output element count from the compiled model.
    size_t output_size() const {
        auto ps = _compiled.output().get_partial_shape();
        size_t n = 1;
        for (auto& d : ps) n *= d.is_dynamic() ? 1 : d.get_length();
        return n;
    }

    /// Load an OpenVINO IR model with BGR uint8 NHWC preprocess fused in via
    /// PrePostProcessor. On success, encode_raw() accepts raw BGR frames directly.
    /// Falls back to plain load() if PPP setup fails.
    /// @param src_h/src_w  Source frame size (0 = dynamic, no resize source hint).
    /// @param dst_h/dst_w  Codec target size (resize destination, fixed at load time).
    bool load_with_preprocess(const std::string& model_path,
                              const std::string& device,
                              bool f32_precision,
                              int src_h, int src_w,
                              int dst_h, int dst_w);

    /// Run inference on a raw BGR uint8 HWC frame.
    /// Only valid after a successful load_with_preprocess(); throws otherwise.
    /// @param bgr_data  Pointer to BGR uint8 interleaved data.
    /// @param src_h/src_w  Frame dimensions (must match load_with_preprocess src).
    std::vector<float> encode_raw(const uint8_t* bgr_data, int src_h, int src_w);

    bool has_preprocess() const { return _has_preprocess; }

private:
    ov::Core          _core;
    ov::CompiledModel _compiled;
    ov::InferRequest  _ireq;
    ov::Shape         _input_shape;
    ov::element::Type _input_type;
    size_t            _output_size = 0;
    std::vector<float> _output_buffer;
    bool              _has_preprocess = false;
};
