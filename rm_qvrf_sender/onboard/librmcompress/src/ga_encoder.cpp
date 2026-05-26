#include "rmcompress/ga_encoder.h"

#include <stdexcept>

namespace rmcompress {

GaEncoder::~GaEncoder() = default;

bool GaEncoder::load_openvino(const std::string& model_path,
                              const std::string& device,
                              bool f32_precision) {
    if (!_openvino.load(model_path, device, f32_precision)) {
        return false;
    }
    _backend = GaBackend::OpenVINO;
    _backend_name = "openvino";
    _device_name = device;
    _tensorrt.reset();
    return true;
}

bool GaEncoder::load_openvino_with_preprocess(const std::string& model_path,
                                               const std::string& device,
                                               bool f32_precision,
                                               int src_h, int src_w,
                                               int dst_h, int dst_w) {
    if (_openvino.load_with_preprocess(model_path, device, f32_precision,
                                       src_h, src_w, dst_h, dst_w)) {
        _backend = GaBackend::OpenVINO;
        _backend_name = "openvino+preprocess";
        _device_name = device;
        _tensorrt.reset();
        return true;
    }
    // Fallback to plain OpenVINO load
    return load_openvino(model_path, device, f32_precision);
}

bool GaEncoder::load_tensorrt(const std::string& engine_path,
                              int device_id,
                              int input_c,
                              int input_h,
                              int input_w,
                              int output_c,
                              int output_h,
                              int output_w) {
    auto trt = std::make_unique<TensorRtGaEncoder>();
    if (!trt->load(engine_path, device_id, input_c, input_h, input_w,
                   output_c, output_h, output_w)) {
        return false;
    }
    _backend = GaBackend::TensorRT;
    _backend_name = "tensorrt";
    _device_name = "cuda:" + std::to_string(device_id);
    _tensorrt = std::move(trt);
    return true;
}

bool GaEncoder::setup_cuda_preprocess(int src_h, int src_w) {
    if (_backend != GaBackend::TensorRT || !_tensorrt) return false;
    return _tensorrt->setup_cuda_preprocess(src_h, src_w);
}

bool GaEncoder::has_preprocess() const {
    if (_backend == GaBackend::TensorRT)
        return _tensorrt && _tensorrt->has_preprocess();
    return _openvino.has_preprocess();
}

std::vector<float> GaEncoder::encode(const float* data, int C, int H, int W) {
    if (_backend == GaBackend::TensorRT) {
        if (!_tensorrt)
            throw std::runtime_error("TensorRT g_a backend selected but not loaded");
        return _tensorrt->encode(data, C, H, W);
    }
    return _openvino.encode(data, C, H, W);
}

std::vector<float> GaEncoder::encode_raw(const uint8_t* bgr_data,
                                          int src_h, int src_w) {
    if (_backend == GaBackend::TensorRT) {
        if (!_tensorrt)
            throw std::runtime_error("TensorRT g_a backend selected but not loaded");
        return _tensorrt->encode_raw(bgr_data, src_h, src_w);
    }
    return _openvino.encode_raw(bgr_data, src_h, src_w);
}

}  // namespace rmcompress
