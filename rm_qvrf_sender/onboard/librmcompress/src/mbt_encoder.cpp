#include "rmcompress/mbt_encoder.h"

#include <openvino/core/preprocess/pre_post_process.hpp>
#include <cstring>
#include <stdexcept>

bool MbtEncoder::load(const std::string& model_path, const std::string& device,
                       bool f32_precision) {
    // Read the IR model (.xml)
    auto model = _core.read_model(model_path);

    // Compile with low-latency single-stream settings for real-time encode.
    // h_s MUST use FP32: FP16 differences push values across quantization
    // boundaries in build_indexes(), causing wrong CDF table selection and
    // corrupting the bitstream (96%+ elements wrong). g_a/g_s are safe at FP16.
    _compiled = _core.compile_model(
        model, device,
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
        ov::hint::inference_precision(
            f32_precision ? ov::element::f32 : ov::element::f16),
        ov::streams::num(1));

    _ireq = _compiled.create_infer_request();
    ov::PartialShape input_ps = _compiled.input().get_partial_shape();
    _input_shape.clear();
    for (auto& d : input_ps) {
        _input_shape.push_back(d.is_dynamic() ? 1 : d.get_length());
    }
    _input_type = _compiled.input().get_element_type();

    ov::PartialShape output_ps = _compiled.output().get_partial_shape();
    _output_size = 1;
    for (auto& d : output_ps) {
        _output_size *= d.is_dynamic() ? 1 : d.get_length();
    }
    _output_buffer.resize(_output_size);
    return true;
}

std::vector<float> MbtEncoder::encode(const float* data, int C, int H, int W) {
    // Build input shape: use explicit dims for dynamic axes, model dims for fixed axes
    ov::Shape shape = _input_shape;
    auto input_ps = _compiled.input().get_partial_shape();
    for (size_t i = 0; i < shape.size() && i < input_ps.size(); i++) {
        if (input_ps[i].is_dynamic()) {
            if (i == 1 && C > 0) shape[i] = C;        // channels
            else if (i == 2 && H > 0) shape[i] = H;   // height
            else if (i == 3 && W > 0) shape[i] = W;   // width
            else if (i == 0) shape[i] = 1;            // batch
        }
    }

    ov::Tensor input_tensor(_input_type, shape, const_cast<float*>(data));

    _ireq.set_input_tensor(input_tensor);
    _ireq.infer();

    const ov::Tensor& output_tensor = _ireq.get_output_tensor();
    const size_t      total         = output_tensor.get_size();

    if (_output_buffer.size() != total) {
        _output_buffer.resize(total);
    }
    std::memcpy(_output_buffer.data(), output_tensor.data<float>(), total * sizeof(float));
    return _output_buffer;
}

bool MbtEncoder::load_with_preprocess(const std::string& model_path,
                                      const std::string& device,
                                      bool f32_precision,
                                      int src_h, int src_w,
                                      int dst_h, int dst_w) {
    try {
        auto model = _core.read_model(model_path);

        ov::preprocess::PrePostProcessor ppp(model);
        auto& inp = ppp.input();

        // Declare the tensor arriving from the caller: BGR uint8 NHWC
        inp.tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);

        // Fused preprocess steps (executed on the target device)
        auto& pre = inp.preprocess();
        pre.convert_color(ov::preprocess::ColorFormat::RGB);
        if (dst_h > 0 && dst_w > 0) {
            pre.resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR, dst_h, dst_w);
        }
        pre.convert_element_type(ov::element::f32)
           .scale(255.0f);

        // Model expects NCHW float
        inp.model().set_layout("NCHW");

        model = ppp.build();

        _compiled = _core.compile_model(
            model, device,
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
            ov::hint::inference_precision(
                f32_precision ? ov::element::f32 : ov::element::f16),
            ov::streams::num(1));

        _ireq = _compiled.create_infer_request();

        // After PPP the compiled input is u8 NHWC; record output size
        ov::PartialShape output_ps = _compiled.output().get_partial_shape();
        _output_size = 1;
        for (auto& d : output_ps)
            _output_size *= d.is_dynamic() ? 1 : d.get_length();
        _output_buffer.resize(_output_size);

        _has_preprocess = true;
        return true;
    } catch (const std::exception& e) {
        // Fallback: caller should retry with plain load()
        _has_preprocess = false;
        return false;
    }
}

std::vector<float> MbtEncoder::encode_raw(const uint8_t* bgr_data,
                                          int src_h, int src_w) {
    if (!_has_preprocess)
        throw std::runtime_error("encode_raw called but preprocess not loaded");

    // Wrap caller buffer as u8 NHWC tensor [1, src_h, src_w, 3]
    ov::Shape shape{1, (size_t)src_h, (size_t)src_w, 3};
    ov::Tensor input_tensor(ov::element::u8, shape,
                            const_cast<uint8_t*>(bgr_data));

    _ireq.set_input_tensor(input_tensor);
    _ireq.infer();

    const ov::Tensor& out = _ireq.get_output_tensor();
    const size_t total = out.get_size();
    if (_output_buffer.size() != total)
        _output_buffer.resize(total);
    std::memcpy(_output_buffer.data(), out.data<float>(), total * sizeof(float));
    return _output_buffer;
}
