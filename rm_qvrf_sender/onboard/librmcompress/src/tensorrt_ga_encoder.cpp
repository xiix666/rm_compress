#include "rmcompress/ga_encoder.h"

#ifndef RMCOMPRESS_ENABLE_TENSORRT

#include <stdexcept>

namespace rmcompress {

class TensorRtGaEncoder::Impl {};

TensorRtGaEncoder::TensorRtGaEncoder() = default;
TensorRtGaEncoder::~TensorRtGaEncoder() = default;

bool TensorRtGaEncoder::load(const std::string&,
                             int,
                             int,
                             int,
                             int,
                             int,
                             int,
                             int) {
    throw std::runtime_error(
        "TensorRT g_a backend requested, but rmcompress was built without "
        "RMCOMPRESS_ENABLE_TENSORRT. Reconfigure onboard with TensorRT and "
        "-DRMCOMPRESS_ENABLE_TENSORRT=ON.");
}

std::vector<float> TensorRtGaEncoder::encode(const float*, int, int, int) {
    throw std::runtime_error("TensorRT g_a backend is not compiled");
}

bool TensorRtGaEncoder::setup_cuda_preprocess(int, int) {
    throw std::runtime_error("TensorRT g_a backend is not compiled");
}

bool TensorRtGaEncoder::has_preprocess() const { return false; }

std::vector<float> TensorRtGaEncoder::encode_raw(const uint8_t*, int, int) {
    throw std::runtime_error("TensorRT g_a backend is not compiled");
}

}  // namespace rmcompress

#else

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "rmcompress/preprocess_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace rmcompress {

namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

template <typename T>
struct TrtDestroy {
    void operator()(T* ptr) const {
        delete ptr;
    }
};

static void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::string msg = std::string(what) + ": " + cudaGetErrorString(status);
        throw std::runtime_error(msg);
    }
}

static std::vector<char> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("TensorRT engine does not exist or cannot be opened: " + path);
    }
    std::streamsize size = in.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + path);
    }
    in.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!in.read(data.data(), size)) {
        throw std::runtime_error("Failed to read TensorRT engine: " + path);
    }
    return data;
}

static size_t volume(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0) {
        throw std::runtime_error("TensorRT binding has invalid rank");
    }
    size_t total = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("TensorRT binding has unresolved dynamic dimension");
        }
        total *= static_cast<size_t>(dims.d[i]);
    }
    return total;
}

static std::string dims_to_string(const nvinfer1::Dims& dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i) oss << ",";
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

static void require_nchw(const nvinfer1::Dims& dims,
                         int c,
                         int h,
                         int w,
                         const char* name) {
    if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != c ||
        dims.d[2] != h || dims.d[3] != w) {
        std::ostringstream oss;
        oss << "TensorRT " << name << " shape mismatch: expected [1,"
            << c << "," << h << "," << w << "], got " << dims_to_string(dims);
        throw std::runtime_error(oss.str());
    }
}

}  // namespace

class TensorRtGaEncoder::Impl {
public:
    ~Impl() {
        if (_raw_input_device) cudaFree(_raw_input_device);
        if (_input_device) cudaFree(_input_device);
        if (_output_device) cudaFree(_output_device);
        if (_stream) cudaStreamDestroy(_stream);
    }

    bool load(const std::string& engine_path,
              int device_id,
              int input_c,
              int input_h,
              int input_w,
              int output_c,
              int output_h,
              int output_w) {
        if (device_id < 0) {
            throw std::runtime_error("TensorRT CUDA device id must be >= 0");
        }
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_id >= device_count) {
            std::ostringstream oss;
            oss << "TensorRT CUDA device " << device_id
                << " unavailable; CUDA device count is " << device_count;
            throw std::runtime_error(oss.str());
        }
        check_cuda(cudaSetDevice(device_id), "cudaSetDevice");

        auto engine_data = read_file(engine_path);
        _runtime.reset(nvinfer1::createInferRuntime(_logger));
        if (!_runtime) {
            throw std::runtime_error("createInferRuntime failed");
        }
        _engine.reset(_runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!_engine) {
            throw std::runtime_error("deserializeCudaEngine failed for: " + engine_path);
        }
        _context.reset(_engine->createExecutionContext());
        if (!_context) {
            throw std::runtime_error("createExecutionContext failed");
        }

        int input_count = 0;
        int output_count = 0;
        const int nb = _engine->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            const char* name = _engine->getIOTensorName(i);
            auto mode = _engine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_count++;
                _input_name = name;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                output_count++;
                _output_name = name;
            }
        }
        if (input_count != 1 || output_count != 1) {
            throw std::runtime_error("TensorRT engine must have one input and one output tensor");
        }
        if (_engine->getTensorDataType(_input_name.c_str()) != nvinfer1::DataType::kFLOAT ||
            _engine->getTensorDataType(_output_name.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("TensorRT g_a engine input/output must be FP32");
        }

        nvinfer1::Dims input_dims = _engine->getTensorShape(_input_name.c_str());
        nvinfer1::Dims output_dims = _engine->getTensorShape(_output_name.c_str());
        if (std::any_of(input_dims.d, input_dims.d + input_dims.nbDims, [](int d) { return d < 0; })) {
            nvinfer1::Dims fixed{};
            fixed.nbDims = 4;
            fixed.d[0] = 1;
            fixed.d[1] = input_c;
            fixed.d[2] = input_h;
            fixed.d[3] = input_w;
            if (!_context->setInputShape(_input_name.c_str(), fixed)) {
                throw std::runtime_error("TensorRT failed to set dynamic input shape");
            }
            input_dims = _context->getTensorShape(_input_name.c_str());
            output_dims = _context->getTensorShape(_output_name.c_str());
        }
        require_nchw(input_dims, input_c, input_h, input_w, "input");
        require_nchw(output_dims, output_c, output_h, output_w, "output");

        _input_elems = volume(input_dims);
        _output_elems = volume(output_dims);
        _output_host.resize(_output_elems);
        check_cuda(cudaMalloc(&_input_device, _input_elems * sizeof(float)), "cudaMalloc input");
        check_cuda(cudaMalloc(&_output_device, _output_elems * sizeof(float)), "cudaMalloc output");
        check_cuda(cudaStreamCreate(&_stream), "cudaStreamCreate");

        if (!_context->setTensorAddress(_input_name.c_str(), _input_device) ||
            !_context->setTensorAddress(_output_name.c_str(), _output_device)) {
            throw std::runtime_error("TensorRT failed to bind tensor addresses");
        }

        fprintf(stderr,
                "TensorRT g_a loaded: backend=tensorrt device=cuda:%d engine=%s "
                "input=%s output=%s (TensorRT/NVIDIA GPU)\n",
                device_id, engine_path.c_str(),
                dims_to_string(input_dims).c_str(),
                dims_to_string(output_dims).c_str());
        return true;
    }

    bool setup_cuda_preprocess(int src_h, int src_w) {
        if (_raw_input_device) {
            cudaFree(_raw_input_device);
            _raw_input_device = nullptr;
        }
        cudaError_t err = cudaMalloc(&_raw_input_device,
                                     static_cast<size_t>(src_h) * src_w * 3);
        if (err != cudaSuccess) {
            fprintf(stderr, "TensorRT CUDA preprocess alloc failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        _src_h = src_h;
        _src_w = src_w;
        _has_cuda_preprocess = true;
        fprintf(stderr,
                "TensorRT g_a: CUDA preprocess enabled src=%dx%d dst=%dx%d\n",
                src_w, src_h,
                static_cast<int>(_input_elems > 0
                    ? static_cast<int>(std::sqrt(_input_elems / 3)) : 0),
                static_cast<int>(_input_elems > 0
                    ? static_cast<int>(std::sqrt(_input_elems / 3)) : 0));
        return true;
    }

    bool has_preprocess() const { return _has_cuda_preprocess; }

    std::vector<float> encode_raw(const uint8_t* bgr_host, int src_h, int src_w) {
        if (!bgr_host) throw std::runtime_error("TensorRT encode_raw: null input");
        // H2D: raw BGR uint8 frame
        check_cuda(cudaMemcpyAsync(_raw_input_device, bgr_host,
                                   static_cast<size_t>(src_h) * src_w * 3,
                                   cudaMemcpyHostToDevice, _stream),
                   "encode_raw H2D");
        // Derive codec (dst) dimensions from _input_elems = 3 * dst_h * dst_w
        // The engine was loaded with explicit input_h/input_w; recover from dims.
        const nvinfer1::Dims dims = _context->getTensorShape(_input_name.c_str());
        const int dst_h = static_cast<int>(dims.d[2]);
        const int dst_w = static_cast<int>(dims.d[3]);
        // CUDA kernel: BGR→RGB + bilinear resize + normalize → TRT input buffer
        launch_bgr_resize_normalize(
            static_cast<const uint8_t*>(_raw_input_device),
            src_h, src_w,
            static_cast<float*>(_input_device),
            dst_h, dst_w,
            _stream);
        // TRT inference (input already in _input_device)
        if (!_context->enqueueV3(_stream)) {
            throw std::runtime_error("TensorRT enqueueV3 failed (encode_raw)");
        }
        check_cuda(cudaMemcpyAsync(_output_host.data(), _output_device,
                                   _output_elems * sizeof(float),
                                   cudaMemcpyDeviceToHost, _stream),
                   "encode_raw D2H");
        check_cuda(cudaStreamSynchronize(_stream), "encode_raw sync");
        auto bad = std::find_if(_output_host.begin(), _output_host.end(),
                                [](float v) { return !std::isfinite(v); });
        if (bad != _output_host.end()) {
            std::ostringstream oss;
            oss << "TensorRT g_a (encode_raw) non-finite output at offset "
                << std::distance(_output_host.begin(), bad);
            throw std::runtime_error(oss.str());
        }
        return _output_host;
    }

    std::vector<float> encode(const float* data, int C, int H, int W) {
        if (!data) {
            throw std::runtime_error("TensorRT g_a input is null");
        }
        if (C > 0 && H > 0 && W > 0) {
            const size_t requested = static_cast<size_t>(C) * H * W;
            if (requested != _input_elems) {
                std::ostringstream oss;
                oss << "TensorRT g_a input size mismatch: requested "
                    << requested << " floats, engine expects " << _input_elems;
                throw std::runtime_error(oss.str());
            }
        }
        check_cuda(cudaMemcpyAsync(_input_device, data, _input_elems * sizeof(float),
                                   cudaMemcpyHostToDevice, _stream),
                   "cudaMemcpyAsync H2D");
        if (!_context->enqueueV3(_stream)) {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }
        check_cuda(cudaMemcpyAsync(_output_host.data(), _output_device,
                                   _output_elems * sizeof(float),
                                   cudaMemcpyDeviceToHost, _stream),
                   "cudaMemcpyAsync D2H");
        check_cuda(cudaStreamSynchronize(_stream), "cudaStreamSynchronize");
        auto bad = std::find_if(_output_host.begin(), _output_host.end(),
                                [](float v) { return !std::isfinite(v); });
        if (bad != _output_host.end()) {
            std::ostringstream oss;
            oss << "TensorRT g_a produced non-finite output at latent offset "
                << std::distance(_output_host.begin(), bad);
            throw std::runtime_error(oss.str());
        }
        return _output_host;
    }

private:
    TrtLogger _logger;
    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> _runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>> _engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>> _context;
    std::string _input_name;
    std::string _output_name;
    void* _input_device = nullptr;
    void* _output_device = nullptr;
    void* _raw_input_device = nullptr;
    cudaStream_t _stream = nullptr;
    size_t _input_elems = 0;
    size_t _output_elems = 0;
    std::vector<float> _output_host;
    int _src_h = 0;
    int _src_w = 0;
    bool _has_cuda_preprocess = false;
};

TensorRtGaEncoder::TensorRtGaEncoder() = default;
TensorRtGaEncoder::~TensorRtGaEncoder() = default;

bool TensorRtGaEncoder::load(const std::string& engine_path,
                             int device_id,
                             int input_c,
                             int input_h,
                             int input_w,
                             int output_c,
                             int output_h,
                             int output_w) {
    _impl = std::make_unique<Impl>();
    return _impl->load(engine_path, device_id, input_c, input_h, input_w,
                       output_c, output_h, output_w);
}

std::vector<float> TensorRtGaEncoder::encode(const float* data, int C, int H, int W) {
    if (!_impl) throw std::runtime_error("TensorRT g_a encoder has not been loaded");
    return _impl->encode(data, C, H, W);
}

bool TensorRtGaEncoder::setup_cuda_preprocess(int src_h, int src_w) {
    if (!_impl) throw std::runtime_error("TensorRT g_a encoder has not been loaded");
    return _impl->setup_cuda_preprocess(src_h, src_w);
}

bool TensorRtGaEncoder::has_preprocess() const {
    return _impl && _impl->has_preprocess();
}

std::vector<float> TensorRtGaEncoder::encode_raw(const uint8_t* bgr_host, int src_h, int src_w) {
    if (!_impl) throw std::runtime_error("TensorRT g_a encoder has not been loaded");
    return _impl->encode_raw(bgr_host, src_h, src_w);
}

}  // namespace rmcompress

#endif
