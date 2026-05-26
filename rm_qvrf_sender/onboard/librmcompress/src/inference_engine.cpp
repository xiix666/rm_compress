#include "rmcompress/inference_engine.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

InferenceEngine::InferenceEngine(const std::string& preferred) {
    // Discover available devices.
    const std::vector<std::string> devices = _core.get_available_devices();

    // Device preference order: requested -> GPU.0 -> GPU -> CPU.
    auto has = [&](const std::string& d) {
        return std::find(devices.begin(), devices.end(), d) != devices.end();
    };

    if (!preferred.empty() && has(preferred)) {
        _device = preferred;
    } else if (has("GPU.0")) {
        _device = "GPU.0";
    } else if (has("GPU")) {
        _device = "GPU";
    } else if (has("CPU")) {
        _device = "CPU";
    } else if (!devices.empty()) {
        _device = devices.front();
    } else {
        throw std::runtime_error("InferenceEngine: no OpenVINO devices found");
    }

    // Enable model caching for faster startup on subsequent loads.
    try {
        _core.set_property(ov::cache_dir("."));
    } catch (...) {
        // Non-fatal: caching is best-effort.
    }
}

ov::CompiledModel InferenceEngine::compile(const std::string& model_path,
                                           const std::string& hint,
                                           int                streams) {
    auto model = _core.read_model(model_path);

    ov::AnyMap config;
    config["PERFORMANCE_HINT"] =
        (hint == "THROUGHPUT") ? ov::hint::PerformanceMode::THROUGHPUT
                                : ov::hint::PerformanceMode::LATENCY;
    config["INFERENCE_PRECISION_HINT"] = ov::element::f16;
    config["NUM_STREAMS"]              = streams;

    return _core.compile_model(model, _device, config);
}
