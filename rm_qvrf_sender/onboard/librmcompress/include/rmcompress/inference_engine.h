#pragma once

#include <openvino/openvino.hpp>

#include <string>

/// Thin wrapper around OpenVINO Core with automatic device selection.
///
/// Prefers GPU.0 when available, falls back to GPU, then CPU.
/// Mirrors the Python OpenVINOBackend in client/src/rm_stream/inference_engine.py.
class InferenceEngine {
public:
    /// Construct the engine and select the best available device.
    /// @param preferred  Preferred device (e.g. "GPU.0").  Actual device may
    ///                   differ if the preferred one is not available.
    explicit InferenceEngine(const std::string& preferred = "GPU.0");

    /// Read and compile an OpenVINO IR model.
    /// @param model_path  Path to .xml file.
    /// @param hint        Performance hint: "LATENCY" or "THROUGHPUT".
    /// @param streams     Number of inference streams (1 = single-stream).
    /// @return Compiled model ready for inference.
    ov::CompiledModel compile(const std::string& model_path,
                              const std::string& hint    = "LATENCY",
                              int                streams = 1);

    /// The device that is actually being used.
    const std::string& device() const { return _device; }

    /// Direct access to the underlying Core object (for advanced use).
    ov::Core&       core()       { return _core; }
    const ov::Core& core() const { return _core; }

private:
    ov::Core    _core;
    std::string _device;
};
