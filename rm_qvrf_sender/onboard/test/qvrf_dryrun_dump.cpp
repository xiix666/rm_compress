#include "rmcompress/entropy_coder.h"
#include "rmcompress/ga_encoder.h"
#include "rmcompress/mbt_encoder.h"
#include "rmcompress/preprocess.h"
#include "rmcompress/protocol.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
void write_scalar(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_bytes(std::ofstream& out, const std::vector<uint8_t>& data) {
    uint64_t n = data.size();
    write_scalar(out, n);
    if (n) out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(n));
}

void write_string_vector(std::ofstream& out, const std::vector<std::string>& strings) {
    uint32_t n = static_cast<uint32_t>(strings.size());
    write_scalar(out, n);
    for (const auto& s : strings) {
        uint64_t len = s.size();
        write_scalar(out, len);
        if (len) out.write(s.data(), static_cast<std::streamsize>(len));
    }
}

template <typename T>
void write_vector(std::ofstream& out, const std::vector<T>& values) {
    uint64_t n = values.size();
    write_scalar(out, n);
    if (n) out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

std::vector<uint8_t> read_raw(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open input: " + path);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

std::string arg_value(int& i, int argc, char** argv) {
    if (++i >= argc) throw std::runtime_error("missing value for " + std::string(argv[i - 1]));
    return argv[i];
}

}  // namespace

int main(int argc, char** argv) {
    std::string input_path;
    std::string output_path;
    std::string models_dir = "models";
    std::string device = "CPU";
    std::string ga_backend = "openvino";
    std::string trt_engine;
    int trt_device = 0;
    int width = 192;
    int height = 192;
    int codec_size = 192;
    float gain = 0.8f;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--input") input_path = arg_value(i, argc, argv);
            else if (a == "--output") output_path = arg_value(i, argc, argv);
            else if (a == "--models-dir") models_dir = arg_value(i, argc, argv);
            else if (a == "--device") device = arg_value(i, argc, argv);
            else if (a == "--tx-ga-backend") ga_backend = arg_value(i, argc, argv);
            else if (a == "--tx-trt-engine") trt_engine = arg_value(i, argc, argv);
            else if (a == "--tx-trt-device") trt_device = std::stoi(arg_value(i, argc, argv));
            else if (a == "--width") width = std::stoi(arg_value(i, argc, argv));
            else if (a == "--height") height = std::stoi(arg_value(i, argc, argv));
            else if (a == "--codec-size") codec_size = std::stoi(arg_value(i, argc, argv));
            else if (a == "--gain") gain = std::stof(arg_value(i, argc, argv));
            else throw std::runtime_error("unknown argument: " + a);
        }
        if (input_path.empty() || output_path.empty()) {
            throw std::runtime_error("--input and --output are required");
        }
        if (!models_dir.empty() && models_dir.back() != '/') models_dir.push_back('/');
        if (ga_backend != "openvino" && ga_backend != "tensorrt") {
            throw std::runtime_error("--tx-ga-backend must be openvino or tensorrt");
        }
        if (ga_backend == "tensorrt" && trt_engine.empty()) {
            throw std::runtime_error("--tx-trt-engine is required when --tx-ga-backend tensorrt");
        }

        auto bgr = read_raw(input_path);
        const size_t expected = static_cast<size_t>(width) * height * 3;
        if (bgr.size() != expected) {
            throw std::runtime_error("input size does not match width*height*3");
        }

        rmcompress::GaEncoder g_a;
        MbtEncoder h_a;
        MbtEncoder h_s;
        const std::string entropy_param_device = "CPU";
        const int c_y = 192;
        const int h_y = codec_size / 16;
        const int w_y = codec_size / 16;
        if (ga_backend == "tensorrt") {
            if (!g_a.load_tensorrt(trt_engine, trt_device, 3, codec_size, codec_size, c_y, h_y, w_y)) {
                throw std::runtime_error("load TensorRT g_a failed");
            }
        } else if (!g_a.load_openvino(models_dir + "msssim_g_a_fp32.xml", device, true)) {
            throw std::runtime_error("load OpenVINO g_a failed");
        }
        if (!h_a.load(models_dir + "msssim_h_a_fp32.xml", entropy_param_device, true)) throw std::runtime_error("load h_a failed");
        if (!h_s.load(models_dir + "msssim_h_s_fp32.xml", entropy_param_device, true)) throw std::runtime_error("load h_s failed");

        rmcompress::EntropyCoder entropy;
        if (!entropy.load_cdfs(models_dir + "msssim_cdfs.bin")) throw std::runtime_error("load cdfs failed");

        auto x = preprocess_rgb(bgr.data(), width, height, codec_size, codec_size);
        auto y = g_a.encode(x.data(), 3, codec_size, codec_size);
        const int c_z = 128;
        const int h_z = codec_size / 64;
        const int w_z = codec_size / 64;

        auto z = h_a.encode(y.data(), c_y, h_y, w_y);
        auto z_strings = entropy.compress_bottleneck(z.data(), c_z, h_z, w_z);
        auto z_hat = entropy.decompress_bottleneck(z_strings, c_z, h_z, w_z);
        auto gp = h_s.encode(z_hat.data(), c_z, h_z, w_z);

        const int total_y = c_y * h_y * w_y;
        std::vector<float> scales(total_y), means(total_y), y_scaled(total_y);
        for (int i = 0; i < total_y; ++i) {
            scales[i] = gp[i] * gain;
            means[i] = gp[i + total_y] * gain;
            y_scaled[i] = y[i] * gain;
        }
        auto indexes = entropy.build_indexes(scales.data(), c_y, h_y, w_y);
        auto symbols = entropy.quantize_symbols(y_scaled.data(), means.data(), total_y);
        auto y_strings = entropy.compress_gaussian(y_scaled.data(), indexes.data(), means.data(), c_y, h_y, w_y);
        auto packed = pack_msssim_qvrf(y_strings, z_strings, h_z, w_z, gain);

        std::ofstream out(output_path, std::ios::binary);
        if (!out) throw std::runtime_error("failed to open output: " + output_path);
        out.write("QVD1", 4);
        write_scalar(out, static_cast<uint32_t>(codec_size));
        write_scalar(out, gain);
        write_scalar(out, static_cast<uint32_t>(h_z));
        write_scalar(out, static_cast<uint32_t>(w_z));
        write_vector(out, x);
        write_vector(out, y);
        write_vector(out, z);
        write_string_vector(out, z_strings);
        write_vector(out, z_hat);
        write_vector(out, gp);
        write_vector(out, scales);
        write_vector(out, means);
        write_vector(out, indexes);
        write_vector(out, symbols);
        write_string_vector(out, y_strings);
        write_bytes(out, packed);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "qvrf_dryrun_dump: " << e.what() << "\n";
        return 1;
    }
}
