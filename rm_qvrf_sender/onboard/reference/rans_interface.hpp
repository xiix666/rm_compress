// Reference: CompressAI rANS interface (from GitHub InterDigitalInc/CompressAI)
// compressai/cpp_exts/rans/rans_interface.hpp
// Copyright (c) 2021-2025, InterDigital Communications, Inc
// BSD-3-Clause-Clear license

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <cstdint>
#include "rans64.h"

namespace py = pybind11;

struct RansSymbol {
    uint32_t start;
    uint32_t range;
    bool bypass;
};

class BufferedRansEncoder {
public:
    void encode_with_indexes(
        const std::vector<int32_t>& symbols,
        const std::vector<int32_t>& indexes,
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& cdfs_sizes,
        const std::vector<int32_t>& offsets);
    py::bytes flush();
private:
    std::vector<RansSymbol> _syms;
};

class RansEncoder {
public:
    py::bytes encode_with_indexes(
        const std::vector<int32_t>& symbols,
        const std::vector<int32_t>& indexes,
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& cdfs_sizes,
        const std::vector<int32_t>& offsets);
};

class RansDecoder {
public:
    std::vector<int32_t> decode_with_indexes(
        const std::string& encoded,
        const std::vector<int32_t>& indexes,
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& cdfs_sizes,
        const std::vector<int32_t>& offsets);
    void set_stream(const std::string& encoded);
    std::vector<int32_t> decode_stream(
        const std::vector<int32_t>& indexes,
        const std::vector<std::vector<int32_t>>& cdfs,
        const std::vector<int32_t>& cdfs_sizes,
        const std::vector<int32_t>& offsets);
private:
    Rans64State _rans;
    std::string _stream;
    uint32_t* _ptr;
};
