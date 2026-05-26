#pragma once

#include <cstdint>
#include <vector>

struct PreprocessRgbScratch {
    int src_w = 0;
    int src_h = 0;
    int dst_w = 0;
    int dst_h = 0;
    std::vector<float> r_plane;
    std::vector<float> g_plane;
    std::vector<float> b_plane;
    std::vector<float> result;
    std::vector<int> x0;
    std::vector<int> x1;
    std::vector<int> y0;
    std::vector<int> y1;
    std::vector<float> fx;
    std::vector<float> fy;
};

/// Preprocess an interleaved BGR uint8 image into the NCHW float32 format
/// required by MbtEncoder.
///
/// Steps: BGR→RGB, bilinear resize to codec size, normalise to [0, 1].
///
/// @param bgr_data  Pointer to interleaved BGR pixel data (b, g, r, b, g, r, …).
/// @param src_w     Source image width in pixels.
/// @param src_h     Source image height in pixels.
/// @return          (3, dst_h, dst_w) float32 tensor in channel-first (NCHW
///                  without the batch dim) layout, values in [0, 1].
std::vector<float> preprocess_rgb(const uint8_t* bgr_data, int src_w, int src_h,
                                  int dst_w = 128, int dst_h = 128);

/// Reusable-buffer variant of preprocess_rgb().
///
/// The returned reference remains valid until `scratch` is reused or destroyed.
const std::vector<float>& preprocess_rgb(const uint8_t* bgr_data,
                                         int src_w,
                                         int src_h,
                                         int dst_w,
                                         int dst_h,
                                         PreprocessRgbScratch& scratch);
