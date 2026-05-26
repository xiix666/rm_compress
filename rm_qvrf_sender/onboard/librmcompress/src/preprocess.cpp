#include "rmcompress/preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

/// Bilinear resize a single channel from (sw, sh) to (dw, dh).
/// Reads from `src` with stride `src_stride` (>= sw).
/// Writes to `dst` contiguously.
void bilinear_channel(const float* src, int sw, int sh, int src_stride,
                      float* dst, int dw, int dh) {
    // "align_corners = False" semantics — matches PyTorch F.interpolate and OpenCV.
    //   src_x = (dx + 0.5) * sw / dw - 0.5
    //   src_y = (dy + 0.5) * sh / dh - 0.5
    const float sx_scale = (dw > 0) ? (static_cast<float>(sw) / dw) : 0.0f;
    const float sy_scale = (dh > 0) ? (static_cast<float>(sh) / dh) : 0.0f;

    for (int dy = 0; dy < dh; ++dy) {
        const float sy = (dy + 0.5f) * sy_scale - 0.5f;
        const int   y0 = std::max(0, std::min(static_cast<int>(std::floor(sy)), sh - 1));
        const int   y1 = std::min(y0 + 1, sh - 1);
        const float fy = sy - static_cast<float>(y0);

        float* dst_row = dst + dy * dw;
        const float* src_row0 = src + y0 * src_stride;
        const float* src_row1 = src + y1 * src_stride;

        for (int dx = 0; dx < dw; ++dx) {
            const float sx = (dx + 0.5f) * sx_scale - 0.5f;
            const int   x0 = std::max(0, std::min(static_cast<int>(std::floor(sx)), sw - 1));
            const int   x1 = std::min(x0 + 1, sw - 1);
            const float fx = sx - static_cast<float>(x0);

            // Bilinear interpolation
            const float v00 = src_row0[x0];
            const float v01 = src_row0[x1];
            const float v10 = src_row1[x0];
            const float v11 = src_row1[x1];

            const float top    = v00 + (v01 - v00) * fx;
            const float bottom = v10 + (v11 - v10) * fx;
            dst_row[dx]        = top + (bottom - top) * fy;
        }
    }
}

void build_resize_maps(int src_w, int src_h, int dst_w, int dst_h,
                       PreprocessRgbScratch& scratch) {
    scratch.src_w = src_w;
    scratch.src_h = src_h;
    scratch.dst_w = dst_w;
    scratch.dst_h = dst_h;
    scratch.x0.resize(dst_w);
    scratch.x1.resize(dst_w);
    scratch.fx.resize(dst_w);
    scratch.y0.resize(dst_h);
    scratch.y1.resize(dst_h);
    scratch.fy.resize(dst_h);

    const float sx_scale = (dst_w > 0) ? (static_cast<float>(src_w) / dst_w) : 0.0f;
    const float sy_scale = (dst_h > 0) ? (static_cast<float>(src_h) / dst_h) : 0.0f;

    for (int dx = 0; dx < dst_w; ++dx) {
        const float sx = (dx + 0.5f) * sx_scale - 0.5f;
        const int x0 = std::max(0, std::min(static_cast<int>(std::floor(sx)), src_w - 1));
        scratch.x0[dx] = x0;
        scratch.x1[dx] = std::min(x0 + 1, src_w - 1);
        scratch.fx[dx] = sx - static_cast<float>(x0);
    }
    for (int dy = 0; dy < dst_h; ++dy) {
        const float sy = (dy + 0.5f) * sy_scale - 0.5f;
        const int y0 = std::max(0, std::min(static_cast<int>(std::floor(sy)), src_h - 1));
        scratch.y0[dy] = y0;
        scratch.y1[dy] = std::min(y0 + 1, src_h - 1);
        scratch.fy[dy] = sy - static_cast<float>(y0);
    }
}

void bilinear_channel_mapped(const float* src,
                             int src_stride,
                             float* dst,
                             int dst_w,
                             int dst_h,
                             const PreprocessRgbScratch& scratch) {
    for (int dy = 0; dy < dst_h; ++dy) {
        const float fy = scratch.fy[dy];
        float* dst_row = dst + dy * dst_w;
        const float* src_row0 = src + scratch.y0[dy] * src_stride;
        const float* src_row1 = src + scratch.y1[dy] * src_stride;

        for (int dx = 0; dx < dst_w; ++dx) {
            const int x0 = scratch.x0[dx];
            const int x1 = scratch.x1[dx];
            const float fx = scratch.fx[dx];

            const float v00 = src_row0[x0];
            const float v01 = src_row0[x1];
            const float v10 = src_row1[x0];
            const float v11 = src_row1[x1];

            const float top    = v00 + (v01 - v00) * fx;
            const float bottom = v10 + (v11 - v10) * fx;
            dst_row[dx]        = top + (bottom - top) * fy;
        }
    }
}

}  // anonymous namespace

std::vector<float> preprocess_rgb(const uint8_t* bgr_data, int src_w, int src_h,
                                  int dst_w, int dst_h) {
    // 1. Extract R, G, B channels from interleaved BGR into float planes.
    const int num_pixels = src_w * src_h;
    std::vector<float> r_plane(num_pixels);
    std::vector<float> g_plane(num_pixels);
    std::vector<float> b_plane(num_pixels);

    for (int i = 0; i < num_pixels; ++i) {
        b_plane[i] = bgr_data[i * 3 + 0] / 255.0f;  // B
        g_plane[i] = bgr_data[i * 3 + 1] / 255.0f;  // G
        r_plane[i] = bgr_data[i * 3 + 2] / 255.0f;  // R
    }

    // 2. Bilinear resize each channel to codec size.
    std::vector<float> r_resized(dst_w * dst_h);
    std::vector<float> g_resized(dst_w * dst_h);
    std::vector<float> b_resized(dst_w * dst_h);

    bilinear_channel(r_plane.data(), src_w, src_h, src_w, r_resized.data(), dst_w, dst_h);
    bilinear_channel(g_plane.data(), src_w, src_h, src_w, g_resized.data(), dst_w, dst_h);
    bilinear_channel(b_plane.data(), src_w, src_h, src_w, b_resized.data(), dst_w, dst_h);

    // 3. Pack into NCHW layout: (3, dst_h, dst_w).
    std::vector<float> result(3 * dst_w * dst_h);
    float* ch_r = result.data();
    float* ch_g = ch_r + dst_w * dst_h;
    float* ch_b = ch_g + dst_w * dst_h;

    std::memcpy(ch_r, r_resized.data(), dst_w * dst_h * sizeof(float));
    std::memcpy(ch_g, g_resized.data(), dst_w * dst_h * sizeof(float));
    std::memcpy(ch_b, b_resized.data(), dst_w * dst_h * sizeof(float));

    return result;
}

const std::vector<float>& preprocess_rgb(const uint8_t* bgr_data,
                                         int src_w,
                                         int src_h,
                                         int dst_w,
                                         int dst_h,
                                         PreprocessRgbScratch& scratch) {
    const int num_pixels = src_w * src_h;
    if (scratch.src_w != src_w || scratch.src_h != src_h ||
        scratch.dst_w != dst_w || scratch.dst_h != dst_h ||
        static_cast<int>(scratch.x0.size()) != dst_w ||
        static_cast<int>(scratch.y0.size()) != dst_h) {
        build_resize_maps(src_w, src_h, dst_w, dst_h, scratch);
    }

    scratch.r_plane.resize(num_pixels);
    scratch.g_plane.resize(num_pixels);
    scratch.b_plane.resize(num_pixels);

    for (int i = 0; i < num_pixels; ++i) {
        scratch.b_plane[i] = bgr_data[i * 3 + 0] / 255.0f;
        scratch.g_plane[i] = bgr_data[i * 3 + 1] / 255.0f;
        scratch.r_plane[i] = bgr_data[i * 3 + 2] / 255.0f;
    }

    const int dst_pixels = dst_w * dst_h;
    scratch.result.resize(3 * dst_pixels);
    float* ch_r = scratch.result.data();
    float* ch_g = ch_r + dst_pixels;
    float* ch_b = ch_g + dst_pixels;

    bilinear_channel_mapped(scratch.r_plane.data(), src_w, ch_r, dst_w, dst_h, scratch);
    bilinear_channel_mapped(scratch.g_plane.data(), src_w, ch_g, dst_w, dst_h, scratch);
    bilinear_channel_mapped(scratch.b_plane.data(), src_w, ch_b, dst_w, dst_h, scratch);

    return scratch.result;
}
