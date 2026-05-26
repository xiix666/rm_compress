#include "rmcompress/preprocess_cuda.h"

#ifdef RMCOMPRESS_ENABLE_TENSORRT

// Each thread handles one output pixel (dx, dy) across all 3 channels.
// align_corners=False: src_x = (dx + 0.5) * (src_w / dst_w) - 0.5
__global__ void bgr_u8_to_rgb_f32_resize_nchw(
    const uint8_t* __restrict__ src,
    int src_h, int src_w,
    float* __restrict__ dst,
    int dst_h, int dst_w)
{
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    const float sx_scale = (float)src_w / (float)dst_w;
    const float sy_scale = (float)src_h / (float)dst_h;

    const float src_x = (dx + 0.5f) * sx_scale - 0.5f;
    const float src_y = (dy + 0.5f) * sy_scale - 0.5f;

    const int x0 = max(0, min((int)floorf(src_x), src_w - 1));
    const int x1 = min(x0 + 1, src_w - 1);
    const int y0 = max(0, min((int)floorf(src_y), src_h - 1));
    const int y1 = min(y0 + 1, src_h - 1);

    const float fx = src_x - (float)x0;
    const float fy = src_y - (float)y0;

    const int plane = dst_h * dst_w;
    // BGR -> RGB: channel order reversal (src index 0=B,1=G,2=R -> dst 0=R,1=G,2=B)
    for (int c = 0; c < 3; ++c) {
        const int src_c = 2 - c;  // R=src[2], G=src[1], B=src[0]
        const float v00 = src[(y0 * src_w + x0) * 3 + src_c];
        const float v01 = src[(y0 * src_w + x1) * 3 + src_c];
        const float v10 = src[(y1 * src_w + x0) * 3 + src_c];
        const float v11 = src[(y1 * src_w + x1) * 3 + src_c];
        const float val = (v00 + (v01 - v00) * fx) * (1.0f - fy)
                        + (v10 + (v11 - v10) * fx) * fy;
        dst[c * plane + dy * dst_w + dx] = val / 255.0f;
    }
}

void launch_bgr_resize_normalize(
    const uint8_t* src_device, int src_h, int src_w,
    float* dst_device, int dst_h, int dst_w,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    bgr_u8_to_rgb_f32_resize_nchw<<<grid, block, 0, stream>>>(
        src_device, src_h, src_w, dst_device, dst_h, dst_w);
}

#endif  // RMCOMPRESS_ENABLE_TENSORRT
