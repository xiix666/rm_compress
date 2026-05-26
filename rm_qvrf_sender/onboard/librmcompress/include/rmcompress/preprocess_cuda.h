#pragma once

#ifdef RMCOMPRESS_ENABLE_TENSORRT

#include <cuda_runtime_api.h>
#include <cstdint>

/// Launch CUDA kernel: BGR uint8 NHWC → RGB float32 NCHW, bilinear resize,
/// normalize to [0,1]. align_corners=False (matches PyTorch/OpenCV).
///
/// src_device: device pointer to BGR uint8 [src_h * src_w * 3]
/// dst_device: device pointer to float32 NCHW [1 * 3 * dst_h * dst_w]
///             (typically the TRT _input_device buffer)
void launch_bgr_resize_normalize(
    const uint8_t* src_device, int src_h, int src_w,
    float* dst_device, int dst_h, int dst_w,
    cudaStream_t stream);

#endif  // RMCOMPRESS_ENABLE_TENSORRT
