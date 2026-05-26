#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "MvCameraControl.h"
#include "rmcompress/shm_ring.h"

namespace {

volatile std::sig_atomic_t g_stop = 0;

void on_signal(int) {
    g_stop = 1;
}

struct CameraOptions {
    const char* shm_name = "/rm_camera_frames";
    int slots = 4;
    unsigned int device_index = 0;
    int max_frames = 0;
    int roi_size = 1080;
    bool auto_square_roi = false;
    double fps = 24.0;
    double exposure_us = 20000.0;
    bool allow_adjust_roi = false;
};

struct SessionResult {
    bool reconnect = false;
    int frames = 0;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  --shm-name NAME       Shared memory ring name (default: /rm_camera_frames)\n"
        "  --slots N             Ring slots (default: 4)\n"
        "  --device-index N      Camera index from MVS enumeration (default: 0)\n"
        "  --frames N            Stop after N frames, 0 means run forever (default: 0)\n"
        "  --roi-size N          Fixed center ROI size (default: 1080)\n"
        "  --roi-mode MODE       fixed|max-square; max-square uses min(max width, max height)\n"
        "  --auto-square-roi     Alias for --roi-mode max-square\n"
        "  --fps FPS             Acquisition frame rate (default: 24)\n"
        "  --exposure-us US      Exposure time in microseconds (default: 20000)\n"
        "  --allow-adjust-roi    Align unsupported ROI size down to camera increment\n"
        "  --help                Show this help\n",
        prog);
}

bool check(int ret, const char* op) {
    if (ret == MV_OK) return true;
    std::fprintf(stderr, "%s failed: 0x%x\n", op, ret);
    return false;
}

int64_t align_down(int64_t value, int64_t min_value, int64_t inc) {
    if (inc <= 0) inc = 1;
    if (value < min_value) return min_value;
    return min_value + ((value - min_value) / inc) * inc;
}

int64_t align_square_down(int64_t value,
                          const MVCC_INTVALUE_EX& width_rng,
                          const MVCC_INTVALUE_EX& height_rng) {
    int64_t side = std::min<int64_t>(value, std::min(width_rng.nMax, height_rng.nMax));
    side = std::min(align_down(side, width_rng.nMin, width_rng.nInc),
                    align_down(side, height_rng.nMin, height_rng.nInc));
    while (side >= width_rng.nMin && side >= height_rng.nMin) {
        const int64_t w = align_down(side, width_rng.nMin, width_rng.nInc);
        const int64_t h = align_down(side, height_rng.nMin, height_rng.nInc);
        if (w == side && h == side) return side;
        side = std::min(w, h) - 1;
    }
    return 0;
}

bool get_int(void* handle, const char* key, MVCC_INTVALUE_EX* out) {
    std::memset(out, 0, sizeof(*out));
    return check(MV_CC_GetIntValueEx(handle, key, out), key);
}

bool set_int(void* handle, const char* key, int64_t value) {
    return check(MV_CC_SetIntValueEx(handle, key, value), key);
}

void warn_if_fail(int ret, const char* op) {
    if (ret != MV_OK) {
        std::fprintf(stderr, "Warning: %s failed: 0x%x\n", op, ret);
    }
}

bool configure_center_roi(void* handle, int requested_size, bool auto_square, bool allow_adjust,
                          uint32_t* out_w, uint32_t* out_h) {
    if (!set_int(handle, "OffsetX", 0) || !set_int(handle, "OffsetY", 0))
        return false;

    MVCC_INTVALUE_EX width_rng{}, height_rng{}, ox_rng{}, oy_rng{};
    if (!get_int(handle, "Width", &width_rng) ||
        !get_int(handle, "Height", &height_rng) ||
        !get_int(handle, "OffsetX", &ox_rng) ||
        !get_int(handle, "OffsetY", &oy_rng)) {
        return false;
    }
    const int64_t full_w = width_rng.nMax;
    const int64_t full_h = height_rng.nMax;

    int64_t roi_w = requested_size;
    int64_t roi_h = requested_size;
    if (auto_square) {
        const int64_t requested_auto = std::min(full_w, full_h);
        const int64_t side = align_square_down(requested_auto, width_rng, height_rng);
        if (side <= 0) {
            std::fprintf(stderr,
                "Cannot derive max-square ROI from camera ranges "
                "(Width min=%ld max=%ld inc=%ld, Height min=%ld max=%ld inc=%ld)\n",
                static_cast<long>(width_rng.nMin), static_cast<long>(width_rng.nMax),
                static_cast<long>(width_rng.nInc),
                static_cast<long>(height_rng.nMin), static_cast<long>(height_rng.nMax),
                static_cast<long>(height_rng.nInc));
            return false;
        }
        roi_w = side;
        roi_h = side;
        if (side != requested_auto) {
            std::printf("Auto max-square ROI: shortest sensor side %ld aligned down to %ld\n",
                        static_cast<long>(requested_auto), static_cast<long>(side));
        } else {
            std::printf("Auto max-square ROI: using shortest sensor side %ld\n",
                        static_cast<long>(side));
        }
    }

    const bool exact_w = roi_w >= width_rng.nMin && roi_w <= width_rng.nMax &&
                         align_down(roi_w, width_rng.nMin, width_rng.nInc) == roi_w;
    const bool exact_h = roi_h >= height_rng.nMin && roi_h <= height_rng.nMax &&
                         align_down(roi_h, height_rng.nMin, height_rng.nInc) == roi_h;
    if (!exact_w || !exact_h) {
        if (!allow_adjust) {
            std::fprintf(stderr,
                "Requested ROI %d is not supported by camera increments "
                "(Width min=%ld max=%ld inc=%ld, Height min=%ld max=%ld inc=%ld). "
                "Use --allow-adjust-roi to align down.\n",
                requested_size,
                static_cast<long>(width_rng.nMin), static_cast<long>(width_rng.nMax),
                static_cast<long>(width_rng.nInc),
                static_cast<long>(height_rng.nMin), static_cast<long>(height_rng.nMax),
                static_cast<long>(height_rng.nInc));
            return false;
        }
        const int64_t side = align_square_down(requested_size, width_rng, height_rng);
        if (side <= 0) {
            std::fprintf(stderr, "Cannot align requested ROI %d to a supported square ROI\n",
                         requested_size);
            return false;
        }
        roi_w = side;
        roi_h = side;
        std::fprintf(stderr, "Adjusted ROI from %d to %ldx%ld for camera increments\n",
                     requested_size, static_cast<long>(roi_w), static_cast<long>(roi_h));
    }

    if (!set_int(handle, "Width", roi_w) || !set_int(handle, "Height", roi_h))
        return false;

    if (!get_int(handle, "OffsetX", &ox_rng) || !get_int(handle, "OffsetY", &oy_rng))
        return false;
    const int64_t ox = align_down((full_w - roi_w) / 2, ox_rng.nMin, ox_rng.nInc);
    const int64_t oy = align_down((full_h - roi_h) / 2, oy_rng.nMin, oy_rng.nInc);
    if (!set_int(handle, "OffsetX", ox) || !set_int(handle, "OffsetY", oy))
        return false;

    MVCC_INTVALUE_EX actual_w{}, actual_h{}, actual_ox{}, actual_oy{};
    if (!get_int(handle, "Width", &actual_w) ||
        !get_int(handle, "Height", &actual_h) ||
        !get_int(handle, "OffsetX", &actual_ox) ||
        !get_int(handle, "OffsetY", &actual_oy)) {
        return false;
    }
    std::printf("ROI: %ldx%ld at offset (%ld,%ld), full max %ldx%ld\n",
                static_cast<long>(actual_w.nCurValue), static_cast<long>(actual_h.nCurValue),
                static_cast<long>(actual_ox.nCurValue), static_cast<long>(actual_oy.nCurValue),
                static_cast<long>(full_w), static_cast<long>(full_h));
    *out_w = static_cast<uint32_t>(actual_w.nCurValue);
    *out_h = static_cast<uint32_t>(actual_h.nCurValue);
    return true;
}

bool print_device(MV_CC_DEVICE_INFO* info, unsigned int index) {
    if (!info) return false;
    std::printf("[device %u] ", index);
    if (info->nTLayerType == MV_GIGE_DEVICE) {
        std::printf("%s %s\n", info->SpecialInfo.stGigEInfo.chModelName,
                    info->SpecialInfo.stGigEInfo.chSerialNumber);
    } else if (info->nTLayerType == MV_USB_DEVICE) {
        std::printf("%s %s\n", info->SpecialInfo.stUsb3VInfo.chModelName,
                    info->SpecialInfo.stUsb3VInfo.chSerialNumber);
    } else {
        std::printf("transport=0x%x\n", info->nTLayerType);
    }
    return true;
}

SessionResult run_camera_session(const CameraOptions& opt, int frames_remaining) {
    SessionResult result{};
    void* handle = nullptr;
    rmcompress::ShmRing ring;
    int ret = MV_OK;

    auto cleanup = [&]() {
        if (handle) {
            MV_CC_StopGrabbing(handle);
            MV_CC_CloseDevice(handle);
            MV_CC_DestroyHandle(handle);
            handle = nullptr;
        }
        ring.close();
    };

    MV_CC_DEVICE_INFO_LIST devices{};
    ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE |
                            MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE |
                            MV_GENTL_XOF_DEVICE, &devices);
    if (!check(ret, "MV_CC_EnumDevices")) {
        result.reconnect = true;
        return result;
    }
    if (devices.nDeviceNum == 0 || opt.device_index >= devices.nDeviceNum) {
        std::fprintf(stderr, "No camera at index %u (found %u devices)\n",
                     opt.device_index, devices.nDeviceNum);
        result.reconnect = true;
        return result;
    }
    for (unsigned int i = 0; i < devices.nDeviceNum; ++i) {
        print_device(devices.pDeviceInfo[i], i);
    }

    ret = MV_CC_CreateHandle(&handle, devices.pDeviceInfo[opt.device_index]);
    if (!check(ret, "MV_CC_CreateHandle")) {
        result.reconnect = true;
        cleanup();
        return result;
    }
    ret = MV_CC_OpenDevice(handle, MV_ACCESS_Exclusive, 0);
    if (!check(ret, "MV_CC_OpenDevice")) {
        result.reconnect = true;
        cleanup();
        return result;
    }

    if (devices.pDeviceInfo[opt.device_index]->nTLayerType == MV_GIGE_DEVICE) {
        int pkt = MV_CC_GetOptimalPacketSize(handle);
        if (pkt > 0) warn_if_fail(MV_CC_SetIntValueEx(handle, "GevSCPSPacketSize", pkt),
                                  "GevSCPSPacketSize");
    }

    uint32_t width = 0, height = 0;
    if (!configure_center_roi(handle, opt.roi_size, opt.auto_square_roi,
                              opt.allow_adjust_roi, &width, &height)) {
        result.reconnect = true;
        cleanup();
        return result;
    }

    warn_if_fail(MV_CC_SetEnumValue(handle, "TriggerMode", MV_TRIGGER_MODE_OFF),
                 "TriggerMode Off");
    warn_if_fail(MV_CC_SetEnumValue(handle, "GainAuto", MV_GAIN_MODE_CONTINUOUS),
                 "GainAuto Continuous");
    warn_if_fail(MV_CC_SetEnumValue(handle, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF),
                 "ExposureAuto Off");
    warn_if_fail(MV_CC_SetFloatValue(handle, "ExposureTime", static_cast<float>(opt.exposure_us)),
                 "ExposureTime");
    warn_if_fail(MV_CC_SetBoolValue(handle, "AcquisitionFrameRateEnable", true),
                 "AcquisitionFrameRateEnable");
    warn_if_fail(MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", static_cast<float>(opt.fps)),
                 "AcquisitionFrameRate");
    warn_if_fail(MV_CC_SetBayerCvtQuality(handle, 1), "BayerCvtQuality");

    const uint32_t stride = width * 3;
    std::string error;
    if (!ring.create(opt.shm_name, static_cast<uint32_t>(opt.slots), width, height,
                     stride, rmcompress::SHM_PIXFMT_BGR8, &error)) {
        std::fprintf(stderr, "Failed to create shm ring %s: %s\n",
                     opt.shm_name, error.c_str());
        result.reconnect = true;
        cleanup();
        return result;
    }
    std::vector<uint8_t> bgr(static_cast<size_t>(stride) * height);

    ret = MV_CC_StartGrabbing(handle);
    if (!check(ret, "MV_CC_StartGrabbing")) {
        result.reconnect = true;
        cleanup();
        return result;
    }

    std::printf("Camera capture started: %ux%u BGR8 %.2f fps exposure %.1fus -> shm %s\n",
                width, height, opt.fps, opt.exposure_us, opt.shm_name);

    uint64_t first_ns = rmcompress::monotonic_time_ns();
    uint64_t last_report_ns = first_ns;
    int consecutive_errors = 0;
    while (!g_stop && (frames_remaining <= 0 || result.frames < frames_remaining)) {
        MV_FRAME_OUT frame{};
        ret = MV_CC_GetImageBuffer(handle, &frame, 1000);
        if (ret != MV_OK) {
            std::fprintf(stderr, "MV_CC_GetImageBuffer timeout/error: 0x%x\n", ret);
            consecutive_errors++;
            if (consecutive_errors >= 3) {
                std::fprintf(stderr, "Camera grab failed %d times; reconnecting camera\n",
                             consecutive_errors);
                result.reconnect = true;
                break;
            }
            continue;
        }
        consecutive_errors = 0;

        MV_CC_PIXEL_CONVERT_PARAM_EX cvt{};
        cvt.nWidth = frame.stFrameInfo.nExtendWidth;
        cvt.nHeight = frame.stFrameInfo.nExtendHeight;
        cvt.enSrcPixelType = frame.stFrameInfo.enPixelType;
        cvt.pSrcData = frame.pBufAddr;
        cvt.nSrcDataLen = static_cast<unsigned int>(frame.stFrameInfo.nFrameLenEx);
        cvt.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
        cvt.pDstBuffer = bgr.data();
        cvt.nDstBufferSize = static_cast<unsigned int>(bgr.size());
        ret = MV_CC_ConvertPixelTypeEx(handle, &cvt);
        MV_CC_FreeImageBuffer(handle, &frame);
        if (ret != MV_OK || cvt.nDstLen != bgr.size()) {
            std::fprintf(stderr, "MV_CC_ConvertPixelTypeEx failed/short: 0x%x len=%u\n",
                         ret, cvt.nDstLen);
            consecutive_errors++;
            if (consecutive_errors >= 3) {
                std::fprintf(stderr, "Pixel conversion failed %d times; reconnecting camera\n",
                             consecutive_errors);
                result.reconnect = true;
                break;
            }
            continue;
        }

        ring.write_latest(bgr.data(), static_cast<uint32_t>(bgr.size()),
                          rmcompress::monotonic_time_ns());
        result.frames++;

        uint64_t now = rmcompress::monotonic_time_ns();
        if (now - last_report_ns >= 1000000000ULL) {
            double elapsed = static_cast<double>(now - first_ns) / 1.0e9;
            std::printf("  captured=%d actual_fps=%.2f camera_frame=%u\n",
                        result.frames, result.frames / std::max(0.001, elapsed),
                        frame.stFrameInfo.nFrameNum);
            last_report_ns = now;
        }
    }

    cleanup();
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    CameraOptions opt;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shm-name") == 0 && i + 1 < argc) {
            opt.shm_name = argv[++i];
        } else if (std::strcmp(argv[i], "--slots") == 0 && i + 1 < argc) {
            opt.slots = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--device-index") == 0 && i + 1 < argc) {
            opt.device_index = static_cast<unsigned int>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            opt.max_frames = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--roi-size") == 0 && i + 1 < argc) {
            opt.roi_size = std::atoi(argv[++i]);
            opt.auto_square_roi = false;
        } else if (std::strcmp(argv[i], "--roi-mode") == 0 && i + 1 < argc) {
            const char* mode = argv[++i];
            if (std::strcmp(mode, "fixed") == 0) {
                opt.auto_square_roi = false;
            } else if (std::strcmp(mode, "max-square") == 0) {
                opt.auto_square_roi = true;
                opt.allow_adjust_roi = true;
            } else {
                std::fprintf(stderr, "Unknown roi mode: %s\n", mode);
                usage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--auto-square-roi") == 0) {
            opt.auto_square_roi = true;
            opt.allow_adjust_roi = true;
        } else if (std::strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            opt.fps = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--exposure-us") == 0 && i + 1 < argc) {
            opt.exposure_us = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--allow-adjust-roi") == 0) {
            opt.allow_adjust_roi = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    int ret = MV_CC_Initialize();
    if (!check(ret, "MV_CC_Initialize")) return 1;

    int reconnect_attempt = 0;
    int total_frames = 0;

    while (!g_stop && (opt.max_frames <= 0 || total_frames < opt.max_frames)) {
        const int remaining = opt.max_frames > 0 ? (opt.max_frames - total_frames) : 0;
        SessionResult sr = run_camera_session(opt, remaining);
        total_frames += sr.frames;
        if (sr.frames > 0) reconnect_attempt = 0;
        if (sr.reconnect && !g_stop && (opt.max_frames <= 0 || total_frames < opt.max_frames)) {
            reconnect_attempt++;
            int delay_s = std::min(5, reconnect_attempt);
            std::fprintf(stderr, "Reconnecting camera in %ds (attempt %d)\n",
                         delay_s, reconnect_attempt);
            for (int i = 0; i < delay_s && !g_stop; ++i) {
                sleep(1);
            }
            continue;
        }
        break;
    }

    MV_CC_Finalize();
    return 0;
}
