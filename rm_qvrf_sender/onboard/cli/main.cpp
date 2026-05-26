#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <map>
#include <limits>
#include <csignal>
#include <numeric>
#include <cerrno>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include "rmcompress/rm_compress.h"
#include "rmcompress/protocol.h"
#include "rmcompress/shm_ring.h"
#include "serial/serial_tx.h"

static std::atomic<bool> g_stop{false};

static void handle_signal(int) {
    g_stop.store(true);
}

static bool open_shm_with_retry(rmcompress::ShmRing& shm_ring, const char* shm_name) {
    std::string error;
    auto last_log = std::chrono::steady_clock::now() - std::chrono::seconds(2);
    while (!g_stop.load()) {
        if (shm_ring.open(shm_name, &error)) {
            return true;
        }
        auto now = std::chrono::steady_clock::now();
        if (now - last_log >= std::chrono::seconds(1)) {
            fprintf(stderr, "Waiting for shared memory ring %s: %s\n",
                    shm_name, error.c_str());
            last_log = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        error.clear();
    }
    return false;
}

// ---------------------------------------------------------------------------
// Test pattern generator: 128x128 RGB, moving color bar with frame counter
// ---------------------------------------------------------------------------
static void generate_test_frame(uint8_t* rgb, int width, int height, int frame_idx) {
    // Moving vertical bar
    int bar_pos = (frame_idx * 2) % (width + 64) - 32;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            if (x >= bar_pos && x < bar_pos + 16) {
                // White bar with frame counter in red channel
                rgb[idx + 0] = static_cast<uint8_t>(std::min(255, 200 + (frame_idx % 56)));
                rgb[idx + 1] = 255;
                rgb[idx + 2] = 255;
            } else {
                // Color gradient background: encodes position
                rgb[idx + 0] = static_cast<uint8_t>(x * 2);
                rgb[idx + 1] = static_cast<uint8_t>(y * 2);
                rgb[idx + 2] = static_cast<uint8_t>(((x + y + frame_idx) % 256));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Read raw BGR frames from a file (width*height*3 bytes per frame)
// ---------------------------------------------------------------------------
static bool read_raw_frame(FILE* f, uint8_t* rgb, int size) {
    size_t n = fread(rgb, 1, size, f);
    return n == static_cast<size_t>(size);
}

struct QueuedChunk {
    uint32_t frame_id;
    uint8_t chunk_id;
    uint8_t chunk_count;
    std::vector<uint8_t> data;
    std::chrono::steady_clock::time_point enqueue_time;
    double capture_age_ms;
    double encode_ms;
};

static long long wall_ms_now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static int connect_tcp_with_retry(const char* host, int port) {
    char port_buf[16];
    std::snprintf(port_buf, sizeof(port_buf), "%d", port);

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    while (!g_stop.load()) {
        addrinfo* res = nullptr;
        int gai = getaddrinfo(host, port_buf, &hints, &res);
        if (gai != 0) {
            fprintf(stderr, "IPC resolve %s:%d failed: %s\n", host, port, gai_strerror(gai));
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
            int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (fd < 0) continue;
            if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
                freeaddrinfo(res);
                return fd;
            }
            close(fd);
        }
        freeaddrinfo(res);
        fprintf(stderr, "IPC %s:%d unavailable; retrying in 1s...\n", host, port);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return -1;
}

static bool write_all(int fd, const uint8_t* data, size_t len) {
    size_t off = 0;
    while (off < len && !g_stop.load()) {
        ssize_t n = send(fd, data + off, len - off, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        off += static_cast<size_t>(n);
    }
    return off == len;
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  -p, --port PORT       Serial port, or auto to scan by-id/ttyUSB/ttyACM (default: auto)\n"
        "  -b, --baudrate RATE   Baud rate (default: 921600)\n"
        "  -n, --frames N        Number of frames to send, 0 means forever (default: 0)\n"
        "  -w, --width W         Frame width (default: 128)\n"
        "  -H, --height H        Frame height (default: 128)\n"
        "  --codec-size S        Codec preprocess size, 128/192/256 (default: 128)\n"
        "  --chunks-per-frame N  Fixed physical chunks per frame (default: 2)\n"
        "  --fec-data-chunks N   Reserve remaining fixed chunks for FEC (0 disables, default: 0)\n"
        "  --codec NAME          Codec: mbt (default) or msssim_qvrf with --qvrf-cpp-sender.\n"
        "  --qvrf-cpp-sender     Use the supported C++ QVRF sender for --codec msssim_qvrf\n"
        "  --qvrf-cpp-experiment Compatibility alias for --qvrf-cpp-sender\n"
        "  --tx-ga-backend NAME  Sender g_a backend: openvino or tensorrt (default: openvino)\n"
        "  --tx-trt-engine PATH  TensorRT engine for g_a when --tx-ga-backend tensorrt\n"
        "  --tx-trt-device N     CUDA device id for TensorRT g_a (default: 0)\n"
        "  --mode NAME           legacy128x2x24, codec192x4x12, codec256x4x12, codec256x6x8, codec320x8x6, or codec448x9x5\n"
        "  -m, --model PATH      Path to mbt_g_a.xml dir (default: auto)\n"
        "  -d, --device DEV      OpenVINO device: GPU.0, GPU, CPU (default: GPU.0)\n"
        "  -r, --redundancy N    Send each chunk N times for lossy links (default: 1)\n"
        "  --fps FPS             Target FPS (default: 24, max: 48)\n"
        "  --dry-run             Don't open serial, just compress + print stats\n"
        "  --ipc-host HOST       Send raw 300B R1V1 chunks to TCP host instead of serial\n"
        "  --ipc-port PORT       TCP port for --ipc-host (default: 49031)\n"
        "  --serial-wait         Wait/reconnect forever if serial is absent or unplugged\n"
        "  --input FILE          Read raw BGR frames from file instead of test pattern\n"
        "  --shm-input           Read latest BGR frames from shared memory ring\n"
        "  --shm-name NAME       Shared memory ring name (default: /rm_camera_frames)\n"
        "  --save-bitstreams FILE  Save compressed bitstreams for bench_consumer.py\n"
        "  --profile             Print per-stage compression timing summary (keeps last 600 frames)\n"
        "  --prebuffer-chunks N   Buffer N chunks before serial TX starts (default: 4)\n"
        "  --tail-flush-chunks N  Send N non-video 0x0310 flush packets at end (default: 4)\n"
        "  --chunk-rate-hz HZ    Physical 0x0310 send rate (default: 48)\n"
        "  --max-queue-chunks N  Drop oldest queued chunks above this limit, 0 disables (default: 16)\n"
        "  --chunk-order ORDER   4-chunk send order, e.g. 0123 or 0312 (default: 0312)\n"
        "  --single-packet-mode   Explicit diagnostic only: one physical video chunk per frame.\n"
        "                         Lower quality; do not use as the normal fix path.\n"
        "  --help                Show this help\n"
        "\n"
        "Sends compressed video frames over serial using 0x0310 protocol.\n"
        "The 0x0310 link is LOSSY — use --redundancy 2 (or higher) for critical data.\n"
        "Max recommended send rate: 48 Hz (50 Hz causes burst loss per commu testing).\n",
        prog);
}

static bool require_arg(int i, int argc, const char* opt) {
    if (i + 1 < argc) return true;
    fprintf(stderr, "%s requires a value\n", opt);
    return false;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    const char* port = "auto";
    int baudrate = 921600;
    int num_frames = 0;
    int width = 128;
    int height = 128;
    int codec_size = 128;
    int fixed_chunks = FIXED_CHUNKS_PER_FRAME;
    int fec_data_chunks = 0;
    const char* model_path = nullptr;
    const char* device = "GPU.0";
    int redundancy = 1;
    double target_fps = 24.0;
    bool dry_run = false;
    const char* ipc_host = nullptr;
    int ipc_port = 49031;
    bool serial_wait = false;
    const char* input_file = nullptr;
    bool shm_input = false;
    const char* shm_name = "/rm_camera_frames";
    const char* save_bitstreams_file = nullptr;
    bool profile = false;
    const char* codec = "mbt";
    int prebuffer_chunks = 4;
    int tail_flush_chunks = 4;
    double chunk_rate_hz = 48.0;
    int max_queue_chunks = 16;
    std::string chunk_order_arg = "0312";
    bool single_packet_mode = false;
    bool qvrf_cpp_sender = false;
    const char* tx_ga_backend = "openvino";
    const char* tx_trt_engine = nullptr;
    int tx_trt_device = 0;
    const char* qvrf_env = std::getenv("RM_QVRF_CPP_EXPERIMENT");
    if (qvrf_env && std::strcmp(qvrf_env, "1") == 0) {
        qvrf_cpp_sender = true;
    }
    const char* qvrf_sender_env = std::getenv("RM_QVRF_CPP_SENDER");
    if (qvrf_sender_env && std::strcmp(qvrf_sender_env, "1") == 0) {
        qvrf_cpp_sender = true;
    }
    const bool e2e_trace = []() {
        const char* env = std::getenv("RM_STREAM_E2E_TRACE");
        return env && std::strcmp(env, "1") == 0;
    }();

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) {
            if (++i < argc) port = argv[i];
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--baudrate") == 0) {
            if (++i < argc) baudrate = atoi(argv[i]);
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--frames") == 0) {
            if (++i < argc) num_frames = atoi(argv[i]);
        } else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--width") == 0) {
            if (++i < argc) width = atoi(argv[i]);
        } else if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--height") == 0) {
            if (++i < argc) height = atoi(argv[i]);
        } else if (strcmp(argv[i], "--codec-size") == 0) {
            if (++i < argc) codec_size = atoi(argv[i]);
        } else if (strcmp(argv[i], "--chunks-per-frame") == 0) {
            if (++i < argc) fixed_chunks = atoi(argv[i]);
        } else if (strcmp(argv[i], "--fec-data-chunks") == 0) {
            if (++i < argc) fec_data_chunks = atoi(argv[i]);
        } else if (strcmp(argv[i], "--mode") == 0) {
            if (++i < argc) {
                if (strcmp(argv[i], "legacy128x2x24") == 0) {
                    codec_size = 128;
                    fixed_chunks = 2;
                    target_fps = 24.0;
                } else if (strcmp(argv[i], "codec192x4x12") == 0) {
                    codec_size = 192;
                    fixed_chunks = 4;
                    target_fps = 12.0;
                } else if (strcmp(argv[i], "codec256x4x12") == 0) {
                    codec_size = 256;
                    fixed_chunks = 4;
                    target_fps = 12.0;
                } else if (strcmp(argv[i], "codec256x6x8") == 0) {
                    codec_size = 256;
                    fixed_chunks = 6;
                    target_fps = 8.0;
                } else if (strcmp(argv[i], "codec320x8x6") == 0) {
                    codec_size = 320;
                    fixed_chunks = 8;
                    target_fps = 6.0;
                } else if (strcmp(argv[i], "codec448x9x5") == 0) {
                    codec_size = 448;
                    fixed_chunks = 9;
                    target_fps = 5.0;
                } else {
                    fprintf(stderr, "Unknown mode: %s\n", argv[i]);
                    usage(argv[0]);
                    return 1;
                }
            }
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i < argc) model_path = argv[i];
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) {
            if (++i < argc) device = argv[i];
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--redundancy") == 0) {
            if (++i < argc) redundancy = atoi(argv[i]);
        } else if (strcmp(argv[i], "--fps") == 0) {
            if (++i < argc) target_fps = atof(argv[i]);
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            dry_run = true;
        } else if (strcmp(argv[i], "--ipc-host") == 0) {
            if (++i < argc) ipc_host = argv[i];
        } else if (strcmp(argv[i], "--ipc-port") == 0) {
            if (++i < argc) ipc_port = atoi(argv[i]);
        } else if (strcmp(argv[i], "--serial-wait") == 0) {
            serial_wait = true;
        } else if (strcmp(argv[i], "--input") == 0) {
            if (++i < argc) input_file = argv[i];
        } else if (strcmp(argv[i], "--shm-input") == 0) {
            shm_input = true;
        } else if (strcmp(argv[i], "--shm-name") == 0) {
            if (++i < argc) shm_name = argv[i];
        } else if (strcmp(argv[i], "--save-bitstreams") == 0) {
            if (++i < argc) save_bitstreams_file = argv[i];
        } else if (strcmp(argv[i], "--codec") == 0) {
            if (++i < argc) {
                codec = argv[i];
            }
        } else if (strcmp(argv[i], "--qvrf-cpp-sender") == 0 ||
                   strcmp(argv[i], "--qvrf-cpp-experiment") == 0) {
            qvrf_cpp_sender = true;
        } else if (strcmp(argv[i], "--tx-ga-backend") == 0) {
            if (!require_arg(i, argc, argv[i])) return 1;
            tx_ga_backend = argv[++i];
        } else if (strcmp(argv[i], "--tx-trt-engine") == 0) {
            if (!require_arg(i, argc, argv[i])) return 1;
            tx_trt_engine = argv[++i];
        } else if (strcmp(argv[i], "--tx-trt-device") == 0) {
            if (!require_arg(i, argc, argv[i])) return 1;
            tx_trt_device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--profile") == 0) {
            profile = true;
        } else if (strcmp(argv[i], "--prebuffer-chunks") == 0) {
            if (++i < argc) prebuffer_chunks = atoi(argv[i]);
        } else if (strcmp(argv[i], "--tail-flush-chunks") == 0) {
            if (++i < argc) tail_flush_chunks = atoi(argv[i]);
        } else if (strcmp(argv[i], "--chunk-rate-hz") == 0) {
            if (++i < argc) chunk_rate_hz = atof(argv[i]);
        } else if (strcmp(argv[i], "--max-queue-chunks") == 0) {
            if (++i < argc) max_queue_chunks = atoi(argv[i]);
        } else if (strcmp(argv[i], "--chunk-order") == 0) {
            if (++i < argc) chunk_order_arg = argv[i];
        } else if (strcmp(argv[i], "--single-packet-mode") == 0) {
            single_packet_mode = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    // Clamp FPS (max 48 per commu testing)
    if (target_fps > 48.0) {
        fprintf(stderr, "Warning: FPS %.1f > 48 may cause burst loss. Clamping to 48.\n", target_fps);
        target_fps = 48.0;
    }
    if (redundancy < 1) redundancy = 1;
    if (codec_size <= 0 || codec_size % 64 != 0) {
        fprintf(stderr, "--codec-size must be positive and divisible by 64\n");
        return 1;
    }
    if (fixed_chunks < 1 || fixed_chunks > 255) {
        fprintf(stderr, "--chunks-per-frame must be in [1,255]\n");
        return 1;
    }
    if (fec_data_chunks < 0 || fec_data_chunks >= fixed_chunks) {
        if (fec_data_chunks != 0) {
            fprintf(stderr, "--fec-data-chunks must be 0 or in [1, chunks-per-frame-1]\n");
            return 1;
        }
    }
    if (prebuffer_chunks < 0) prebuffer_chunks = 0;
    if (tail_flush_chunks < 0) tail_flush_chunks = 0;
    if (chunk_rate_hz <= 0.0) chunk_rate_hz = 48.0;
    if (max_queue_chunks < 0) max_queue_chunks = 0;
    std::vector<size_t> chunk_order;
    if (fixed_chunks == 4) {
        if (chunk_order_arg.size() != 4) {
            fprintf(stderr, "--chunk-order must contain exactly 4 digits for 4-chunk mode\n");
            return 1;
        }
        bool seen[4] = {false, false, false, false};
        for (char ch : chunk_order_arg) {
            if (ch < '0' || ch > '3') {
                fprintf(stderr, "--chunk-order may only contain digits 0..3\n");
                return 1;
            }
            int idx = ch - '0';
            if (seen[idx]) {
                fprintf(stderr, "--chunk-order must not repeat chunk ids\n");
                return 1;
            }
            seen[idx] = true;
            chunk_order.push_back(static_cast<size_t>(idx));
        }
    }
    if (shm_input && input_file) {
        fprintf(stderr, "--shm-input and --input are mutually exclusive\n");
        return 1;
    }
    if (strcmp(codec, "mbt") != 0 && strcmp(codec, "msssim_qvrf") != 0) {
        fprintf(stderr, "Unsupported codec for rm_compress_cli: %s\n", codec);
        usage(argv[0]);
        return 1;
    }
    if (strcmp(codec, "msssim_qvrf") == 0 && !qvrf_cpp_sender) {
        fprintf(stderr,
                "C++ QVRF sender requires --qvrf-cpp-sender. "
                "The launcher default QVRF path uses experiments/sender_msssim_qvrf_v2.py.\n");
        return 1;
    }
    if (strcmp(tx_ga_backend, "openvino") != 0 && strcmp(tx_ga_backend, "tensorrt") != 0) {
        fprintf(stderr, "--tx-ga-backend must be openvino or tensorrt\n");
        return 1;
    }
    if (strcmp(tx_ga_backend, "tensorrt") == 0) {
        if (!tx_trt_engine || tx_trt_engine[0] == '\0') {
            fprintf(stderr, "--tx-trt-engine is required when --tx-ga-backend tensorrt\n");
            return 1;
        }
        if (tx_trt_device < 0) {
            fprintf(stderr, "--tx-trt-device must be >= 0\n");
            return 1;
        }
    }

    // --- Open input ---
    FILE* raw_input = nullptr;
    rmcompress::ShmRing shm_ring;
    uint64_t shm_last_sequence = 0;
    if (input_file) {
        raw_input = fopen(input_file, "rb");
        if (!raw_input) {
            perror("open input file");
            return 1;
        }
        printf("Reading raw BGR frames from: %s\n", input_file);
    } else if (shm_input) {
        if (!open_shm_with_retry(shm_ring, shm_name)) {
            return 1;
        }
        const rmcompress::ShmRingHeader* h = shm_ring.header();
        if (h->pixfmt != rmcompress::SHM_PIXFMT_BGR8 || h->stride < h->width * 3) {
            fprintf(stderr, "Unsupported shm frame format: pixfmt=0x%x stride=%u width=%u\n",
                    h->pixfmt, h->stride, h->width);
            return 1;
        }
        width = static_cast<int>(h->width);
        height = static_cast<int>(h->height);
        printf("Reading latest BGR frames from shared memory: %s (%ux%u stride=%u cap=%u)\n",
               shm_name, h->width, h->height, h->stride, h->capacity);
    } else {
        printf("Using test pattern generator (moving bar)\n");
    }

    // --- Create compressor ---
    rm_compressor_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.mbt_ir_path = model_path;
    cfg.width = width;
    cfg.height = height;
    cfg.codec_width = codec_size;
    cfg.codec_height = codec_size;
    cfg.device = device;
    cfg.ga_backend = tx_ga_backend;
    cfg.trt_engine_path = tx_trt_engine;
    cfg.trt_device = tx_trt_device;
    const int transport_data_chunks = (fec_data_chunks > 0) ? fec_data_chunks : fixed_chunks;
    cfg.max_packed_bytes = single_packet_mode ? MAX_PAYLOAD : MAX_PAYLOAD * transport_data_chunks;

    // --- Codec selection ---
    bool codec_is_msssim = (strcmp(codec, "msssim_qvrf") == 0);
    if (codec_is_msssim) {
        cfg.codec = 1;
        cfg.msssim_ga_path = "models/msssim_g_a_fp32.xml";
        cfg.msssim_ha_path = "models/msssim_h_a_fp32.xml";
        cfg.msssim_hs_path = "models/msssim_h_s_fp32.xml";
        cfg.msssim_cdf_path = "models/msssim_cdfs.bin";
    }

    rm_compressor_t* compressor = rm_compressor_create(&cfg);
    if (!compressor) {
        if (strcmp(tx_ga_backend, "tensorrt") == 0) {
            fprintf(stderr, "Failed to create compressor. Check:\n"
                            "  - rmcompress was built with TensorRT/CUDA support?\n"
                            "  - TensorRT engine exists and matches this codec/input shape?\n"
                            "  - Engine input/output are FP32 NCHW?\n"
                            "  - CUDA device id is available?\n"
                            "  - h_a/h_s OpenVINO FP32 CPU models and CDFs still exist?\n");
        } else if (codec_is_msssim) {
            fprintf(stderr, "Failed to create compressor. Check:\n"
                            "  - OpenVINO installed?\n"
                            "  - Models: msssim_g_a_fp32.xml, msssim_h_a_fp32.xml, msssim_h_s_fp32.xml exist?\n"
                            "  - CDFs: msssim_cdfs.bin exists?\n");
        } else {
            fprintf(stderr, "Failed to create compressor. Check:\n"
                            "  - OpenVINO installed?\n"
                            "  - Models: mbt_g_a.xml, mbt_h_a.xml, mbt_h_s.xml exist?\n"
                            "  - CDFs: mbt_cdfs.bin exists?\n");
        }
        if (raw_input) fclose(raw_input);
        return 1;
    }

    const bool ipc_output = ipc_host != nullptr && ipc_host[0] != '\0';

    // --- Open output ---
    SerialTx serial(port, baudrate);
    int ipc_fd = -1;
    if (ipc_output) {
        ipc_fd = connect_tcp_with_retry(ipc_host, ipc_port);
        if (ipc_fd < 0) {
            rm_compressor_destroy(compressor);
            if (raw_input) fclose(raw_input);
            return 1;
        }
        printf("IPC output connected to %s:%d (raw 300B R1V1 chunks)\n", ipc_host, ipc_port);
    } else if (!dry_run) {
        while (!serial.open()) {
            if (!serial_wait) {
                break;
            }
            fprintf(stderr, "Serial port %s unavailable; retrying in 1s...\n", port);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (!serial.is_open()) {
            fprintf(stderr, "Failed to open serial port %s (try --dry-run to test without serial)\n", port);
            rm_compressor_destroy(compressor);
            if (raw_input) fclose(raw_input);
            return 1;
        }
        printf("Serial port %s opened at %d baud, 8N1\n", serial.active_port().c_str(), baudrate);
    } else {
        printf("Dry-run mode: serial not opened\n");
    }

    // --- Allocate buffers ---
    const int frame_bytes = width * height * 3;
    std::vector<uint8_t> frame_rgb(frame_bytes);
    std::vector<uint8_t> shm_frame;
    std::vector<uint8_t> bitstream(8192);  // compressed output (generous)

    // FPS pacing
    const double frame_interval_us = 1.0e6 / target_fps;
    const double chunk_interval_us = 1.0e6 / chunk_rate_hz;

    printf("Starting: %s frames at %.1f FPS, input=%dx%d codec=%dx%d, codec=%s, redundancy=%d\n",
           num_frames > 0 ? std::to_string(num_frames).c_str() : "unlimited",
           target_fps, width, height, codec_size, codec_size, codec, redundancy);
    printf("Sender g_a:   backend=%s device=%s%s%s\n",
           tx_ga_backend,
           strcmp(tx_ga_backend, "tensorrt") == 0 ? "cuda:" : device,
           strcmp(tx_ga_backend, "tensorrt") == 0 ? std::to_string(tx_trt_device).c_str() : "",
           strcmp(tx_ga_backend, "tensorrt") == 0 ? " (TensorRT/NVIDIA GPU)" : " (OpenVINO)");
    printf("h_a/h_s:      OpenVINO FP32 CPU (hard requirement)\n");
    printf("Entropy:      CPU/host CompressAI CDF/rANS contract\n");
    if (codec_is_msssim) {
        printf("QVRF C++:     sender=%s\n", qvrf_cpp_sender ? "enabled" : "disabled");
    }
    printf("Chunk pacing: %.1f chunks/s max (309B each)\n", chunk_rate_hz);
    printf("TX prebuffer: %d chunks\n", prebuffer_chunks);
    printf("Tail flush: %d chunks\n", tail_flush_chunks);
    printf("TX max queue: %d chunks%s\n", max_queue_chunks, max_queue_chunks == 0 ? " (disabled)" : "");
    printf("Packet mode: %s\n", single_packet_mode ? "single physical chunk (diagnostic)" : (std::to_string(fixed_chunks) + " fixed chunks").c_str());
    if (!single_packet_mode && fixed_chunks == 4) {
        printf("Chunk order: %s\n", chunk_order_arg.c_str());
    }
    if (!single_packet_mode && fec_data_chunks > 0) {
        printf("FEC mode: %d data chunks + %d parity chunks, payload budget=%dB\n",
               fec_data_chunks, fixed_chunks - fec_data_chunks, MAX_PAYLOAD * fec_data_chunks);
    }
    uint16_t stream_id = static_cast<uint16_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() & FLAG_SESSION_MASK);
    if (stream_id == 0) stream_id = 1;

    auto start_time = std::chrono::steady_clock::now();
    std::atomic<int> total_chunks{0};
    std::atomic<int> total_errors{0};
    std::atomic<int> queue_dropped_chunks{0};
    std::atomic<int> tx_underruns{0};
    std::atomic<int> over_budget_frames{0};
    std::atomic<int> total_bytes{0};
    std::vector<double> chunk_send_intervals_ms;
    std::mutex chunk_send_stats_mutex;
    double max_compress_ms = 0;
    double total_compress_ms = 0;
    double last_compress_ms = 0;
    std::vector<rm_compress_stats_t> profile_stats;
    std::vector<int> profile_frame_ids;
    if (profile) {
        if (num_frames > 0) {
            profile_stats.reserve(static_cast<size_t>(num_frames));
            profile_frame_ids.reserve(static_cast<size_t>(num_frames));
        }
    }
    std::atomic<int> max_queue_depth{0};
    std::vector<std::pair<uint32_t, std::vector<uint8_t>>> saved_bitstreams;
    if (save_bitstreams_file && num_frames > 0) {
        saved_bitstreams.reserve(static_cast<size_t>(num_frames));
    }

    std::deque<QueuedChunk> send_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool enqueue_done = false;

    auto sender = [&]() {
        bool started = false;
        auto next_chunk_time = std::chrono::steady_clock::now();
        auto last_send_start = std::chrono::steady_clock::time_point{};
        const auto chunk_step = std::chrono::microseconds(static_cast<long long>(
            chunk_interval_us * (redundancy > 1 ? redundancy : 1)));
        while (true) {
            QueuedChunk item{};
            bool have_item = false;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (!started) {
                    queue_cv.wait(lock, [&]() {
                        return enqueue_done ||
                            prebuffer_chunks <= 0 ||
                            static_cast<int>(send_queue.size()) >= prebuffer_chunks;
                    });
                    if (send_queue.empty() && enqueue_done) break;
                    started = true;
                    next_chunk_time = std::chrono::steady_clock::now();
                }
            }

            auto now = std::chrono::steady_clock::now();
            if (now < next_chunk_time) {
                std::this_thread::sleep_until(next_chunk_time);
            }
            const auto send_start = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (!send_queue.empty()) {
                    item = std::move(send_queue.front());
                    send_queue.pop_front();
                    have_item = true;
                } else if (enqueue_done) {
                    break;
                }
            }

            if (!have_item) {
                tx_underruns++;
                next_chunk_time += chunk_step;
                const auto after_tick = std::chrono::steady_clock::now();
                if (next_chunk_time < after_tick) {
                    next_chunk_time = after_tick + chunk_step;
                }
                continue;
            }

            if (last_send_start.time_since_epoch().count() != 0) {
                const double interval_ms = std::chrono::duration<double, std::milli>(
                    send_start - last_send_start).count();
                if (profile) {
                    std::lock_guard<std::mutex> lock(chunk_send_stats_mutex);
                    chunk_send_intervals_ms.push_back(interval_ms);
                }
            }
            last_send_start = send_start;
            if (e2e_trace) {
                const double enqueue_to_send_ms = std::chrono::duration<double, std::milli>(
                    send_start - item.enqueue_time).count();
                std::fprintf(stderr,
                    "E2E_TX frame=%u cid=%u/%u wall_ms=%lld capture_age_ms=%.1f "
                    "encode_ms=%.1f enqueue_to_send_ms=%.1f\n",
                    item.frame_id,
                    static_cast<unsigned>(item.chunk_id),
                    static_cast<unsigned>(item.chunk_count),
                    wall_ms_now(),
                    item.capture_age_ms,
                    item.encode_ms,
                    enqueue_to_send_ms);
            }

            if (ipc_output) {
                if (ipc_fd < 0 || !write_all(ipc_fd, item.data.data(), item.data.size())) {
                    total_errors++;
                    if (ipc_fd >= 0) close(ipc_fd);
                    ipc_fd = connect_tcp_with_retry(ipc_host, ipc_port);
                    if (ipc_fd >= 0 && write_all(ipc_fd, item.data.data(), item.data.size())) {
                        total_chunks++;
                        total_bytes += static_cast<int>(item.data.size());
                    } else {
                        total_errors++;
                    }
                } else {
                    total_chunks++;
                    total_bytes += static_cast<int>(item.data.size());
                }
            } else if (!dry_run) {
                int written;
                if (redundancy > 1) {
                    written = serial.send_0310_redundant(item.data.data(),
                        static_cast<int>(item.data.size()), redundancy);
                } else {
                    written = serial.send_0310(item.data.data(),
                        static_cast<int>(item.data.size()));
                }
                while (written < 0 && serial_wait) {
                    fprintf(stderr, "serial write failed; reconnecting...\n");
                    serial.close();
                    while (!g_stop.load() && !serial.open()) {
                        fprintf(stderr, "Serial port %s unavailable; retrying in 1s...\n", port);
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    }
                    if (g_stop.load()) break;
                    fprintf(stderr, "Serial port %s reconnected; retrying current chunk\n",
                            serial.active_port().c_str());
                    if (redundancy > 1) {
                        written = serial.send_0310_redundant(item.data.data(),
                            static_cast<int>(item.data.size()), redundancy);
                    } else {
                        written = serial.send_0310(item.data.data(),
                            static_cast<int>(item.data.size()));
                    }
                }
                if (written < 0) {
                    total_errors++;
                } else {
                    total_chunks += (redundancy > 1 ? redundancy : 1);
                    total_bytes += written;
                }
            } else {
                total_chunks += (redundancy > 1 ? redundancy : 1);
                total_bytes += 309 * (redundancy > 1 ? redundancy : 1);
            }

            next_chunk_time += chunk_step;
            const auto after_send = std::chrono::steady_clock::now();
            if (next_chunk_time < after_send) {
                next_chunk_time = after_send + chunk_step;
            }
        }
    };

    std::thread sender_thread(sender);

    int frames_processed = 0;
    constexpr size_t PROFILE_KEEP_LAST = 600;
    for (int frame_id = 0; !g_stop.load() && (num_frames <= 0 || frame_id < num_frames); frame_id++) {
        auto frame_start = std::chrono::steady_clock::now();
        double capture_age_ms = 0.0;

        // 1. Get frame (file or test pattern)
        if (raw_input) {
            if (!read_raw_frame(raw_input, frame_rgb.data(), frame_bytes)) {
                fprintf(stderr, "Frame %d: EOF or read error\n", frame_id);
                break;
            }
        } else if (shm_input) {
            rmcompress::ShmFrameHeader fh{};
            auto wait_start = std::chrono::steady_clock::now();
            while (!shm_ring.read_latest(shm_frame, &fh, &shm_last_sequence)) {
                if (g_stop.load()) goto finish_frames;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (std::chrono::steady_clock::now() - wait_start > std::chrono::seconds(2)) {
                    const rmcompress::ShmRingHeader* h = shm_ring.header();
                    fprintf(stderr,
                            "Frame %d: timed out waiting for shared memory frame; reopening %s "
                            "(last_seq=%llu write_seq=%llu frames_written=%llu)\n",
                            frame_id, shm_name,
                            static_cast<unsigned long long>(shm_last_sequence),
                            static_cast<unsigned long long>(h ? h->write_sequence : 0),
                            static_cast<unsigned long long>(h ? h->frames_written : 0));
                    shm_ring.close();
                    if (!open_shm_with_retry(shm_ring, shm_name)) {
                        goto finish_frames;
                    }
                    shm_last_sequence = 0;
                    wait_start = std::chrono::steady_clock::now();
                }
            }
            if (fh.width != static_cast<uint32_t>(width) ||
                fh.height != static_cast<uint32_t>(height) ||
                fh.data_bytes < static_cast<uint32_t>(frame_bytes)) {
                fprintf(stderr, "Frame %d: unexpected shm frame geometry %ux%u bytes=%u\n",
                        frame_id, fh.width, fh.height, fh.data_bytes);
                break;
            }
            std::memcpy(frame_rgb.data(), shm_frame.data(), frame_bytes);
            if (fh.timestamp_ns > 0) {
                const uint64_t now_ns = rmcompress::monotonic_time_ns();
                if (now_ns >= fh.timestamp_ns) {
                    capture_age_ms = static_cast<double>(now_ns - fh.timestamp_ns) / 1e6;
                }
            }
        } else {
            generate_test_frame(frame_rgb.data(), width, height, frame_id);
        }
        auto frame_acquired = std::chrono::steady_clock::now();

        // 2. Compress
        auto t0 = std::chrono::steady_clock::now();
        int bitstream_len = static_cast<int>(bitstream.size());
        int ret = rm_compress_frame(compressor, frame_rgb.data(),
                                    bitstream.data(), &bitstream_len);
        auto t1 = std::chrono::steady_clock::now();
        double compress_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_compress_ms = compress_ms;
        total_compress_ms += compress_ms;
        if (compress_ms > max_compress_ms) max_compress_ms = compress_ms;
        if (profile) {
            if (const rm_compress_stats_t* st = rm_compressor_last_stats(compressor)) {
                profile_stats.push_back(*st);
                profile_frame_ids.push_back(frame_id);
                if (profile_stats.size() > PROFILE_KEEP_LAST) {
                    profile_stats.erase(profile_stats.begin());
                    profile_frame_ids.erase(profile_frame_ids.begin());
                }
            }
        }

        bool over_budget = false;
        if (ret != 0) {
            if (strcmp(tx_ga_backend, "tensorrt") == 0) {
                fprintf(stderr,
                        "Frame %d: TensorRT g_a compression failed (ret=%d); "
                        "aborting instead of sending a fallback marker\n",
                        frame_id, ret);
                total_errors++;
                break;
            }
            fprintf(stderr, "Frame %d: compression over budget or failed (ret=%d); sending marker\n",
                    frame_id, ret);
            over_budget = true;
            over_budget_frames++;
            auto marker = pack_over_budget_marker(
                static_cast<uint32_t>(frame_id), bitstream_len, MAX_PAYLOAD * fixed_chunks, 0.0f);
            bitstream_len = static_cast<int>(marker.size());
            std::copy(marker.begin(), marker.end(), bitstream.begin());
        }

        if (save_bitstreams_file && !over_budget) {
            saved_bitstreams.emplace_back(
                static_cast<uint32_t>(frame_id),
                std::vector<uint8_t>(bitstream.begin(), bitstream.begin() + bitstream_len));
        }

        // 3. Packetize. Default realtime mode is two chunks per image. The
        // single-packet path is diagnostic-only and must be explicitly enabled.
        auto chunks = single_packet_mode
            ? pack_frame(static_cast<uint32_t>(frame_id),
                         bitstream.data(), bitstream_len, stream_id)
            : ((fec_data_chunks > 0)
                ? pack_frame_fixed_n(static_cast<uint32_t>(frame_id),
                                     bitstream.data(), bitstream_len, fixed_chunks,
                                     fec_data_chunks, stream_id)
                : pack_frame_fixed_n(static_cast<uint32_t>(frame_id),
                                     bitstream.data(), bitstream_len, fixed_chunks, stream_id));

        // 4. Enqueue chunks for the serial sender thread.
        int queue_depth_after_enqueue = 0;
        std::vector<size_t> enqueue_order;
        enqueue_order.reserve(chunks.size());
        if (!single_packet_mode && fixed_chunks == 4 && chunks.size() == 4 && chunk_order.size() == 4) {
            // The 0x0310 bridge can lose a consistent packet position in a 4-packet burst.
            // Keep this configurable so tests can place the tail chunk in the weak slot.
            enqueue_order = chunk_order;
        } else {
            for (size_t i = 0; i < chunks.size(); ++i) {
                enqueue_order.push_back(i);
            }
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (max_queue_chunks > 0) {
                const int incoming = static_cast<int>(chunks.size());
                while (!send_queue.empty() &&
                       static_cast<int>(send_queue.size()) + incoming > max_queue_chunks) {
                    const uint32_t drop_frame_id = send_queue.front().frame_id;
                    while (!send_queue.empty() && send_queue.front().frame_id == drop_frame_id) {
                        send_queue.pop_front();
                        queue_dropped_chunks++;
                    }
                }
            }
            const auto enqueue_time = std::chrono::steady_clock::now();
            for (size_t idx : enqueue_order) {
                send_queue.push_back(QueuedChunk{
                    static_cast<uint32_t>(frame_id),
                    static_cast<uint8_t>(idx),
                    static_cast<uint8_t>(chunks.size()),
                    std::move(chunks[idx]),
                    enqueue_time,
                    capture_age_ms,
                    compress_ms,
                });
            }
            int depth = static_cast<int>(send_queue.size());
            queue_depth_after_enqueue = depth;
            int prev = max_queue_depth.load();
            while (depth > prev && !max_queue_depth.compare_exchange_weak(prev, depth)) {}
        }
        queue_cv.notify_one();

        // 5. Progress every ~1 second (24 frames)
        if ((frame_id + 1) % 24 == 0 || frame_id == num_frames - 1) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            int queue_depth = 0;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                queue_depth = static_cast<int>(send_queue.size());
            }
            printf("  Frame %d/%s | %d chunks | %.0f FPS actual | %.1f KB/s | q=%d maxq=%d dropq=%d | comp=%.1fms max=%.1fms | over=%d err=%d\n",
                   frame_id + 1, num_frames > 0 ? std::to_string(num_frames).c_str() : "inf", total_chunks.load(),
                   (frame_id + 1) / elapsed,
                   total_bytes.load() / elapsed / 1000.0,
                   queue_depth, max_queue_depth.load(), queue_dropped_chunks.load(),
                   last_compress_ms, max_compress_ms,
                   over_budget_frames.load(),
                   total_errors.load());
        }

        // 6. Input pacing. The serial sender is the authoritative 48Hz clock.
        // If the TX queue falls below the jitter-buffer target, refill it
        // immediately instead of creating a gap that the VTX bridge may drop.
        auto frame_end = std::chrono::steady_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(
            frame_end - frame_start).count();
        double remaining_us = frame_interval_us - elapsed_us;
        if (queue_depth_after_enqueue >= prebuffer_chunks && remaining_us > 100) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(static_cast<long long>(remaining_us)));
        }
        frames_processed = frame_id + 1;
    }

finish_frames:
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        enqueue_done = true;
    }
    queue_cv.notify_one();
    sender_thread.join();

    int tail_flush_sent = 0;
    if (tail_flush_chunks > 0 && !ipc_output) {
        std::vector<uint8_t> flush_payload(300, 0);
        const char marker[] = "R1FLUSH";
        std::memcpy(flush_payload.data(), marker, sizeof(marker) - 1);
        for (int i = 0; i < tail_flush_chunks; ++i) {
            flush_payload[8] = static_cast<uint8_t>(i & 0xFF);
            int written = -1;
            if (!dry_run) {
                written = serial.send_0310(flush_payload.data(), static_cast<int>(flush_payload.size()));
                if (written < 0) {
                    total_errors++;
                    break;
                }
            }
            tail_flush_sent++;
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long long>(chunk_interval_us)));
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    if (save_bitstreams_file) {
        FILE* out = fopen(save_bitstreams_file, "wb");
        if (!out) {
            perror("save bitstreams");
            total_errors++;
        } else {
            uint32_t count = static_cast<uint32_t>(saved_bitstreams.size());
            fwrite(&count, sizeof(count), 1, out);
            for (const auto& item : saved_bitstreams) {
                uint32_t frame_id = item.first;
                uint32_t size = static_cast<uint32_t>(item.second.size());
                fwrite(&frame_id, sizeof(frame_id), 1, out);
                fwrite(&size, sizeof(size), 1, out);
                fwrite(item.second.data(), 1, item.second.size(), out);
            }
            fclose(out);
            printf("Saved %u bitstreams to %s\n", count, save_bitstreams_file);
        }
    }
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();

    printf("\n--- Transmission complete ---\n");
    printf("Frames:       %d\n", frames_processed);
    printf("Chunks sent:  %d (redundancy=%d)\n", total_chunks.load(), redundancy);
    printf("Total bytes:  %d (%.1f KB)\n", total_bytes.load(), total_bytes.load() / 1000.0);
    printf("Errors:       %d\n", total_errors.load());
    printf("Queue drops:  %d chunks\n", queue_dropped_chunks.load());
    printf("TX underruns: %d ticks\n", tx_underruns.load());
    printf("Over budget:  %d\n", over_budget_frames.load());
    printf("Tail flush:   %d\n", tail_flush_sent);
    printf("Duration:     %.2f s (%.1f FPS actual)\n", total_sec, frames_processed / total_sec);
    if (ipc_output) {
        printf("IPC stats:    %d bytes, %d chunks\n",
               total_bytes.load(), total_chunks.load());
    } else if (!dry_run) {
        printf("Serial stats: %lu bytes, %lu frames\n",
               serial.bytes_written(), serial.frames_sent());
    }
    if (ipc_fd >= 0) {
        close(ipc_fd);
    }
    printf("Compress:     max=%.1f ms, avg=%.1f ms\n",
           max_compress_ms, total_compress_ms / std::max(1, frames_processed));
    if (profile && !profile_stats.empty()) {
        std::map<int, int> rc_counts;
        double sum_pre = 0, sum_ga = 0, sum_p1 = 0, sum_p2 = 0, sum_p3 = 0, sum_total = 0;
        double sum_bytes = 0, sum_beta = 0;
        int min_bytes = std::numeric_limits<int>::max();
        int max_bytes = 0;
        float min_beta = std::numeric_limits<float>::max();
        float max_beta = 0.0f;
        int low_bytes = 0;
        int longest_low_run = 0;
        int current_low_run = 0;
        std::vector<int> packed_sizes;
        std::vector<float> betas;
        std::vector<float> pre_ms;
        std::vector<float> ga_ms;
        std::vector<float> p1_ms;
        std::vector<float> p2_ms;
        std::vector<float> p3_ms;
        std::vector<float> total_ms;
        for (const auto& st : profile_stats) {
            rc_counts[st.rc_passes]++;
            sum_pre += st.preprocess_ms;
            sum_ga += st.g_a_ms;
            sum_p1 += st.pass1_ms;
            sum_p2 += st.pass2_ms;
            sum_p3 += st.pass3_ms;
            sum_total += st.total_ms;
            sum_bytes += st.packed_bytes;
            sum_beta += st.beta;
            min_bytes = std::min(min_bytes, st.packed_bytes);
            max_bytes = std::max(max_bytes, st.packed_bytes);
            min_beta = std::min(min_beta, st.beta);
            max_beta = std::max(max_beta, st.beta);
            packed_sizes.push_back(st.packed_bytes);
            betas.push_back(st.beta);
            pre_ms.push_back(st.preprocess_ms);
            ga_ms.push_back(st.g_a_ms);
            p1_ms.push_back(st.pass1_ms);
            p2_ms.push_back(st.pass2_ms);
            p3_ms.push_back(st.pass3_ms);
            total_ms.push_back(st.total_ms);
            if (st.packed_bytes > 0 && st.packed_bytes < 220) {
                low_bytes++;
                current_low_run++;
                longest_low_run = std::max(longest_low_run, current_low_run);
            } else {
                current_low_run = 0;
            }
        }
        const double denom = std::max<size_t>(1, profile_stats.size());
        auto percentile_int = [](std::vector<int> values, double q) {
            if (values.empty()) return 0;
            std::sort(values.begin(), values.end());
            size_t idx = static_cast<size_t>(std::round(q * (values.size() - 1)));
            return values[std::min(idx, values.size() - 1)];
        };
        auto percentile_float = [](std::vector<float> values, double q) {
            if (values.empty()) return 0.0f;
            std::sort(values.begin(), values.end());
            size_t idx = static_cast<size_t>(std::round(q * (values.size() - 1)));
            return values[std::min(idx, values.size() - 1)];
        };
        auto percentile_double = [](std::vector<float> values, double q) {
            if (values.empty()) return 0.0;
            std::sort(values.begin(), values.end());
            size_t idx = static_cast<size_t>(std::round(q * (values.size() - 1)));
            return static_cast<double>(values[std::min(idx, values.size() - 1)]);
        };
        printf("\n--- Compression profile ---\n");
        printf("RC passes:    ");
        for (const auto& kv : rc_counts) {
            printf("%d-pass=%d ", kv.first, kv.second);
        }
        printf("\n");
        printf("Packed bytes: min=%d p10=%d avg=%.1f p50=%d p90=%d max=%d low<220=%d longest_low_run=%d\n",
               min_bytes == std::numeric_limits<int>::max() ? 0 : min_bytes,
               percentile_int(packed_sizes, 0.10), sum_bytes / denom,
               percentile_int(packed_sizes, 0.50), percentile_int(packed_sizes, 0.90),
               max_bytes, low_bytes, longest_low_run);
        printf("Beta:         min=%.2f p50=%.2f avg=%.2f p90=%.2f max=%.2f\n",
               min_beta == std::numeric_limits<float>::max() ? 0.0f : min_beta,
               percentile_float(betas, 0.50), sum_beta / denom,
               percentile_float(betas, 0.90), max_beta);
        printf("Avg stages:   prep=%.2fms g_a=%.2fms pass1=%.2fms pass2=%.2fms pass3=%.2fms total=%.2fms\n",
               sum_pre / denom, sum_ga / denom, sum_p1 / denom,
               sum_p2 / denom, sum_p3 / denom, sum_total / denom);
        printf("Stage p50:    prep=%.2fms g_a=%.2fms pass1=%.2fms pass2=%.2fms pass3=%.2fms total=%.2fms\n",
               percentile_double(pre_ms, 0.50), percentile_double(ga_ms, 0.50),
               percentile_double(p1_ms, 0.50), percentile_double(p2_ms, 0.50),
               percentile_double(p3_ms, 0.50), percentile_double(total_ms, 0.50));
        printf("Stage p90:    prep=%.2fms g_a=%.2fms pass1=%.2fms pass2=%.2fms pass3=%.2fms total=%.2fms\n",
               percentile_double(pre_ms, 0.90), percentile_double(ga_ms, 0.90),
               percentile_double(p1_ms, 0.90), percentile_double(p2_ms, 0.90),
               percentile_double(p3_ms, 0.90), percentile_double(total_ms, 0.90));
        printf("Stage p99:    prep=%.2fms g_a=%.2fms pass1=%.2fms pass2=%.2fms pass3=%.2fms total=%.2fms\n",
               percentile_double(pre_ms, 0.99), percentile_double(ga_ms, 0.99),
               percentile_double(p1_ms, 0.99), percentile_double(p2_ms, 0.99),
               percentile_double(p3_ms, 0.99), percentile_double(total_ms, 0.99));
        printf("Stage max:    prep=%.2fms g_a=%.2fms pass1=%.2fms pass2=%.2fms pass3=%.2fms total=%.2fms\n",
               percentile_double(pre_ms, 1.00), percentile_double(ga_ms, 1.00),
               percentile_double(p1_ms, 1.00), percentile_double(p2_ms, 1.00),
               percentile_double(p3_ms, 1.00), percentile_double(total_ms, 1.00));
        std::vector<int> order(profile_stats.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return profile_stats[a].total_ms > profile_stats[b].total_ms;
        });
        int top_n = std::min<int>(10, order.size());
        printf("Top spikes:\n");
        for (int i = 0; i < top_n; ++i) {
            int idx = order[i];
            const auto& st = profile_stats[idx];
            printf("  f=%d total=%.2fms prep=%.2f g_a=%.2f p1=%.2f p2=%.2f p3=%.2f rc=%d bytes=%d beta=%.2f over=%d\n",
                   profile_frame_ids[idx], st.total_ms, st.preprocess_ms, st.g_a_ms,
                   st.pass1_ms, st.pass2_ms, st.pass3_ms, st.rc_passes,
                   st.packed_bytes, st.beta, st.over_budget);
        }
    }
    if (profile) {
        std::vector<double> intervals;
        {
            std::lock_guard<std::mutex> lock(chunk_send_stats_mutex);
            intervals = chunk_send_intervals_ms;
        }
        if (!intervals.empty()) {
            std::sort(intervals.begin(), intervals.end());
            auto percentile_double = [&](double q) {
                size_t idx = static_cast<size_t>(std::round(q * (intervals.size() - 1)));
                return intervals[std::min(idx, intervals.size() - 1)];
            };
            const double avg_interval = std::accumulate(intervals.begin(), intervals.end(), 0.0) /
                static_cast<double>(intervals.size());
            const double target_ms = 1000.0 / chunk_rate_hz;
            int slow_150 = 0;
            int slow_200 = 0;
            for (double interval : intervals) {
                if (interval > target_ms * 1.5) slow_150++;
                if (interval > target_ms * 2.0) slow_200++;
            }
            printf("Chunk intervals: n=%zu target=%.2fms min=%.2f avg=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f slow>1.5x=%d slow>2x=%d\n",
                   intervals.size(), target_ms, intervals.front(), avg_interval,
                   percentile_double(0.50), percentile_double(0.90),
                   percentile_double(0.99), intervals.back(), slow_150, slow_200);
        }
    }

    // --- Cleanup ---
    if (!dry_run && !ipc_output) serial.close();
    rm_compressor_destroy(compressor);
    if (raw_input) fclose(raw_input);

    return (total_errors.load() > 0) ? 1 : 0;
}
