#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace rmcompress {

constexpr uint32_t SHM_RING_MAGIC = 0x524d5348;  // RMSH
constexpr uint32_t SHM_RING_VERSION = 1;
constexpr uint32_t SHM_PIXFMT_BGR8 = 0x42475238;  // BGR8

struct ShmFrameHeader {
    uint64_t sequence;
    uint64_t timestamp_ns;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t pixfmt;
    uint32_t data_bytes;
    uint32_t reserved;
};

struct ShmRingHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t capacity;
    uint32_t slot_bytes;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t pixfmt;
    uint64_t write_sequence;
    uint64_t frames_written;
    uint64_t frames_dropped;
};

class ShmRing {
public:
    ShmRing();
    ~ShmRing();

    ShmRing(const ShmRing&) = delete;
    ShmRing& operator=(const ShmRing&) = delete;

    bool create(const std::string& name, uint32_t capacity, uint32_t width,
                uint32_t height, uint32_t stride, uint32_t pixfmt,
                std::string* error);
    bool open(const std::string& name, std::string* error);
    void close();

    bool write_latest(const uint8_t* data, uint32_t data_bytes,
                      uint64_t timestamp_ns);
    bool read_latest(std::vector<uint8_t>& out, ShmFrameHeader* frame_header,
                     uint64_t* last_sequence);

    const ShmRingHeader* header() const { return header_; }

private:
    uint8_t* slot_ptr(uint32_t index) const;
    size_t total_size() const;

    int fd_;
    uint8_t* mapping_;
    size_t mapping_size_;
    ShmRingHeader* header_;
    bool owner_;
    std::string name_;
};

uint64_t monotonic_time_ns();

}  // namespace rmcompress
