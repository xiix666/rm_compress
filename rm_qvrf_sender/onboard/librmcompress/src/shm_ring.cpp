#include "rmcompress/shm_ring.h"

#include <cerrno>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>

namespace rmcompress {

namespace {

std::string errno_msg(const char* what) {
    return std::string(what) + ": " + std::strerror(errno);
}

std::string normalize_name(const std::string& name) {
    if (!name.empty() && name[0] == '/') return name;
    return "/" + name;
}

}  // namespace

ShmRing::ShmRing()
    : fd_(-1), mapping_(nullptr), mapping_size_(0), header_(nullptr),
      owner_(false) {}

ShmRing::~ShmRing() {
    close();
}

size_t ShmRing::total_size() const {
    if (!header_) return 0;
    return sizeof(ShmRingHeader) +
           static_cast<size_t>(header_->capacity) * header_->slot_bytes;
}

uint8_t* ShmRing::slot_ptr(uint32_t index) const {
    return mapping_ + sizeof(ShmRingHeader) +
           static_cast<size_t>(index) * header_->slot_bytes;
}

bool ShmRing::create(const std::string& name, uint32_t capacity, uint32_t width,
                     uint32_t height, uint32_t stride, uint32_t pixfmt,
                     std::string* error) {
    close();
    if (capacity < 2 || width == 0 || height == 0 || stride == 0) {
        if (error) *error = "invalid shm ring geometry";
        return false;
    }

    name_ = normalize_name(name);
    owner_ = true;
    shm_unlink(name_.c_str());
    fd_ = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    if (fd_ < 0) {
        if (error) *error = errno_msg("shm_open create");
        return false;
    }

    const uint32_t data_bytes = stride * height;
    const uint32_t slot_bytes =
        static_cast<uint32_t>(sizeof(ShmFrameHeader) + data_bytes);
    mapping_size_ = sizeof(ShmRingHeader) +
                    static_cast<size_t>(capacity) * slot_bytes;

    if (ftruncate(fd_, static_cast<off_t>(mapping_size_)) != 0) {
        if (error) *error = errno_msg("ftruncate");
        close();
        return false;
    }

    mapping_ = static_cast<uint8_t*>(
        mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        if (error) *error = errno_msg("mmap create");
        close();
        return false;
    }

    header_ = reinterpret_cast<ShmRingHeader*>(mapping_);
    std::memset(mapping_, 0, mapping_size_);
    header_->magic = SHM_RING_MAGIC;
    header_->version = SHM_RING_VERSION;
    header_->capacity = capacity;
    header_->slot_bytes = slot_bytes;
    header_->width = width;
    header_->height = height;
    header_->stride = stride;
    header_->pixfmt = pixfmt;
    __sync_synchronize();
    return true;
}

bool ShmRing::open(const std::string& name, std::string* error) {
    close();
    name_ = normalize_name(name);
    fd_ = shm_open(name_.c_str(), O_RDWR, 0666);
    if (fd_ < 0) {
        if (error) *error = errno_msg("shm_open open");
        return false;
    }

    struct stat st {};
    if (fstat(fd_, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(ShmRingHeader))) {
        if (error) *error = errno_msg("fstat");
        close();
        return false;
    }

    mapping_size_ = static_cast<size_t>(st.st_size);
    mapping_ = static_cast<uint8_t*>(
        mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        if (error) *error = errno_msg("mmap open");
        close();
        return false;
    }

    header_ = reinterpret_cast<ShmRingHeader*>(mapping_);
    if (header_->magic != SHM_RING_MAGIC || header_->version != SHM_RING_VERSION) {
        if (error) *error = "shared memory header magic/version mismatch";
        close();
        return false;
    }
    if (total_size() > mapping_size_) {
        if (error) *error = "shared memory mapping is smaller than ring header declares";
        close();
        return false;
    }
    return true;
}

void ShmRing::close() {
    if (mapping_) {
        munmap(mapping_, mapping_size_);
        mapping_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    if (owner_ && !name_.empty()) {
        shm_unlink(name_.c_str());
    }
    mapping_size_ = 0;
    header_ = nullptr;
    owner_ = false;
}

bool ShmRing::write_latest(const uint8_t* data, uint32_t data_bytes,
                           uint64_t timestamp_ns) {
    if (!header_ || !data || data_bytes != header_->stride * header_->height)
        return false;

    const uint64_t seq = __atomic_load_n(&header_->write_sequence, __ATOMIC_ACQUIRE) + 1;
    const uint32_t index = static_cast<uint32_t>(seq % header_->capacity);
    uint8_t* slot = slot_ptr(index);
    auto* fh = reinterpret_cast<ShmFrameHeader*>(slot);
    uint8_t* dst = slot + sizeof(ShmFrameHeader);

    __atomic_store_n(&fh->sequence, seq | 1ULL, __ATOMIC_RELEASE);  // odd means writer in progress
    std::memcpy(dst, data, data_bytes);
    fh->timestamp_ns = timestamp_ns;
    fh->width = header_->width;
    fh->height = header_->height;
    fh->stride = header_->stride;
    fh->pixfmt = header_->pixfmt;
    fh->data_bytes = data_bytes;
    __atomic_store_n(&fh->sequence, seq << 1, __ATOMIC_RELEASE);  // even means complete
    __atomic_store_n(&header_->write_sequence, seq, __ATOMIC_RELEASE);
    __atomic_add_fetch(&header_->frames_written, 1ULL, __ATOMIC_RELAXED);
    if (seq > header_->capacity) __atomic_add_fetch(&header_->frames_dropped, 1ULL, __ATOMIC_RELAXED);
    return true;
}

bool ShmRing::read_latest(std::vector<uint8_t>& out,
                          ShmFrameHeader* frame_header,
                          uint64_t* last_sequence) {
    if (!header_ || !last_sequence) return false;
    const uint64_t seq = __atomic_load_n(&header_->write_sequence, __ATOMIC_ACQUIRE);
    if (seq == 0 || seq == *last_sequence) return false;

    const uint32_t index = static_cast<uint32_t>(seq % header_->capacity);
    uint8_t* slot = slot_ptr(index);
    auto* fh = reinterpret_cast<ShmFrameHeader*>(slot);

    const uint64_t a = __atomic_load_n(&fh->sequence, __ATOMIC_ACQUIRE);
    if ((a & 1ULL) != 0 || (a >> 1) != seq) return false;

    out.resize(fh->data_bytes);
    std::memcpy(out.data(), slot + sizeof(ShmFrameHeader), fh->data_bytes);
    const uint64_t b = __atomic_load_n(&fh->sequence, __ATOMIC_ACQUIRE);
    if (a != b || (b & 1ULL) != 0 || (b >> 1) != seq) return false;

    if (frame_header) *frame_header = *fh;
    *last_sequence = seq;
    return true;
}

uint64_t monotonic_time_ns() {
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
}

}  // namespace rmcompress
