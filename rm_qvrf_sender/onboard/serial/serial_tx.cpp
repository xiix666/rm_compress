#include "serial_tx.h"

#include <cstring>
#include <cstdio>
#include <cerrno>
#include <vector>
#include <algorithm>

#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <glob.h>

// ---------------------------------------------------------------------------
// CRC-8: poly=G(x)=x^8+x^5+x^4+1 (0x31), init=0xFF
// Table from official RoboMaster protocol appendix — matches commu serial_comm.py
// ---------------------------------------------------------------------------
static const uint8_t _CRC8_TABLE[256] = {
    0x00, 0x5E, 0xBC, 0xE2, 0x61, 0x3F, 0xDD, 0x83, 0xC2, 0x9C, 0x7E, 0x20, 0xA3, 0xFD, 0x1F, 0x41,
    0x9D, 0xC3, 0x21, 0x7F, 0xFC, 0xA2, 0x40, 0x1E, 0x5F, 0x01, 0xE3, 0xBD, 0x3E, 0x60, 0x82, 0xDC,
    0x23, 0x7D, 0x9F, 0xC1, 0x42, 0x1C, 0xFE, 0xA0, 0xE1, 0xBF, 0x5D, 0x03, 0x80, 0xDE, 0x3C, 0x62,
    0xBE, 0xE0, 0x02, 0x5C, 0xDF, 0x81, 0x63, 0x3D, 0x7C, 0x22, 0xC0, 0x9E, 0x1D, 0x43, 0xA1, 0xFF,
    0x46, 0x18, 0xFA, 0xA4, 0x27, 0x79, 0x9B, 0xC5, 0x84, 0xDA, 0x38, 0x66, 0xE5, 0xBB, 0x59, 0x07,
    0xDB, 0x85, 0x67, 0x39, 0xBA, 0xE4, 0x06, 0x58, 0x19, 0x47, 0xA5, 0xFB, 0x78, 0x26, 0xC4, 0x9A,
    0x65, 0x3B, 0xD9, 0x87, 0x04, 0x5A, 0xB8, 0xE6, 0xA7, 0xF9, 0x1B, 0x45, 0xC6, 0x98, 0x7A, 0x24,
    0xF8, 0xA6, 0x44, 0x1A, 0x99, 0xC7, 0x25, 0x7B, 0x3A, 0x64, 0x86, 0xD8, 0x5B, 0x05, 0xE7, 0xB9,
    0x8C, 0xD2, 0x30, 0x6E, 0xED, 0xB3, 0x51, 0x0F, 0x4E, 0x10, 0xF2, 0xAC, 0x2F, 0x71, 0x93, 0xCD,
    0x11, 0x4F, 0xAD, 0xF3, 0x70, 0x2E, 0xCC, 0x92, 0xD3, 0x8D, 0x6F, 0x31, 0xB2, 0xEC, 0x0E, 0x50,
    0xAF, 0xF1, 0x13, 0x4D, 0xCE, 0x90, 0x72, 0x2C, 0x6D, 0x33, 0xD1, 0x8F, 0x0C, 0x52, 0xB0, 0xEE,
    0x32, 0x6C, 0x8E, 0xD0, 0x53, 0x0D, 0xEF, 0xB1, 0xF0, 0xAE, 0x4C, 0x12, 0x91, 0xCF, 0x2D, 0x73,
    0xCA, 0x94, 0x76, 0x28, 0xAB, 0xF5, 0x17, 0x49, 0x08, 0x56, 0xB4, 0xEA, 0x69, 0x37, 0xD5, 0x8B,
    0x57, 0x09, 0xEB, 0xB5, 0x36, 0x68, 0x8A, 0xD4, 0x95, 0xCB, 0x29, 0x77, 0xF4, 0xAA, 0x48, 0x16,
    0xE9, 0xB7, 0x55, 0x0B, 0x88, 0xD6, 0x34, 0x6A, 0x2B, 0x75, 0x97, 0xC9, 0x4A, 0x14, 0xF6, 0xA8,
    0x74, 0x2A, 0xC8, 0x96, 0x15, 0x4B, 0xA9, 0xF7, 0xB6, 0xE8, 0x0A, 0x54, 0xD7, 0x89, 0x6B, 0x35,
};

// ---------------------------------------------------------------------------
// CRC-16/MCRF4XX: poly=0x1021, init=0xFFFF, refin=True, refout=True
// Reflected poly = 0x8408. Matches commu serial_comm.py.
// ---------------------------------------------------------------------------
static uint16_t _CRC16_VTX_TABLE[256];

static void _make_crc16_vtx_table() {
    for (int i = 0; i < 256; i++) {
        uint16_t crc = static_cast<uint16_t>(i);
        for (int j = 0; j < 8; j++) {
            if (crc & 0x01) {
                crc = (crc >> 1) ^ 0x8408;
            } else {
                crc >>= 1;
            }
        }
        _CRC16_VTX_TABLE[i] = crc;
    }
}

static bool _tables_inited = false;

static void _init_tables() {
    if (_tables_inited) return;
    _make_crc16_vtx_table();
    _tables_inited = true;
}

// ---------------------------------------------------------------------------
// Protocol constants — matches commu SerialFrameBuilder
// ---------------------------------------------------------------------------
static const uint8_t  SOF                    = 0xA5;
static const uint16_t CMD_ROBOT_TO_CUSTOM_CLIENT = 0x0310;
static const int      MAX_0310_PAYLOAD       = 300;   // VTX requires exactly 300B
static const int      FRAME_TOTAL            = 309;   // 5 header + 2 cmd + 300 data + 2 crc16

static bool _baudrate_to_speed(int baudrate, speed_t* speed) {
    if (!speed) return false;
    switch (baudrate) {
    case 115200: *speed = B115200; return true;
    case 230400: *speed = B230400; return true;
    case 460800: *speed = B460800; return true;
    case 921600: *speed = B921600; return true;
#ifdef B1000000
    case 1000000: *speed = B1000000; return true;
#endif
#ifdef B1500000
    case 1500000: *speed = B1500000; return true;
#endif
#ifdef B2000000
    case 2000000: *speed = B2000000; return true;
#endif
    default:
        return false;
    }
}

static void _append_glob_matches(std::vector<std::string>* out, const char* pattern) {
    if (!out) return;
    glob_t matches{};
    if (glob(pattern, 0, nullptr, &matches) != 0) {
        globfree(&matches);
        return;
    }
    for (size_t i = 0; i < matches.gl_pathc; ++i) {
        std::string path = matches.gl_pathv[i];
        if (std::find(out->begin(), out->end(), path) == out->end()) {
            out->push_back(path);
        }
    }
    globfree(&matches);
}

static std::vector<std::string> _serial_candidates(const std::string& configured_port) {
    if (configured_port != "auto") {
        return {configured_port};
    }
    std::vector<std::string> ports;
    _append_glob_matches(&ports, "/dev/serial/by-id/*");
    _append_glob_matches(&ports, "/dev/ttyUSB*");
    _append_glob_matches(&ports, "/dev/ttyACM*");
    return ports;
}

// ---------------------------------------------------------------------------
// SerialTx implementation
// ---------------------------------------------------------------------------

SerialTx::SerialTx(const std::string& port, int baudrate)
    : _port(port), _baudrate(baudrate)
{
    _init_tables();
}

SerialTx::~SerialTx() {
    close();
}

bool SerialTx::open_path(const std::string& path) {
    _fd = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (_fd < 0) {
        return false;
    }

    // Set blocking mode
    int flags = fcntl(_fd, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(_fd, F_SETFL, flags & ~O_NONBLOCK);
    }

    struct termios options;
    if (tcgetattr(_fd, &options) != 0) {
        fprintf(stderr, "tcgetattr %s: %s\n", path.c_str(), strerror(errno));
        mark_failed();
        return false;
    }

    // 8N1, default deployment is 921600 baud.
    speed_t speed = B921600;
    if (!_baudrate_to_speed(_baudrate, &speed)) {
        fprintf(stderr, "SerialTx: unsupported baudrate %d\n", _baudrate);
        mark_failed();
        return false;
    }
    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);

    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag |= CREAD | CLOCAL;

    // Raw mode
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);

    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 5;

    if (tcsetattr(_fd, TCSANOW, &options) != 0) {
        fprintf(stderr, "tcsetattr %s: %s\n", path.c_str(), strerror(errno));
        mark_failed();
        return false;
    }
    if (tcflush(_fd, TCIOFLUSH) != 0) {
        fprintf(stderr, "tcflush %s: %s\n", path.c_str(), strerror(errno));
        mark_failed();
        return false;
    }

    _active_port = path;
    return true;
}

bool SerialTx::open() {
    const auto candidates = _serial_candidates(_port);
    if (candidates.empty()) {
        if (_port == "auto") {
            fprintf(stderr, "SerialTx: no serial candidates found under /dev/serial/by-id, /dev/ttyUSB*, /dev/ttyACM*\n");
        } else {
            fprintf(stderr, "SerialTx: no serial candidate for %s\n", _port.c_str());
        }
        return false;
    }
    for (const auto& path : candidates) {
        if (open_path(path)) {
            return true;
        }
        if (_port != "auto") {
            fprintf(stderr, "open serial port %s: %s\n", path.c_str(), strerror(errno));
        }
    }
    if (_port == "auto") {
        fprintf(stderr, "SerialTx: failed to open any serial candidate (%zu found)\n", candidates.size());
    }
    return false;
}

void SerialTx::close() {
    if (_fd >= 0) {
        ::close(_fd);
        _fd = -1;
    }
    _active_port.clear();
}

bool SerialTx::is_open() const {
    return _fd >= 0;
}

void SerialTx::mark_failed() {
    close();
}

int SerialTx::send_raw(const uint8_t* data, int len) {
    if (_fd < 0) {
        fprintf(stderr, "SerialTx: port not open\n");
        return -1;
    }
    int total = 0;
    while (total < len) {
        ssize_t written = ::write(_fd, data + total, static_cast<size_t>(len - total));
        if (written < 0) {
            if (errno == EINTR) continue;
            perror("serial write");
            mark_failed();
            return -1;
        }
        if (written == 0) {
            fprintf(stderr, "serial write returned 0\n");
            mark_failed();
            return -1;
        }
        total += static_cast<int>(written);
    }
    _bytes_written += total;
    return total;
}

void SerialTx::drain() {
    if (_fd >= 0) {
        if (tcdrain(_fd) != 0) {
            perror("tcdrain");
            mark_failed();
        }
    }
}

uint8_t SerialTx::crc8(const uint8_t* data, int len) {
    uint8_t crc = 0xFF;
    for (int i = 0; i < len; i++) {
        crc = _CRC8_TABLE[(crc ^ data[i]) & 0xFF];
    }
    return crc;
}

uint16_t SerialTx::crc16(const uint8_t* data, int len) {
    uint16_t crc = 0xFFFF;
    for (int i = 0; i < len; i++) {
        crc = (crc >> 8) ^ _CRC16_VTX_TABLE[(crc ^ data[i]) & 0xFF];
    }
    return crc;
}

// Build a single 0x0310 frame into a buffer (309 bytes). Used by send_0310.
static void _build_0310_frame(uint8_t* frame, const uint8_t* payload,
                               int payload_len, uint8_t seq) {
    int data_len = MAX_0310_PAYLOAD;  // VTX requires exactly 300B data field

    // CRC8 over SOF + data_len(2) + seq = 4 bytes
    uint8_t header_crc_input[4];
    header_crc_input[0] = SOF;
    header_crc_input[1] = static_cast<uint8_t>(data_len & 0xFF);
    header_crc_input[2] = static_cast<uint8_t>((data_len >> 8) & 0xFF);
    header_crc_input[3] = seq;

    // Compute CRC8 (matches commu serial_comm.py)
    uint8_t header_crc = 0xFF;
    for (int i = 0; i < 4; i++) {
        header_crc = _CRC8_TABLE[(header_crc ^ header_crc_input[i]) & 0xFF];
    }

    int pos = 0;
    frame[pos++] = SOF;
    frame[pos++] = static_cast<uint8_t>(data_len & 0xFF);
    frame[pos++] = static_cast<uint8_t>((data_len >> 8) & 0xFF);
    frame[pos++] = seq;
    frame[pos++] = header_crc;

    // cmd_id (2B LE)
    frame[pos++] = static_cast<uint8_t>(CMD_ROBOT_TO_CUSTOM_CLIENT & 0xFF);
    frame[pos++] = static_cast<uint8_t>((CMD_ROBOT_TO_CUSTOM_CLIENT >> 8) & 0xFF);

    // Payload data, zero-padded to exactly 300 bytes
    int copy_len = payload_len;
    if (copy_len > 0) {
        memcpy(frame + pos, payload, copy_len);
    }
    pos += copy_len;
    if (copy_len < MAX_0310_PAYLOAD) {
        memset(frame + pos, 0, MAX_0310_PAYLOAD - copy_len);
        pos += MAX_0310_PAYLOAD - copy_len;
    }

    // CRC16 over everything up to this point (307 bytes)
    uint16_t frame_crc = 0xFFFF;
    for (int i = 0; i < pos; i++) {
        frame_crc = (frame_crc >> 8) ^ _CRC16_VTX_TABLE[(frame_crc ^ frame[i]) & 0xFF];
    }
    frame[pos++] = static_cast<uint8_t>(frame_crc & 0xFF);
    frame[pos++] = static_cast<uint8_t>((frame_crc >> 8) & 0xFF);
}

int SerialTx::send_0310(const uint8_t* payload, int payload_len) {
    if (payload_len > MAX_0310_PAYLOAD) {
        fprintf(stderr, "SerialTx: 0x0310 payload max %d bytes, got %d\n",
                MAX_0310_PAYLOAD, payload_len);
        return -1;
    }

    uint8_t frame[FRAME_TOTAL];
    _build_0310_frame(frame, payload, payload_len, _seq);

    int written = send_raw(frame, FRAME_TOTAL);
    if (written != FRAME_TOTAL) {
        if (written >= 0) {
            fprintf(stderr, "SerialTx: short write %d/%d\n", written, FRAME_TOTAL);
            mark_failed();
        }
        return -1;
    }

    if (written > 0) {
        // tcdrain for deterministic transmission — matches commu pattern
        // of os.write + termios.tcdrain (see HANDOFF_0310.md)
        if (tcdrain(_fd) != 0) {
            perror("tcdrain");
            mark_failed();
            return -1;
        }
        _seq = (_seq + 1) & 0xFF;
        _frames_sent++;
    }
    return written;
}

int SerialTx::send_0310_redundant(const uint8_t* payload, int payload_len,
                                   int copies) {
    if (copies <= 0) copies = 1;
    int total = 0;
    for (int i = 0; i < copies; i++) {
        int written = send_0310(payload, payload_len);
        if (written < 0) return -1;
        total += written;
    }
    return total;
}
