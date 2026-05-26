#pragma once
#include <cstdint>
#include <string>
#include <vector>

/// Serial transmitter for RoboMaster 0x0310 protocol over /dev/ttyUSB0.
///
/// Based on patterns from `commu/src/rm_custom_client/serial_comm.py`:
///   - 921600 baud 8N1
///   - 0x0310 frames with 300-byte fixed data field (VTX requires exactly 300B)
///   - CRC-8 (poly 0x31, init 0xFF) over SOF+len+seq
///   - CRC-16/MCRF4XX (poly 0x1021, init 0xFFFF, reflected) over full frame
///   - tcdrain after write for deterministic transmission (matches os.write+termios.tcdrain)
///   - Max 48 Hz send rate (50 Hz causes burst loss on 0x0310 link)
class SerialTx {
public:
    SerialTx(const std::string& port = "/dev/ttyUSB0", int baudrate = 921600);
    ~SerialTx();

    bool open();
    void close();
    bool is_open() const;
    const std::string& active_port() const { return _active_port; }

    /// Send raw bytes over serial. Returns bytes written or -1 on error.
    int send_raw(const uint8_t* data, int len);

    /// Build and send a 0x0310 frame (payload padded to 300 bytes if needed).
    /// Returns total bytes written (309) or -1 on error.
    /// Includes tcdrain() after write — see commu HANDOFF_0310.md.
    int send_0310(const uint8_t* payload, int payload_len);

    /// Send the same 0x0310 frame N times for lossy link redundancy.
    /// Returns total bytes written or -1 on first error.
    /// Use for critical frames (testing shows 0x0310 link is lossy).
    int send_0310_redundant(const uint8_t* payload, int payload_len, int copies);

    /// Drain the serial output buffer (call after batch of sends).
    void drain();

    /// Current sequence number
    uint8_t seq() const { return _seq; }

    /// Statistics
    uint64_t bytes_written() const { return _bytes_written; }
    uint64_t frames_sent()   const { return _frames_sent;   }

private:
    std::string _port;
    std::string _active_port;
    int _baudrate;
    int _fd = -1;
    uint8_t _seq = 0;
    uint64_t _bytes_written = 0;
    uint64_t _frames_sent = 0;

    uint8_t crc8(const uint8_t* data, int len);
    uint16_t crc16(const uint8_t* data, int len);
    bool open_path(const std::string& path);
    void mark_failed();
};
