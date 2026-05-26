from __future__ import annotations

import logging
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import serial

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CRC implementations per RoboMaster official protocol appendix
# ---------------------------------------------------------------------------

# CRC-8: poly=G(x)=x^8+x^5+x^4+1 (0x31), init=0xFF
# Table from official protocol appendix
_CRC8_TABLE: list[int] = [
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
]

CRC8_INIT = 0xFF


def crc8(data: bytes) -> int:
    crc = CRC8_INIT
    for byte in data:
        crc = _CRC8_TABLE[(crc ^ byte) & 0xFF]
    return crc


# CRC-16 for all frames (serial + VTX): MCRF4XX
# poly=0x1021, init=0xFFFF, refin=True, refout=True (same as official protocol appendix 1)


def _make_crc16_vtx_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x01:
                crc = (crc >> 1) ^ 0x8408  # reflected poly
            else:
                crc >>= 1
        table.append(crc)
    return table


_CRC16_VTX_TABLE = _make_crc16_vtx_table()


def crc16_vtx(data: bytes) -> int:
    """CRC-16/MCRF4XX per protocol appendix 1 — used for both VTX and serial frames."""
    crc = 0xFFFF
    for byte in data:
        crc = (crc >> 8) ^ _CRC16_VTX_TABLE[(crc ^ byte) & 0xFF]
    return crc


class CrcValidationError(ValueError):
    """Raised when CRC validation fails on a received frame."""
    pass


# ---------------------------------------------------------------------------
# VTX remote control frame (21 bytes, ~14ms interval)
# ---------------------------------------------------------------------------

VTX_FRAME_LENGTH = 21
VTX_HEADER = b'\xA9\x53'

KEY_NAMES = [
    "W", "S", "A", "D", "Shift", "Ctrl",
    "Q", "E", "R", "F", "G",
    "Z", "X", "C", "V", "B",
]


@dataclass
class VtxRemoteControl:
    channel_0: int       # right stick H, 11-bit, center 1024
    channel_1: int       # right stick V, 11-bit, center 1024
    channel_2: int       # left stick V, 11-bit, center 1024
    channel_3: int       # left stick H, 11-bit, center 1024
    mode_switch: int     # 2-bit (0=C, 1=N, 2=S)
    pause: bool
    button_left: bool
    button_right: bool
    dial: int            # 11-bit
    trigger: bool
    mouse_x: int         # 16-bit signed
    mouse_y: int         # 16-bit signed
    mouse_z: int         # 16-bit signed
    mouse_left: bool
    mouse_right: bool
    mouse_mid: bool
    keyboard_value: int  # 16-bit bitmask
    raw: bytes = field(repr=False)
    timestamp: float = 0.0

    @property
    def keyboard_states(self) -> dict[str, bool]:
        return {name: bool(self.keyboard_value & (1 << i)) for i, name in enumerate(KEY_NAMES)}

    @property
    def channels(self) -> tuple[int, int, int, int]:
        return (self.channel_0, self.channel_1, self.channel_2, self.channel_3)


class VtxFrameParser:
    """Parse 21-byte VTX remote control frames from the UART."""

    @staticmethod
    def parse(data: bytes) -> VtxRemoteControl | None:
        if len(data) != VTX_FRAME_LENGTH:
            return None
        if data[:2] != VTX_HEADER:
            return None

        # CRC16 over bytes 0..18, crc stored in bytes 19..20 (little-endian)
        expected_crc = data[19] | (data[20] << 8)
        computed_crc = crc16_vtx(data[:19])
        if expected_crc != computed_crc:
            return None

        return VtxFrameParser._extract_fields(data)

    @staticmethod
    def _extract_fields(data: bytes) -> VtxRemoteControl:
        # Build a bit buffer from bytes 2..18 (17 bytes = 136 bits of payload)
        payload = data[2:19]

        def bits(offset: int, width: int) -> int:
            value = 0
            for i in range(width):
                bit_pos = offset + i
                byte_idx = bit_pos // 8
                bit_idx = 7 - (bit_pos % 8)
                if byte_idx < len(payload):
                    if payload[byte_idx] & (1 << bit_idx):
                        value |= (1 << (width - 1 - i))
            return value

        def signed(value: int, width: int) -> int:
            if value & (1 << (width - 1)):
                return value - (1 << width)
            return value

        # Bit positions per official VTX manual (VT03&VT13 User Guide v1.0).
        # Frame: 2-byte header + 17-byte payload + 2-byte CRC16 = 21 bytes.
        # The uint64_t bitfield (channels+controls) uses 61 bits of 8 bytes,
        # leaving 3 bits padding. Mouse buttons (6 bits) live in a uint8_t,
        # leaving 2 bits padding before the keyboard field.
        return VtxRemoteControl(
            channel_0=bits(0, 11),
            channel_1=bits(11, 11),
            channel_2=bits(22, 11),
            channel_3=bits(33, 11),
            mode_switch=bits(44, 2),
            pause=bool(bits(46, 1)),
            button_left=bool(bits(47, 1)),
            button_right=bool(bits(48, 1)),
            dial=bits(49, 11),
            trigger=bool(bits(60, 1)),
            # 3 bits uint64_t padding (61–63)
            mouse_x=signed(bits(64, 16), 16),
            mouse_y=signed(bits(80, 16), 16),
            mouse_z=signed(bits(96, 16), 16),
            mouse_left=bool(bits(112, 2)),
            mouse_right=bool(bits(114, 2)),
            mouse_mid=bool(bits(116, 2)),
            # 2 bits uint8_t padding (118–119)
            keyboard_value=bits(120, 16),
            raw=data,
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
# RoboMaster serial command frames
# ---------------------------------------------------------------------------

SOF = 0xA5

CMD_ROBOT_TO_CUSTOM_CLIENT = 0x0310   # max 300 bytes payload
CMD_ROBOT_TO_CUSTOM_CONTROLLER = 0x0309  # max 30 bytes
CMD_CUSTOM_CLIENT_TO_ROBOT = 0x0311   # max 30 bytes

CMD_NAMES: dict[int, str] = {
    0x0310: "Robot→CustomClient",
    0x0309: "Robot→CustomController",
    0x0311: "CustomClient→Robot",
    0x0302: "CustomController→Robot",
}


class SerialFrameHeader(NamedTuple):
    sof: int          # 0xA5
    data_length: int  # payload length, not including cmd_id
    seq: int          # sequence number
    crc8_val: int     # CRC8 over SOF + data_length + seq


@dataclass
class SerialFrame:
    cmd_id: int
    data: bytes
    seq: int
    raw: bytes = field(repr=False)
    timestamp: float = 0.0


class SerialFrameBuilder:
    """Build RoboMaster serial command frames."""

    def __init__(self) -> None:
        self._seq: int = 0

    def build(self, cmd_id: int, data: bytes) -> bytes:
        """Build a complete serial frame.

        Frame structure per protocol section 1.1:
          frame_header: SOF(1B) + data_length(2B,little-endian) + seq(1B) + CRC8(1B)
          cmd_id(2B,little-endian) + data(nB) + CRC16(2B,little-endian)

        data_length = len(data)  (NOT including cmd_id)
        CRC8  computed over SOF(1) + data_length(2) + seq(1) = 4 bytes
        CRC16 computed over entire frame excluding the 2 CRC16 bytes
        """
        data_len = len(data)  # just the data payload, NOT including cmd_id
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFF

        # CRC8 over SOF(1) + data_length(2) + seq(1) = 4 bytes
        header_crc_input = bytes([SOF]) + struct.pack("<HB", data_len, seq)
        header_crc = crc8(header_crc_input)

        # prefix = SOF + data_length(2) + seq(1) + CRC8(1)
        prefix = header_crc_input + bytes([header_crc])
        body = struct.pack("<H", cmd_id) + data
        frame_crc = crc16_vtx(prefix + body)
        frame_tail = struct.pack("<H", frame_crc)

        return prefix + body + frame_tail

    def build_0310(self, payload: bytes, pad_to_max: bool = True) -> bytes:
        """Build a robot-to-custom-client frame.

        Live VTX-to-MQTT forwarding in this setup required a final 300-byte
        0x0310 data field. Keep the default padding for real sends.
        ``pad_to_max=False`` is only for local unit tests, historical probes,
        or explicit negative tests of non-forwarded lengths.
        """
        if len(payload) > 300:
            raise ValueError(f"0x0310 payload max 300 bytes, got {len(payload)}")
        if pad_to_max:
            payload = payload.ljust(300, b"\x00")
        return self.build(CMD_ROBOT_TO_CUSTOM_CLIENT, payload)

    def build_0311(self, payload: bytes) -> bytes:
        if len(payload) > 30:
            raise ValueError(f"0x0311 payload max 30 bytes, got {len(payload)}")
        return self.build(CMD_CUSTOM_CLIENT_TO_ROBOT, payload)


class SerialFrameParser:
    """Parse RoboMaster serial command frames from raw bytes."""

    HEADER_LENGTH = 5

    @staticmethod
    def parse(data: bytes) -> SerialFrame | None:
        if len(data) < 9:  # header(5) + cmd_id(2) + crc16(2)
            return None
        if data[0] != SOF:
            return None

        data_len = data[1] | (data[2] << 8)
        seq = data[3]
        crc8_val = data[4]

        # CRC8 over SOF(1) + data_length(2) + seq(1) = 4 bytes (per protocol appendix 1)
        expected_crc8 = crc8(data[:4])
        if crc8_val != expected_crc8:
            return None

        # frame = frame_header(5) + cmd_id(2) + data(data_len) + CRC16(2)
        total_len = 5 + 2 + data_len + 2
        if len(data) < total_len:
            return None

        expected_crc16 = data[total_len - 2] | (data[total_len - 1] << 8)  # little-endian
        computed_crc16 = crc16_vtx(data[:total_len - 2])
        if expected_crc16 != computed_crc16:
            return None

        cmd_id = data[5] | (data[6] << 8)
        payload = data[7:total_len - 2]

        return SerialFrame(
            cmd_id=cmd_id,
            data=payload,
            seq=seq,
            raw=data[:total_len],
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
# Serial communication manager
# ---------------------------------------------------------------------------


@dataclass
class SerialStats:
    bytes_rx: int = 0
    bytes_tx: int = 0
    frames_rx: int = 0
    frames_tx: int = 0
    crc_errors: int = 0
    parse_errors: int = 0
    first_rx_at: float | None = None
    last_rx_at: float | None = None
    connected: bool = False


class SerialComm:
    """Manages bidirectional UART communication with the VTX."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 921600,
        vtx_queue: queue.Queue[VtxRemoteControl] | None = None,
        frame_queue: queue.Queue[SerialFrame] | None = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.vtx_queue = vtx_queue or queue.Queue(maxsize=300)
        self.frame_queue = frame_queue or queue.Queue(maxsize=300)
        self.stats = SerialStats()
        self._ser: serial.Serial | None = None
        self._running = threading.Event()
        self._read_thread: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._frame_builder = SerialFrameBuilder()
        self._frame_parser = SerialFrameParser()
        self._vtx_parser = VtxFrameParser()

    def open(self) -> bool:
        if not os.access(self.port, os.R_OK | os.W_OK):
            LOGGER.error(
                "Cannot access %s. Run: sudo usermod -a -G dialout $USER and re-login.",
                self.port,
            )
            return False
        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
            )
        except serial.SerialException as exc:
            LOGGER.error("Failed to open %s: %s", self.port, exc)
            return False
        self.stats.connected = True
        LOGGER.info("Serial port %s opened at %d baud", self.port, self.baudrate)
        return True

    def close(self) -> None:
        self._running.set()
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            self._ser.close()
        self.stats.connected = False
        LOGGER.info("Serial port %s closed", self.port)

    def send_raw(self, data: bytes) -> int:
        if not self._ser or not self._ser.is_open:
            LOGGER.warning("Serial not open, cannot send")
            return 0
        with self._write_lock:
            written = self._ser.write(data)
            self.stats.bytes_tx += written
            self.stats.frames_tx += 1
            return written

    def send_frame(self, cmd_id: int, data: bytes) -> int:
        frame = self._frame_builder.build(cmd_id, data)
        return self.send_raw(frame)

    def start_reading(self) -> None:
        if self._read_thread and self._read_thread.is_alive():
            return
        self._running.clear()
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True, name="serial-reader")
        self._read_thread.start()
        LOGGER.info("Serial read thread started")

    def stop(self) -> None:
        self._running.set()
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _read_loop(self) -> None:
        if self._ser is None:
            return
        buf = bytearray()
        while not self._running.is_set():
            try:
                chunk = self._ser.read(128)
            except serial.SerialException as exc:
                LOGGER.error("Serial read error: %s", exc)
                self.stats.connected = False
                break
            if not chunk:
                continue

            self.stats.bytes_rx += len(chunk)
            now = time.time()
            if self.stats.first_rx_at is None:
                self.stats.first_rx_at = now
            self.stats.last_rx_at = now

            buf.extend(chunk)

            # Process frames: VTX remote control (0xA9 0x53) and serial command (0xA5)
            while len(buf) >= 2:
                byte0, byte1 = buf[0], buf[1]

                # --- VTX remote control frame (21 bytes, header 0xA9 0x53) ---
                if byte0 == VTX_HEADER[0] and byte1 == VTX_HEADER[1]:
                    if len(buf) < VTX_FRAME_LENGTH:
                        break
                    frame = buf[:VTX_FRAME_LENGTH]
                    buf = buf[VTX_FRAME_LENGTH:]
                    vtx = self._vtx_parser.parse(bytes(frame))
                    if vtx is not None:
                        self.stats.frames_rx += 1
                        try:
                            self.vtx_queue.put_nowait(vtx)
                        except queue.Full:
                            pass
                    else:
                        self.stats.crc_errors += 1
                    continue

                # --- RoboMaster serial command frame (header 0xA5, variable length) ---
                if byte0 == SOF and len(buf) >= SerialFrameParser.HEADER_LENGTH:
                    data_len = buf[1] | (buf[2] << 8)
                    total_len = SerialFrameParser.HEADER_LENGTH + 2 + data_len + 2
                    if total_len <= SerialFrameParser.HEADER_LENGTH + 2 + 2:
                        # Invalid: total_len must be at least header + cmd_id + crc16
                        buf = buf[1:]
                        self.stats.parse_errors += 1
                        continue
                    if len(buf) < total_len:
                        break
                    frame_data = bytes(buf[:total_len])
                    rm_frame = SerialFrameParser.parse(frame_data)
                    if rm_frame is not None:
                        buf = buf[total_len:]
                        self.stats.frames_rx += 1
                        try:
                            self.frame_queue.put_nowait(rm_frame)
                        except queue.Full:
                            pass
                    else:
                        # CRC failure — skip SOF byte and rescan
                        buf = buf[1:]
                        self.stats.crc_errors += 1
                    continue

                # Neither header matches at position 0 — skip this byte
                buf = buf[1:]

    def build_and_send_0310(self, payload: bytes) -> int:
        frame = self._frame_builder.build_0310(payload, pad_to_max=True)
        return self.send_raw(frame)
