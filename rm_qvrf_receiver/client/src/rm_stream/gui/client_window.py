"""Main PyQt5 window — integrates MQTT, decoder, SR, and display panels.

Uses paho.mqtt directly (like the proven E2E test pattern) with a thread-safe
queue + QTimer poll to bridge MQTT callbacks into the Qt main thread.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
import os

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import paho.mqtt.client as mqtt

from rm_stream.frame_assembler import FrameAssembler
from rm_stream.protocol import CHUNK_SIZE, parse_chunk_header
from rm_stream.gui.console_widget import ConsoleWidget
from rm_stream.gui.competition_panel import CompetitionPanel
from rm_stream.gui.debug_panel import DebugPanel
from rm_stream.gui.decode_worker import DecodeWorker
from rm_stream.gui.video_panel import VideoPanel


def _format_last_missing(frame_id: int | None, missing: tuple[int, ...]) -> str:
    if frame_id is None or not missing:
        return "-"
    chunks = ",".join(f"C{i}" for i in missing)
    return f"f{frame_id}: {chunks}"


# MQTT v5 reason codes (paho VERSION2 callback API)
_MQTT_CONNECT_REASONS = {
    0: "Success",
    128: "Unspecified error",
    129: "Malformed packet",
    130: "Protocol error",
    131: "Implementation specific error",
    132: "Unsupported protocol version",
    133: "Client identifier not valid",
    134: "Bad username or password",
    135: "Not authorized",
    136: "Server unavailable",
    137: "Server busy",
    138: "Banned",
    140: "Bad authentication method",
    144: "Topic name invalid",
    149: "Packet too large",
    151: "Quota exceeded",
    153: "Payload format invalid",
    154: "Retain not supported",
    155: "QoS not supported",
    156: "Use another server",
    157: "Server moved",
    158: "Shared subscriptions not supported",
    159: "Connection rate exceeded",
    160: "Maximum connect time",
    161: "Subscription identifiers not supported",
    162: "Wildcard subscriptions not supported",
}


def _mqtt_connect_reason(rc) -> str:
    """Map paho ReasonCode (enum or int) to human-readable string."""
    try:
        rc_int = rc.value if hasattr(rc, 'value') else int(rc)
    except (TypeError, ValueError):
        return str(rc)
    return _MQTT_CONNECT_REASONS.get(rc_int, f"Unknown reason ({rc_int})")

# =========================================================================
# Dark theme -- Catppuccin Mocha palette
# =========================================================================

DARK_QSS = """
QMainWindow { background-color: #1e1e2e; }
QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: "DejaVu Sans"; font-size: 13px; }
QMenu { background-color: #313244; border: 1px solid #45475a; }
QMenu::item:selected { background-color: #45475a; }
QSplitter::handle { background-color: #313244; width: 2px; }
QGroupBox { border: 1px solid #313244; border-radius: 6px; margin-top: 12px; padding-top: 12px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #89b4fa; }
QLabel { color: #cdd6f4; }
QStatusBar { background-color: #11111b; color: #a6adc8; border-top: 1px solid #313244; }
QStatusBar::item { border: none; }
QPushButton { background-color: #45475a; color: #cdd6f4; border: none; border-radius: 4px; padding: 8px 16px; font-weight: bold; }
QPushButton:hover { background-color: #585b70; }
QPushButton:pressed { background-color: #313244; }
QLineEdit { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 6px; font-size: 13px; }
QTextEdit { background-color: #181825; color: #cdd6f4; border: 1px solid #313244; font-family: "DejaVu Sans Mono"; font-size: 12px; }
QListWidget { background-color: #181825; color: #cdd6f4; border: 1px solid #313244; font-family: "DejaVu Sans Mono"; font-size: 11px; }
QListWidget::item { padding: 2px 4px; }
QListWidget::item:selected { background-color: #45475a; }
QScrollBar:vertical { background-color: #1e1e2e; width: 10px; border: none; }
QScrollBar::handle:vertical { background-color: #45475a; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background-color: #1e1e2e; height: 10px; border: none; }
QScrollBar::handle:horizontal { background-color: #45475a; border-radius: 4px; min-width: 20px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
"""


class ClientWindow(QtWidgets.QMainWindow):
    """Main window integrating all GUI components for the RM custom client."""

    console_log_requested = QtCore.pyqtSignal(str, str)

    def __init__(
        self,
        mbt_ckpt: str,
        sr_ckpt: str,
        mqtt_host: str = "192.168.12.1",
        mqtt_port: int = 3333,
        client_id: str = "1",
        enable_sr: bool = False,
        sr_backend: str = "none",
        sr_scale: int = 2,
        realesr_model: str = "",
        rlfn_model: str = "",
        sr_engine: str = "torch",
        sr_trt_engine: str = "",
        sr_trt_device: int = 0,
        rx_gs_backend: str = "",
        rx_gs_trt_engine: str = "",
        rx_gs_trt_device: int = 0,
        rx_fused_sr_trt_engine: str = "",
        rx_fused_sr_trt_device: int = 0,
        codec_size: int = 128,
        display_size: int = 256,
        codec: str = "mbt",
        msssim_gain: float = 0.0,
        receive_mode: str = "mqtt",
        ipc_host: str = "127.0.0.1",
        ipc_port: int = 49031,
    ) -> None:
        super().__init__()

        # --- Window properties ---
        title = "RM Custom Client"
        if codec == "msssim_qvrf":
            title += " [MS-SSIM+QVRF]"
        if receive_mode == "ipc":
            title += " [IPC]"
        self.setWindowTitle(title)
        self.setMinimumSize(900, 500)
        self.resize(1100, 600)
        self.setStyleSheet(DARK_QSS)

        # --- Child widgets ---
        self.video_panel = VideoPanel(self)
        self.debug_panel = DebugPanel(self)
        self.competition_panel = CompetitionPanel(self)
        self.console = ConsoleWidget(self)
        self.console_log_requested.connect(self._append_console_log)

        # --- Central splitter ---
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(self.debug_panel)
        right_layout.addWidget(self.competition_panel)
        right_layout.addWidget(self.console, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.video_panel)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([770, 330])
        self.setCentralWidget(splitter)

        # --- Status bar ---
        self._status = QtWidgets.QStatusBar()
        self.setStatusBar(self._status)
        self._mqtt_label = QtWidgets.QLabel("MQTT: --")
        self._frame_label = QtWidgets.QLabel("Frames: 0")
        self._uptime_label = QtWidgets.QLabel("Uptime: 00:00")
        for lbl in (self._mqtt_label, self._frame_label, self._uptime_label):
            lbl.setStyleSheet("color: #a6adc8; padding: 0 8px;")
        self._status.addWidget(self._mqtt_label)
        self._status.addWidget(self._frame_label)
        self._status.addPermanentWidget(self._uptime_label)

        # --- Receive path ---
        self._receive_mode = receive_mode
        self._ipc_host = ipc_host
        self._ipc_port = ipc_port
        self._ipc_stop = threading.Event()
        self._ipc_thread: threading.Thread | None = None
        self._ipc_connected = False
        self._ipc_disconnects = 0

        # --- MQTT (paho directly, like proven E2E test) ---
        self._mqtt_host = mqtt_host
        self._mqtt_port = mqtt_port
        self._mqtt_queue: queue.Queue = queue.Queue(maxsize=5000)
        self._mqtt_connected = False
        self._mqtt_raw_count = 0
        self._mqtt_queue_drops = 0
        self._mqtt_disconnects = 0
        self._last_disconnect_reason = ""
        self._mqtt_client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        self._mqtt_client.on_connect = self._on_mqtt_connect
        self._mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self._mqtt_client.on_message = self._on_mqtt_message_raw
        self._rx_debug_log = os.environ.get("RM_STREAM_DEBUG_RX_LOG", "")
        self._rx_debug_chunks = os.environ.get("RM_STREAM_DEBUG_RX_CHUNKS", "0") == "1"
        self._rx_log_lock = threading.Lock()
        self._rx_log_file = None
        if self._rx_debug_log:
            try:
                self._rx_log_file = open(self._rx_debug_log, "a", encoding="utf-8", buffering=1)
            except Exception:
                self._rx_log_file = None
        self._e2e_trace = os.environ.get("RM_STREAM_E2E_TRACE", "0") == "1"
        self._frame_first_chunk_perf: dict[int, float] = {}
        self._frame_assembled_perf: dict[int, float] = {}

        # --- Frame assembler ---
        self._assembler = FrameAssembler(timeout_sec=0.5, require_start_frame=True)
        self._chunks_recv = 0
        self._mqtt_msg_count = 0
        self._ignored_custom_data = 0

        # --- Decode worker (QThread for heavy model loading + inference) ---
        self.decode_worker = DecodeWorker(
            mbt_ckpt,
            sr_ckpt,
            enable_sr=enable_sr,
            sr_backend=sr_backend,
            sr_scale=sr_scale,
            realesr_model_path=realesr_model,
            rlfn_model_path=rlfn_model,
            sr_engine=sr_engine,
            sr_trt_engine=sr_trt_engine,
            sr_trt_device=sr_trt_device,
            rx_gs_backend=rx_gs_backend,
            rx_gs_trt_engine=rx_gs_trt_engine,
            rx_gs_trt_device=rx_gs_trt_device,
            rx_fused_sr_trt_engine=rx_fused_sr_trt_engine,
            rx_fused_sr_trt_device=rx_fused_sr_trt_device,
            codec_size=codec_size,
            display_size=display_size,
            codec=codec,
            msssim_qvrf_gain=msssim_gain,
        )
        self._decode_thread = QtCore.QThread()
        self.decode_worker.moveToThread(self._decode_thread)
        self._decode_thread.started.connect(self.decode_worker.start)
        self.decode_worker.frame_ready.connect(self._on_frame_ready)
        self.decode_worker.frame_skipped.connect(self._on_frame_skipped)
        self.decode_worker.decode_error.connect(self._on_decode_error)

        # --- Stats ---
        self._frame_count = 0
        self._start_time = 0.0
        self._last_fps = 0.0
        self._fps_times: list[float] = []
        self._over_budget_frames = 0
        self._sample_frame_dir = os.environ.get("RM_STREAM_SAVE_SAMPLE_FRAMES", "").strip()
        self._sample_frames_saved = 0
        if self._sample_frame_dir:
            os.makedirs(self._sample_frame_dir, exist_ok=True)
        self._last_complete_time = 0.0
        self._last_chunk_time = 0.0

        # --- Poll timer: drain MQTT queue in main thread (every 5ms) ---
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_mqtt_queue)
        self._poll_timer.setInterval(5)
        self._recover_timer = QtCore.QTimer(self)
        self._recover_timer.timeout.connect(self._poll_recoverable_frames)
        self._recover_timer.setInterval(50)

        # --- Stats timer (500ms) ---
        self._stats_timer = QtCore.QTimer(self)
        self._stats_timer.timeout.connect(self._on_stats_tick)
        self._stats_timer.setInterval(500)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Connect receiver, show window, start all timers + threads."""
        if self._receive_mode == "ipc":
            self._start_ipc_server()
        else:
            self._mqtt_client.connect_async(self._mqtt_host, self._mqtt_port)
            self._mqtt_client.loop_start()

        self.show()
        self._decode_thread.start()
        self._poll_timer.start()
        self._recover_timer.start()
        self._stats_timer.start()
        self._start_time = time.time()
        self._log_console("Client started")
        return QtWidgets.QApplication.instance().exec()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Gracefully stop all workers, timers, and threads."""
        self._poll_timer.stop()
        self._recover_timer.stop()
        self._stats_timer.stop()

        try:
            if self._receive_mode == "ipc":
                self._stop_ipc_server()
            else:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
        except Exception:
            pass
        try:
            if self._rx_log_file is not None:
                self._rx_log_file.close()
                self._rx_log_file = None
        except Exception:
            pass

        try:
            self.decode_worker.stop()
            self._decode_thread.quit()
            self._decode_thread.wait(2000)
        except Exception:
            pass

        super().closeEvent(event)

    # ------------------------------------------------------------------
    # IPC receiver (local offline debug path)
    # ------------------------------------------------------------------

    def _append_console_log(self, message: str, level: str) -> None:
        self.console.log(message, level)

    def _log_console(self, message: str, level: str = "info") -> None:
        self.console_log_requested.emit(message, level)

    def _start_ipc_server(self) -> None:
        self._ipc_stop.clear()
        self._ipc_thread = threading.Thread(target=self._ipc_server_loop, name="rm-ipc-rx", daemon=True)
        self._ipc_thread.start()
        self._rx_log(f"IPC listen {self._ipc_host}:{self._ipc_port}")

    def _stop_ipc_server(self) -> None:
        self._ipc_stop.set()
        try:
            with socket.create_connection((self._ipc_host, self._ipc_port), timeout=0.2):
                pass
        except Exception:
            pass
        if self._ipc_thread is not None:
            self._ipc_thread.join(timeout=1.0)
        self._ipc_thread = None

    def _ipc_server_loop(self) -> None:
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self._ipc_host, self._ipc_port))
            srv.listen(1)
            srv.settimeout(0.5)
        except Exception as exc:
            self._rx_log(f"IPC listen failed {exc}")
            self._log_console(f"IPC listen failed: {exc}", "error")
            return

        with srv:
            while not self._ipc_stop.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                except Exception as exc:
                    if not self._ipc_stop.is_set():
                        self._rx_log(f"IPC accept failed {exc}")
                    continue
                self._ipc_connected = True
                self._rx_log(f"IPC connect {addr[0]}:{addr[1]}")
                with conn:
                    conn.settimeout(0.5)
                    buf = bytearray()
                    while not self._ipc_stop.is_set():
                        try:
                            data = conn.recv(4096)
                        except socket.timeout:
                            continue
                        except Exception as exc:
                            self._rx_log(f"IPC recv failed {exc}")
                            break
                        if not data:
                            break
                        buf.extend(data)
                        while len(buf) >= CHUNK_SIZE:
                            chunk = bytes(buf[:CHUNK_SIZE])
                            del buf[:CHUNK_SIZE]
                            self._process_ipc_chunk(chunk)
                self._ipc_connected = False
                self._ipc_disconnects += 1
                self._rx_log(f"IPC disconnect disconnects={self._ipc_disconnects}")

    def _process_ipc_chunk(self, chunk: bytes) -> None:
        header = parse_chunk_header(chunk)
        if header is None:
            self._rx_log(f"ipc_bad_chunk len={len(chunk)} head={chunk[:16].hex(' ')}")
            return
        t_chunk = time.perf_counter()
        self._chunks_recv += 1
        self._last_chunk_time = time.time()
        self._frame_first_chunk_perf.setdefault(header.frame_id, t_chunk)
        bitstream = self._assembler.add_chunk(chunk)
        if self._rx_debug_chunks or bitstream is not None:
            self._rx_log(
                f"ipc chunk frame={header.frame_id} cid={header.chunk_id}/{header.chunk_count} "
                f"plen={header.payload_len} flags={header.flags} assembled={bitstream is not None} "
                f"done={self._assembler.completed_frames} stale={self._assembler.dropped_stale} "
                f"session={self._assembler.dropped_session}"
            )
        if bitstream is not None:
            self._frame_assembled_perf[header.frame_id] = time.perf_counter()
            if self._e2e_trace:
                first = self._frame_first_chunk_perf.get(header.frame_id, t_chunk)
                self._rx_log(
                    f"E2E_RX_ASSEMBLE frame={header.frame_id} wall_ms={int(time.time() * 1000)} "
                    f"first_to_assembled_ms={(self._frame_assembled_perf[header.frame_id] - first) * 1000:.1f}"
                )
            self._last_complete_time = time.time()
            self.decode_worker.enqueue(header.frame_id, bitstream)

    # ------------------------------------------------------------------
    # MQTT callbacks (run in paho thread — use queue for thread safety)
    # ------------------------------------------------------------------

    def _on_mqtt_connect(self, client, userdata, flags, reason_code, properties):
        rc_int = reason_code.value if hasattr(reason_code, 'value') else int(reason_code)
        self._mqtt_connected = (rc_int == 0)
        client.subscribe("CustomByteBlock", qos=1)
        reason_str = _mqtt_connect_reason(reason_code)
        if rc_int == 0:
            self._rx_log(f"MQTT connect OK ({reason_str})")
            self._log_console(f"MQTT connected to {self._mqtt_host}:{self._mqtt_port}", "info")
        else:
            self._rx_log(f"MQTT connect FAILED rc={rc_int} ({reason_str})")
            cid = self._mqtt_client._client_id
            cid_str = cid.decode() if isinstance(cid, bytes) else str(cid)
            self._log_console(
                f"MQTT connect failed: {reason_str} (rc={rc_int}). "
                f"Check broker at {self._mqtt_host}:{self._mqtt_port}, "
                f"client_id='{cid_str}'",
                "error",
            )
            if rc_int == 133:  # Client identifier not valid
                self._log_console(
                    "Client ID rejected. Try a different --client-id (e.g. 'gui-2' or 'test-1'). "
                    "The broker may already have a client with this ID connected.",
                    "error",
                )

    def _on_mqtt_disconnect(self, client, userdata, flags, reason_code, properties):
        self._mqtt_connected = False
        self._mqtt_disconnects += 1
        rc_int = reason_code.value if hasattr(reason_code, 'value') else int(reason_code)
        reason_str = _mqtt_connect_reason(reason_code) if rc_int != 0 else "Clean disconnect"
        self._last_disconnect_reason = reason_str
        self._rx_log(f"MQTT disconnect rc={rc_int} ({reason_str}) disconnects={self._mqtt_disconnects}")
        if rc_int != 0:
            self._log_console(f"MQTT disconnected: {reason_str} (rc={rc_int})", "warn")

    def _on_mqtt_message_raw(self, client, userdata, msg):
        """Called from paho thread.

        Video chunks are parsed and assembled here so Qt UI stalls cannot delay
        chunk pairing. Non-video messages are queued for main-thread widgets.
        """
        self._mqtt_raw_count += 1
        if self._process_stream_message(msg.topic, bytes(msg.payload)):
            return
        try:
            self._mqtt_queue.put_nowait((msg.topic, msg.payload))
        except queue.Full:
            self._mqtt_queue_drops += 1
            if self._mqtt_queue_drops <= 5 or self._mqtt_queue_drops % 50 == 0:
                self._rx_log(
                    f"mqtt_queue_full drops={self._mqtt_queue_drops} raw={self._mqtt_raw_count}"
                )

    # ------------------------------------------------------------------
    # Main-thread MQTT message processing (polled by QTimer)
    # ------------------------------------------------------------------

    def _poll_mqtt_queue(self):
        """Drain the MQTT queue and process messages in the main thread."""
        for _ in range(20):  # Process up to 20 messages per tick
            try:
                topic, payload = self._mqtt_queue.get_nowait()
            except queue.Empty:
                break
            self._process_mqtt_message(topic, payload)

    def _process_mqtt_message(self, topic: str, payload: bytes):
        """Process one MQTT message in the main thread."""
        self._mqtt_msg_count += 1

        if self._mqtt_msg_count <= 3:
            self._log_console(f"MQTT: {topic} ({len(payload)}B)", "info")

        # Lazy-load protobuf on first call
        if not hasattr(self, '_pb'):
            try:
                from rm_custom_client.proto import rm_custom_client_pb2
                self._pb = rm_custom_client_pb2
            except ImportError:
                self._pb = None
                self._log_console("protobuf import failed", "warn")
                return

        if self._pb is None:
            return

        # Try CustomByteBlock
        try:
            cbb = self._pb.CustomByteBlock()
            cbb.ParseFromString(payload)
            data = bytes(cbb.data)
            chunk = self._extract_stream_chunk(data)             #  1
            if chunk is not None:
                header = parse_chunk_header(chunk)
                if header is not None:
                    self._chunks_recv += 1
                    bitstream = self._assembler.add_chunk(chunk) #  2
                    self._rx_log(
                        f"chunk frame={header.frame_id} cid={header.chunk_id}/{header.chunk_count} "
                        f"plen={header.payload_len} flags={header.flags} assembled={bitstream is not None} "
                        f"done={self._assembler.completed_frames} stale={self._assembler.dropped_stale} "
                        f"session={self._assembler.dropped_session}"
                    )
                    if bitstream is not None:
                        self.decode_worker.enqueue(header.frame_id, bitstream)
                    return
            elif data:
                self._ignored_custom_data += 1
                self._rx_log(f"custom data ignored len={len(data)} head={data[:16].hex(' ')}")
            self.competition_panel.update_custom_byte_block(cbb)
            return
        except Exception:
            pass

    def _poll_recoverable_frames(self) -> None:
        """Recover single-chunk frames whose placeholder chunk timed out."""
        bitstream = self._assembler.poll_recovered()
        if bitstream is not None:
            self._last_complete_time = time.time()
            self.decode_worker.enqueue(-1, bitstream)
            self._rx_log(
                f"frame_recovered_single done={self._assembler.completed_frames} "
                f"recovered={self._assembler.recovered_single_chunk}"
            )

        # Try GameStatus
        try:
            gs = self._pb.GameStatus()
            gs.ParseFromString(payload)
            if getattr(gs, 'game_time', 0) > 0:
                self.competition_panel.update_game_status(gs)
                return
        except Exception:
            pass

    def _process_stream_message(self, topic: str, payload: bytes) -> bool:
        """Fast video path. Returns True if the message was an R1V1 stream chunk."""
        if topic != "CustomByteBlock":
            return False
        try:
            from rm_custom_client.proto import rm_custom_client_pb2 as pb
            cbb = pb.CustomByteBlock()
            cbb.ParseFromString(payload)
            data = bytes(cbb.data)
        except Exception:
            return False

        chunk = self._extract_stream_chunk(data)
        if chunk is None:
            return False

        header = parse_chunk_header(chunk)
        if header is None:
            return False

        self._chunks_recv += 1
        self._last_chunk_time = time.time()
        bitstream = self._assembler.add_chunk(chunk)
        if self._rx_debug_chunks or bitstream is not None:
            self._rx_log(
                f"chunk frame={header.frame_id} cid={header.chunk_id}/{header.chunk_count} "
                f"plen={header.payload_len} flags={header.flags} assembled={bitstream is not None} "
                f"done={self._assembler.completed_frames} stale={self._assembler.dropped_stale} "
                f"session={self._assembler.dropped_session}"
            )
        if bitstream is not None:
            self._last_complete_time = time.time()
            self.decode_worker.enqueue(header.frame_id, bitstream)
        return True

    def _extract_stream_chunk(self, data: bytes) -> bytes | None:
        """Find one valid 300B R1V1 chunk inside CustomByteBlock.data."""
        if not data:
            return None
        for off in range(0, max(1, len(data) - 3)):
            if data[off:off + 4] != b"R1V1":
                continue
            chunk = data[off:off + CHUNK_SIZE]
            if len(chunk) != CHUNK_SIZE:
                self._rx_log(f"R1V1 offset={off} but only {len(chunk)}B available")
                continue
            header = parse_chunk_header(chunk)
            if header is not None:
                if off != 0 or len(data) != CHUNK_SIZE:
                    self._rx_log(f"extracted chunk offset={off} outer_data_len={len(data)}")
                return chunk
            self._rx_log(f"R1V1 offset={off} failed header parse len={len(data)}")
        return None

    def _rx_log(self, message: str) -> None:
        if self._rx_log_file is None:
            return
        try:
            with self._rx_log_lock:
                if self._rx_log_file is not None:
                    self._rx_log_file.write(f"{time.time():.3f} {message}\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Decode callbacks
    # ------------------------------------------------------------------

    def _on_frame_ready(self, rgb_image, stats_dict):    #  3
        """Handle a decoded + SR frame from the DecodeWorker."""
        t_display0 = time.perf_counter()
        self._frame_count += 1
        self._fps_times.append(time.perf_counter())
        if len(self._fps_times) > 60:
            self._fps_times.pop(0)
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            self._last_fps = (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0

        self.video_panel.show_frame(rgb_image)
        # self.video_panel.show_frame(rgb_image[:, :, ::-1].copy())
        if self._sample_frame_dir and self._sample_frames_saved < 5:
            import cv2

            sample_path = os.path.join(
                self._sample_frame_dir,
                f"frame_{self._frame_count:04d}.png",
            )
            cv2.imwrite(sample_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            self._sample_frames_saved += 1
        display_ms = (time.perf_counter() - t_display0) * 1000

        merged = dict(stats_dict) if stats_dict else {}
        merged.update({
            "fps": round(self._last_fps, 1),
            "chunk_count": self._chunks_recv,
            "display_ms": display_ms,
            "assembled_frames": self._assembler.completed_frames,
            "stale_drops": self._assembler.dropped_stale,
            "session_drops": self._assembler.dropped_session,
            "mqtt_raw": self._mqtt_raw_count,
            "mqtt_queue": self._mqtt_queue.qsize(),
            "mqtt_queue_drops": self._mqtt_queue_drops,
            "mqtt_disconnects": self._mqtt_disconnects,
            "valid_chunks": self._chunks_recv,
            "incomplete_frames": self._assembler.dropped_stale,
            "missing_chunk0": self._assembler.stale_missing_chunk0,
            "missing_chunk1": self._assembler.stale_missing_chunk1,
            "missing_chunk2": self._assembler.stale_missing_chunks.get(2, 0),
            "missing_chunk3": self._assembler.stale_missing_chunks.get(3, 0),
            "missing_chunks": dict(self._assembler.stale_missing_chunks),
            "last_missing": _format_last_missing(
                self._assembler.stale_last_frame_id,
                self._assembler.stale_last_missing,
            ),
            "pending_frames": self._assembler.pending_frames,
        })
        self.debug_panel.update(merged)
        self._frame_label.setText(f"Frames: {self._frame_count}")
        self._rx_log(
            f"frame_ready count={self._frame_count} fps={self._last_fps:.1f} "
            f"codec_ms={merged.get('codec_ms', 0):.1f} sr_ms={merged.get('sr_ms', 0):.1f} "
            f"display_ms={display_ms:.1f} sr_backend={merged.get('sr_backend', 'none')} "
            f"sr_scale={merged.get('sr_scale', 1)} output_shape={merged.get('output_shape', '-')} "
            f"codec_win_avg={merged.get('codec_win_avg_ms', 0):.1f} "
            f"codec_win_p90={merged.get('codec_win_p90_ms', 0):.1f} "
            f"codec_win_p99={merged.get('codec_win_p99_ms', 0):.1f} "
            f"sr_win_avg={merged.get('sr_win_avg_ms', 0):.1f} "
            f"sr_win_p90={merged.get('sr_win_p90_ms', 0):.1f} "
            f"sr_win_p99={merged.get('sr_win_p99_ms', 0):.1f} "
            f"decode_q_drop={merged.get('decode_queue_drops', 0)} "
            f"sr_q_drop={merged.get('sr_queue_drops', 0)}"
        )
        if self._e2e_trace:
            frame_id = merged.get("frame_id")
            assembled = self._frame_assembled_perf.pop(frame_id, None) if frame_id is not None else None
            first = self._frame_first_chunk_perf.pop(frame_id, None) if frame_id is not None else None
            now_perf = time.perf_counter()
            self._rx_log(
                f"E2E_RX_READY frame={frame_id} wall_ms={int(time.time() * 1000)} "
                f"first_to_ready_ms={((now_perf - first) * 1000 if first is not None else -1):.1f} "
                f"assembled_to_ready_ms={((now_perf - assembled) * 1000 if assembled is not None else -1):.1f} "
                f"codec_ms={merged.get('codec_ms', 0):.1f} sr_ms={merged.get('sr_ms', 0):.1f} "
                f"display_ms={display_ms:.1f}"
            )

    def _on_decode_error(self, msg: str):
        self._log_console(msg, "error")
        self._rx_log(f"decode_error {msg}")

    def _on_frame_skipped(self, stats_dict):
        """Handle over-budget marker. Keep the previous video frame visible."""
        self._over_budget_frames += 1
        merged = dict(stats_dict) if stats_dict else {}
        merged.update({
            "fps": round(self._last_fps, 1),
            "chunk_count": self._chunks_recv,
            "assembled_frames": self._assembler.completed_frames,
            "stale_drops": self._assembler.dropped_stale,
            "session_drops": self._assembler.dropped_session,
            "mqtt_raw": self._mqtt_raw_count,
            "mqtt_queue": self._mqtt_queue.qsize(),
            "mqtt_queue_drops": self._mqtt_queue_drops,
            "mqtt_disconnects": self._mqtt_disconnects,
            "valid_chunks": self._chunks_recv,
            "incomplete_frames": self._assembler.dropped_stale,
            "missing_chunk0": self._assembler.stale_missing_chunk0,
            "missing_chunk1": self._assembler.stale_missing_chunk1,
            "missing_chunk2": self._assembler.stale_missing_chunks.get(2, 0),
            "missing_chunk3": self._assembler.stale_missing_chunks.get(3, 0),
            "missing_chunks": dict(self._assembler.stale_missing_chunks),
            "last_missing": _format_last_missing(
                self._assembler.stale_last_frame_id,
                self._assembler.stale_last_missing,
            ),
            "pending_frames": self._assembler.pending_frames,
            "over_budget_frames": self._over_budget_frames,
        })
        self.debug_panel.update(merged)
        self._log_console(
            f"Over-budget frame {merged.get('over_budget_last', '?')}: "
            f"{merged.get('over_budget_bytes', '?')}/{merged.get('over_budget_max', '?')} B",
            "warn",
        )
        self._rx_log(f"frame_skipped {merged}")

    # ------------------------------------------------------------------
    # Periodic stats
    # ------------------------------------------------------------------

    def _on_stats_tick(self):
        """Update overlays every 500ms."""
        if self._mqtt_connected:
            self._mqtt_label.setText("MQTT: Connected")
            self._mqtt_label.setStyleSheet(
                "color: #a6e3a1; font-weight: bold; padding: 0 8px;"
            )
        elif self._receive_mode == "ipc" and self._ipc_connected:
            self._mqtt_label.setText("IPC: Connected")
            self._mqtt_label.setStyleSheet(
                "color: #a6e3a1; font-weight: bold; padding: 0 8px;"
            )
        elif self._receive_mode == "ipc":
            self._mqtt_label.setText("IPC: Waiting")
            self._mqtt_label.setStyleSheet(
                "color: #f9e2af; font-weight: bold; padding: 0 8px;"
            )
        else:
            reason = self._last_disconnect_reason or "Not connected"
            self._mqtt_label.setText(f"MQTT: {reason}")
            self._mqtt_label.setStyleSheet(
                "color: #f38ba8; font-weight: bold; padding: 0 8px;"
            )
        self.video_panel.update_overlay(self._last_fps, self._frame_count)
        now = time.time()
        seconds_since_complete = (
            now - self._last_complete_time if self._last_complete_time > 0 else 0.0
        )
        seconds_since_chunk = (
            now - self._last_chunk_time if self._last_chunk_time > 0 else 0.0
        )
        self.debug_panel.update({
            "mqtt_raw": self._mqtt_raw_count,
            "mqtt_queue": self._mqtt_queue.qsize(),
            "mqtt_queue_drops": self._mqtt_queue_drops,
            "mqtt_disconnects": self._mqtt_disconnects,
            "ipc_disconnects": self._ipc_disconnects,
            "valid_chunks": self._chunks_recv,
            "assembled_frames": self._assembler.completed_frames,
            "incomplete_frames": self._assembler.dropped_stale,
            "missing_chunk0": self._assembler.stale_missing_chunk0,
            "missing_chunk1": self._assembler.stale_missing_chunk1,
            "missing_chunk2": self._assembler.stale_missing_chunks.get(2, 0),
            "missing_chunk3": self._assembler.stale_missing_chunks.get(3, 0),
            "missing_chunks": dict(self._assembler.stale_missing_chunks),
            "last_missing": _format_last_missing(
                self._assembler.stale_last_frame_id,
                self._assembler.stale_last_missing,
            ),
            "pending_frames": self._assembler.pending_frames,
            "seconds_since_complete": seconds_since_complete,
            "seconds_since_chunk": seconds_since_chunk,
        })
        if (
            seconds_since_complete > 2.0
            and seconds_since_chunk < 1.0
            and self._assembler.dropped_stale > 0
        ):
            last_missing = _format_last_missing(
                self._assembler.stale_last_frame_id,
                self._assembler.stale_last_missing,
            )
            self._status.showMessage(
                f"Receiving chunks but no complete frame; last missing {last_missing}",
                1500,
            )

        if self._start_time > 0:
            secs = int(time.time() - self._start_time)
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            if h > 0:
                self._uptime_label.setText(f"Uptime: {h:02d}:{m:02d}:{s:02d}")
            else:
                self._uptime_label.setText(f"Uptime: {m:02d}:{s:02d}")

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------

    @staticmethod
    def launch(
        mbt_ckpt: str,
        sr_ckpt: str,
        mqtt_host: str = "192.168.12.1",
        mqtt_port: int = 3333,
        client_id: str = "1",
        enable_sr: bool = False,
        sr_backend: str = "none",
        sr_scale: int = 2,
        realesr_model: str = "",
        rlfn_model: str = "",
        sr_engine: str = "torch",
        sr_trt_engine: str = "",
        sr_trt_device: int = 0,
        rx_gs_backend: str = "",
        rx_gs_trt_engine: str = "",
        rx_gs_trt_device: int = 0,
        rx_fused_sr_trt_engine: str = "",
        rx_fused_sr_trt_device: int = 0,
        codec_size: int = 128,
        display_size: int = 256,
        codec: str = "mbt",
        msssim_gain: float = 0.0,
        receive_mode: str = "mqtt",
        ipc_host: str = "127.0.0.1",
        ipc_port: int = 49031,
    ) -> int:
        import sys as _sys
        from PyQt5.QtCore import QLibraryInfo

        pyqt_plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(pyqt_plugins, "platforms")
        os.environ["QT_PLUGIN_PATH"] = pyqt_plugins
        qapp = QtWidgets.QApplication(_sys.argv)
        qapp.setApplicationName("RM Custom Client")
        window = ClientWindow(
            mbt_ckpt=mbt_ckpt, sr_ckpt=sr_ckpt,
            mqtt_host=mqtt_host, mqtt_port=mqtt_port, client_id=client_id,
            enable_sr=enable_sr,
            sr_backend=sr_backend,
            sr_scale=sr_scale,
            realesr_model=realesr_model,
            rlfn_model=rlfn_model,
            sr_engine=sr_engine,
            sr_trt_engine=sr_trt_engine,
            sr_trt_device=sr_trt_device,
            rx_gs_backend=rx_gs_backend,
            rx_gs_trt_engine=rx_gs_trt_engine,
            rx_gs_trt_device=rx_gs_trt_device,
            rx_fused_sr_trt_engine=rx_fused_sr_trt_engine,
            rx_fused_sr_trt_device=rx_fused_sr_trt_device,
            codec_size=codec_size,
            display_size=display_size,
            codec=codec,
            msssim_gain=msssim_gain,
            receive_mode=receive_mode,
            ipc_host=ipc_host,
            ipc_port=ipc_port,
        )
        return window.run()
