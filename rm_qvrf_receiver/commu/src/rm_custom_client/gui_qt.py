from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

from PyQt5 import QtCore, QtGui, QtWidgets

from rm_custom_client.video_decoder import HevcPreviewDecoder
from rm_custom_client.video_receiver import VideoReceiver
from rm_custom_client.mqtt_receiver import DEFAULT_TOPICS, parse_payload
from rm_custom_client.serial_comm import (
    SerialComm,
    SerialFrameBuilder,
    VtxRemoteControl,
    KEY_NAMES,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dark theme (Catppuccin Mocha inspired)
# ---------------------------------------------------------------------------

DARK_QSS = """
QMainWindow { background-color: #1e1e2e; }
QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: "DejaVu Sans"; font-size: 13px; }
QMenu { background-color: #313244; border: 1px solid #45475a; }
QMenu::item:selected { background-color: #45475a; }
QTreeWidget { background-color: #181825; alternate-background-color: #1e1e2e;
              border: 1px solid #313244; outline: none; }
QTreeWidget::item { padding: 3px 0; }
QTreeWidget::item:selected { background-color: #45475a; color: #cdd6f4; }
QHeaderView::section { background-color: #313244; color: #cdd6f4;
                       border: none; padding: 4px 8px; font-weight: bold; }
QTextEdit { background-color: #181825; color: #cdd6f4; border: 1px solid #313244;
            font-family: "DejaVu Sans Mono"; font-size: 12px; }
QListWidget { background-color: #181825; color: #cdd6f4; border: 1px solid #313244;
              font-family: "DejaVu Sans Mono"; font-size: 11px; }
QListWidget::item { padding: 2px 4px; }
QListWidget::item:selected { background-color: #45475a; }
QLineEdit { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
            border-radius: 4px; padding: 6px; font-size: 13px; }
QPushButton { background-color: #45475a; color: #cdd6f4; border: none;
              border-radius: 4px; padding: 8px 16px; font-weight: bold; }
QPushButton:hover { background-color: #585b70; }
QPushButton:pressed { background-color: #313244; }
QStatusBar { background-color: #11111b; color: #a6adc8; }
QSplitter::handle { background-color: #313244; width: 2px; }
QGroupBox { border: 1px solid #313244; border-radius: 6px; margin-top: 12px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #bac2de; }
QTabWidget::pane { border: 1px solid #313244; background-color: #1e1e2e; }
QTabBar::tab { background-color: #313244; color: #a6adc8; padding: 6px 14px;
              border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background-color: #1e1e2e; color: #cdd6f4; }
QComboBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
            border-radius: 4px; padding: 4px 8px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background-color: #313244; selection-background-color: #45475a; }
"""


# ---------------------------------------------------------------------------
# Video bridge
# ---------------------------------------------------------------------------

class VideoBridge(QtCore.QObject):
    hevc_frame_ready = QtCore.pyqtSignal(bytes)

    def __call__(self, hevc_data: bytes) -> None:
        self.hevc_frame_ready.emit(hevc_data)


# ---------------------------------------------------------------------------
# QThread workers
# ---------------------------------------------------------------------------

class MqttWorker(QtCore.QThread):
    mqtt_message = QtCore.pyqtSignal(dict)
    mqtt_status = QtCore.pyqtSignal(bool, str)

    def __init__(self, host: str, port: int, client_id: str, subscribe_all: bool = False) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.subscribe_all = subscribe_all
        self._stop_event = threading.Event()

    def run(self) -> None:
        import paho.mqtt.client as mqtt
        from google.protobuf.message import DecodeError

        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
            protocol=mqtt.MQTTv311,
        )

        def on_connect(_client, _userdata, _flags, reason_code, _properties):
            connected = reason_code == 0
            self.mqtt_status.emit(connected, str(reason_code))
            if not connected:
                return
            topics = ["#"] if self.subscribe_all else list(DEFAULT_TOPICS)
            for t in topics:
                _client.subscribe(t, qos=1)

        def on_disconnect(_client, _userdata, _disconnect_flags, reason_code, _properties):
            self.mqtt_status.emit(False, str(reason_code))

        def on_message(_client, _userdata, msg):
            payload = bytes(msg.payload)
            try:
                parsed = parse_payload(msg.topic, payload)
                error = None
            except (DecodeError, ValueError) as exc:
                parsed = {"hex": payload.hex(" ", 1)}
                error = str(exc)
            self.mqtt_message.emit({
                "topic": msg.topic,
                "bytes": len(payload),
                "payload": parsed,
                "error": error,
                "time": time.strftime("%H:%M:%S"),
            })

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message

        try:
            client.connect(self.host, self.port, keepalive=30)
            client.loop_start()
            while not self._stop_event.is_set():
                time.sleep(0.3)
            client.loop_stop()
            client.disconnect()
        except OSError as exc:
            self.mqtt_status.emit(False, str(exc))

    def stop(self) -> None:
        self._stop_event.set()


class SerialWorker(QtCore.QThread):
    vtx_data_ready = QtCore.pyqtSignal(object)
    serial_status = QtCore.pyqtSignal(bool, str)
    serial_stats_updated = QtCore.pyqtSignal(object)

    def __init__(self, port: str, baudrate: int) -> None:
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._vtx_queue: queue.Queue[VtxRemoteControl] = queue.Queue(maxsize=300)
        self._write_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self._serial: SerialComm | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        self._serial = SerialComm(
            port=self.port,
            baudrate=self.baudrate,
            vtx_queue=self._vtx_queue,
        )
        if not self._serial.open():
            self.serial_status.emit(False, f"Cannot open {self.port}")
            return
        self.serial_status.emit(True, f"Connected {self.port}@{self.baudrate}")
        self._serial.start_reading()

        last_stats_time = time.time()
        while not self._stop_event.is_set():
            # Poll VTX data
            try:
                vtx = self._vtx_queue.get_nowait()
                self.vtx_data_ready.emit(vtx)
            except queue.Empty:
                pass

            # Send pending writes
            try:
                data = self._write_queue.get_nowait()
                if self._serial:
                    self._serial.send_raw(data)
            except queue.Empty:
                pass

            # Emit stats periodically
            now = time.time()
            if now - last_stats_time >= 1.0 and self._serial:
                self.serial_stats_updated.emit(self._serial.stats)
                last_stats_time = now

            time.sleep(0.01)

        if self._serial:
            self._serial.stop()
            self._serial.close()
        self.serial_status.emit(False, "Disconnected")

    def stop(self) -> None:
        self._stop_event.set()

    def send(self, data: bytes) -> None:
        try:
            self._write_queue.put_nowait(data)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------

class JoystickWidget(QtWidgets.QWidget):
    """Dual joystick visualization with QPainter."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ch = [1024, 1024, 1024, 1024]  # ch0..ch3
        self._min = 364
        self._max = 1684
        self.setMinimumSize(240, 160)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

    def set_channels(self, ch0: int, ch1: int, ch2: int, ch3: int) -> None:
        self._ch = [ch0, ch1, ch2, ch3]
        self.update()

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        mid_w = w // 2

        # Left joystick: ch2 (Y), ch3 (X)  -- left stick
        self._draw_joystick(p, QtCore.QRect(4, 4, mid_w - 12, h - 8),
                            self._ch[2], self._ch[3], "#f38ba8", "LEFT")
        # Right joystick: ch0 (X), ch1 (Y) -- right stick
        self._draw_joystick(p, QtCore.QRect(mid_w + 8, 4, mid_w - 12, h - 8),
                            self._ch[0], self._ch[1], "#89b4fa", "RIGHT")

    def _draw_joystick(self, p: QtGui.QPainter, rect: QtCore.QRect,
                       ch_y: int, ch_x: int, color: str, label: str) -> None:
        p.save()
        p.translate(rect.left(), rect.top())
        rw = rect.width()
        rh = rect.height()

        # Background circle
        cx = rw // 2
        cy = rh // 2
        radius = min(rw, rh) // 2 - 10
        p.setPen(QtGui.QPen(QtGui.QColor("#45475a"), 1))
        p.setBrush(QtGui.QColor("#181825"))
        p.drawEllipse(QtCore.QPoint(cx, cy), radius, radius)

        # Crosshair
        p.setPen(QtGui.QPen(QtGui.QColor("#45475a"), 1, QtCore.Qt.DashLine))
        p.drawLine(cx - radius, cy, cx + radius, cy)
        p.drawLine(cx, cy - radius, cx, radius + cy)

        # Stick position
        x_range = self._max - self._min
        nx = (ch_x - self._min) / x_range if x_range else 0.5
        ny = (ch_y - self._min) / x_range if x_range else 0.5
        dot_x = int(cx + (nx - 0.5) * 2 * radius)
        dot_y = int(cy + (ny - 0.5) * 2 * radius)

        p.setPen(QtGui.QPen(QtGui.QColor(color), 2))
        p.setBrush(QtGui.QColor(color))
        p.drawEllipse(QtCore.QPoint(dot_x, dot_y), 6, 6)

        # Label
        font = QtGui.QFont("DejaVu Sans", 9)
        p.setFont(font)
        p.setPen(QtGui.QColor("#bac2de"))
        label_rect = QtCore.QRect(0, rh - 16, rw, 14)
        p.drawText(label_rect, QtCore.Qt.AlignCenter, label)
        val_text = f"CH:{ch_x},{ch_y}"
        val_rect = QtCore.QRect(0, rh - 30, rw, 14)
        p.drawText(val_rect, QtCore.Qt.AlignCenter, val_text)

        p.restore()


class KeyboardDisplay(QtWidgets.QWidget):
    """Shows keyboard key states as colored indicators."""

    key_pressed = QtCore.pyqtSignal(int, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(3)
        layout.setContentsMargins(0, 0, 0, 0)
        self._labels: dict[str, QtWidgets.QLabel] = {}
        key_rows = [
            ["W", "S", "A", "D"],
            ["Shift", "Ctrl", "Q", "E"],
            ["R", "F", "G", "Z"],
            ["X", "C", "V", "B"],
        ]
        for row_idx, row_keys in enumerate(key_rows):
            for col_idx, name in enumerate(row_keys):
                lbl = QtWidgets.QLabel(name)
                lbl.setAlignment(QtCore.Qt.AlignCenter)
                lbl.setMinimumSize(36, 24)
                lbl.setMaximumSize(52, 28)
                lbl.setStyleSheet(
                    "background-color: #313244; color: #6c7086; "
                    "border-radius: 3px; padding: 2px 4px; font-size: 10px; font-weight: bold;"
                )
                layout.addWidget(lbl, row_idx, col_idx)
                self._labels[name] = lbl

    def update_keys(self, keyboard_value: int) -> None:
        for i, name in enumerate(KEY_NAMES):
            if name not in self._labels:
                continue
            pressed = bool(keyboard_value & (1 << i))
            lbl = self._labels[name]
            if pressed:
                lbl.setStyleSheet(
                    "background-color: #a6e3a1; color: #1e1e2e; "
                    "border-radius: 3px; padding: 2px 4px; font-size: 10px; font-weight: bold;"
                )
            else:
                lbl.setStyleSheet(
                    "background-color: #313244; color: #6c7086; "
                    "border-radius: 3px; padding: 2px 4px; font-size: 10px; font-weight: bold;"
                )


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class RmCustomClientQtWindow(QtWidgets.QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.setWindowTitle("红三实时监控")
        self.resize(1500, 900)
        self.setMinimumSize(1100, 700)

        # State
        self.topic_counts: Counter[str] = Counter()
        self.latest_payloads: dict[str, dict[str, Any]] = {}
        self.recent_messages: deque[dict[str, Any]] = deque(maxlen=300)
        self.vtx_data: VtxRemoteControl | None = None
        self.mqtt_msg_count = 0
        self.video_frame_count = 0
        self.video_pkt_count = 0
        self._last_mqtt_count = 0
        self._last_video_pkt = 0
        self._last_vtx_count = 0
        self._last_rate_at = time.time()
        self.mqtt_rate = 0.0
        self.video_rate = 0.0
        self.vtx_rate = 0.0

        # Workers
        self.mqtt_worker: MqttWorker | None = None
        self.serial_worker: SerialWorker | None = None
        self.video_bridge = VideoBridge()

        # Video
        self._video_queue: queue.Queue = queue.Queue(maxsize=3)
        self.decoder = HevcPreviewDecoder(self._video_queue)
        self.video_receiver = VideoReceiver(
            bind_host=args.bind,
            port=args.video_port,
            output=args.video_output,
            endian=args.endian,
            frame_callback=self.video_bridge,
        )

        self._build_ui()
        self._connect_signals()
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(DARK_QSS)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # --- Header ---
        header = QtWidgets.QWidget()
        header.setFixedHeight(36)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(6, 0, 6, 0)

        title = QtWidgets.QLabel("红三实时监控")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f5c2e7;")
        header_layout.addWidget(title)

        header_layout.addStretch()
        self.mqtt_badge = self._make_badge("MQTT")
        self.video_badge = self._make_badge("图传")
        self.serial_badge = self._make_badge("串口")
        self.msg_count_label = self._make_badge("消息 0")
        header_layout.addWidget(self.mqtt_badge)
        header_layout.addWidget(self.video_badge)
        header_layout.addWidget(self.serial_badge)
        header_layout.addWidget(self.msg_count_label)
        root.addWidget(header)

        # --- Main splitter ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        # --- Left panel ---
        left = QtWidgets.QWidget()
        left.setMinimumWidth(340)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)

        # Summary cards
        summary_group = QtWidgets.QGroupBox("比赛信息")
        summary_grid = QtWidgets.QGridLayout(summary_group)
        summary_grid.setSpacing(4)
        self.summary_labels: dict[str, tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = {}
        card_items = [
            ("stage", "阶段"), ("score", "比分"), ("time", "时间"),
            ("robot", "机器人"), ("health", "血量/热量"), ("position", "坐标"),
            ("module", "图传模块"), ("video", "UDP 图传"),
        ]
        for idx, (key, label) in enumerate(card_items):
            l = QtWidgets.QLabel(label)
            l.setStyleSheet("color: #6c7086; font-size: 11px;")
            v = QtWidgets.QLabel("--")
            v.setStyleSheet("color: #cdd6f4; font-size: 13px; font-weight: bold;")
            row = idx // 2
            col = (idx % 2) * 2
            summary_grid.addWidget(l, row, col)
            summary_grid.addWidget(v, row, col + 1)
            self.summary_labels[key] = (l, v)
        left_layout.addWidget(summary_group)

        # Topic tree
        topic_group = QtWidgets.QGroupBox("MQTT Topics")
        topic_layout = QtWidgets.QVBoxLayout(topic_group)
        self.topic_tree = QtWidgets.QTreeWidget()
        self.topic_tree.setHeaderLabels(["Topic", "次数"])
        self.topic_tree.setColumnWidth(0, 220)
        self.topic_tree.setColumnWidth(1, 60)
        self.topic_tree.setMaximumHeight(220)
        self.topic_tree.itemClicked.connect(self._on_topic_clicked)
        topic_layout.addWidget(self.topic_tree)
        left_layout.addWidget(topic_group)

        # Recent messages
        recent_group = QtWidgets.QGroupBox("最近消息")
        recent_layout = QtWidgets.QVBoxLayout(recent_group)
        self.recent_list = QtWidgets.QListWidget()
        self.recent_list.currentRowChanged.connect(self._on_recent_selected)
        recent_layout.addWidget(self.recent_list)
        left_layout.addWidget(recent_group)

        splitter.addWidget(left)

        # --- Center panel (video + detail) ---
        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(4, 0, 4, 0)
        center_layout.setSpacing(6)

        video_group = QtWidgets.QGroupBox("图传预览")
        video_layout = QtWidgets.QVBoxLayout(video_group)
        self.video_label = QtWidgets.QLabel("等待 UDP 3334 HEVC 数据...")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(400, 280)
        self.video_label.setStyleSheet(
            "background-color: #11111b; color: #6c7086; font-size: 16px; border-radius: 4px;"
        )
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        video_layout.addWidget(self.video_label)
        center_layout.addWidget(video_group, 1)

        detail_group = QtWidgets.QGroupBox("数据详情")
        detail_layout = QtWidgets.QVBoxLayout(detail_group)
        self.detail_text = QtWidgets.QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(200)
        detail_layout.addWidget(self.detail_text)
        center_layout.addWidget(detail_group)

        splitter.addWidget(center)

        # --- Right panel ---
        right = QtWidgets.QWidget()
        right.setMinimumWidth(280)
        right.setMaximumWidth(380)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(6)

        # Joystick
        joy_group = QtWidgets.QGroupBox("遥控摇杆")
        joy_layout = QtWidgets.QVBoxLayout(joy_group)
        self.joystick = JoystickWidget()
        self.joystick.setMinimumHeight(160)
        joy_layout.addWidget(self.joystick)
        right_layout.addWidget(joy_group)

        # Buttons & switches
        btn_group = QtWidgets.QGroupBox("按钮/开关状态")
        btn_layout = QtWidgets.QGridLayout(btn_group)
        btn_layout.setSpacing(4)
        self._btn_labels: dict[str, QtWidgets.QLabel] = {}
        btn_items = [
            ("mode", "模式"), ("pause", "暂停"), ("btn_l", "左按键"),
            ("btn_r", "右按键"), ("trig", "扳机"),
        ]
        for idx, (key, label) in enumerate(btn_items):
            l = QtWidgets.QLabel(label)
            l.setStyleSheet("color: #6c7086; font-size: 11px;")
            v = QtWidgets.QLabel("--")
            v.setStyleSheet("color: #cdd6f4; font-size: 13px; font-weight: bold;")
            btn_layout.addWidget(l, idx, 0)
            btn_layout.addWidget(v, idx, 1)
            self._btn_labels[key] = v
        right_layout.addWidget(btn_group)

        # Mouse display
        mouse_group = QtWidgets.QGroupBox("鼠标")
        mouse_layout = QtWidgets.QGridLayout(mouse_group)
        mouse_layout.setSpacing(4)
        self._mouse_labels: dict[str, QtWidgets.QLabel] = {}
        for idx, key in enumerate(["pos", "btn"]):
            l = QtWidgets.QLabel({"pos": "位置", "btn": "按键"}[key])
            l.setStyleSheet("color: #6c7086; font-size: 11px;")
            v = QtWidgets.QLabel("--")
            v.setStyleSheet("color: #cdd6f4; font-size: 12px; font-weight: bold;")
            mouse_layout.addWidget(l, idx, 0)
            mouse_layout.addWidget(v, idx, 1)
            self._mouse_labels[key] = v
        right_layout.addWidget(mouse_group)

        # Keyboard
        kb_group = QtWidgets.QGroupBox("键盘")
        kb_layout = QtWidgets.QVBoxLayout(kb_group)
        self.keyboard_display = KeyboardDisplay()
        kb_layout.addWidget(self.keyboard_display)
        right_layout.addWidget(kb_group)

        # Serial send panel
        send_group = QtWidgets.QGroupBox("串口发送 (0x0310)")
        send_layout = QtWidgets.QVBoxLayout(send_group)
        input_row = QtWidgets.QHBoxLayout()
        self.serial_input = QtWidgets.QLineEdit()
        self.serial_input.setPlaceholderText("输入数据 (text 或 hex)...")
        input_row.addWidget(self.serial_input, 1)
        self.send_btn = QtWidgets.QPushButton("发送")
        self.send_btn.setFixedWidth(70)
        self.send_btn.clicked.connect(self._on_send_serial)
        input_row.addWidget(self.send_btn)
        send_layout.addLayout(input_row)

        mode_row = QtWidgets.QHBoxLayout()
        self.send_mode = QtWidgets.QComboBox()
        self.send_mode.addItems(["文本 (UTF-8)", "十六进制"])
        mode_row.addWidget(self.send_mode)
        send_layout.addLayout(mode_row)
        self.send_status = QtWidgets.QLabel("")
        self.send_status.setStyleSheet("color: #a6adc8; font-size: 11px;")
        send_layout.addWidget(self.send_status)
        right_layout.addWidget(send_group)

        right_layout.addStretch()
        splitter.addWidget(right)

        splitter.setSizes([340, 780, 340])

        # --- Status bar ---
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self._status_mqtt = QtWidgets.QLabel("MQTT: --")
        self._status_video = QtWidgets.QLabel("视频: --")
        self._status_serial = QtWidgets.QLabel("串口: --")
        self.status_bar.addWidget(self._status_mqtt)
        self.status_bar.addWidget(self._status_video)
        self.status_bar.addWidget(self._status_serial)

    def _make_badge(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 2px 8px;")
        return lbl

    def _connect_signals(self) -> None:
        # Video bridge
        self.video_bridge.hevc_frame_ready.connect(self._on_hevc_frame)

        # Timers
        self._rate_timer = QtCore.QTimer()
        self._rate_timer.timeout.connect(self._refresh_rates)
        self._rate_timer.start(1000)

        self._video_timer = QtCore.QTimer()
        self._video_timer.timeout.connect(self._poll_video_frames)
        self._video_timer.start(33)

        self._summary_timer = QtCore.QTimer()
        self._summary_timer.timeout.connect(self._refresh_summary)
        self._summary_timer.start(500)

    def start(self) -> None:
        # Video
        self.video_receiver.start()

        # MQTT
        self.mqtt_worker = MqttWorker(
            host=self.args.host,
            port=self.args.mqtt_port,
            client_id=self.args.client_id,
            subscribe_all=self.args.subscribe_all,
        )
        self.mqtt_worker.mqtt_message.connect(self._on_mqtt_message)
        self.mqtt_worker.mqtt_status.connect(self._on_mqtt_status)
        self.mqtt_worker.start()

        # Serial
        if self.args.serial:
            self.serial_worker = SerialWorker(
                port=self.args.serial_port,
                baudrate=self.args.serial_baud,
            )
            self.serial_worker.vtx_data_ready.connect(self._on_vtx_data)
            self.serial_worker.serial_status.connect(self._on_serial_status)
            self.serial_worker.serial_stats_updated.connect(self._on_serial_stats)
            self.serial_worker.start()
            self.send_btn.setEnabled(True)
        else:
            self.send_btn.setEnabled(False)
            self._status_serial.setText("串口: 未启用")

    # ---- Slots ----

    def _on_mqtt_status(self, connected: bool, reason: str) -> None:
        if connected:
            self.mqtt_badge.setText("MQTT 已连接")
            self.mqtt_badge.setStyleSheet("color: #a6e3a1; font-size: 11px; font-weight: bold; padding: 2px 8px;")
            self._status_mqtt.setText(f"MQTT: 已连接 | {self.mqtt_rate:.1f}/s")
            self._status_mqtt.setStyleSheet("color: #a6e3a1;")
        else:
            self.mqtt_badge.setText("MQTT 断开")
            self.mqtt_badge.setStyleSheet("color: #f38ba8; font-size: 11px; font-weight: bold; padding: 2px 8px;")
            self._status_mqtt.setText(f"MQTT: 断开 ({reason})")
            self._status_mqtt.setStyleSheet("color: #f38ba8;")

    def _on_mqtt_message(self, event: dict) -> None:
        self.mqtt_msg_count += 1
        topic = event["topic"]
        self.topic_counts[topic] = self.topic_counts.get(topic, 0) + 1
        self.latest_payloads[topic] = event["payload"]
        self.recent_messages.append(event)
        self.msg_count_label.setText(f"消息 {self.mqtt_msg_count}")

    def _on_hevc_frame(self, hevc_data: bytes) -> None:
        self.video_frame_count += 1
        self.decoder.feed(hevc_data)

    def _on_vtx_data(self, vtx: VtxRemoteControl) -> None:
        self.vtx_data = vtx
        self.joystick.set_channels(vtx.channel_0, vtx.channel_1, vtx.channel_2, vtx.channel_3)

        # Button states
        mode_names = {0: "C", 1: "N", 2: "S"}
        self._btn_labels["mode"].setText(mode_names.get(vtx.mode_switch, str(vtx.mode_switch)))
        self._btn_labels["pause"].setText("按下" if vtx.pause else "释放")
        self._btn_labels["btn_l"].setText("按下" if vtx.button_left else "释放")
        self._btn_labels["btn_r"].setText("按下" if vtx.button_right else "释放")
        self._btn_labels["trig"].setText("按下" if vtx.trigger else "释放")

        for key, label in self._btn_labels.items():
            if key == "mode":
                continue
            val = label.text()
            if val == "按下":
                label.setStyleSheet("color: #a6e3a1; font-size: 13px; font-weight: bold;")
            else:
                label.setStyleSheet("color: #cdd6f4; font-size: 13px; font-weight: bold;")

        # Mouse
        self._mouse_labels["pos"].setText(
            f"X:{vtx.mouse_x:6d}  Y:{vtx.mouse_y:6d}  Z:{vtx.mouse_z:6d}"
        )
        mouse_btns = []
        if vtx.mouse_left:
            mouse_btns.append("左")
        if vtx.mouse_right:
            mouse_btns.append("右")
        if vtx.mouse_mid:
            mouse_btns.append("中")
        self._mouse_labels["btn"].setText(",".join(mouse_btns) if mouse_btns else "无")

        # Keyboard
        self.keyboard_display.update_keys(vtx.keyboard_value)

    def _on_serial_status(self, connected: bool, msg: str) -> None:
        if connected:
            self.serial_badge.setText("串口 已连接")
            self.serial_badge.setStyleSheet("color: #a6e3a1; font-size: 11px; font-weight: bold; padding: 2px 8px;")
        else:
            self.serial_badge.setText("串口 断开")
            self.serial_badge.setStyleSheet("color: #f38ba8; font-size: 11px; font-weight: bold; padding: 2px 8px;")

    def _on_serial_stats(self, stats) -> None:
        self._status_serial.setText(
            f"串口: RX {stats.frames_rx}fr/{stats.bytes_rx}B  TX {stats.frames_tx}fr/{stats.bytes_tx}B"
        )

    def _on_send_serial(self) -> None:
        if not self.serial_worker:
            self.send_status.setText("串口未启用")
            return
        text = self.serial_input.text()
        if not text:
            return
        if self.send_mode.currentIndex() == 0:
            data = text.encode("utf-8")
        else:
            try:
                data = bytes.fromhex(text.replace(" ", ""))
            except ValueError:
                self.send_status.setText("无效的十六进制")
                return

        if len(data) > 300:
            self.send_status.setText(f"数据过长: {len(data)} > 300 字节")
            return

        builder = SerialFrameBuilder()
        frame = builder.build_0310(data)
        self.serial_worker.send(frame)
        self.send_status.setText(
            f"已发送 0x0310: {len(data)}B → {data[:32].hex(' ')}{'...' if len(data) > 32 else ''}"
        )

    def _on_topic_clicked(self, item, _col) -> None:
        topic = item.text(0)
        payload = self.latest_payloads.get(topic, {})
        self._show_detail(topic, payload)

    def _on_recent_selected(self, row: int) -> None:
        if row < 0:
            return
        recent = list(self.recent_messages)[-80:]
        if row < len(recent):
            event = recent[row]
            self._show_detail(event["topic"], event["payload"])

    def _show_detail(self, topic: str, payload: dict) -> None:
        self.detail_text.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

    # ---- Periodic refresh ----

    def _poll_video_frames(self) -> None:
        self.video_pkt_count = self.video_receiver.stats.packets
        try:
            image = self._video_queue.get_nowait()
        except queue.Empty:
            image = None

        if image is not None:
            w = max(320, self.video_label.width())
            h = max(240, self.video_label.height())
            image.thumbnail((w, h))
            data = image.tobytes("raw", "RGB")
            qimage = QtGui.QImage(data, image.width, image.height, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.video_label.setPixmap(
                pixmap.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio,
                              QtCore.Qt.SmoothTransformation)
            )
        elif self.video_frame_count > 0 and self.decoder.frames_decoded > 0:
            pass  # keep last frame
        elif self.video_pkt_count == 0:
            self.video_label.setText("等待 UDP 3334 HEVC 数据...")
        elif self.decoder.waiting_for_keyframe:
            self.video_label.setText("等待关键帧解码...")

    def _refresh_rates(self) -> None:
        now = time.time()
        elapsed = now - self._last_rate_at
        if elapsed < 0.5:
            return
        self.mqtt_rate = (self.mqtt_msg_count - self._last_mqtt_count) / elapsed
        self.video_rate = (self.video_receiver.stats.packets - self._last_video_pkt) / elapsed
        self._last_mqtt_count = self.mqtt_msg_count
        self._last_video_pkt = self.video_receiver.stats.packets
        self._last_rate_at = now

        # MQTT badge
        if self.mqtt_worker:
            self._status_mqtt.setText(f"MQTT: {self.mqtt_rate:.1f} msg/s | {self.mqtt_msg_count} 总")
        # Video badge
        decoded = self.decoder.frames_decoded
        if decoded:
            self.video_badge.setText(f"图传 {self.video_rate:.1f} pkt/s 解码 {decoded}")
            self.video_badge.setStyleSheet("color: #a6e3a1; font-size: 11px; font-weight: bold; padding: 2px 8px;")
        elif self.video_pkt_count:
            self.video_badge.setText(f"图传 {self.video_rate:.1f} pkt/s 等待")
            self.video_badge.setStyleSheet("color: #f9e2af; font-size: 11px; font-weight: bold; padding: 2px 8px;")
        self._status_video.setText(f"视频: {self.video_pkt_count}pkt {self.video_frame_count}fr")

    def _refresh_summary(self) -> None:
        game = self.latest_payloads.get("GameStatus", {})
        dynamic = self.latest_payloads.get("RobotDynamicStatus", {})
        static = self.latest_payloads.get("RobotStaticStatus", {})
        module = self.latest_payloads.get("RobotModuleStatus", {})
        position = self.latest_payloads.get("RobotPosition", {})

        stage_names = {0: "未开始", 1: "准备", 2: "自检", 3: "倒计时", 4: "比赛中", 5: "结算"}
        stage = game.get("current_stage")
        stage_text = stage_names.get(stage, str(stage)) if stage is not None else "--"
        if game.get("is_paused"):
            stage_text += " 暂停"
        self._set_summary("stage", stage_text)
        self._set_summary("score", f"红 {game.get('red_score','--')} : 蓝 {game.get('blue_score','--')}")
        cd = game.get("stage_countdown_sec", "--")
        el = game.get("stage_elapsed_sec", "--")
        self._set_summary("time", f"剩 {cd}s / 已 {el}s")

        rid = static.get("robot_id", self.args.client_id)
        lv = static.get("level", "--")
        alive = static.get("alive_state", "--")
        self._set_summary("robot", f"ID {rid}  Lv {lv}  状态 {alive}")

        hp = dynamic.get("current_health", "--")
        heat = dynamic.get("current_heat", "--")
        ammo = dynamic.get("remaining_ammo", "--")
        self._set_summary("health", f"HP {hp}  热 {heat}  弹 {ammo}")

        x = position.get("x", "--")
        y = position.get("y", "--")
        yaw = position.get("yaw", "--")
        self._set_summary("position", f"x {self._fmt(x)}  y {self._fmt(y)}  yaw {self._fmt(yaw)}")

        vt = module.get("video_transmission")
        vt_text = {0: "离线", 1: "在线", 2: "安装异常"}.get(vt, "--")
        self._set_summary("module", vt_text)
        self._set_summary("video", f"{self.video_pkt_count}pkt {self.video_frame_count}fr 显示{self.decoder.frames_decoded}")

    def _set_summary(self, key: str, value: str) -> None:
        if key in self.summary_labels:
            _, vlabel = self.summary_labels[key]
            vlabel.setText(str(value))

    @staticmethod
    def _fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def closeEvent(self, event) -> None:
        self._rate_timer.stop()
        self._video_timer.stop()
        self._summary_timer.stop()
        if self.mqtt_worker:
            self.mqtt_worker.stop()
            self.mqtt_worker.wait(3000)
        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker.wait(3000)
        self.video_receiver.stop()
        self.video_receiver.join(timeout=2.0)
        self.decoder.stop()
        event.accept()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoboMaster custom client PyQt5 GUI")
    parser.add_argument("--host", default="192.168.12.1")
    parser.add_argument("--mqtt-port", type=int, default=3333)
    parser.add_argument("--video-port", type=int, default=3334)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--client-id", default="3")
    parser.add_argument("--subscribe-all", action="store_true")
    parser.add_argument("--video-output", type=Path, default=Path("captures/red3_gui.hevc"))
    parser.add_argument("--endian", choices=("little", "big"), default="big")
    parser.add_argument("--serial", action="store_true", help="enable VTX serial on /dev/ttyUSB0")
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--serial-baud", type=int, default=921600)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.video_output.exists() and args.video_output.is_dir():
        args.video_output = args.video_output / "red3_gui.hevc"
    elif str(args.video_output).endswith("/"):
        args.video_output = args.video_output / "red3_gui.hevc"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    qapp = QtWidgets.QApplication(sys.argv)
    qapp.setApplicationName("RoboMaster Custom Client")
    window = RmCustomClientQtWindow(args)
    window.show()
    window.start()
    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
