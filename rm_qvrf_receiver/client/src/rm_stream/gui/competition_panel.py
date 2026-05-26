"""Competition telemetry panel for RoboMaster MQTT protobuf messages."""

from __future__ import annotations

from PyQt5.QtWidgets import QGroupBox, QFormLayout, QLabel, QWidget

# ---------------------------------------------------------------------------
# Game state constants (per RoboMaster protocol)
# ---------------------------------------------------------------------------
_GAME_STATE_TEXT: dict[int, str] = {
    1: "Running",
    2: "Pause",
    3: "End",
    4: "Pre-match",
}


class CompetitionPanel(QGroupBox):
    """Displays RoboMaster competition telemetry from MQTT protobuf data.

    Protobuf imports are lazy so the panel works even when the commu
    package is not installed.
    """

    MAX_HP = 2000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Match Info", parent)

        self._labels: dict[str, QLabel] = {}
        self._data_bytes_received: int = 0

        self._setup_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_game_status(self, msg) -> None:
        """Parse a GameStatus protobuf message and update the display.

        Args:
            msg: Protobuf object with ``game_time`` (int, seconds) and
                 ``game_state`` (int).
        """
        try:
            game_time = getattr(msg, "game_time", None)
            if game_time is not None:
                minutes = int(game_time) // 60
                seconds = int(game_time) % 60
                self._set_label("game_time", f"{minutes}:{seconds:02d}")

            game_state = getattr(msg, "game_state", None)
            if game_state is not None:
                text = _GAME_STATE_TEXT.get(int(game_state), f"Unknown ({game_state})")
                self._set_label("status", text)
        except Exception:
            pass  # Silently ignore parse failures

    def update_robot_status(self, msg) -> None:
        """Parse a RobotDynamicStatus protobuf message and update the display.

        Args:
            msg: Protobuf object with ``robot_hp`` (int) and optionally
                 ``ammo`` (int).
        """
        try:
            robot_hp = getattr(msg, "robot_hp", None)
            if robot_hp is not None:
                self._set_label("robot_hp", f"{int(robot_hp)}/{self.MAX_HP}")

            ammo = getattr(msg, "ammo", None)
            if ammo is not None:
                self._set_label("ammo", str(int(ammo)))
        except Exception:
            pass  # Silently ignore parse failures

    def update_custom_byte_block(self, msg) -> None:
        """Handle a CustomByteBlock protobuf message.

        Args:
            msg: Protobuf object with optional ``data`` (bytes-like).
        """
        try:
            data = getattr(msg, "data", None)
            if data is not None:
                self._data_bytes_received += len(data)
                self._set_label("data", f"Data: {self._data_bytes_received} bytes")
        except Exception:
            pass  # Silently ignore parse failures

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _FIELD_SPEC: list[tuple[str, str]] = [
        ("game_time", "Game Time"),
        ("robot_hp", "Robot HP"),
        ("ammo", "Ammo"),
        ("status", "Status"),
        ("data", "Custom Data"),
    ]

    _STYLE = """
        QGroupBox {
            color: #e0e0e0;
            background-color: #16213e;
            border: 1px solid #2a2a4a;
            border-radius: 4px;
            margin-top: 1em;
            padding-top: 0.5em;
            font-size: 10pt;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QLabel {
            color: #c0c0c0;
            background-color: transparent;
            font-size: 9pt;
            font-family: monospace;
        }
    """

    def _setup_ui(self) -> None:
        self.setStyleSheet(self._STYLE)

        layout = QFormLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 14, 10, 8)

        for key, display_name in self._FIELD_SPEC:
            label = QLabel("—")
            label.setTextInteractionFlags(label.textInteractionFlags())
            layout.addRow(f"{display_name}:", label)
            self._labels[key] = label

    def _set_label(self, key: str, text: str) -> None:
        if key in self._labels:
            self._labels[key].setText(text)
