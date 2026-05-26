"""Debug statistics panel for compression/transport metrics."""

from __future__ import annotations

from PyQt5.QtWidgets import QGroupBox, QFormLayout, QLabel, QWidget


class DebugPanel(QGroupBox):
    """Shows real-time compression and transport statistics in a form layout."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Debug", parent)

        self._labels: dict[str, QLabel] = {}

        self._setup_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, stats: dict) -> None:  # noqa: A003
        """Update all stat labels from a dictionary.

        Args:
            stats: May contain keys: fps, bpp, bitstream_bytes, chunk_count,
                   total_chunks, decode_ms, codec_ms, sr_ms, display_ms,
                   assembled_frames, stale_drops, session_drops, beta, delivery_pct.
                   Missing keys display as "—".
        """
        if "fps" in stats:
            self._set_label("fps", f"{stats['fps']:.1f}")
        if "bpp" in stats:
            self._set_label("bpp", f"{stats['bpp']:.4f}")
        if "bitstream_bytes" in stats:
            self._set_label("bitstream", f"{int(stats['bitstream_bytes'])} B")
        if "chunk_count" in stats and "total_chunks" in stats:
            text = f"{stats['chunk_count']}/{stats['total_chunks']}"
            self._set_label("chunks", text)
        if "decode_ms" in stats:
            self._set_label("decode", f"{int(stats['decode_ms'])} ms")
        if "codec_ms" in stats:
            self._set_label("codec", f"{int(stats['codec_ms'])} ms")
        if "sr_ms" in stats:
            self._set_label("sr", f"{int(stats['sr_ms'])} ms")
        if "display_ms" in stats:
            self._set_label("display", f"{int(stats['display_ms'])} ms")
        if "assembled_frames" in stats:
            self._set_label("assembled", f"{int(stats['assembled_frames'])}")
        if "stale_drops" in stats:
            self._set_label("stale_drops", f"{int(stats['stale_drops'])}")
        if "session_drops" in stats:
            self._set_label("session_drops", f"{int(stats['session_drops'])}")
        if "mqtt_raw" in stats:
            self._set_label("mqtt_raw", f"{int(stats['mqtt_raw'])}")
        if "mqtt_queue" in stats:
            self._set_label("mqtt_queue", f"{int(stats['mqtt_queue'])}")
        if "mqtt_queue_drops" in stats:
            self._set_label("mqtt_queue_drops", f"{int(stats['mqtt_queue_drops'])}")
        if "mqtt_disconnects" in stats:
            self._set_label("mqtt_disconnects", f"{int(stats['mqtt_disconnects'])}")
        if "valid_chunks" in stats:
            self._set_label("valid_chunks", f"{int(stats['valid_chunks'])}")
        if "incomplete_frames" in stats:
            self._set_label("incomplete_frames", f"{int(stats['incomplete_frames'])}")
        if "missing_chunk0" in stats:
            self._set_label("missing_chunk0", f"{int(stats['missing_chunk0'])}")
        if "missing_chunk1" in stats:
            self._set_label("missing_chunk1", f"{int(stats['missing_chunk1'])}")
        if "missing_chunk2" in stats:
            self._set_label("missing_chunk2", f"{int(stats['missing_chunk2'])}")
        if "missing_chunk3" in stats:
            self._set_label("missing_chunk3", f"{int(stats['missing_chunk3'])}")
        if "last_missing" in stats:
            self._set_label("last_missing", str(stats["last_missing"]))
        if "pending_frames" in stats:
            self._set_label("pending_frames", f"{int(stats['pending_frames'])}")
        if "seconds_since_complete" in stats:
            self._set_label("since_complete", f"{stats['seconds_since_complete']:.1f} s")
        if "seconds_since_chunk" in stats:
            self._set_label("since_chunk", f"{stats['seconds_since_chunk']:.1f} s")
        if "beta" in stats:
            self._set_label("rc_beta", f"{stats['beta']:.2f}")
        if "delivery_pct" in stats:
            self._set_label("delivery", f"{stats['delivery_pct']:.1f}%")
        if "over_budget_frames" in stats:
            self._set_label("over_budget", f"{int(stats['over_budget_frames'])}")
        if "over_budget_last" in stats:
            self._set_label("over_budget_last", f"{int(stats['over_budget_last'])}")
        if "over_budget_bytes" in stats and "over_budget_max" in stats:
            self._set_label("over_budget_bytes",
                            f"{int(stats['over_budget_bytes'])}/{int(stats['over_budget_max'])} B")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _FIELD_SPEC = [
        ("fps", "FPS"),
        ("bpp", "BPP"),
        ("bitstream", "Bitstream"),
        ("chunks", "Chunks"),
        ("decode", "Decode"),
        ("codec", "Codec"),
        ("sr", "SR"),
        ("display", "Display"),
        ("assembled", "Assembled"),
        ("stale_drops", "Stale Drops"),
        ("session_drops", "Session Drops"),
        ("mqtt_raw", "MQTT Raw"),
        ("mqtt_queue", "MQTT Queue"),
        ("mqtt_queue_drops", "MQTT Q Drops"),
        ("mqtt_disconnects", "MQTT Disc"),
        ("valid_chunks", "Valid Chunks"),
        ("incomplete_frames", "Incomplete"),
        ("missing_chunk0", "Missing C0"),
        ("missing_chunk1", "Missing C1"),
        ("missing_chunk2", "Missing C2"),
        ("missing_chunk3", "Missing C3"),
        ("last_missing", "Last Missing"),
        ("pending_frames", "Pending"),
        ("since_complete", "Since Frame"),
        ("since_chunk", "Since Chunk"),
        ("rc_beta", "RC Beta"),
        ("delivery", "Delivery"),
        ("over_budget", "Over Budget"),
        ("over_budget_last", "Over Last"),
        ("over_budget_bytes", "Over Bytes"),
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
