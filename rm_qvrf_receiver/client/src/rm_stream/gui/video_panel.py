"""Video display panel with FPS overlay."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap, QColor, QFont
from PyQt5.QtWidgets import QWidget, QLabel, QScrollArea, QVBoxLayout


class VideoPanel(QWidget):
    """Displays a 256x256 RGB video frame with an FPS overlay.

    Caches the last QPixmap to avoid flicker during repaint events.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._last_pixmap: QPixmap | None = None
        self._overlay_fps: float = 0.0
        self._overlay_frame_count: int = 0

        # Scroll area with dark background
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setBackgroundRole(self._scroll_area.palette().Dark)
        self._scroll_area.setStyleSheet("QScrollArea { background-color: #1a1a2e; border: none; }")
        self._scroll_area.setWidgetResizable(True)

        # Image label
        self._label = QLabel("Waiting for video...")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            "QLabel { color: #aaaaaa; background-color: #1a1a2e; font-size: 12pt; }"
        )
        self._label.setMinimumSize(256, 256)
        self._scroll_area.setWidget(self._label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll_area)

        self.setMinimumSize(280, 280)

    def show_frame(self, rgb_image: np.ndarray) -> None:
        """Display a numpy uint8 RGB (256,256,3) frame in the panel.

        Args:
            rgb_image: numpy uint8 array of shape (256, 256, 3) in RGB order.
        """
        # Ensure C-contiguous for QImage
        rgb = np.ascontiguousarray(rgb_image)
        h, w, c = rgb.shape
        bytes_per_line = c * w
        qimage = QImage(rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        self._last_pixmap = QPixmap.fromImage(qimage)
        self._redraw_pixmap()
        self.update()

    def update_overlay(self, fps: float, frame_count: int) -> None:
        """Store overlay text for the next paint event.

        Args:
            fps: Current frames per second.
            frame_count: Total frames received.
        """
        self._overlay_fps = fps
        self._overlay_frame_count = frame_count
        self.update()

    def clear(self) -> None:
        """Remove the current pixmap and show a placeholder."""
        self._last_pixmap = None
        self._label.clear()
        self._label.setText("Waiting for video...")
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Rescale the current pixmap to fit the new label size."""
        super().resizeEvent(event)
        self._redraw_pixmap()

    def paintEvent(self, event) -> None:  # noqa: N802
        """Draw the FPS overlay in the top-left corner."""
        super().paintEvent(event)

        if self._overlay_frame_count <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        text = f"FPS: {self._overlay_fps:.1f} | Frame: {self._overlay_frame_count}"

        font = QFont("monospace", 10)
        font.setBold(True)
        painter.setFont(font)

        fm = painter.fontMetrics()
        text_rect = fm.boundingRect(text)
        padding = 6

        bg_rect = text_rect.translated(8, 8)
        bg_rect.adjust(-padding, -padding, padding, padding)

        painter.fillRect(bg_rect, QColor(0, 0, 0, 180))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(8 + padding, 8 + padding + fm.ascent(), text)

        painter.end()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _redraw_pixmap(self) -> None:
        """Scale the cached pixmap to fit the label and apply it."""
        if self._last_pixmap is None:
            return
        label_size = self._label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return
        scaled = self._last_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        self._label.setPixmap(scaled)
