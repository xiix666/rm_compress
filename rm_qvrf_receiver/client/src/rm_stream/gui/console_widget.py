"""Console widget — read-only, dark-themed, timestamped log output."""

from datetime import datetime

from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QFont


class ConsoleWidget(QTextEdit):
    """A read-only QTextEdit widget for timestamped log messages.

    Auto-scrolls to the bottom and keeps at most ``max_lines`` lines
    (trims from the top).  Supports ``info``, ``warn``, and ``error``
    severity levels with distinct colours.
    """

    _COLORS = {
        "info":  "#cccccc",
        "warn":  "#ffcc00",
        "error": "#ff4444",
    }

    def __init__(self, parent=None, max_lines=500):
        super().__init__(parent)
        self._max_lines = int(max_lines)
        self.setReadOnly(True)
        self.setFont(QFont("monospace", 10))
        self.setStyleSheet(
            "QTextEdit {"
            "  background-color: #16213e;"
            "  color: #cccccc;"
            "  border: 1px solid #1a1a2e;"
            "}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, message, level="info"):
        """Append a timestamped message with a colour based on *level*.

        Parameters
        ----------
        message : str
            The log text.
        level : str
            One of ``"info"``, ``"warn"``, or ``"error"``.
        """
        colour = self._COLORS.get(level, self._COLORS["info"])
        ts = datetime.now().strftime("%H:%M:%S")
        self.append(f'<span style="color:{colour};">[{ts}] {message}</span>')
        self._trim_if_needed()

    def log_error(self, message):
        """Shortcut for ``log(message, "error")``."""
        self.log(message, "error")

    def log_warning(self, message):
        """Shortcut for ``log(message, "warn")``."""
        self.log(message, "warn")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim_if_needed(self):
        """Remove oldest lines until we are at or below *max_lines*."""
        doc = self.document()
        while doc.blockCount() > self._max_lines:
            block = doc.findBlockByNumber(0)
            if block.isValid():
                cursor = self.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.movePosition(cursor.EndOfBlock, cursor.KeepAnchor)
                cursor.movePosition(cursor.NextBlock, cursor.KeepAnchor)
                cursor.removeSelectedText()
                cursor.deleteChar()  # remove trailing newline
                self.setTextCursor(cursor)
