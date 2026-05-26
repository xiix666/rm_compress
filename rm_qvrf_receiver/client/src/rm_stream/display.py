"""OpenCV display window with FPS counter."""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

WINDOW_NAME = "RM Stream (M0)"


class Display:
    """OpenCV window showing 256×256 RGB frames with FPS overlay."""

    def __init__(self, window_name: str = WINDOW_NAME) -> None:
        self._window = window_name
        self._frame_times: deque[float] = deque(maxlen=60)
        self._frame_count = 0
        self._start_time = time.time()
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

    def show(self, frame_rgb: np.ndarray) -> bool:
        """Display a frame. Returns False if window was closed."""
        self._frame_count += 1
        now = time.time()
        self._frame_times.append(now)

        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        fps = self._compute_fps()
        cv2.putText(
            bgr, f"FPS: {fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.putText(
            bgr, f"Frame: {self._frame_count}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )

        cv2.imshow(self._window, bgr)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q") and key != 27

    def _compute_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / elapsed

    def close(self) -> None:
        cv2.destroyWindow(self._window)

    @property
    def frame_count(self) -> int:
        return self._frame_count
