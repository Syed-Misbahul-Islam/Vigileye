"""Video input.

For a webcam we read in a background thread and always hand out the newest
frame. If inference is slower than the camera, a queued reader would build
up latency until the alerts refer to something that happened seconds ago -
for a safety system, dropping stale frames is strictly correct.

For a video *file* we read synchronously so that no frames are skipped and
offline feature extraction is deterministic.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoStream:
    def __init__(self, source, width=640, height=480, fps=30, flip=False) -> None:
        self.is_file = isinstance(source, str) and Path(source).exists()
        self.flip = flip and not self.is_file

        self.cap = cv2.VideoCapture(source if self.is_file else int(source))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.frame_index = 0
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

        if not self.is_file:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            for _ in range(50):          # wait for the first frame
                if self._frame is not None:
                    break
                time.sleep(0.02)

    def _loop(self) -> None:
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Return ``(ok, frame_bgr, timestamp_seconds)``."""
        if self.is_file:
            ok, frame = self.cap.read()
            if not ok:
                return False, None, 0.0
            ts = self.frame_index / max(self.src_fps, 1.0)
            self.frame_index += 1
            return True, frame, ts

        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            return False, None, time.time()
        self.frame_index += 1
        return True, frame, time.time()

    def release(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()
