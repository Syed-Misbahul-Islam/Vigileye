"""Eye and mouth metrics: EAR, MAR, PERCLOS, blinks, micro-sleeps, yawns.

All classes here are pure Python + NumPy and take explicit timestamps, so
they behave identically on a live camera and on an offline video decoded
faster than real time. That property is what makes the training pipeline
(``scripts/collect_features.py``) produce features consistent with what
the live system sees.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Sequence, Tuple

import numpy as np

from .landmarks import (
    LEFT_EYE_EAR,
    MOUTH_CORNERS,
    MOUTH_VERTICAL,
    RIGHT_EYE_EAR,
)

EPS = 1e-6


# --------------------------------------------------------------------------
# Instantaneous geometric ratios
# --------------------------------------------------------------------------
def eye_aspect_ratio(points: np.ndarray, idx: Sequence[int]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|).

    Scale-invariant: the denominator is the eye width, so the value does not
    change when the driver leans towards or away from the camera.
    """
    p1, p2, p3, p4, p5, p6 = (points[i, :2] for i in idx)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    return float(vertical / (horizontal + EPS))


def both_eyes_ear(points: np.ndarray) -> Tuple[float, float, float]:
    """Return ``(mean_ear, left_ear, right_ear)``."""
    left = eye_aspect_ratio(points, LEFT_EYE_EAR)
    right = eye_aspect_ratio(points, RIGHT_EYE_EAR)
    return (left + right) / 2.0, left, right


def mouth_aspect_ratio(points: np.ndarray) -> float:
    """MAR averaged over three vertical lip pairs, normalised by mouth width."""
    vertical = 0.0
    for top, bottom in MOUTH_VERTICAL:
        vertical += float(np.linalg.norm(points[top, :2] - points[bottom, :2]))
    left, right = MOUTH_CORNERS
    width = float(np.linalg.norm(points[left, :2] - points[right, :2]))
    return vertical / (3.0 * width + EPS)


# --------------------------------------------------------------------------
# PERCLOS
# --------------------------------------------------------------------------
class PerclosTracker:
    """P80 PERCLOS: fraction of a rolling window with the eyes closed.

    Samples are time-stamped rather than frame-counted, so a dropped frame
    or a variable frame rate does not bias the measurement.
    """

    def __init__(self, window_s: float = 60.0) -> None:
        self.window_s = float(window_s)
        self._samples: Deque[Tuple[float, float]] = deque()  # (ts, closed 0/1)

    def update(self, ts: float, closed: bool) -> float:
        self._samples.append((ts, 1.0 if closed else 0.0))
        cutoff = ts - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        return self.value

    @property
    def value(self) -> float:
        if not self._samples:
            return 0.0
        return float(sum(s for _, s in self._samples) / len(self._samples))

    @property
    def coverage_s(self) -> float:
        """Seconds of data currently in the window (for warm-up gating)."""
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    def reset(self) -> None:
        self._samples.clear()


# --------------------------------------------------------------------------
# Blink / micro-sleep state machine
# --------------------------------------------------------------------------
@dataclass
class BlinkState:
    is_closed: bool = False
    blink_count: int = 0
    microsleep_count: int = 0
    last_blink_duration: float = 0.0
    last_microsleep_ts: float = -1e9
    closed_duration: float = 0.0        # current, ongoing closure
    blink_rate_per_min: float = 0.0


class BlinkDetector:
    """Detects blinks and micro-sleeps from a stream of EAR values.

    A closure is a blink if it lasts between ``blink_min_s`` and
    ``blink_max_s``; anything longer than ``microsleep_s`` is escalated to a
    micro-sleep, which is the single strongest drowsiness cue we have.
    """

    def __init__(
        self,
        blink_min_s: float = 0.06,
        blink_max_s: float = 0.5,
        microsleep_s: float = 0.8,
        rate_window_s: float = 60.0,
    ) -> None:
        self.blink_min_s = blink_min_s
        self.blink_max_s = blink_max_s
        self.microsleep_s = microsleep_s
        self.rate_window_s = rate_window_s
        self.state = BlinkState()
        self._closed_since: float | None = None
        self._blink_ts: Deque[float] = deque()
        self._microsleep_reported = False

    def update(self, ts: float, ear: float, threshold: float) -> BlinkState:
        closed = ear < threshold
        st = self.state

        if closed and self._closed_since is None:
            self._closed_since = ts
            self._microsleep_reported = False

        if closed:
            # NOTE: must be `is not None`, not `or ts` - a closure starting at
            # timestamp 0.0 (every video file) would otherwise read as "not
            # started" and micro-sleeps would never fire offline.
            st.closed_duration = ts - self._closed_since if self._closed_since is not None else 0.0
            # Fire the micro-sleep *while* the eyes are still shut - waiting
            # for them to reopen would defeat the purpose of the warning.
            if st.closed_duration >= self.microsleep_s and not self._microsleep_reported:
                st.microsleep_count += 1
                st.last_microsleep_ts = ts
                self._microsleep_reported = True
        else:
            if self._closed_since is not None:
                duration = ts - self._closed_since
                st.last_blink_duration = duration
                if self.blink_min_s <= duration <= self.blink_max_s:
                    st.blink_count += 1
                    self._blink_ts.append(ts)
            self._closed_since = None
            st.closed_duration = 0.0

        st.is_closed = closed

        cutoff = ts - self.rate_window_s
        while self._blink_ts and self._blink_ts[0] < cutoff:
            self._blink_ts.popleft()
        span = min(self.rate_window_s, max(ts - (self._blink_ts[0] if self._blink_ts else ts), 1.0))
        st.blink_rate_per_min = len(self._blink_ts) * 60.0 / max(span, 1.0)
        return st

    def microsleep_recent(self, ts: float, within_s: float = 10.0) -> bool:
        return (ts - self.state.last_microsleep_ts) <= within_s


# --------------------------------------------------------------------------
# Yawn detection
# --------------------------------------------------------------------------
@dataclass
class YawnState:
    is_yawning: bool = False
    yawn_count: int = 0
    yawn_rate_per_min: float = 0.0
    last_yawn_ts: float = -1e9
    open_duration: float = 0.0


class YawnDetector:
    """A yawn is a sustained mouth opening, not a single high-MAR frame.

    Requiring ``min_duration_s`` of continuous opening removes the false
    positives caused by speech, laughing and singing.
    """

    def __init__(self, min_duration_s: float = 1.2, window_s: float = 120.0) -> None:
        self.min_duration_s = min_duration_s
        self.window_s = window_s
        self.state = YawnState()
        self._open_since: float | None = None
        self._counted = False
        self._yawn_ts: Deque[float] = deque()

    def update(self, ts: float, mar: float, threshold: float) -> YawnState:
        st = self.state
        is_open = mar > threshold

        if is_open:
            if self._open_since is None:
                self._open_since = ts
                self._counted = False
            st.open_duration = ts - self._open_since
            if st.open_duration >= self.min_duration_s and not self._counted:
                st.yawn_count += 1
                st.last_yawn_ts = ts
                self._yawn_ts.append(ts)
                self._counted = True
        else:
            self._open_since = None
            self._counted = False
            st.open_duration = 0.0

        st.is_yawning = is_open and st.open_duration >= self.min_duration_s

        cutoff = ts - self.window_s
        while self._yawn_ts and self._yawn_ts[0] < cutoff:
            self._yawn_ts.popleft()
        st.yawn_rate_per_min = len(self._yawn_ts) * 60.0 / self.window_s
        return st


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
class RollingRatio:
    """Fraction of a time window in which a boolean condition held."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._samples: Deque[Tuple[float, float]] = deque()

    def update(self, ts: float, flag: bool) -> float:
        self._samples.append((ts, 1.0 if flag else 0.0))
        cutoff = ts - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        return self.value

    @property
    def value(self) -> float:
        if not self._samples:
            return 0.0
        return float(sum(v for _, v in self._samples) / len(self._samples))


class SustainedFlag:
    """True once a condition has held continuously for ``min_duration_s``."""

    def __init__(self, min_duration_s: float) -> None:
        self.min_duration_s = min_duration_s
        self._since: float | None = None
        self.duration = 0.0

    def update(self, ts: float, condition: bool) -> bool:
        if condition:
            if self._since is None:
                self._since = ts
            self.duration = ts - self._since
        else:
            self._since = None
            self.duration = 0.0
        return self.duration >= self.min_duration_s


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
