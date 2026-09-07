"""Gaze estimation from MediaPipe iris landmarks.

We do not attempt a full 3-D gaze vector. Instead we compute where the iris
centre sits *inside its own eye opening*, normalised to [0, 1] on both axes:

    h_ratio = 0.0 -> iris fully to the image-left corner
    h_ratio = 0.5 -> iris centred (looking straight ahead)
    v_ratio = 0.0 -> iris at the upper lid

This is robust, needs no per-user gaze calibration to be *useful*, and is
exactly the signal needed for an "eyes-off-road" flag. It is combined with
head yaw in the fusion stage, because a driver can look away either by
moving their eyes or by turning their head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .landmarks import (
    LEFT_EYE_CORNERS,
    LEFT_EYE_LIDS,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_LIDS,
    RIGHT_IRIS_CENTER,
)
from .metrics import RollingRatio

EPS = 1e-6


@dataclass
class Gaze:
    found: bool = False
    h_ratio: float = 0.5
    v_ratio: float = 0.5
    off_road: bool = False


def _eye_ratio(
    points: np.ndarray,
    iris_idx: int,
    corners: Sequence[int],
    lids: Sequence[int],
) -> Tuple[float, float]:
    iris = points[iris_idx, :2]
    c0 = points[corners[0], :2]
    c1 = points[corners[1], :2]
    top = points[lids[0], :2]
    bottom = points[lids[1], :2]

    x_min, x_max = (c0[0], c1[0]) if c0[0] <= c1[0] else (c1[0], c0[0])
    h = (iris[0] - x_min) / (x_max - x_min + EPS)

    y_min, y_max = (top[1], bottom[1]) if top[1] <= bottom[1] else (bottom[1], top[1])
    v = (iris[1] - y_min) / (y_max - y_min + EPS)
    return float(h), float(v)


class GazeEstimator:
    def __init__(self, cfg) -> None:
        self.h_lo, self.h_hi = float(cfg.h_range[0]), float(cfg.h_range[1])
        self.v_lo, self.v_hi = float(cfg.v_range[0]), float(cfg.v_range[1])
        self.off_tracker = RollingRatio(float(cfg.window_s))
        self.off_ratio_warn = float(cfg.off_ratio_warn)
        self._h_bias = 0.0    # set by calibration
        self._v_bias = 0.0

    def set_baseline(self, h: float, v: float) -> None:
        """Shift the neutral point to the driver's calibrated straight-ahead."""
        self._h_bias = h - 0.5
        self._v_bias = v - 0.5

    def estimate(self, points: np.ndarray) -> Gaze:
        if points.shape[0] <= RIGHT_IRIS_CENTER:
            # refine_landmarks was off - no iris points available
            return Gaze(found=False)

        lh, lv = _eye_ratio(points, LEFT_IRIS_CENTER, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)
        rh, rv = _eye_ratio(points, RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS)

        h = (lh + rh) / 2.0 - self._h_bias
        v = (lv + rv) / 2.0 - self._v_bias
        off = not (self.h_lo <= h <= self.h_hi and self.v_lo <= v <= self.v_hi)
        return Gaze(found=True, h_ratio=h, v_ratio=v, off_road=off)

    def update_window(self, ts: float, off_road: bool) -> float:
        """Return the fraction of the recent window spent looking away."""
        return self.off_tracker.update(ts, off_road)

    def off_road_score(self) -> float:
        return float(min(1.0, self.off_tracker.value / max(self.off_ratio_warn, EPS)))
