"""Per-driver calibration.

A fixed EAR threshold of 0.21 is the single biggest source of false
positives in published drowsiness systems: eye shape varies enormously
between individuals, and people wearing glasses or with naturally narrow
eye apertures sit permanently below the threshold.

The fix is ten seconds of "look at the road normally" at the start of each
session. We record the baseline EAR, MAR and head pose, then derive
personalised thresholds from them. This also captures camera placement,
which is what makes the head-pose thresholds meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Baseline:
    ear_open: float = 0.30
    ear_threshold: float = 0.21
    mar_closed: float = 0.25
    mar_threshold: float = 0.55
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gaze_h: float = 0.5
    gaze_v: float = 0.5
    samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "Baseline":
        b = Baseline()
        for k, v in d.items():
            if hasattr(b, k):
                setattr(b, k, v)
        return b


class Calibrator:
    """Collects samples for ``duration_s`` then derives thresholds.

    Uses medians rather than means so that a couple of blinks during the
    calibration window cannot drag the baseline down.
    """

    def __init__(self, cfg, eye_cfg, mouth_cfg) -> None:
        self.duration_s = float(cfg.duration_s)
        self.calib_ratio = float(eye_cfg.calib_ratio)
        self.mar_multiplier = float(mouth_cfg.calib_multiplier)
        self.fallback_ear = float(eye_cfg.ear_threshold)
        self.fallback_mar = float(mouth_cfg.mar_threshold)

        self._ear: List[float] = []
        self._mar: List[float] = []
        self._yaw: List[float] = []
        self._pitch: List[float] = []
        self._roll: List[float] = []
        self._gh: List[float] = []
        self._gv: List[float] = []
        self._start: Optional[float] = None
        self.done = False
        self.result: Optional[Baseline] = None

    def progress(self, ts: float) -> float:
        if self._start is None:
            return 0.0
        return min(1.0, (ts - self._start) / self.duration_s)

    def update(self, ts, ear, mar, yaw, pitch, roll, gaze_h, gaze_v) -> bool:
        """Feed one frame. Returns True once calibration has completed."""
        if self.done:
            return True
        if self._start is None:
            self._start = ts

        self._ear.append(ear)
        self._mar.append(mar)
        self._yaw.append(yaw)
        self._pitch.append(pitch)
        self._roll.append(roll)
        self._gh.append(gaze_h)
        self._gv.append(gaze_v)

        if ts - self._start >= self.duration_s:
            self.result = self._finish()
            self.done = True
        return self.done

    def _finish(self) -> Baseline:
        if len(self._ear) < 5:
            return Baseline(ear_threshold=self.fallback_ear, mar_threshold=self.fallback_mar)

        ear = np.asarray(self._ear)
        # Ignore the lowest 25% of frames: those are blinks, not "eyes open".
        ear_open = float(np.median(ear[ear >= np.percentile(ear, 25)]))
        mar_closed = float(np.median(self._mar))

        return Baseline(
            ear_open=ear_open,
            ear_threshold=max(0.10, ear_open * self.calib_ratio),
            mar_closed=mar_closed,
            mar_threshold=max(0.35, mar_closed * self.mar_multiplier),
            yaw=float(np.median(self._yaw)),
            pitch=float(np.median(self._pitch)),
            roll=float(np.median(self._roll)),
            gaze_h=float(np.median(self._gh)),
            gaze_v=float(np.median(self._gv)),
            samples=len(self._ear),
        )
