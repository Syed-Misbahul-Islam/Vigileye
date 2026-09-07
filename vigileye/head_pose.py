"""Head pose estimation via ``cv2.solvePnP``.

We solve for the rotation that maps a generic 3-D face model onto the six
detected 2-D landmarks, then decompose it into yaw / pitch / roll.

Two behavioural detectors sit on top of the raw angles:

* :class:`NodDetector`     - downward pitch excursions => drowsiness
* :class:`HeadTurnDetector`- sustained yaw excursions  => distraction

Both are expressed relative to a *calibrated baseline* pose, because a
driver's neutral head position depends entirely on where the dash camera
was mounted.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from .landmarks import MODEL_POINTS_3D, POSE_LANDMARKS
from .metrics import SustainedFlag


@dataclass
class HeadPose:
    found: bool = False
    yaw: float = 0.0      # +ve = turning to the driver's left in image space
    pitch: float = 0.0    # +ve = looking up, -ve = looking down
    roll: float = 0.0
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None


def _camera_matrix(w: int, h: int) -> np.ndarray:
    """Pinhole approximation: focal length ~= image width, centre = image centre.

    Good enough for pose *change* detection. Replace with a real intrinsic
    matrix from ``cv2.calibrateCamera`` if you need absolute accuracy.
    """
    f = float(w)
    return np.array(
        [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _rotation_to_euler(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a rotation matrix into (yaw, pitch, roll) in degrees."""
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0.0
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)

    # Wrap pitch so that a forward-facing head sits near 0 instead of +/-180.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return yaw, pitch, roll


class HeadPoseEstimator:
    def __init__(self) -> None:
        self._cam: Optional[np.ndarray] = None
        self._dist = np.zeros((4, 1), dtype=np.float64)
        self._last_rvec: Optional[np.ndarray] = None
        self._last_tvec: Optional[np.ndarray] = None

    def estimate(self, points: np.ndarray, w: int, h: int) -> HeadPose:
        if self._cam is None:
            self._cam = _camera_matrix(w, h)

        image_points = np.array(
            [
                points[POSE_LANDMARKS["nose_tip"], :2],
                points[POSE_LANDMARKS["chin"], :2],
                points[POSE_LANDMARKS["left_eye_outer"], :2],
                points[POSE_LANDMARKS["right_eye_outer"], :2],
                points[POSE_LANDMARKS["left_mouth"], :2],
                points[POSE_LANDMARKS["right_mouth"], :2],
            ],
            dtype=np.float64,
        )

        use_guess = self._last_rvec is not None
        ok, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            self._cam,
            self._dist,
            rvec=self._last_rvec.copy() if use_guess else None,
            tvec=self._last_tvec.copy() if use_guess else None,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return HeadPose(found=False)

        self._last_rvec, self._last_tvec = rvec, tvec
        rmat, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = _rotation_to_euler(rmat)
        return HeadPose(True, yaw, pitch, roll, rvec, tvec)


class NodDetector:
    """Counts head-drop events relative to the calibrated neutral pitch."""

    def __init__(self, drop_deg: float = 15.0, min_s: float = 0.5, memory_s: float = 20.0) -> None:
        self.drop_deg = drop_deg
        self.memory_s = memory_s
        self._sustained = SustainedFlag(min_s)
        self._counted = False
        self.nod_count = 0
        self._nod_ts: Deque[float] = deque()

    def update(self, ts: float, pitch: float, baseline_pitch: float) -> bool:
        dropped = (pitch - baseline_pitch) < -self.drop_deg
        held = self._sustained.update(ts, dropped)
        if held and not self._counted:
            self.nod_count += 1
            self._nod_ts.append(ts)
            self._counted = True
        if not dropped:
            self._counted = False

        cutoff = ts - self.memory_s
        while self._nod_ts and self._nod_ts[0] < cutoff:
            self._nod_ts.popleft()
        return held

    def recent_count(self) -> int:
        return len(self._nod_ts)


class HeadTurnDetector:
    """True while the head has been turned away from centre long enough."""

    def __init__(self, yaw_deg: float = 25.0, min_s: float = 2.0) -> None:
        self.yaw_deg = yaw_deg
        self._sustained = SustainedFlag(min_s)
        self.deviation = 0.0

    def update(self, ts: float, yaw: float, baseline_yaw: float) -> bool:
        self.deviation = abs(yaw - baseline_yaw)
        return self._sustained.update(ts, self.deviation > self.yaw_deg)

    @property
    def duration(self) -> float:
        return self._sustained.duration
