"""Dashboard HUD overlay drawn with OpenCV."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .fusion import DriverState, FusionResult, Signals
from .landmarks import LEFT_EYE_EAR, MOUTH_CORNERS, MOUTH_VERTICAL, RIGHT_EYE_EAR

STATE_COLORS = {
    DriverState.ALERT: (80, 200, 80),
    DriverState.DROWSY: (40, 60, 235),
    DriverState.DISTRACTED: (30, 170, 245),
    DriverState.NO_DRIVER: (150, 150, 150),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(img, x, y, w, h, alpha=0.55):
    roi = img[y : y + h, x : x + w]
    if roi.size == 0:
        return
    overlay = np.full(roi.shape, 20, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _bar(img, x, y, w, h, value, color, label):
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
    fill = int(w * max(0.0, min(1.0, value)))
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + fill - 1, y + h - 1), color, -1)
    cv2.putText(img, f"{label} {value:.2f}", (x + w + 8, y + h - 2), FONT, 0.42, (230, 230, 230), 1)


def draw_landmarks(frame, points: np.ndarray) -> None:
    for idx in list(LEFT_EYE_EAR) + list(RIGHT_EYE_EAR):
        cv2.circle(frame, tuple(points[idx, :2].astype(int)), 1, (0, 255, 255), -1)
    for top, bottom in MOUTH_VERTICAL:
        cv2.circle(frame, tuple(points[top, :2].astype(int)), 1, (255, 180, 0), -1)
        cv2.circle(frame, tuple(points[bottom, :2].astype(int)), 1, (255, 180, 0), -1)
    for idx in MOUTH_CORNERS:
        cv2.circle(frame, tuple(points[idx, :2].astype(int)), 2, (255, 180, 0), -1)


def draw_pose_axis(frame, points, pose, cam_focal: Optional[float] = None) -> None:
    """Project a short 3-D axis from the nose tip to visualise head pose."""
    if not pose.found or pose.rvec is None:
        return
    h, w = frame.shape[:2]
    f = cam_focal or float(w)
    cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    axis = np.float64([[60, 0, 0], [0, 60, 0], [0, 0, 60]])
    projected, _ = cv2.projectPoints(axis, pose.rvec, pose.tvec, cam, np.zeros((4, 1)))
    origin = tuple(points[1, :2].astype(int))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for p, c in zip(projected.reshape(-1, 2), colors):
        cv2.line(frame, origin, (int(p[0]), int(p[1])), c, 2)


def draw_wheel_roi(frame, roi: List[float]) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
    cv2.putText(frame, "wheel", (x1 + 4, y1 + 16), FONT, 0.4, (120, 120, 120), 1)


def draw_detections(frame, detections) -> None:
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(frame, f"{det.label} {det.conf:.2f}", (x1, max(14, y1 - 6)),
                    FONT, 0.45, (0, 140, 255), 1)


def draw_hud(
    frame,
    signals: Signals,
    result: FusionResult,
    fps: float,
    flashing: bool = False,
    calibrating: float = -1.0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    color = STATE_COLORS[result.state]

    _panel(frame, 0, 0, w, 92)
    cv2.putText(frame, result.state.value, (14, 40), FONT, 1.0, color, 2)
    cv2.putText(frame, f"{fps:5.1f} FPS  |  {result.source}", (14, 66), FONT, 0.45, (200, 200, 200), 1)

    _bar(frame, 200, 20, 130, 12, result.drowsy_score, (40, 60, 235), "drowsy")
    _bar(frame, 200, 40, 130, 12, result.distract_score, (30, 170, 245), "distract")
    _bar(frame, 200, 60, 130, 12, min(1.0, signals.perclos / 0.3), (200, 200, 60), "perclos")

    lines = [
        f"EAR {signals.ear:.3f} / th {signals.ear_threshold:.3f}",
        f"MAR {signals.mar:.3f}  yawns {signals.yawn_count}",
        f"blinks/min {signals.blink_rate:4.1f}  microsleeps {signals.microsleep_count}",
        f"yaw {signals.yaw:6.1f}  pitch {signals.pitch:6.1f}  roll {signals.roll:6.1f}",
        f"gaze {signals.gaze_h:.2f},{signals.gaze_v:.2f}  off {signals.gaze_off_ratio:.2f}",
        f"phone {signals.phone_conf:.2f}  hands {signals.hands_on_wheel}/{signals.hands_detected}",
    ]
    _panel(frame, 0, h - 22 * len(lines) - 12, 330, 22 * len(lines) + 12)
    for i, line in enumerate(lines):
        y = h - 22 * len(lines) + 22 * i
        cv2.putText(frame, line, (12, y), FONT, 0.46, (225, 225, 225), 1)

    if calibrating >= 0.0:
        cv2.rectangle(frame, (w // 2 - 150, h // 2 - 30), (w // 2 + 150, h // 2 + 30), (0, 0, 0), -1)
        cv2.putText(frame, "CALIBRATING - look ahead", (w // 2 - 140, h // 2 - 6),
                    FONT, 0.55, (255, 255, 255), 1)
        cv2.rectangle(frame, (w // 2 - 140, h // 2 + 8), (w // 2 + 140, h // 2 + 22), (90, 90, 90), 1)
        cv2.rectangle(frame, (w // 2 - 139, h // 2 + 9),
                      (w // 2 - 139 + int(278 * calibrating), h // 2 + 21), (80, 200, 80), -1)

    if flashing:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 10)
    return frame
