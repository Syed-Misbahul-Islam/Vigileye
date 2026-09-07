"""Object-level distraction cues.

Two independent detectors:

1. :class:`PhoneDetector` - YOLOv8n on the COCO class ``cell phone``.
   YOLO is the most expensive component in the pipeline, so it runs on
   1-in-N frames and its output is *held* between runs. A phone must also
   persist for ``phone_persist_s`` before it counts, which suppresses the
   single-frame false positives that plague small-object detection.

2. :class:`HandsOnWheelDetector` - MediaPipe Hands + a steering-wheel ROI.
   COCO has no "hand" class, so wiring YOLO to this would require a custom
   dataset. Landmark-based hand detection gives a usable signal on day one;
   swap in a fine-tuned YOLO head later if you collect the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .landmarks import HAND_MODEL_URL, MODEL_DIR, detect_backend, ensure_model
from .metrics import SustainedFlag

COCO_CELL_PHONE = 67


@dataclass
class Detection:
    label: str
    conf: float
    xyxy: Tuple[float, float, float, float]


@dataclass
class ObjectSignals:
    phone_visible: bool = False
    phone_conf: float = 0.0
    phone_sustained: bool = False
    hands_detected: int = 0
    hands_on_wheel: int = 0
    hands_off_wheel: bool = False
    detections: List[Detection] = field(default_factory=list)


class PhoneDetector:
    """Frame-skipping YOLOv8n wrapper with temporal persistence."""

    def __init__(self, cfg) -> None:
        self.enabled = bool(cfg.enabled)
        self.conf = float(cfg.conf)
        self.every_n = max(1, int(cfg.every_n_frames))
        self._sustained = SustainedFlag(float(cfg.phone_persist_s))
        self._frame_i = 0
        self._cached: List[Detection] = []
        self._model = None

        if self.enabled:
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(cfg.model))
            except Exception as exc:  # pragma: no cover - runtime environment
                print(f"[VigilEye] YOLO unavailable ({exc}); phone detection disabled.")
                self.enabled = False

    def update(self, frame_bgr: np.ndarray, ts: float) -> Tuple[bool, float, List[Detection]]:
        if not self.enabled or self._model is None:
            return False, 0.0, []

        if self._frame_i % self.every_n == 0:
            self._cached = self._infer(frame_bgr)
        self._frame_i += 1

        best = max((d.conf for d in self._cached if d.label == "cell phone"), default=0.0)
        visible = best >= self.conf
        sustained = self._sustained.update(ts, visible)
        return sustained, best, self._cached

    def _infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            frame_bgr, conf=self.conf, verbose=False, classes=[COCO_CELL_PHONE]
        )
        out: List[Detection] = []
        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                out.append(
                    Detection(
                        label=names.get(cls, str(cls)),
                        conf=float(box.conf[0]),
                        xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                    )
                )
        return out


class HandsOnWheelDetector:
    """Counts hands whose centroid falls inside the steering-wheel ROI.

    Supports both MediaPipe APIs (see ``vigileye.landmarks`` for why):
    legacy ``mp.solutions.hands`` and the modern Tasks ``HandLandmarker``.
    """

    def __init__(self, cfg) -> None:
        self.enabled = bool(cfg.enabled)
        self.roi = [float(v) for v in cfg.wheel_roi]
        self._sustained = SustainedFlag(float(cfg.hands_off_min_s))
        self._hands = None
        self.backend = "none"
        self._ts_ms = 0
        self.last_points: List[np.ndarray] = []

        if not self.enabled:
            return

        try:
            import mediapipe as mp

            self._mp = mp
            self.backend = detect_backend()

            if self.backend == "solutions":
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=int(cfg.max_num_hands),
                    min_detection_confidence=float(cfg.min_detection_confidence),
                    min_tracking_confidence=0.5,
                )
            elif self.backend == "tasks":
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                model_path = ensure_model(HAND_MODEL_URL, MODEL_DIR / "hand_landmarker.task")
                options = vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=int(cfg.max_num_hands),
                    min_hand_detection_confidence=float(cfg.min_detection_confidence),
                    min_tracking_confidence=0.5,
                )
                self._hands = vision.HandLandmarker.create_from_options(options)
            else:
                raise RuntimeError("no usable MediaPipe backend")
        except Exception as exc:  # pragma: no cover
            print(f"[VigilEye] MediaPipe Hands unavailable ({exc}); disabled.")
            self.enabled = False
            self._hands = None

    def set_roi(self, roi: List[float]) -> None:
        self.roi = [float(v) for v in roi]

    def _in_roi(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.roi
        return x1 <= x <= x2 and y1 <= y <= y2

    def update(self, rgb: np.ndarray, ts: float) -> Tuple[int, int, bool]:
        """Return ``(hands_detected, hands_on_wheel, hands_off_sustained)``."""
        if not self.enabled or self._hands is None:
            return 0, 0, False

        h, w = rgb.shape[:2]

        if self.backend == "solutions":
            res = self._hands.process(rgb)
            hands = res.multi_hand_landmarks or []
            hands = [[(lm.x, lm.y) for lm in hand.landmark] for hand in hands]
        else:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
            )
            self._ts_ms += 33
            res = self._hands.detect_for_video(mp_image, self._ts_ms)
            hands = [[(lm.x, lm.y) for lm in hand] for hand in (res.hand_landmarks or [])]

        self.last_points = []
        detected = on_wheel = 0
        for hand in hands:
            pts = np.asarray(hand, dtype=np.float32)
            self.last_points.append(pts * np.array([w, h], dtype=np.float32))
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            detected += 1
            if self._in_roi(cx, cy):
                on_wheel += 1

        hands_off = self._sustained.update(ts, on_wheel == 0)
        return detected, on_wheel, hands_off

    def close(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
