"""The VigilEye perception pipeline.

One class, one method: ``process(frame_bgr, ts) -> (Signals, FusionResult)``.

Both ``run.py`` (live) and ``scripts/collect_features.py`` (offline) use
this exact object, which guarantees the features the LSTM is trained on
are byte-for-byte the features it sees at inference time. Any divergence
there is the classic cause of a model that scores 97% offline and fails in
the vehicle.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .calibration import Baseline, Calibrator
from .config import Cfg, load_calibration, save_calibration
from .distraction import HandsOnWheelDetector, PhoneDetector
from .fusion import DriverState, FusionResult, HybridFusion, RuleFusion, Signals
from .gaze import GazeEstimator
from .head_pose import HeadPoseEstimator, HeadTurnDetector, NodDetector
from .landmarks import FaceMeshDetector
from .cnn_inference import CNNDrowsinessPredictor

from .metrics import (
    BlinkDetector,
    PerclosTracker,
    YawnDetector,
    both_eyes_ear,
    mouth_aspect_ratio,
)


class VigilEyePipeline:
    def __init__(self, cfg: Cfg, calibrate: bool = True, use_objects: Optional[bool] = None) -> None:
        self.cfg = cfg

        self.face = FaceMeshDetector(cfg.face_mesh)
        self.pose = HeadPoseEstimator()
        self.gaze = GazeEstimator(cfg.gaze)
        # CNN drowsiness classifier. Optional: without a trained checkpoint
        # the rule-based physiological cues carry the drowsiness score alone.
        try:
            self.cnn = CNNDrowsinessPredictor("models/drowsiness_model.pth")
        except Exception as exc:
            print(f"[VigilEye] CNN drowsiness model disabled: {exc}")
            self.cnn = None

        self.perclos = PerclosTracker(cfg.eye.perclos_window_s)
        self.blinks = BlinkDetector(
            blink_min_s=cfg.eye.blink_min_s,
            blink_max_s=cfg.eye.blink_max_s,
            microsleep_s=cfg.eye.microsleep_s,
        )
        self.yawns = YawnDetector(cfg.mouth.yawn_min_s, cfg.mouth.yawn_window_s)
        self.nods = NodDetector(cfg.head.pitch_nod_deg, cfg.head.nod_min_s, cfg.head.nod_memory_s)
        self.turns = HeadTurnDetector(cfg.head.yaw_threshold_deg, cfg.head.off_road_min_s)

        objects_on = cfg.objects.enabled if use_objects is None else use_objects
        obj_cfg = Cfg(dict(cfg.objects))
        obj_cfg["enabled"] = objects_on
        self.phone = PhoneDetector(obj_cfg)
        hands_cfg = Cfg(dict(cfg.objects.hands))
        hands_cfg["enabled"] = bool(cfg.objects.hands.enabled) and objects_on
        self.hands = HandsOnWheelDetector(hands_cfg)

        rule = RuleFusion(cfg.fusion, cfg.eye, cfg.head)
        self.fusion = rule
        self.temporal = None
        if cfg.fusion.mode in ("temporal", "hybrid"):
            try:
                from .temporal_model import TemporalFusion

                self.temporal = TemporalFusion(cfg.temporal.checkpoint, cfg.temporal.window)
                if self.temporal.available:
                    blend = 1.0 if cfg.fusion.mode == "temporal" else float(cfg.temporal.blend)
                    self.fusion = HybridFusion(rule, self.temporal, blend)
                else:
                    self.temporal = None
            except Exception as exc:
                print(f"[VigilEye] temporal model disabled: {exc}")
                self.temporal = None

        # Calibration -----------------------------------------------------
        self.baseline = Baseline(
            ear_threshold=cfg.eye.ear_threshold, mar_threshold=cfg.mouth.mar_threshold
        )
        self.calibrator: Optional[Calibrator] = None
        if calibrate and cfg.eye.use_calibration:
            saved = load_calibration(cfg.calibration.file)
            if saved:
                self.baseline = Baseline.from_dict(saved)
                self.gaze.set_baseline(self.baseline.gaze_h, self.baseline.gaze_v)
                print(f"[VigilEye] loaded calibration: EAR threshold {self.baseline.ear_threshold:.3f}")
            else:
                self.calibrator = Calibrator(cfg.calibration, cfg.eye, cfg.mouth)

        self.frame_count = 0
        self.last_detections: list = []
        self.last_points: Optional[np.ndarray] = None
        self.last_pose = None

    # ------------------------------------------------------------------
    @property
    def calibrating(self) -> bool:
        return self.calibrator is not None and not self.calibrator.done

    def recalibrate(self) -> None:
        self.calibrator = Calibrator(self.cfg.calibration, self.cfg.eye, self.cfg.mouth)

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, ts: Optional[float] = None) -> Tuple[Signals, FusionResult]:
        ts = time.time() if ts is None else float(ts)
        self.frame_count += 1
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        face = self.face.process(rgb)
        s = Signals(ts=ts, face_found=face.found)

        # Object cues run regardless of face detection: a driver looking
        # completely away may lose the face mesh but still hold a phone.
        phone_sustained, phone_conf, detections = self.phone.update(frame_bgr, ts)
        self.last_detections = detections
        s.phone_sustained, s.phone_conf = phone_sustained, phone_conf
        s.hands_detected, s.hands_on_wheel, s.hands_off = self.hands.update(rgb, ts)

        if not face.found:
            self.last_points = None
            self.last_pose = None
            return s, self.fusion.update(s)

        pts = face.points
        self.last_points = pts
                # -- CNN drowsiness ----------------------------------------------
        # Build a bounding box from the MediaPipe face landmarks.
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]

        x_min = max(0, int(np.min(x_coords)) - 30)
        y_min = max(0, int(np.min(y_coords)) - 30)
        x_max = min(frame_bgr.shape[1], int(np.max(x_coords)) + 30)
        y_max = min(frame_bgr.shape[0], int(np.max(y_coords)) + 30)

        face_crop = frame_bgr[y_min:y_max, x_min:x_max]

        if self.cnn is not None and face_crop.size > 0:
            _, _, cnn_drowsy_prob = self.cnn.predict(face_crop)
            s.cnn_drowsy_prob = cnn_drowsy_prob
            s.cnn_available = True

        # -- eyes ---------------------------------------------------------
        ear, _, _ = both_eyes_ear(pts)
        mar = mouth_aspect_ratio(pts)
        s.ear, s.mar = ear, mar
        s.ear_threshold = self.baseline.ear_threshold
        s.mar_threshold = self.baseline.mar_threshold

        # -- head & gaze ---------------------------------------------------
        pose = self.pose.estimate(pts, face.frame_w, face.frame_h)
        s.yaw, s.pitch, s.roll = pose.yaw, pose.pitch, pose.roll
        self.last_pose = pose

        gaze = self.gaze.estimate(pts)
        s.gaze_h, s.gaze_v = gaze.h_ratio, gaze.v_ratio
        s.gaze_off = gaze.off_road

        # -- calibration branch --------------------------------------------
        if self.calibrating:
            self.calibrator.update(ts, ear, mar, pose.yaw, pose.pitch, pose.roll,
                                   gaze.h_ratio, gaze.v_ratio)
            if self.calibrator.done and self.calibrator.result is not None:
                self.baseline = self.calibrator.result
                self.gaze.set_baseline(self.baseline.gaze_h, self.baseline.gaze_v)
                save_calibration(self.cfg.calibration.file, self.baseline.to_dict())
                print(f"[VigilEye] calibrated: EAR open {self.baseline.ear_open:.3f}, "
                      f"threshold {self.baseline.ear_threshold:.3f}")
            s.ear_threshold = self.baseline.ear_threshold
            return s, FusionResult(state=DriverState.ALERT, source="calibrating")

        # -- temporal aggregation -------------------------------------------
        blink = self.blinks.update(ts, ear, self.baseline.ear_threshold)
        s.eye_closed = blink.is_closed
        s.closed_duration = blink.closed_duration
        s.blink_rate = blink.blink_rate_per_min
        s.microsleep_count = blink.microsleep_count
        s.microsleep_recent = self.blinks.microsleep_recent(ts)
        s.perclos = self.perclos.update(ts, blink.is_closed)
        s.perclos_coverage = self.perclos.coverage_s

        yawn = self.yawns.update(ts, mar, self.baseline.mar_threshold)
        s.is_yawning = yawn.is_yawning
        s.yawn_rate = yawn.yawn_rate_per_min
        s.yawn_count = yawn.yawn_count

        s.nod_active = self.nods.update(ts, pose.pitch, self.baseline.pitch)
        s.nod_recent_count = self.nods.recent_count()
        s.head_turned = self.turns.update(ts, pose.yaw, self.baseline.yaw)

        s.gaze_off_ratio = self.gaze.update_window(ts, gaze.off_road or s.head_turned)

        return s, self.fusion.update(s)

    def time_in_state(self, ts: float) -> float:
        return self.fusion.time_in_state(ts)

    def close(self) -> None:
        self.face.close()
        self.hands.close()
