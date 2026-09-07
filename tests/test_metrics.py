"""Unit tests for the parts that do not need a camera.

Run with:  pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vigileye.config import load_config
from vigileye.fusion import FEATURE_NAMES, DriverState, RuleFusion, Signals
from vigileye.metrics import (
    BlinkDetector,
    PerclosTracker,
    RollingRatio,
    SustainedFlag,
    YawnDetector,
    eye_aspect_ratio,
    mouth_aspect_ratio,
)


def synthetic_eye(openness: float) -> np.ndarray:
    """Build a 6-point eye where openness scales the vertical aperture."""
    pts = np.zeros((478, 3), dtype=np.float32)
    #        p1        p2          p3          p4        p5           p6
    coords = [(0, 0), (10, -5), (20, -5), (30, 0), (20, 5), (10, 5)]
    for i, (x, y) in enumerate(coords):
        pts[i, 0] = x
        pts[i, 1] = y * openness
    return pts


class TestEAR:
    def test_open_eye_higher_than_closed(self):
        idx = (0, 1, 2, 3, 4, 5)
        assert eye_aspect_ratio(synthetic_eye(1.0), idx) > eye_aspect_ratio(synthetic_eye(0.1), idx)

    def test_fully_closed_is_near_zero(self):
        assert eye_aspect_ratio(synthetic_eye(0.0), (0, 1, 2, 3, 4, 5)) == pytest.approx(0.0, abs=1e-5)

    def test_scale_invariance(self):
        """EAR must not change when the driver moves closer to the camera."""
        idx = (0, 1, 2, 3, 4, 5)
        near = synthetic_eye(1.0)
        far = near.copy() * 0.4
        assert eye_aspect_ratio(near, idx) == pytest.approx(eye_aspect_ratio(far, idx), rel=1e-4)


class TestMAR:
    def test_open_mouth_scores_higher(self):
        def mouth(gap):
            p = np.zeros((478, 3), dtype=np.float32)
            p[78] = (0, 0, 0)
            p[308] = (40, 0, 0)
            for top, bottom in ((81, 178), (13, 14), (311, 402)):
                p[top] = (20, -gap / 2, 0)
                p[bottom] = (20, gap / 2, 0)
            return p

        assert mouth_aspect_ratio(mouth(30)) > mouth_aspect_ratio(mouth(3))


class TestPerclos:
    def test_half_closed_window(self):
        t = PerclosTracker(window_s=10.0)
        for i in range(100):
            t.update(i * 0.1, closed=(i % 2 == 0))
        assert t.value == pytest.approx(0.5, abs=0.02)

    def test_old_samples_expire(self):
        t = PerclosTracker(window_s=1.0)
        for i in range(10):
            t.update(i * 0.1, closed=True)
        for i in range(10, 30):
            t.update(i * 0.1, closed=False)
        assert t.value < 0.2


class TestBlink:
    def test_normal_blink_counted(self):
        d = BlinkDetector(blink_min_s=0.06, blink_max_s=0.5, microsleep_s=0.8)
        ts = 0.0
        for _ in range(10):   # open
            d.update(ts, 0.30, 0.21); ts += 0.033
        for _ in range(6):    # closed ~0.2 s
            d.update(ts, 0.10, 0.21); ts += 0.033
        for _ in range(10):   # open again
            d.update(ts, 0.30, 0.21); ts += 0.033
        assert d.state.blink_count == 1
        assert d.state.microsleep_count == 0

    def test_long_closure_is_microsleep(self):
        d = BlinkDetector(microsleep_s=0.8)
        ts = 0.0
        for _ in range(40):   # ~1.3 s closed
            d.update(ts, 0.10, 0.21); ts += 0.033
        assert d.state.microsleep_count == 1

    def test_microsleep_fires_before_reopening(self):
        """The warning must not wait for the eyes to open again."""
        d = BlinkDetector(microsleep_s=0.5)
        ts = 0.0
        for _ in range(30):
            d.update(ts, 0.05, 0.21); ts += 0.033
        assert d.state.is_closed and d.state.microsleep_count == 1


class TestYawn:
    def test_brief_opening_is_not_a_yawn(self):
        y = YawnDetector(min_duration_s=1.2)
        ts = 0.0
        for _ in range(15):   # 0.5 s - talking, not yawning
            y.update(ts, 0.9, 0.55); ts += 0.033
        assert y.state.yawn_count == 0

    def test_sustained_opening_is_a_yawn(self):
        y = YawnDetector(min_duration_s=1.2)
        ts = 0.0
        for _ in range(60):   # ~2 s
            y.update(ts, 0.9, 0.55); ts += 0.033
        assert y.state.yawn_count == 1


class TestHelpers:
    def test_sustained_flag(self):
        f = SustainedFlag(1.0)
        assert not f.update(0.0, True)
        assert not f.update(0.5, True)
        assert f.update(1.1, True)
        assert not f.update(1.2, False)

    def test_rolling_ratio(self):
        r = RollingRatio(5.0)
        for i in range(50):
            r.update(i * 0.1, i % 4 == 0)
        assert r.value == pytest.approx(0.25, abs=0.05)


class TestFusion:
    @staticmethod
    def _engine():
        cfg = load_config(None)
        return RuleFusion(cfg.fusion, cfg.eye, cfg.head)

    def test_feature_vector_matches_names(self):
        assert len(Signals().to_vector()) == len(FEATURE_NAMES)

    def test_alert_when_all_signals_clean(self):
        f = self._engine()
        for i in range(60):
            r = f.update(Signals(ts=i * 0.033, face_found=True, ear=0.3, perclos=0.02))
        assert r.state == DriverState.ALERT

    def test_high_perclos_becomes_drowsy(self):
        f = self._engine()
        s = Signals(face_found=True, perclos=0.45, perclos_coverage=60.0,
                    microsleep_recent=True, yawn_rate=2.0, nod_recent_count=2)
        for i in range(60):
            s.ts = i * 0.033
            r = f.update(s)
        assert r.state == DriverState.DROWSY
        assert r.drowsy_score > r.distract_score

    def test_perclos_ignored_before_window_fills(self):
        """A cold-start window must not raise the score on one closed frame."""
        f = self._engine()
        cold = Signals(ts=0.1, face_found=True, perclos=1.0, perclos_coverage=0.1)
        warm = Signals(ts=0.1, face_found=True, perclos=1.0, perclos_coverage=60.0)
        assert f.score(cold)[0] < 0.05
        assert f.score(warm)[0] > 0.35

    def test_phone_becomes_distracted(self):
        f = self._engine()
        s = Signals(face_found=True, phone_conf=0.9, phone_sustained=True,
                    gaze_off_ratio=0.6, head_turned=True, hands_off=True)
        for i in range(60):
            s.ts = i * 0.033
            r = f.update(s)
        assert r.state == DriverState.DISTRACTED

    def test_hysteresis_ignores_single_frame_spike(self):
        """One bad frame must never flip the state - this is the anti-nag test."""
        f = self._engine()
        clean = Signals(face_found=True, perclos=0.01, perclos_coverage=60.0)
        for i in range(60):
            clean.ts = i * 0.033
            f.update(clean)
        spike = Signals(ts=2.0, face_found=True, perclos=0.9,
                        perclos_coverage=60.0, microsleep_recent=True)
        r = f.update(spike)
        assert r.state == DriverState.ALERT
        assert r.raw_state == DriverState.DROWSY   # raw fired, output held

    def test_no_face_reports_no_driver(self):
        f = self._engine()
        for i in range(60):
            r = f.update(Signals(ts=i * 0.033, face_found=False))
        assert r.state == DriverState.NO_DRIVER
