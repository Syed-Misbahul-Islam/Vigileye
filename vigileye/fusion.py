"""Multimodal fusion: many noisy cues in, one stable state out.

Design notes
------------
The hard part of a drowsiness system is not detecting eye closure - it is
*not screaming at the driver every eight seconds*. Three mechanisms handle
that here:

1. **Normalised sub-scores.** Every cue is mapped to [0, 1] against a
   physically meaningful saturation point (e.g. PERCLOS 0.30), so weights
   are comparable and interpretable.

2. **Two separate scores.** Drowsiness and distraction are scored
   independently rather than as one "risk" number. They demand different
   interventions - a drowsy driver needs to stop and rest, a distracted
   driver needs to look up *now* - so collapsing them would destroy the
   most useful thing the system knows.

3. **Hysteresis + dwell time.** A state change requires ``confirm_frames``
   of agreement, returning to ALERT requires ``release_frames``, and every
   state is held for at least ``min_state_s``. This is what turns a jittery
   per-frame classification into something a human can tolerate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .metrics import clip01


class DriverState(str, Enum):
    ALERT = "ALERT"
    DROWSY = "DROWSY"
    DISTRACTED = "DISTRACTED"
    NO_DRIVER = "NO_DRIVER"


#: Ordered feature names. This ordering is the contract between the live
#: pipeline, the feature dumper and the LSTM - do not reorder casually.
FEATURE_NAMES: List[str] = [
    "ear",
    "mar",
    "perclos",
    "eye_closed",
    "closed_duration",
    "blink_rate",
    "microsleep_recent",
    "yawn_rate",
    "is_yawning",
    "yaw",
    "pitch",
    "roll",
    "nod_active",
    "head_turned",
    "gaze_h",
    "gaze_v",
    "gaze_off",
    "gaze_off_ratio",
    "phone_conf",
    "hands_off",
]


@dataclass
class Signals:
    """Everything the perception layer knows about one frame."""

    ts: float = 0.0
    face_found: bool = False

    ear: float = 0.0
    mar: float = 0.0
    ear_threshold: float = 0.21
    mar_threshold: float = 0.55
    perclos: float = 0.0
    perclos_coverage: float = 0.0   # seconds of data in the window (warm-up gate)
    eye_closed: bool = False
    closed_duration: float = 0.0
    blink_rate: float = 0.0
    microsleep_recent: bool = False
    microsleep_count: int = 0

    yawn_rate: float = 0.0
    is_yawning: bool = False
    yawn_count: int = 0

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    nod_active: bool = False
    nod_recent_count: int = 0
    head_turned: bool = False

    gaze_h: float = 0.5
    gaze_v: float = 0.5
    gaze_off: bool = False
    gaze_off_ratio: float = 0.0

    phone_conf: float = 0.0
    phone_sustained: bool = False
    cnn_drowsy_prob: float = 0.0
    cnn_available: bool = False
    hands_detected: int = 0
    hands_on_wheel: int = 0
    hands_off: bool = False

    def to_vector(self) -> np.ndarray:
        """Normalised feature vector for the temporal model."""
        return np.array(
            [
                self.ear,
                self.mar,
                self.perclos,
                float(self.eye_closed),
                min(self.closed_duration, 3.0) / 3.0,
                min(self.blink_rate, 60.0) / 60.0,
                float(self.microsleep_recent),
                min(self.yawn_rate, 6.0) / 6.0,
                float(self.is_yawning),
                np.clip(self.yaw, -90, 90) / 90.0,
                np.clip(self.pitch, -90, 90) / 90.0,
                np.clip(self.roll, -90, 90) / 90.0,
                float(self.nod_active),
                float(self.head_turned),
                np.clip(self.gaze_h, -1, 2),
                np.clip(self.gaze_v, -1, 2),
                float(self.gaze_off),
                self.gaze_off_ratio,
                self.phone_conf,
                float(self.hands_off),
            ],
            dtype=np.float32,
        )


@dataclass
class FusionResult:
    state: DriverState = DriverState.ALERT
    drowsy_score: float = 0.0
    distract_score: float = 0.0
    confidence: float = 0.0
    contributions: Dict[str, float] = field(default_factory=dict)
    raw_state: DriverState = DriverState.ALERT   # pre-hysteresis
    source: str = "rule"


class RuleFusion:
    """Weighted linear scoring + a hysteresis state machine."""

    def __init__(self, cfg, eye_cfg, head_cfg) -> None:
        self.w_drowsy = dict(cfg.weights.drowsy)
        self.w_distract = dict(cfg.weights.distract)
        self.drowsy_th = float(cfg.drowsy_threshold)
        self.distract_th = float(cfg.distract_threshold)
        self.confirm_frames = int(cfg.confirm_frames)
        self.release_frames = int(cfg.release_frames)
        self.min_state_s = float(cfg.min_state_s)

        self.perclos_critical = float(eye_cfg.perclos_critical)
        self.perclos_warmup_s = 15.0
        self.nod_saturation = 3.0
        self.yawn_saturation = 3.0

        self.state = DriverState.ALERT
        self._candidate: Optional[DriverState] = None
        self._candidate_frames = 0
        self._state_since = 0.0

    # -- scoring ---------------------------------------------------------
    def score(self, s: Signals) -> tuple[float, float, Dict[str, float]]:
        c: Dict[str, float] = {}

        # PERCLOS is a rolling-minute statistic and is meaningless until the
        # window has filled: one closed frame at t=0 would otherwise read as
        # 100% closure. Ramp its influence in over the first `perclos_warmup_s`.
        coverage = clip01(s.perclos_coverage / self.perclos_warmup_s)
        c["perclos"] = coverage * clip01(
            s.perclos / max(self.perclos_critical, 1e-6)
        )

        c["cnn"] = clip01(s.cnn_drowsy_prob)

        c["microsleep"] = 1.0 if s.microsleep_recent else 0.0
        c["yawn"] = clip01(s.yawn_rate / self.yawn_saturation)
        c["nod"] = clip01(s.nod_recent_count / self.nod_saturation)

        c["phone"] = s.phone_conf if s.phone_sustained else 0.35 * s.phone_conf
        c["gaze"] = clip01(s.gaze_off_ratio / 0.4)
        c["head"] = 1.0 if s.head_turned else 0.0
        c["hands"] = 1.0 if s.hands_off else 0.0

        drowsy = sum(
            self.w_drowsy[k] * c[k]
            for k in self.w_drowsy
        )

        # CNN provides an additional learned visual cue.
        # Keep it as a supporting signal rather than replacing
        # the interpretable physiological cues.
        if s.cnn_available:
            drowsy = 0.80 * drowsy + 0.20 * c["cnn"]
        distract = sum(self.w_distract[k] * c[k] for k in self.w_distract)

        # A prolonged eye closure is an emergency regardless of the window
        # averages, which are still warming up in the first minute.
        if s.closed_duration >= 1.0:
            drowsy = max(drowsy, 0.95)

        return clip01(drowsy), clip01(distract), c

    # -- state machine ---------------------------------------------------
    def _raw_state(self, drowsy: float, distract: float) -> DriverState:
        if drowsy >= self.drowsy_th and drowsy >= distract:
            return DriverState.DROWSY
        if distract >= self.distract_th:
            return DriverState.DISTRACTED
        return DriverState.ALERT

    def update(self, s: Signals) -> FusionResult:
        if not s.face_found:
            drowsy = distract = 0.0
            contributions: Dict[str, float] = {}
            raw = DriverState.NO_DRIVER
        else:
            drowsy, distract, contributions = self.score(s)
            raw = self._raw_state(drowsy, distract)

        self._advance(raw, s.ts)

        confidence = max(drowsy, distract) if self.state != DriverState.ALERT else 1.0 - max(drowsy, distract)
        return FusionResult(
            state=self.state,
            drowsy_score=drowsy,
            distract_score=distract,
            confidence=clip01(confidence),
            contributions=contributions,
            raw_state=raw,
            source="rule",
        )

    def _advance(self, raw: DriverState, ts: float) -> None:
        if self._state_since == 0.0:
            self._state_since = ts

        if raw == self.state:
            self._candidate = None
            self._candidate_frames = 0
            return

        if raw != self._candidate:
            self._candidate = raw
            self._candidate_frames = 0
        self._candidate_frames += 1

        needed = self.release_frames if raw == DriverState.ALERT else self.confirm_frames
        held_long_enough = (ts - self._state_since) >= self.min_state_s

        if self._candidate_frames >= needed and held_long_enough:
            self.state = raw
            self._state_since = ts
            self._candidate = None
            self._candidate_frames = 0

    def time_in_state(self, ts: float) -> float:
        return max(0.0, ts - self._state_since)


class HybridFusion:
    """Blends the rule score with an LSTM's class probabilities.

    Rationale: the rule engine is fully interpretable and works from frame
    one; the learned model captures temporal patterns the rules miss. The
    blend keeps a sane floor if the model is under-trained, and the rule
    engine's hysteresis is still what produces the final state.
    """

    def __init__(self, rule: RuleFusion, temporal, blend: float = 0.5) -> None:
        self.rule = rule
        self.temporal = temporal
        self.blend = float(np.clip(blend, 0.0, 1.0))

    def update(self, s: Signals) -> FusionResult:
        if not s.face_found:
            return self.rule.update(s)

        drowsy, distract, contributions = self.rule.score(s)
        probs = self.temporal.update(s.to_vector())  # None until window fills

        source = "rule"
        if probs is not None:
            b = self.blend
            drowsy = (1 - b) * drowsy + b * float(probs[DriverState.DROWSY.value])
            distract = (1 - b) * distract + b * float(probs[DriverState.DISTRACTED.value])
            source = "hybrid"

        raw = self.rule._raw_state(drowsy, distract)
        self.rule._advance(raw, s.ts)
        confidence = max(drowsy, distract) if self.rule.state != DriverState.ALERT else 1.0 - max(drowsy, distract)
        return FusionResult(
            state=self.rule.state,
            drowsy_score=clip01(drowsy),
            distract_score=clip01(distract),
            confidence=clip01(confidence),
            contributions=contributions,
            raw_state=raw,
            source=source,
        )

    def time_in_state(self, ts: float) -> float:
        return self.rule.time_in_state(ts)
