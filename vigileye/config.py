"""Configuration loading.

The defaults below mirror ``config.yaml`` so the package still runs if the
YAML file is missing. A user file is deep-merged on top of the defaults,
which means a partial YAML (overriding only two thresholds) is valid.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULTS: Dict[str, Any] = {
    "camera": {"source": 1, "width": 640, "height": 480, "fps": 30, "flip": True},
    "face_mesh": {
        "max_num_faces": 1,
        "refine_landmarks": True,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "eye": {
        "ear_threshold": 0.21,
        "use_calibration": True,
        "calib_ratio": 0.75,
        "perclos_window_s": 60.0,
        "perclos_warn": 0.15,
        "perclos_critical": 0.30,
        "blink_min_s": 0.06,
        "blink_max_s": 0.50,
        "microsleep_s": 0.80,
    },
    "mouth": {
        "mar_threshold": 0.55,
        "use_calibration": True,
        "calib_multiplier": 2.2,
        "yawn_min_s": 1.2,
        "yawn_window_s": 120.0,
    },
    "head": {
        "yaw_threshold_deg": 25.0,
        "pitch_nod_deg": 15.0,
        "nod_min_s": 0.5,
        "off_road_min_s": 2.0,
        "nod_memory_s": 20.0,
    },
    "gaze": {
        "h_range": [0.36, 0.64],
        "v_range": [0.28, 0.75],
        "window_s": 10.0,
        "off_ratio_warn": 0.40,
    },
    "objects": {
        "enabled": True,
        "model": "yolov8n.pt",
        "conf": 0.35,
        "every_n_frames": 5,
        "phone_persist_s": 1.0,
        "hands": {
            "enabled": True,
            "max_num_hands": 2,
            "min_detection_confidence": 0.5,
            "wheel_roi": [0.20, 0.55, 0.90, 1.00],
            "hands_off_min_s": 2.0,
        },
    },
    "fusion": {
        "mode": "rule",
        "drowsy_threshold": 0.45,
        "distract_threshold": 0.45,
        "confirm_frames": 8,
        "release_frames": 20,
        "min_state_s": 1.5,
        "weights": {
            "drowsy": {"perclos": 0.40, "microsleep": 0.20, "yawn": 0.20, "nod": 0.20},
            "distract": {"phone": 0.35, "gaze": 0.25, "head": 0.25, "hands": 0.15},
        },
    },
    "temporal": {"checkpoint": "models/vigileye_lstm.pt", "window": 30, "blend": 0.5},
    "alerts": {
        "enabled": True,
        "cooldown_s": 5.0,
        "escalate_s": 8.0,
        "volume": 0.35,
        "tts": False,
    },
    "logging": {
        "dir": "logs",
        "frame_log": True,
        "db": "logs/vigileye.db",
        "flush_every": 100,
    },
    "calibration": {"duration_s": 10.0, "file": "calibration.json"},
}


class Cfg(dict):
    """A dict that also supports attribute access (``cfg.eye.ear_threshold``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> Cfg:
    """Load ``config.yaml`` merged over the built-in defaults."""
    data = copy.deepcopy(DEFAULTS)
    if path is not None:
        p = Path(path)
        if p.exists():
            if yaml is None:
                raise RuntimeError("PyYAML is required to read a config file")
            user = yaml.safe_load(p.read_text()) or {}
            data = _deep_merge(data, user)
    return _wrap(data)


def load_calibration(path: str | Path) -> Dict[str, float] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_calibration(path: str | Path, values: Dict[str, float]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(values, indent=2))
