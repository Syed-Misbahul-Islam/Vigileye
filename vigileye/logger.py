"""Session logging.

Two sinks, deliberately:

* a **per-frame CSV**, which is the raw material for training the v2 model
  and for offline threshold tuning;
* a **SQLite event store**, which holds only state transitions and alerts
  and is what the fleet dashboard reads. Keeping events separate means the
  dashboard stays fast even after months of driving.
"""
from __future__ import annotations

import csv
import platform
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .fusion import DriverState, FusionResult, Signals

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    driver_id    TEXT,
    vehicle_id   TEXT,
    started_at   TEXT,
    ended_at     TEXT,
    device       TEXT,
    frames       INTEGER DEFAULT 0,
    mean_fps     REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    ts           REAL,
    wall_time    TEXT,
    kind         TEXT,          -- state_change | alert
    state        TEXT,
    level        INTEGER,
    drowsy_score REAL,
    distract_score REAL,
    perclos      REAL,
    message      TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
"""

_FRAME_FIELDS = [
    "ts", "face_found", "ear", "mar", "ear_threshold", "perclos", "eye_closed",
    "closed_duration", "blink_rate", "microsleep_recent", "microsleep_count",
    "yawn_rate", "is_yawning", "yawn_count", "yaw", "pitch", "roll",
    "nod_active", "nod_recent_count", "head_turned", "gaze_h", "gaze_v",
    "gaze_off", "gaze_off_ratio", "phone_conf", "phone_sustained",
    "hands_detected", "hands_on_wheel", "hands_off",
    "state", "drowsy_score", "distract_score", "cnn_drowsy_prob",
]


class SessionLogger:
    def __init__(
        self,
        cfg,
        driver_id: str = "unknown",
        vehicle_id: str = "unknown",
        session_id: Optional[str] = None,
    ) -> None:
        self.dir = Path(cfg.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.driver_id = driver_id
        self.vehicle_id = vehicle_id
        self.frame_log = bool(cfg.frame_log)
        self.flush_every = int(cfg.flush_every)
        self._n = 0
        self._prev_state: Optional[DriverState] = None

        self.db = sqlite3.connect(str(cfg.db), check_same_thread=False)
        self.db.executescript(_SCHEMA)
        self.db.execute(
            "INSERT OR REPLACE INTO sessions"
            "(session_id, driver_id, vehicle_id, started_at, device) VALUES (?,?,?,?,?)",
            (
                self.session_id,
                driver_id,
                vehicle_id,
                datetime.now(timezone.utc).isoformat(),
                platform.node(),
            ),
        )
        self.db.commit()

        self._csv_file = None
        self._csv = None
        if self.frame_log:
            path = self.dir / f"session_{self.session_id}.csv"
            self._csv_file = open(path, "w", newline="")
            self._csv = csv.DictWriter(self._csv_file, fieldnames=_FRAME_FIELDS)
            self._csv.writeheader()
            self.csv_path = path

    # -- writing ---------------------------------------------------------
    def log_frame(self, signals: Signals, result: FusionResult) -> None:
        self._n += 1

        if self._csv is not None:
            row = asdict(signals)
            row["state"] = result.state.value
            row["drowsy_score"] = round(result.drowsy_score, 4)
            row["distract_score"] = round(result.distract_score, 4)
            self._csv.writerow({k: row.get(k) for k in _FRAME_FIELDS})
            if self._n % self.flush_every == 0:
                self._csv_file.flush()

        if result.state != self._prev_state:
            self._event(
                signals.ts, "state_change", result.state, 0,
                result.drowsy_score, result.distract_score, signals.perclos,
                f"{self._prev_state.value if self._prev_state else 'INIT'} -> {result.state.value}",
            )
            self._prev_state = result.state

    def log_alert(self, event, result: FusionResult, perclos: float) -> None:
        self._event(
            event.ts, "alert", event.state, event.level,
            result.drowsy_score, result.distract_score, perclos, event.message,
        )

    def _event(self, ts, kind, state, level, drowsy, distract, perclos, message) -> None:
        self.db.execute(
            "INSERT INTO events(session_id, ts, wall_time, kind, state, level,"
            " drowsy_score, distract_score, perclos, message)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                self.session_id, float(ts), datetime.now(timezone.utc).isoformat(),
                kind, state.value if hasattr(state, "value") else str(state),
                int(level), float(drowsy), float(distract), float(perclos), message,
            ),
        )
        self.db.commit()

    def close(self, mean_fps: float = 0.0) -> None:
        self.db.execute(
            "UPDATE sessions SET ended_at=?, frames=?, mean_fps=? WHERE session_id=?",
            (datetime.now(timezone.utc).isoformat(), self._n, float(mean_fps), self.session_id),
        )
        self.db.commit()
        self.db.close()
        if self._csv_file is not None:
            self._csv_file.close()
