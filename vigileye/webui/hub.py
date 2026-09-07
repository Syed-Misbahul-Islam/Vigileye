"""Shared state between the perception loop and the web UI.

The pipeline thread pushes frames and signals in; HTTP handler threads read
snapshots out. Everything the browser renders is computed here, in Python, so
the page itself stays a thin renderer: the payload is a flat map of element id
-> value (text, bar width, SVG attribute, HTML fragment).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence

import cv2

from ..alerts import AlertEvent
from ..fusion import DriverState, FusionResult, Signals

#: Samples kept per trace. At ~5-15 FPS this is roughly the last 10-25 s.
HISTORY = 150

_STATE_TEXT = {
    DriverState.ALERT: ("Fit to drive", "Driver alert, eyes on the road. No action needed."),
    DriverState.DROWSY: ("Drowsy", "Fatigue signatures rising. Recommend a rest stop."),
    DriverState.DISTRACTED: ("Distracted", "Attention is off the road. Eyes up now."),
    DriverState.NO_DRIVER: ("No driver", "No face in frame. Check the camera framing."),
}

_ALERT_KIND = {
    DriverState.DROWSY: "Fatigue",
    DriverState.DISTRACTED: "Distract",
    DriverState.NO_DRIVER: "Camera",
}


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _spark(values: Sequence[float], lo: float, hi: float,
           w: float = 300.0, h: float = 60.0, pad: float = 8.0,
           close: bool = False) -> str:
    """Polyline path through `values`, newest on the right."""
    if not values:
        return f"M0,{h - pad} L{w},{h - pad}" + (f" L{w},{h} L0,{h} Z" if close else "")

    span = max(hi - lo, 1e-6)
    n = len(values)
    step = w / max(n - 1, 1)
    pts = []
    for i, v in enumerate(values):
        frac = min(max((v - lo) / span, 0.0), 1.0)
        y = (h - pad) - frac * (h - 2 * pad)
        pts.append(f"{i * step:.1f},{y:.1f}")

    # A single sample would collapse to a dot; stretch it into a flat line.
    if n == 1:
        pts.append(f"{w:.1f},{pts[0].split(',')[1]}")

    path = "M" + " L".join(pts)
    if close:
        path += f" L{w},{h} L0,{h} Z"
    return path


def _trace(values: Sequence[float], w: float = 320.0, h: float = 140.0) -> str:
    """Score trace on the 0-1 range, drawn in the deep-dive chart box."""
    return _spark(values, 0.0, 1.0, w=w, h=h, pad=12.0)


class TelemetryHub:
    """Thread-safe latest-frame + latest-telemetry store, plus a command queue."""

    def __init__(self, driver_id: str, vehicle_id: str, jpeg_quality: int = 72) -> None:
        self.driver_id = driver_id
        self.vehicle_id = vehicle_id
        self.session_id = "-"
        self._jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        self._cond = threading.Condition()
        self._frame: Optional[bytes] = None
        self._seq = 0
        self.closed = False

        self._lock = threading.Lock()
        self._payload: Dict[str, object] = {}
        self._commands: List[str] = []

        self._hist: Dict[str, Deque[float]] = {
            k: deque(maxlen=HISTORY)
            for k in ("ear", "perclos", "blink", "mar", "yaw", "pitch", "drowsy", "distract")
        }
        self._alerts: Deque[Dict[str, object]] = deque(maxlen=200)
        self._counts = {"fatigue": 0, "distract": 0, "critical": 0, "camera": 0}
        self._started = time.time()

    # -- frames ---------------------------------------------------------
    def publish_frame(self, frame_bgr) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, self._jpeg_params)
        if not ok:
            return
        with self._cond:
            self._frame = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def frames(self):
        """Yield each new JPEG once, for the MJPEG response."""
        last = -1
        while not self.closed:
            with self._cond:
                if self._seq == last:
                    self._cond.wait(1.0)
                if self._seq == last or self._frame is None:
                    continue
                last = self._seq
                data = self._frame
            yield data

    # -- alerts ---------------------------------------------------------
    def add_alert(self, event: AlertEvent) -> None:
        kind = _ALERT_KIND.get(event.state, "Fatigue")
        key = {"Fatigue": "fatigue", "Distract": "distract", "Camera": "camera"}[kind]
        with self._lock:
            self._counts[key] += 1
            if event.level >= 2:
                self._counts["critical"] += 1
            self._alerts.appendleft(
                {
                    "kind": kind,
                    "title": event.message,
                    "sub": f"Level {event.level} · {event.state.value.title()}",
                    "time": time.strftime("%H:%M:%S", time.localtime(event.ts)),
                    "level": event.level,
                }
            )

    # -- telemetry ------------------------------------------------------
    def publish(self, s: Signals, r: FusionResult, *, fps: float, latency_ms: float,
                calibrating: bool, calib_progress: float, muted: bool,
                backend: str, pico: str) -> None:
        drowsy = float(r.drowsy_score)
        distract = float(r.distract_score)
        eyes_off = float(s.gaze_off_ratio)

        with self._lock:
            if not calibrating:
                self._hist["ear"].append(s.ear)
                self._hist["perclos"].append(s.perclos)
                self._hist["blink"].append(s.blink_rate)
                self._hist["mar"].append(s.mar)
                self._hist["yaw"].append(s.yaw)
                self._hist["pitch"].append(s.pitch)
                self._hist["drowsy"].append(drowsy)
                self._hist["distract"].append(distract)
            hist = {k: list(v) for k, v in self._hist.items()}
            counts = dict(self._counts)
            alerts = list(self._alerts)

        if calibrating:
            headline = "Calibrating"
            note = f"Hold a neutral pose - baseline capture {calib_progress * 100:.0f}% complete."
        else:
            headline, note = _STATE_TEXT.get(r.state, (str(r.state), ""))

        risk = round(100 * max(drowsy, distract))
        risk_label = "Low risk" if risk < 35 else ("Elevated risk" if risk < 65 else "Critical risk")
        c = r.contributions or {}

        text = {
            # header + session
            "hdrDriver": self.driver_id,
            "hdrVehicle": self.vehicle_id,
            "hdrTrip": _clock(time.time() - self._started),
            "hdrFps": f"{fps:.0f}",
            "hdrFeed": "CALIBRATING" if calibrating else "ACTIVE FEED",
            "sideBackend": backend,
            "sideLatency": f"{latency_ms:.1f}ms",
            "footPico": pico,
            "footSession": self.session_id,
            # KPI strip
            "kpiTrip": _clock(time.time() - self._started),
            "kpiLock": "Locked" if s.face_found else "Searching",
            "kpiLockNote": "Face mesh tracking" if s.face_found else "No face in frame",
            "kpiFatigue": str(counts["fatigue"]),
            "kpiDistract": str(counts["distract"]),
            "kpiCritical": str(counts["critical"]),
            # hero
            "heroState": headline,
            "heroNote": note,
            "heroDrowsyVal": f"{drowsy * 100:.0f}% · {_band(drowsy)}",
            "heroDistractVal": f"{distract * 100:.0f}% · {_band(distract)}",
            "heroEyesVal": f"{eyes_off * 100:.0f}% · {_band(eyes_off)}",
            "riskValue": str(risk),
            "riskLabel": risk_label,
            # feed overlay
            "feedEar": f"{s.ear:.2f}",
            "feedMar": f"{s.mar:.2f}",
            "feedMesh": "ACTIVE" if s.face_found else "LOST",
            "feedPose": f"YAW: {s.yaw:+.1f}° • PITCH: {s.pitch:+.1f}°",
            "feedMode": f"CAM_01 [{'CALIBRATING' if calibrating else r.state.value}]",
            "feedRes": f"{fps:.1f} FPS · {r.source}",
            # biometric grid
            "ear": f"{s.ear:.2f}",
            "earLimit": f"{s.ear_threshold:.2f}",
            "earChip": "Closed" if s.eye_closed else "Open",
            "perclos": f"{s.perclos * 100:.1f}%",
            "perclosChip": _band(s.perclos / 0.30 if s.perclos else 0.0),
            "blink": f"{s.blink_rate:.0f}",
            "blinkChip": f"{s.microsleep_count} microsleeps",
            "nods": str(s.nod_recent_count),
            "mar": f"{s.yawn_count}",
            "marValue": f"{s.mar:.2f}",
            "marChip": "Yawning" if s.is_yawning else "Mouth relaxed",
            "yaw": f"{s.yaw:+.1f}°",
            "yawChip": "Off-road" if s.head_turned else "Forward",
            "pitch": f"{s.pitch:+.1f}°",
            "roll": f"Roll {s.roll:+.1f}°",
            "pitchChip": "Nodding" if s.nod_active else "Level",
            # camera & gaze view
            "gazeH": f"{s.gaze_h:.2f}",
            "gazeV": f"{s.gaze_v:.2f}",
            "gazeOff": f"{eyes_off * 100:.0f}%",
            "gazeChip": "Off road" if s.gaze_off else "On road",
            "phoneVal": f"{s.phone_conf * 100:.0f}%",
            "phoneChip": "Sustained" if s.phone_sustained else "Clear",
            "handsVal": f"{s.hands_on_wheel}/{s.hands_detected}",
            "handsChip": "Off wheel" if s.hands_off else "On wheel",
            "cnnVal": f"{s.cnn_drowsy_prob * 100:.0f}%" if s.cnn_available else "n/a",
            "cnnChip": "Model live" if s.cnn_available else "Not trained",
            "closedVal": f"{s.closed_duration:.1f}s",
            # trends view
            "trendPerclos": f"{s.perclos * 100:.1f}%",
            "trendBlink": f"{s.blink_rate:.0f}/min",
            "trendYawn": f"{s.yawn_rate:.1f}/min",
            "trendMicro": str(s.microsleep_count),
            "trendDrowsy": f"{drowsy * 100:.0f}%",
            "trendDistract": f"{distract * 100:.0f}%",
            "alertsAll": f"All {sum(counts[k] for k in ('fatigue', 'distract', 'camera'))}",
            "alertsFatigue": f"Fatigue {counts['fatigue']}",
            "alertsDistract": f"Distract {counts['distract']}",
            "alertsCritical": f"Critical {counts['critical']}",
            "alertsCamera": f"Camera {counts['camera']}",
            "radarTop": _top_contribution(c),
        }

        width = {
            "heroDrowsyBar": f"{drowsy * 100:.0f}%",
            "heroDistractBar": f"{distract * 100:.0f}%",
            "heroEyesBar": f"{eyes_off * 100:.0f}%",
            "earBar": f"{min(s.ear / max(s.ear_threshold * 2, 1e-6), 1.0) * 100:.0f}%",
            "pitchBar": f"{min(abs(s.pitch) / 45.0, 1.0) * 100:.0f}%",
            "calibBar": f"{max(calib_progress, 0.0) * 100:.0f}%",
        }

        attr = {
            "earPath": {"d": _spark(hist["ear"], 0.0, max(0.45, s.ear_threshold * 2))},
            "earFill": {"d": _spark(hist["ear"], 0.0, max(0.45, s.ear_threshold * 2), close=True)},
            "perclosPath": {"d": _spark(hist["perclos"], 0.0, 0.5)},
            "perclosFill": {"d": _spark(hist["perclos"], 0.0, 0.5, close=True)},
            "blinkPath": {"d": _spark(hist["blink"], 0.0, 40.0)},
            "blinkFill": {"d": _spark(hist["blink"], 0.0, 40.0, close=True)},
            "marPath": {"d": _spark(hist["mar"], 0.0, max(0.9, s.mar_threshold * 1.5))},
            "yawPath": {"d": _spark(hist["yaw"], -45.0, 45.0)},
            "yawFill": {"d": _spark(hist["yaw"], -45.0, 45.0, close=True)},
            "pitchPath": {"d": _spark(hist["pitch"], -45.0, 45.0)},
            "pitchFill": {"d": _spark(hist["pitch"], -45.0, 45.0, close=True)},
            "tracePrimary": {"d": _trace(hist["drowsy"])},
            "traceSecondary": {"d": _trace(hist["distract"])},
            "trendEarPath": {"d": _spark(hist["ear"], 0.0, max(0.45, s.ear_threshold * 2), w=640, h=160)},
            "trendPerclosPath": {"d": _spark(hist["perclos"], 0.0, 0.5, w=640, h=160)},
            "trendScorePath": {"d": _spark(hist["drowsy"], 0.0, 1.0, w=640, h=160)},
            "trendDistractPath": {"d": _spark(hist["distract"], 0.0, 1.0, w=640, h=160)},
            "radarShape": {"points": _radar(c)},
            "riskArc": {"stroke-dashoffset": f"{251.2 * (1 - risk / 100.0):.1f}"},
        }

        html = {
            "alertList": _alert_rows(alerts[:12], empty="No alerts this session."),
            "incidentList": _incident_rows(alerts),
        }

        payload = {
            "ts": time.time(),
            "state": "CALIBRATING" if calibrating else r.state.value,
            "level": max((a["level"] for a in alerts[:1]), default=0),
            "calibrating": calibrating,
            "muted": muted,
            "text": text,
            "width": width,
            "attr": attr,
            "html": html,
        }
        with self._lock:
            self._payload = payload

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._payload)

    # -- commands -------------------------------------------------------
    def push_command(self, action: str) -> None:
        with self._lock:
            self._commands.append(action)

    def drain_commands(self) -> List[str]:
        with self._lock:
            pending, self._commands = self._commands, []
        return pending

    def close(self) -> None:
        self.closed = True
        with self._cond:
            self._cond.notify_all()


def _band(value: float) -> str:
    return "Good" if value < 0.25 else ("Watch" if value < 0.5 else "Act")


def _top_contribution(c: Dict[str, float]) -> str:
    labels = {
        "perclos": "Eye Closure (PERCLOS)", "microsleep": "Micro-sleep", "yawn": "Yawning",
        "nod": "Head Nodding", "phone": "Phone Use", "gaze": "Off-road Gaze",
        "head": "Head Turn", "hands": "Hands Off Wheel", "cnn": "CNN Visual Cue",
    }
    if not c:
        return "-"
    key = max(c, key=lambda k: c.get(k, 0.0))
    return labels.get(key, key) if c.get(key, 0.0) > 0 else "Nominal"


def _radar(c: Dict[str, float]) -> str:
    """Hexagon matching the axis labels baked into the radar SVG."""
    axes = (("perclos", 0, -80), ("microsleep", 70, -40), ("nod", 70, 40),
            ("yawn", 0, 80), ("phone", -70, 40), ("gaze", -70, -40))
    pts = []
    for key, dx, dy in axes:
        v = min(max(float(c.get(key, 0.0)), 0.06), 1.0)
        pts.append(f"{100 + dx * v:.0f},{100 + dy * v:.0f}")
    return " ".join(pts)


_TONE = {"Fatigue": "bg-slate-700", "Distract": "bg-slate-600", "Camera": "bg-slate-500"}


def _alert_rows(alerts: List[Dict[str, object]], empty: str) -> str:
    if not alerts:
        return f'<div class="px-1 py-6 text-center text-xs text-slate-400">{empty}</div>'
    rows = []
    for a in alerts:
        tone = _TONE.get(str(a["kind"]), "bg-slate-700")
        searchable = f'{a["kind"]} {a["title"]} {a["sub"]}'.lower()
        rows.append(
            f'<div data-kind="{a["kind"]}" data-level="{a["level"]}" data-text="{searchable}"'
            ' class="alert-row group flex items-center justify-between rounded-xl border '
            'border-slate-200/70 bg-white/70 p-2.5 transition hover:bg-slate-100/80">'
            '<div class="flex items-center gap-2.5">'
            f'<span class="rounded-md {tone} px-2 py-0.5 text-[10px] font-semibold text-white shadow-sm">{a["kind"]}</span>'
            f'<div><div class="text-xs font-semibold text-slate-900">{a["title"]}</div>'
            f'<div class="text-[11px] text-slate-500">{a["sub"]}</div></div></div>'
            f'<div class="font-mono text-[11px] font-semibold text-slate-600">{a["time"]}</div></div>'
        )
    return "".join(rows)


def _incident_rows(alerts: List[Dict[str, object]]) -> str:
    if not alerts:
        return ('<tr><td class="px-4 py-6 text-center text-xs text-slate-400" colspan="4">'
                "No incidents recorded this session.</td></tr>")
    rows = []
    for a in alerts:
        searchable = f'{a["kind"]} {a["title"]} {a["sub"]}'.lower()
        rows.append(
            f'<tr data-kind="{a["kind"]}" data-level="{a["level"]}" data-text="{searchable}"'
            ' class="alert-row border-t border-slate-200/70 hover:bg-white/60">'
            f'<td class="px-4 py-2 font-mono text-xs text-slate-600">{a["time"]}</td>'
            f'<td class="px-4 py-2 text-xs font-semibold text-slate-800">{a["kind"]}</td>'
            f'<td class="px-4 py-2 font-mono text-xs text-slate-600">L{a["level"]}</td>'
            f'<td class="px-4 py-2 text-xs text-slate-700">{a["title"]}</td></tr>'
        )
    return "".join(rows)
