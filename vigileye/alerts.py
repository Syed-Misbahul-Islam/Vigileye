"""Alert delivery with escalation and cooldown.

An alerting policy matters as much as the detector. Rules implemented:

* **Cooldown** - the same alert level cannot fire more than once per
  ``cooldown_s``. Without this, a drowsy driver gets a continuous siren,
  which people respond to by unplugging the device.
* **Escalation** - if the unsafe state persists past ``escalate_s``, the
  alert moves from level 1 (soft chime) to level 2 (urgent repeated tone).
* **Graceful degradation** - if no audio backend exists, alerts still fire
  visually and to the log, so the system is never silently broken.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fusion import DriverState

SAMPLE_RATE = 22050

_MESSAGES = {
    (DriverState.DROWSY, 1): "Drowsiness detected - stay alert",
    (DriverState.DROWSY, 2): "WAKE UP - pull over and rest",
    (DriverState.DISTRACTED, 1): "Eyes on the road",
    (DriverState.DISTRACTED, 2): "EYES ON THE ROAD NOW",
    (DriverState.NO_DRIVER, 1): "Driver not detected - check camera",
}


def _tone(freq: float, seconds: float, volume: float) -> np.ndarray:
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    # 10 ms raised-cosine fade in/out to avoid audible clicks
    fade = max(1, int(0.01 * SAMPLE_RATE))
    envelope = np.ones_like(wave)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (wave * envelope * volume).astype(np.float32)


class AudioBackend:
    """Tries sounddevice, then simpleaudio, then falls back to console."""

    def __init__(self) -> None:
        self.kind = "none"
        try:
            import sounddevice as sd

            self._sd = sd
            self.kind = "sounddevice"
            return
        except Exception:
            pass
        try:
            import simpleaudio as sa

            self._sa = sa
            self.kind = "simpleaudio"
        except Exception:
            pass

    def play(self, samples: np.ndarray) -> None:
        if self.kind == "sounddevice":
            self._sd.play(samples, SAMPLE_RATE)
        elif self.kind == "simpleaudio":
            pcm = (samples * 32767).astype(np.int16)
            self._sa.play_buffer(pcm, 1, 2, SAMPLE_RATE)
        else:
            print("\a", end="", flush=True)


@dataclass
class AlertEvent:
    ts: float
    state: DriverState
    level: int
    message: str


class AlertManager:
    def __init__(self, cfg) -> None:
        self.enabled = bool(cfg.enabled)
        self.cooldown_s = float(cfg.cooldown_s)
        self.escalate_s = float(cfg.escalate_s)
        self.volume = float(cfg.volume)
        self.audio = AudioBackend()
        self._last_fire: dict[tuple[DriverState, int], float] = {}
        self.last_event: Optional[AlertEvent] = None
        self.flash_until = 0.0

        self._tts = None
        if bool(getattr(cfg, "tts", False)):
            try:
                import pyttsx3

                self._tts = pyttsx3.init()
            except Exception:
                self._tts = None

    def update(self, ts: float, state: DriverState, time_in_state: float) -> Optional[AlertEvent]:
        if not self.enabled or state == DriverState.ALERT:
            return None

        level = 2 if time_in_state >= self.escalate_s else 1
        key = (state, level)
        if ts - self._last_fire.get(key, -1e9) < self.cooldown_s:
            return None

        message = _MESSAGES.get(key, f"{state.value} detected")
        event = AlertEvent(ts=ts, state=state, level=level, message=message)
        self._last_fire[key] = ts
        self.last_event = event
        self.flash_until = ts + 1.0
        self._fire(event)
        return event

    def _fire(self, event: AlertEvent) -> None:
        def _run() -> None:
            try:
                if event.level == 1:
                    self.audio.play(_tone(880, 0.25, self.volume))
                else:
                    burst = np.concatenate(
                        [
                            _tone(1200, 0.18, self.volume),
                            np.zeros(int(0.06 * SAMPLE_RATE), dtype=np.float32),
                            _tone(1200, 0.18, self.volume),
                            np.zeros(int(0.06 * SAMPLE_RATE), dtype=np.float32),
                            _tone(1400, 0.30, self.volume),
                        ]
                    )
                    self.audio.play(burst)
                if self._tts is not None:
                    self._tts.say(event.message)
                    self._tts.runAndWait()
            except Exception as exc:  # pragma: no cover
                print(f"[VigilEye] alert playback failed: {exc}")

        threading.Thread(target=_run, daemon=True).start()

    def is_flashing(self, ts: float) -> bool:
        return ts < self.flash_until
