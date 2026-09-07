"""Optional serial link to a Raspberry Pi Pico.

The Pico is an accessory, not a dependency: if ``pyserial`` is missing or no
board is on the port, VigilEye keeps running and simply reports the link as
offline. Failures are announced once rather than on every state change, so a
missing board cannot drown the console or the UI alert feed.

Set ``VIGILEYE_PICO_PORT`` to override the port (default ``COM7``).
"""
from __future__ import annotations

import os

try:
    import serial
except ImportError:  # pyserial not installed
    serial = None

PICO_PORT = os.environ.get("VIGILEYE_PICO_PORT", "COM7")
BAUD = 115200

_pico = None
_status = "pyserial missing" if serial is None else f"{PICO_PORT} not opened"
_announced = False


def _warn_once(message: str) -> None:
    global _announced
    if not _announced:
        print(f"[VigilEye] {message}")
        _announced = True


def get_pico():
    """Open the port on first use. Returns None when unavailable."""
    global _pico, _status
    if serial is None:
        _warn_once("Pico disabled: pyserial not installed (pip install pyserial)")
        return None
    if _pico is None:
        try:
            _pico = serial.Serial(PICO_PORT, BAUD, timeout=1)
            _status = f"{PICO_PORT} linked"
            print(f"[VigilEye] Pico linked on {PICO_PORT}")
        except Exception as exc:
            _status = f"{PICO_PORT} offline"
            _warn_once(f"Pico offline: could not open {PICO_PORT} ({exc})")
    return _pico


def send_to_pico(state: str) -> bool:
    """Write one state name to the board. Returns True when it landed."""
    global _pico, _status
    pico = get_pico()
    if pico is None:
        return False
    try:
        pico.write((state + "\n").encode())
        return True
    except Exception as exc:
        print(f"[VigilEye] Pico write failed, dropping link: {exc}")
        try:
            pico.close()
        except Exception:
            pass
        # Drop the handle so the next state change retries the connection.
        _pico = None
        _status = f"{PICO_PORT} write error"
        return False


def pico_status() -> str:
    return _status


def close_pico() -> None:
    global _pico
    if _pico is not None:
        try:
            _pico.close()
        except Exception:
            pass
        _pico = None
