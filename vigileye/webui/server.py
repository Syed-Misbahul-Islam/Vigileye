"""Local HTTP server for the VigilEye Command Deck.

Deliberately built on the standard library: the whole point of this UI is to
come up automatically with ``python run.py`` on an edge device, so it must not
add a web framework to the install. Three endpoints do the work:

* ``/``               - the dashboard page
* ``/video``          - annotated frames as an MJPEG stream
* ``/api/telemetry``  - the element-id -> value payload the page polls
* ``/api/command``    - recalibrate / mute / snapshot / stop from the browser
* ``/api/fleet``      - historical sessions from the SQLite event store
"""
from __future__ import annotations

import json
import sqlite3
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from .hub import TelemetryHub

_HERE = Path(__file__).parent
_INDEX = _HERE / "index.html"
_BOUNDARY = "vigileyeframe"

_COMMANDS = {"recalibrate", "mute", "unmute", "snapshot", "stop"}


class _Handler(BaseHTTPRequestHandler):
    hub: TelemetryHub = None            # type: ignore[assignment]
    db_path: str = "logs/vigileye.db"
    protocol_version = "HTTP/1.1"

    # Per-frame request logging would drown the console the pipeline uses.
    def log_message(self, *args) -> None:  # noqa: D102
        pass

    # -- helpers --------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    # -- routes ---------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?")[0]

        if route in ("/", "/index.html"):
            try:
                self._send(200, _INDEX.read_bytes(), "text/html; charset=utf-8")
            except OSError as exc:
                self._send(500, f"index.html unreadable: {exc}".encode(), "text/plain")
        elif route == "/video":
            self._stream_video()
        elif route == "/api/telemetry":
            self._json(self.hub.snapshot())
        elif route == "/api/fleet":
            self._json(self._fleet())
        elif route == "/healthz":
            self._json({"ok": True})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/api/command":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            action = str(body.get("action", "")).lower()
        except (ValueError, json.JSONDecodeError):
            self._json({"ok": False, "error": "malformed request"}, 400)
            return

        if action not in _COMMANDS:
            self._json({"ok": False, "error": f"unknown action {action!r}"}, 400)
            return
        self.hub.push_command(action)
        self._json({"ok": True, "action": action})

    # -- MJPEG ----------------------------------------------------------
    def _stream_video(self) -> None:
        # The stream has no Content-Length: it ends when the socket closes, so
        # keep-alive has to be off or the client would wait for a next response.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.end_headers()
        try:
            for jpeg in self.hub.frames():
                self.wfile.write(
                    f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass   # browser navigated away or closed the tab

    # -- fleet history --------------------------------------------------
    def _fleet(self) -> dict:
        if not Path(self.db_path).exists():
            return {"sessions": [], "note": "No event database yet."}
        try:
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT s.session_id, s.driver_id, s.vehicle_id, s.started_at,"
                "       s.frames, s.mean_fps,"
                "       SUM(CASE WHEN e.kind='alert' THEN 1 ELSE 0 END) AS alerts,"
                "       SUM(CASE WHEN e.kind='alert' AND e.level=2 THEN 1 ELSE 0 END) AS critical,"
                "       MAX(e.drowsy_score) AS peak_drowsy"
                " FROM sessions s LEFT JOIN events e ON e.session_id = s.session_id"
                " GROUP BY s.session_id ORDER BY s.started_at DESC LIMIT 40"
            ).fetchall()
            con.close()
        except sqlite3.Error as exc:
            return {"sessions": [], "note": f"Database unreadable: {exc}"}
        return {"sessions": [dict(r) for r in rows]}


class WebUI:
    """Owns the server thread and its lifetime."""

    def __init__(self, hub: TelemetryHub, host: str = "127.0.0.1", port: int = 8000,
                 db_path: str = "logs/vigileye.db") -> None:
        self.hub = hub
        handler = type("VigilEyeHandler", (_Handler,), {"hub": hub, "db_path": db_path})

        # Port already taken (a previous run still closing) -> try the next few.
        last: Optional[OSError] = None
        for candidate in range(port, port + 10):
            try:
                self._srv = ThreadingHTTPServer((host, candidate), handler)
                self.port = candidate
                break
            except OSError as exc:
                last = exc
        else:
            raise RuntimeError(f"no free port in {port}-{port + 9}: {last}")

        self._srv.daemon_threads = True
        self.host = host
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        self._thread.start()

    def open_browser(self) -> None:
        threading.Thread(target=webbrowser.open, args=(self.url,), daemon=True).start()

    def shutdown(self) -> None:
        self.hub.close()
        self._srv.shutdown()
        self._srv.server_close()
