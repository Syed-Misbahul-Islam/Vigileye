#!/usr/bin/env python3
"""VigilEye live runner.

    python run.py                          # webcam + browser cockpit
    python run.py --source data/test.mp4   # run against a video file
    python run.py --mode hybrid            # rules + trained LSTM
    python run.py --no-objects             # skip YOLO/hands (low-power mode)
    python run.py --recalibrate            # force a fresh baseline
    python run.py --window                 # also show the OpenCV window
    python run.py --no-ui                  # OpenCV window only, no web UI
    python run.py --headless               # no window, no browser (embedded)

The browser cockpit exposes the same controls as the desktop window:
recalibrate, mute, snapshot and stop.

Keys (OpenCV window): q quit | c recalibrate | m mute | h hide HUD | s save frame
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2

from vigileye.alerts import AlertManager
from vigileye.config import load_config
from vigileye.fusion import DriverState
from vigileye.logger import SessionLogger
from vigileye.pipeline import VigilEyePipeline
from vigileye.video import VideoStream
from vigileye.pico_alert import close_pico, pico_status, send_to_pico
from vigileye.visualize import (
    draw_detections,
    draw_hud,
    draw_landmarks,
    draw_pose_axis,
    draw_wheel_roi,
)
from vigileye.webui import TelemetryHub, WebUI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VigilEye driver monitoring")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None, help="camera index or video path")
    p.add_argument("--mode", choices=["rule", "temporal", "hybrid"], default=None)
    p.add_argument("--driver-id", default="driver_01")
    p.add_argument("--vehicle-id", default="vehicle_01")
    p.add_argument("--no-objects", action="store_true", help="disable YOLO + hands")
    p.add_argument("--no-alerts", action="store_true")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--headless", action="store_true", help="no window and no browser")
    p.add_argument("--record", default=None, help="write annotated video to this path")
    p.add_argument("--no-ui", action="store_true", help="disable the browser cockpit")
    p.add_argument("--ui-port", type=int, default=8000)
    p.add_argument("--ui-host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true", help="serve the UI but don't open a tab")
    p.add_argument("--window", action="store_true", help="also show the OpenCV window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.source is not None:
        cfg.camera.source = args.source
    if args.mode is not None:
        cfg.fusion.mode = args.mode
    if args.no_alerts:
        cfg.alerts.enabled = False

    if args.recalibrate:
        Path(cfg.calibration.file).unlink(missing_ok=True)

    stream = VideoStream(
        cfg.camera.source, cfg.camera.width, cfg.camera.height,
        cfg.camera.fps, cfg.camera.flip,
    )
    pipeline = VigilEyePipeline(cfg, use_objects=not args.no_objects)
    alerts = AlertManager(cfg.alerts)
    logger = None if args.no_log else SessionLogger(cfg.logging, args.driver_id, args.vehicle_id)

    ui_enabled = not args.no_ui
    show_window = (not args.headless) and (args.window or not ui_enabled)

    hub = None
    webui = None
    if ui_enabled:
        hub = TelemetryHub(args.driver_id, args.vehicle_id)
        if logger:
            hub.session_id = logger.session_id
        webui = WebUI(hub, host=args.ui_host, port=args.ui_port, db_path=str(cfg.logging.db))
        webui.start()
        print(f"[VigilEye] command deck: {webui.url}")
        if not args.headless and not args.no_browser:
            webui.open_browser()

    backend = f"MediaPipe {pipeline.face.backend} · {cfg.fusion.mode}"
    writer = None
    fps_hist = deque(maxlen=30)
    show_hud = True
    muted = False
    frames = 0
    t_start = time.time()
    previous_state = None
    running = True
    want_snapshot = False

    if logger:
        print(f"[VigilEye] session {logger.session_id} started")
    print("[VigilEye] running - press 'q' in the window or Stop in the browser to quit")

    try:
        while running:
            ok, frame, ts = stream.read()
            if not ok:
                break

            t0 = time.perf_counter()
            signals, result = pipeline.process(frame, ts)
            if not pipeline.calibrating:
                current_state = (
                    result.state.value
                    if hasattr(result.state, "value")
                    else str(result.state)
                )

                if current_state != previous_state:
                    send_to_pico(current_state)
                    previous_state = current_state
            latency_ms = (time.perf_counter() - t0) * 1000.0
            fps_hist.append(1000.0 / max(latency_ms, 1e-3))
            fps = sum(fps_hist) / len(fps_hist)
            frames += 1

            event = None
            if not muted and not pipeline.calibrating:
                event = alerts.update(ts, result.state, pipeline.time_in_state(ts))

            if logger:
                logger.log_frame(signals, result)
                if event:
                    logger.log_alert(event, result, signals.perclos)
            if event:
                print(f"[ALERT L{event.level}] {event.message}")
                if hub:
                    hub.add_alert(event)

            # -- commands from the browser ---------------------------------
            if hub:
                for action in hub.drain_commands():
                    if action == "recalibrate":
                        pipeline.recalibrate()
                        print("[VigilEye] recalibrating...")
                    elif action in ("mute", "unmute"):
                        muted = action == "mute"
                        print(f"[VigilEye] alerts {'muted' if muted else 'unmuted'}")
                    elif action == "snapshot":
                        want_snapshot = True
                    elif action == "stop":
                        print("[VigilEye] stop requested from the browser")
                        running = False

            calib_progress = (
                pipeline.calibrator.progress(ts) if pipeline.calibrating else -1.0
            )

            if hub:
                hub.publish(
                    signals, result,
                    fps=fps, latency_ms=latency_ms,
                    calibrating=pipeline.calibrating,
                    calib_progress=max(calib_progress, 0.0),
                    muted=muted, backend=backend, pico=pico_status(),
                )

            if show_window or args.record or hub:
                canvas = frame.copy()
                if show_hud:
                    if pipeline.last_points is not None:
                        draw_landmarks(canvas, pipeline.last_points)
                        if pipeline.last_pose is not None:
                            draw_pose_axis(canvas, pipeline.last_points, pipeline.last_pose)
                    draw_wheel_roi(canvas, pipeline.hands.roi)
                    draw_detections(canvas, pipeline.last_detections)

                # The browser renders its own metrics, so the painted HUD panel
                # is only worth the pixels on the desktop window / recording.
                if show_hud and (show_window or args.record):
                    canvas = draw_hud(
                        canvas, signals, result, fps,
                        flashing=alerts.is_flashing(ts),
                        calibrating=calib_progress,
                    )

                if hub:
                    hub.publish_frame(canvas)

                if want_snapshot:
                    name = f"snapshot_{int(time.time())}.png"
                    cv2.imwrite(name, canvas)
                    print(f"[VigilEye] saved {name}")
                    want_snapshot = False

                if args.record:
                    if writer is None:
                        h, w = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h)
                        )
                    writer.write(canvas)

                if show_window:
                    cv2.imshow("VigilEye", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("c"):
                        pipeline.recalibrate()
                        print("[VigilEye] recalibrating...")
                    if key == ord("m"):
                        muted = not muted
                        print(f"[VigilEye] alerts {'muted' if muted else 'unmuted'}")
                    if key == ord("h"):
                        show_hud = not show_hud
                    if key == ord("s"):
                        want_snapshot = True

    except KeyboardInterrupt:
        print("\n[VigilEye] interrupted")
    finally:
        elapsed = max(time.time() - t_start, 1e-6)
        mean_fps = frames / elapsed
        stream.release()
        pipeline.close()
        close_pico()
        if writer is not None:
            writer.release()
        if logger:
            logger.close(mean_fps)
        if webui is not None:
            webui.shutdown()
        cv2.destroyAllWindows()
        print(f"[VigilEye] {frames} frames in {elapsed:.1f}s ({mean_fps:.1f} FPS)")


if __name__ == "__main__":
    main()
