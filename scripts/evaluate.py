#!/usr/bin/env python3
"""Evaluate VigilEye - accuracy of both fusion modes, and runtime cost.

    # classification quality of the trained LSTM on held-out features
    python scripts/evaluate.py model --features data/features.csv

    # accuracy of the RULE engine against labelled videos (no training needed)
    python scripts/evaluate.py rules --videos data/videos

    # per-stage latency + FPS, i.e. "will this run on a Jetson Nano?"
    python scripts/evaluate.py speed --source 0 --frames 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vigileye.config import load_config          # noqa: E402
from vigileye.fusion import DriverState          # noqa: E402
from vigileye.pipeline import VigilEyePipeline   # noqa: E402
from vigileye.video import VideoStream           # noqa: E402

CLASSES = ["ALERT", "DROWSY", "DISTRACTED"]
STATE_TO_ID = {DriverState.ALERT: 0, DriverState.DROWSY: 1, DriverState.DISTRACTED: 2}


def eval_model(args) -> None:
    import pandas as pd
    import torch
    from sklearn.metrics import classification_report, confusion_matrix

    from vigileye.fusion import FEATURE_NAMES
    from vigileye.temporal_model import TemporalFusion

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_fusion import make_windows

    tf = TemporalFusion(args.checkpoint, window=args.window)
    if not tf.available:
        raise SystemExit(f"No usable checkpoint at {args.checkpoint}")

    df = pd.read_csv(args.features)
    X, y, groups = make_windows(df, tf.window, args.stride)
    Xn = (X - tf.mean) / (tf.std + 1e-6)

    preds = []
    with torch.no_grad():
        for i in range(0, len(Xn), 256):
            batch = torch.from_numpy(Xn[i : i + 256]).float()
            preds.append(tf.model(batch).argmax(dim=-1).numpy())
    preds = np.concatenate(preds)

    print(classification_report(y, preds, target_names=CLASSES, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y, preds))


def eval_rules(args) -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    cfg = load_config(args.config)
    cfg.fusion.mode = "rule"
    root = Path(args.videos)
    labels = {"alert": 0, "drowsy": 1, "distracted": 2}

    y_true, y_pred = [], []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir() or label_dir.name.lower() not in labels:
            continue
        for video in sorted(label_dir.iterdir()):
            if video.suffix.lower() not in {".avi", ".mp4", ".mov", ".mkv"}:
                continue
            pipeline = VigilEyePipeline(cfg, calibrate=False, use_objects=not args.no_objects)
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _, result = pipeline.process(frame, i / fps)
                # Skip the first 5 s: the rolling windows are still filling
                # and every system looks ALERT during warm-up.
                if i / fps > 5.0 and result.state in STATE_TO_ID:
                    y_true.append(labels[label_dir.name.lower()])
                    y_pred.append(STATE_TO_ID[result.state])
                i += 1
            cap.release()
            pipeline.close()
            print(f"processed {video.name}")

    if not y_true:
        raise SystemExit("No frames evaluated.")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    print(confusion_matrix(y_true, y_pred))


def eval_speed(args) -> None:
    cfg = load_config(args.config)
    stream = VideoStream(args.source, cfg.camera.width, cfg.camera.height, cfg.camera.fps, False)

    for label, use_objects in (("face+fusion only", False), ("full (with YOLO+hands)", True)):
        pipeline = VigilEyePipeline(cfg, calibrate=False, use_objects=use_objects)
        times = []
        for _ in range(args.frames):
            ok, frame, ts = stream.read()
            if not ok:
                break
            t0 = time.perf_counter()
            pipeline.process(frame, ts)
            times.append((time.perf_counter() - t0) * 1000.0)
        pipeline.close()

        if times:
            t = np.asarray(times[5:])  # drop warm-up frames
            print(f"\n{label}")
            print(f"  mean {t.mean():7.2f} ms   median {np.median(t):7.2f} ms")
            print(f"  p95  {np.percentile(t,95):7.2f} ms   max {t.max():7.2f} ms")
            print(f"  -> {1000.0/t.mean():.1f} FPS sustained")

    stream.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("model")
    m.add_argument("--features", default="data/features.csv")
    m.add_argument("--checkpoint", default="models/vigileye_lstm.pt")
    m.add_argument("--window", type=int, default=30)
    m.add_argument("--stride", type=int, default=5)
    m.set_defaults(func=eval_model)

    r = sub.add_parser("rules")
    r.add_argument("--videos", required=True)
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--no-objects", action="store_true")
    r.set_defaults(func=eval_rules)

    s = sub.add_parser("speed")
    s.add_argument("--source", default="0")
    s.add_argument("--frames", type=int, default=200)
    s.add_argument("--config", default="config.yaml")
    s.set_defaults(func=eval_speed)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
