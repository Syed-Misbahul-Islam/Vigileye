#!/usr/bin/env python3
"""Extract per-frame feature vectors from labelled videos.

Expected layout (works for NTHU-DDD and YawDD once organised this way):

    data/videos/
        alert/       subject01_noglasses_normal.avi
        drowsy/      subject01_glasses_sleepy.avi
        distracted/  subject03_talking.avi

The label comes from the parent directory; the subject id is parsed from
the filename so that ``train_fusion.py`` can split by subject rather than
by frame. Splitting by frame leaks the same person into train and test and
inflates accuracy by 15-25 points - it is the most common mistake in
published drowsiness results.

Usage
-----
    python scripts/collect_features.py --videos data/videos --out data/features.csv
    python scripts/collect_features.py --videos data/videos --stride 2 --no-objects
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vigileye.config import load_config           # noqa: E402
from vigileye.fusion import FEATURE_NAMES         # noqa: E402
from vigileye.pipeline import VigilEyePipeline    # noqa: E402

LABELS = {"alert": 0, "drowsy": 1, "distracted": 2}
VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".MP4", ".AVI"}


def subject_from_name(path: Path) -> str:
    """Pull a subject id out of the filename; fall back to the stem."""
    m = re.search(r"(subject|sub|p|s)[\s_-]?(\d+)", path.stem, re.IGNORECASE)
    if m:
        return f"subject{int(m.group(2)):03d}"
    return path.stem.split("_")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="root dir with alert/ drowsy/ distracted/")
    ap.add_argument("--out", default="data/features.csv")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stride", type=int, default=1, help="keep 1 of every N frames")
    ap.add_argument("--no-objects", action="store_true", help="skip YOLO/hands (much faster)")
    ap.add_argument("--limit-per-video", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.videos)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    videos = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir() or label_dir.name.lower() not in LABELS:
            continue
        for v in sorted(label_dir.iterdir()):
            if v.suffix in VIDEO_EXT:
                videos.append((v, label_dir.name.lower()))

    if not videos:
        raise SystemExit(f"No videos found under {root}. Expected subdirs: {list(LABELS)}")

    print(f"Found {len(videos)} videos")
    header = ["video", "subject", "label", "label_id", "frame", "ts"] + FEATURE_NAMES
    n_rows = 0

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for vi, (path, label) in enumerate(videos, 1):
            # A fresh pipeline per video resets every rolling window, so
            # PERCLOS from clip N never bleeds into clip N+1.
            pipeline = VigilEyePipeline(
                cfg, calibrate=False, use_objects=not args.no_objects
            )
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_i = kept = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_i % args.stride == 0:
                    ts = frame_i / fps
                    signals, _ = pipeline.process(frame, ts)
                    if signals.face_found:
                        writer.writerow(
                            [path.name, subject_from_name(path), label, LABELS[label],
                             frame_i, round(ts, 4)]
                            + [round(float(v), 6) for v in signals.to_vector()]
                        )
                        kept += 1
                        n_rows += 1
                frame_i += 1
                if args.limit_per_video and kept >= args.limit_per_video:
                    break

            cap.release()
            pipeline.close()
            print(f"[{vi}/{len(videos)}] {path.name:<45} {label:<11} {kept:6d} rows")

    print(f"\nWrote {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
