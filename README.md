# VigilEye

**Real-time multimodal driver drowsiness, fatigue & distraction detection**
Smart India Hackathon 2026

A camera-only driver monitoring system that fuses eye closure, yawning, head
pose, gaze and object cues into one stable state: **ALERT**, **DROWSY** or
**DISTRACTED**.

---

## Table of contents

1. [How it works](#1-how-it-works)
2. [Install](#2-install)
3. [Run it (15 minutes to a working demo)](#3-run-it)
4. [Tuning](#4-tuning)
5. [Datasets](#5-datasets)
6. [Training the v2 fusion model](#6-training-the-v2-fusion-model)
7. [Evaluation](#7-evaluation)
8. [Fleet dashboard](#8-fleet-dashboard)
9. [Edge deployment](#9-edge-deployment)
10. [Build schedule](#10-build-schedule)
11. [Demo script](#11-demo-script)
12. [Known limitations](#12-known-limitations)
13. [File map](#13-file-map)

---

## 1. How it works

```
                    ┌──────────────────────────────────────┐
   dashboard cam →  │  MediaPipe Face Mesh (478 landmarks) │
                    └───────────────┬──────────────────────┘
                                    │
        ┌───────────┬───────────────┼───────────────┬────────────┐
        ▼           ▼               ▼               ▼            ▼
     EAR/PERCLOS  MAR/yawn     solvePnP head    iris gaze    YOLOv8n phone
     blinks       detection    yaw/pitch/roll   h/v ratio    + MediaPipe Hands
     microsleep                nod / turn       eyes-off     hands-off-wheel
        │           │               │               │            │
        └───────────┴───────────────┼───────────────┴────────────┘
                                    ▼
                    ┌──────────────────────────────────────┐
                    │  FUSION                              │
                    │  v1 weighted rules  (works day one)  │
                    │  v2 LSTM over 30-frame windows       │
                    │  hybrid = blend of both              │
                    │  + hysteresis / dwell-time gating    │
                    └───────────────┬──────────────────────┘
                                    ▼
                    ALERT · DROWSY · DISTRACTED · NO_DRIVER
                                    │
                    ┌───────────────┴──────────────────────┐
                    ▼                                      ▼
            escalating audio-visual alerts      SQLite event log → dashboard
```

### The three decisions that matter

**Separate drowsy and distract scores, not one "risk" number.**
They demand opposite interventions — a drowsy driver must stop and rest, a
distracted driver must look up *now*. Collapsing them destroys the most
useful thing the system knows.

**Per-driver calibration instead of fixed thresholds.**
A hard-coded EAR of 0.21 is the biggest false-positive source in published
systems. Eye aperture varies enormously between people; drivers with narrow
eyes or glasses sit permanently below threshold. Ten seconds of baseline
capture personalises the thresholds *and* captures camera placement, without
which head-pose thresholds are meaningless.

**Hysteresis on every state change.**
8 confirming frames to enter an unsafe state, 20 to leave it, 1.5 s minimum
dwell. This is the difference between a system drivers use and one they
unplug. There is an explicit regression test asserting a single bad frame
cannot flip the output.

---

## 2. Install

```bash
git clone <your-repo> vigileye && cd vigileye
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Two install traps, both handled

**1. numpy 2.0.** MediaPipe breaks on `numpy >= 2.0` with a cryptic
`_ARRAY_API not found` error. The pin in `requirements.txt` handles it — do
not upgrade numpy past 2.0.

**2. MediaPipe removed `mp.solutions`.** In the 0.10.2x series MediaPipe
deleted the legacy solutions API that every tutorial (and most published
drowsiness code) still uses. On a current release, `mp.solutions.face_mesh`
raises `AttributeError: module 'mediapipe' has no attribute 'solutions'`.

VigilEye supports **both** APIs and auto-detects which you have. Check with:

```bash
python -m vigileye.landmarks       # prints: solutions | tasks | none
```

* `solutions` — legacy API, nothing more to do.
* `tasks` — modern API, needs a one-time model download:

```bash
python -m vigileye.landmarks --download
```

Both backends produce the identical 478-point topology, so every threshold,
index constant and trained model works unchanged across them.

For training only, the CPU torch build is ~5× smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify the install:

```bash
pytest -q                 # expect: 20 passed
```

---

## 3. Run it

```bash
python run.py
```

That is the whole demo. On first launch it spends 10 seconds calibrating —
**look straight ahead at the road normally** — then starts classifying. The
baseline is cached to `calibration.json` and reused next time.

Useful flags:

```bash
python run.py --source data/clip.mp4   # run against a video file
python run.py --mode hybrid            # rules + trained LSTM
python run.py --no-objects             # skip YOLO/hands (low-power / weak CPU)
python run.py --recalibrate            # force a fresh baseline
python run.py --headless               # no window, for embedded deployment
python run.py --record out.mp4         # save annotated video for your pitch
```

Keys while running: `q` quit · `c` recalibrate · `m` mute · `h` hide HUD ·
`s` save frame.

### Prove it works, in this order

1. **Blink normally** → stays ALERT. (If it flickers, calibration failed —
   press `c` and hold still.)
2. **Close your eyes ~2 s** → DROWSY within a second, alert fires.
3. **Yawn twice** → yawn counter increments on the HUD.
4. **Look at the passenger seat for 3 s** → DISTRACTED.
5. **Hold a phone up** → DISTRACTED with a YOLO box on the phone.

If step 1 fails, nothing downstream matters. Fix calibration first.

---

## 4. Tuning

Every threshold lives in `config.yaml`. Nothing is hard-coded in the
detectors, so you retune for a new camera or vehicle without touching Python.

| Symptom | Change |
|---|---|
| Fires DROWSY while you're alert | lower `eye.calib_ratio` (0.75 → 0.65) |
| Misses obvious eye closure | raise `eye.calib_ratio` (0.75 → 0.82) |
| Talking counts as yawning | raise `mouth.yawn_min_s` (1.2 → 1.8) |
| Distraction too twitchy | raise `head.off_road_min_s` (2.0 → 3.0) |
| Alerts too naggy | raise `alerts.cooldown_s`, `fusion.min_state_s` |
| State flickers | raise `fusion.confirm_frames` |
| Too slow on your machine | raise `objects.every_n_frames` (5 → 10), or `--no-objects` |

`fusion.weights` controls how much each cue contributes. The defaults are
deliberately conservative; retune them against your own labelled footage
rather than by feel.

---

## 5. Datasets

Organise videos so the parent directory is the label:

```
data/videos/
    alert/        subject01_noglasses_normal.avi
    drowsy/       subject01_glasses_sleepy.avi
    distracted/   subject03_talking.avi
```

| Dataset | Contains | Use for |
|---|---|---|
| **NTHU-DDD** | 36 subjects, drowsy/alert, glasses + night IR | main train/test set |
| **YawDD** | dashboard-mounted yawning clips | MAR threshold validation |
| **Your own footage** | Indian lighting, diverse skin tones, real dash placement | robustness claim |

NTHU-DDD requires an academic request form — **start that on day one**, it
can take a week or more to be granted. Build against your own recordings in
the meantime; the pipeline does not care where the video came from.

**Record your own footage.** This is your strongest differentiator and the
one claim in the abstract you cannot currently defend. Aim for 8–10 people,
day and night, with and without glasses, in an actual parked vehicle with the
camera where it would really be mounted. Two hours of this is worth more to
the judges than another model architecture.

---

## 6. Training the v2 fusion model

**Step 1 — extract features.**

```bash
python scripts/collect_features.py \
    --videos data/videos --out data/features.csv --stride 2 --no-objects
```

This runs the *exact same* `VigilEyePipeline` the live system uses, which
guarantees training features match inference features byte for byte.
Divergence there is the classic cause of a model that scores 97% offline and
fails in the vehicle. `--no-objects` skips YOLO and is roughly 4× faster;
use it unless you specifically need phone cues in the training data.

**Step 2 — train.**

```bash
python scripts/train_fusion.py --features data/features.csv --epochs 40
```

The script enforces three things that most published drowsiness results get
wrong:

- **Subject-wise splitting.** Whole *people* are held out, never frames.
  Frame-level shuffling leaks the same face into train and test and inflates
  accuracy by 15–25 points. If a judge asks one hard methodology question,
  it will be this one.
- **Windows never cross video boundaries.** A window spanning the end of one
  clip and the start of another is a fabricated sample.
- **Class weighting + macro-F1 selection.** Real datasets are heavily skewed
  towards ALERT; unweighted training yields a model that always predicts
  ALERT and reports 78% accuracy. Missing a micro-sleep is far worse than a
  false alarm, and plain accuracy hides that entirely.

**Step 3 — run the hybrid.**

```bash
python run.py --mode hybrid
```

`hybrid` blends the rule score with the LSTM's class probabilities
(`temporal.blend`, default 0.5). The rule engine still owns the hysteresis
and still works from frame one, so an under-trained model degrades the system
gracefully instead of breaking it.

### What score to expect

On NTHU-DDD with honest subject-wise splits, **75–88% macro-F1** is a
realistic range. If you see 97%+, you have a leak — check that no subject
appears in both splits.

---

## 7. Evaluation

```bash
# LSTM quality on held-out features
python scripts/evaluate.py model --features data/features.csv

# rule engine accuracy against labelled videos (no training required)
python scripts/evaluate.py rules --videos data/videos

# per-stage latency and sustained FPS — "will this run on a Jetson Nano?"
python scripts/evaluate.py speed --source 0 --frames 200
```

The `speed` benchmark reports with and without YOLO separately, which is the
number you need for the edge-deployment claim. Report **p95 latency**, not
mean — a system that averages 30 FPS but stalls for 400 ms during a
micro-sleep is not a safety system.

---

## 8. Fleet dashboard

```bash
streamlit run dashboard/app.py
```

Reads the SQLite event store and shows alerts by type, alerts by hour of day
(the fatigue-peak story), risk score over time, and a driver risk ranking
normalised per hour driven. This is what makes the system valuable to a fleet
operator rather than only to the driver in the seat — do not skip it in the
pitch.

---

## 9. Edge deployment

Measured design targets on a Jetson Nano / Pi 4 class device:

| Config | Expected |
|---|---|
| Face mesh + fusion only (`--no-objects`) | 25–30 FPS |
| Full pipeline, YOLO every 5th frame | 12–18 FPS |

Tips that actually matter:

- `objects.every_n_frames: 10` on a Pi — YOLO dominates the budget and phone
  detections are held between runs anyway.
- Drop `camera.width/height` to 480×360 before touching anything else.
- Export YOLO to TensorRT on Jetson: `yolo export model=yolov8n.pt format=engine`.
- Run `--headless` and let the dashboard read the SQLite file; rendering the
  HUD costs real frames.
- Set `logging.frame_log: false` in production. Per-frame CSV is for training
  data collection, not for shipping.

---

## 10. Build schedule

A realistic order that keeps you demo-ready at every checkpoint:

| Phase | Work | Checkpoint |
|---|---|---|
| 1 | Install, `python run.py`, verify calibration | live EAR/MAR on screen |
| 2 | Tune `config.yaml` on yourself + 2 teammates | reliable drowsy detection |
| 3 | Request NTHU-DDD; record your own footage | 8–10 subjects captured |
| 4 | `collect_features.py` → `train_fusion.py` | trained checkpoint |
| 5 | `evaluate.py` all three modes | numbers for the slides |
| 6 | Dashboard + edge benchmark + demo video | full pitch |

Phases 1–2 give you a working demo. Everything after is what turns a demo
into a submission.

---

## 11. Demo script

Five minutes, in this order:

1. **Launch, calibrate on a judge.** Shows it adapts to a new face in 10 s.
2. **Normal blinking → stays ALERT.** Leads with the false-positive problem,
   which is what everyone else's demo fails at.
3. **Eyes closed 2 s → DROWSY + alert.** The core capability.
4. **Look away 3 s → DISTRACTED.** Emphasise: *different state, different
   intervention* — this is the differentiator.
5. **Phone in frame → DISTRACTED + YOLO box.**
6. **Cut to the dashboard.** "Here's what the fleet operator sees."

Have `--record` output on standby in case the venue lighting defeats the live
camera. Record it the night before with the actual demo laptop.

---

## 12. Known limitations

State these before a judge finds them. Owning a limitation reads as rigour;
being caught by one reads as overclaiming.

- **The Indian-conditions claim is design intent, not a measured result.**
  Nothing here has been validated across skin tones or Indian lighting yet.
  That needs your own recorded, labelled footage.
- **Hands-off-wheel uses MediaPipe Hands + a configurable ROI**, not YOLO —
  COCO has no "hand" class. It works day one but the ROI
  (`objects.hands.wheel_roi`) must be set per vehicle.
- **PERCLOS is inherently slow.** It is a rolling-minute statistic, so gradual
  fatigue takes ~45–60 s to register. Acute events are caught immediately by
  the micro-sleep and long-closure overrides; this is a deliberate trade-off,
  not a bug.
- **Sunglasses defeat EAR and gaze entirely.** The system falls back to head
  pose and yawning. An IR camera is the real fix.
- **The camera pose model is a pinhole approximation** (focal length ≈ image
  width). Fine for detecting pose *change*; use `cv2.calibrateCamera` if you
  ever need absolute angles.
- **Not a medical or legal device.** It is a driver-assistance aid.
- **The Tasks backend downloads model bundles from Google's CDN on first
  run.** Fetch them ahead of time (`python -m vigileye.landmarks --download`)
  — do not rely on venue wifi during a demo.

---

## 13. File map

```
vigileye/
├── config.yaml                 every tunable threshold
├── requirements.txt
├── run.py                      live runner (CLI, HUD, alerts, logging)
│
├── vigileye/
│   ├── config.py               YAML + defaults, dot-access
│   ├── landmarks.py            MediaPipe wrapper, index constants, 3-D model
│   ├── metrics.py              EAR, MAR, PERCLOS, blinks, micro-sleeps, yawns
│   ├── head_pose.py            solvePnP, Euler decomposition, nod/turn
│   ├── gaze.py                 iris-based gaze, eyes-off-road window
│   ├── distraction.py          YOLOv8n phone + MediaPipe hands-off-wheel
│   ├── fusion.py               scoring, hysteresis state machine, hybrid
│   ├── temporal_model.py       LSTM/GRU + sliding-window inference
│   ├── calibration.py          per-driver baseline
│   ├── alerts.py               escalation, cooldown, audio fallbacks
│   ├── logger.py               frame CSV + SQLite event store
│   ├── visualize.py            HUD overlay
│   ├── video.py                threaded capture (drops stale frames)
│   └── pipeline.py             orchestrates everything
│
├── scripts/
│   ├── collect_features.py     videos → features.csv
│   ├── train_fusion.py         subject-wise LSTM training
│   └── evaluate.py             model / rules / speed benchmarks
│
├── dashboard/app.py            Streamlit fleet analytics
└── tests/test_metrics.py       20 unit tests
```

### Two bugs already found and fixed by the test suite

Worth knowing about, because both are easy to reintroduce:

**Falsy-zero in micro-sleep timing.** `self._closed_since or ts` treats
timestamp `0.0` as "not started", because `0.0` is falsy in Python. Live
webcam use hides this (Unix epoch timestamps), but *every video file starts
at t=0.0* — so offline feature extraction would silently never emit a
micro-sleep, and the v2 model would train on broken labels. Now uses an
explicit `is not None` check, locked behind a regression test.

**PERCLOS cold start.** One closed frame at startup made PERCLOS read 100%,
pushing the drowsy score to 0.40 against a 0.45 threshold — one frame of
noise away from a false alarm on every launch. Its influence now ramps in
over the first 15 seconds of window coverage.
