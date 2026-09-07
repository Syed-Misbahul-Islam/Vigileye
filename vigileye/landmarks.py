"""MediaPipe face landmark extraction — dual backend.

Exposes a single :class:`FaceMeshDetector` that turns a frame into an
``(N, 3)`` array of *pixel-space* landmarks. Everything downstream
(EAR, MAR, head pose, gaze) consumes that array, so the rest of the codebase
has no MediaPipe dependency and is trivially unit-testable.

IMPORTANT — two incompatible MediaPipe APIs
-------------------------------------------
MediaPipe removed the legacy ``mp.solutions`` API in the 0.10.2x series.
Every tutorial and most published drowsiness code still uses it, so a fresh
``pip install mediapipe`` today crashes with::

    AttributeError: module 'mediapipe' has no attribute 'solutions'

This module therefore supports **both**:

* ``solutions``  - legacy ``mp.solutions.face_mesh.FaceMesh`` (<= ~0.10.21)
* ``tasks``      - modern ``mediapipe.tasks.python.vision.FaceLandmarker``

The backend is auto-detected. Both produce the same 478-point topology, so
all index constants below are valid either way.

The Tasks backend needs a model bundle downloaded once::

    python -m vigileye.landmarks --download

Index constants use MediaPipe's canonical 478-point topology. Note that
LEFT/RIGHT naming is *image-space*: LEFT_EYE is the eye appearing on the
left of the image.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# Landmark indices
# --------------------------------------------------------------------------

# EAR six-point sets, ordered p1..p6 as in Soukupova & Cech (2016):
#   p1, p4 = horizontal corners;  p2,p6 and p3,p5 = vertical pairs
LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)

# Eye corners / lids used for gaze normalisation
LEFT_EYE_CORNERS = (33, 133)      # outer, inner
LEFT_EYE_LIDS = (159, 145)        # top, bottom
RIGHT_EYE_CORNERS = (362, 263)    # inner, outer
RIGHT_EYE_LIDS = (386, 374)       # top, bottom

# Iris centres (require refine_landmarks / the full Tasks bundle)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_IRIS = (469, 470, 471, 472)
RIGHT_IRIS = (474, 475, 476, 477)

# Mouth: three vertical pairs + the two corners
MOUTH_VERTICAL = ((81, 178), (13, 14), (311, 402))
MOUTH_CORNERS = (78, 308)

# Six points used for solvePnP head-pose
POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 199,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# Generic 3-D face model (millimetres, nose tip at origin). Values are the
# widely used anthropometric approximation; absolute scale is irrelevant
# because we only need the rotation component.
MODEL_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],           # nose tip
        [0.0, -63.6, -12.5],       # chin
        [-43.3, 32.7, -26.0],      # left eye outer corner
        [43.3, 32.7, -26.0],       # right eye outer corner
        [-28.9, -28.9, -24.1],     # left mouth corner
        [28.9, -28.9, -24.1],      # right mouth corner
    ],
    dtype=np.float64,
)

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def detect_backend() -> str:
    """Return ``'solutions'``, ``'tasks'`` or ``'none'``."""
    try:
        import mediapipe as mp
    except ImportError:
        return "none"
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return "solutions"
    if hasattr(mp, "tasks"):
        return "tasks"
    return "none"


def ensure_model(url: str, dest: Path) -> Path:
    """Download a Tasks model bundle once, if missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"[VigilEye] downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[VigilEye] saved -> {dest}")
    return dest


@dataclass
class FaceResult:
    """Landmarks for one frame."""

    found: bool
    points: Optional[np.ndarray] = None   # (N, 3) pixel coords
    frame_w: int = 0
    frame_h: int = 0

    def xy(self, idx: int) -> np.ndarray:
        return self.points[idx, :2]


class FaceMeshDetector:
    """Backend-agnostic 478-point face landmark detector."""

    def __init__(self, cfg, backend: Optional[str] = None) -> None:
        import mediapipe as mp

        self._mp = mp
        self.backend = backend or detect_backend()
        self._ts_ms = 0

        if self.backend == "solutions":
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=int(cfg.max_num_faces),
                refine_landmarks=bool(cfg.refine_landmarks),
                min_detection_confidence=float(cfg.min_detection_confidence),
                min_tracking_confidence=float(cfg.min_tracking_confidence),
            )

        elif self.backend == "tasks":
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = ensure_model(FACE_MODEL_URL, MODEL_DIR / "face_landmarker.task")
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=int(cfg.max_num_faces),
                min_face_detection_confidence=float(cfg.min_detection_confidence),
                min_tracking_confidence=float(cfg.min_tracking_confidence),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._mesh = vision.FaceLandmarker.create_from_options(options)

        else:
            raise RuntimeError(
                "MediaPipe is not installed or exposes neither the legacy "
                "'solutions' API nor the 'tasks' API. Install with:\n"
                "  pip install 'mediapipe>=0.10.9,<0.10.22'   (legacy API)\n"
                "  pip install mediapipe                       (tasks API)"
            )

    def process(self, rgb: np.ndarray) -> FaceResult:
        """``rgb`` must be an RGB (not BGR) uint8 image."""
        h, w = rgb.shape[:2]

        if self.backend == "solutions":
            rgb.flags.writeable = False
            res = self._mesh.process(rgb)
            rgb.flags.writeable = True
            if not res.multi_face_landmarks:
                return FaceResult(found=False, frame_w=w, frame_h=h)
            landmarks = res.multi_face_landmarks[0].landmark
        else:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
            )
            # detect_for_video requires strictly increasing timestamps
            self._ts_ms += 33
            res = self._mesh.detect_for_video(mp_image, self._ts_ms)
            if not res.face_landmarks:
                return FaceResult(found=False, frame_w=w, frame_h=h)
            landmarks = res.face_landmarks[0]

        pts = np.empty((len(landmarks), 3), dtype=np.float32)
        for i, p in enumerate(landmarks):
            pts[i, 0] = p.x * w
            pts[i, 1] = p.y * h
            pts[i, 2] = p.z * w   # z is scaled like x by MediaPipe convention
        return FaceResult(found=True, points=pts, frame_w=w, frame_h=h)

    def close(self) -> None:
        try:
            self._mesh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MediaPipe backend utilities")
    ap.add_argument("--download", action="store_true", help="fetch Tasks model bundles")
    args = ap.parse_args()

    print(f"Detected MediaPipe backend: {detect_backend()}")
    if args.download:
        ensure_model(FACE_MODEL_URL, MODEL_DIR / "face_landmarker.task")
        ensure_model(HAND_MODEL_URL, MODEL_DIR / "hand_landmarker.task")
        print("Done.")
