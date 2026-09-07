"""The v2 learned fusion model: a small LSTM/GRU over windowed features.

Input : (batch, window, n_features)  - window ~= 1-2 seconds of frames
Output: (batch, 3) logits over [ALERT, DROWSY, DISTRACTED]

The model is deliberately tiny (~40k params). It must run at 30 fps on a
Jetson Nano *alongside* MediaPipe and YOLO, and it is trained on a dataset
of at most a few hundred subjects - a larger network would simply memorise
the subjects.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np

CLASSES = ["ALERT", "DROWSY", "DISTRACTED"]


def _torch():
    import torch

    return torch


class DrowsinessLSTM:
    """Factory shim so the module imports cleanly without torch installed."""

    def __new__(cls, *args, **kwargs):
        return build_model(*args, **kwargs)


def build_model(
    n_features: int,
    hidden: int = 64,
    layers: int = 1,
    n_classes: int = 3,
    dropout: float = 0.3,
    cell: str = "lstm",
):
    torch = _torch()
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_cls = nn.LSTM if cell.lower() == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
                bidirectional=False,
            )
            self.norm = nn.LayerNorm(hidden)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_classes)
            )

        def forward(self, x):
            out, _ = self.rnn(x)
            # Attention-free temporal pooling: mean over the window plus the
            # final state. Mean gives stability, last state gives recency.
            pooled = 0.5 * out.mean(dim=1) + 0.5 * out[:, -1, :]
            return self.head(self.drop(self.norm(pooled)))

    net = _Net()
    net.n_features = n_features
    net.cell = cell
    net.hidden = hidden
    net.layers = layers
    return net


class TemporalFusion:
    """Sliding-window inference wrapper used by the live pipeline."""

    def __init__(self, checkpoint: str | Path, window: int = 30, device: str = "cpu") -> None:
        torch = _torch()
        self.torch = torch
        self.device = torch.device(device)
        self.window = int(window)
        self.available = False
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.window)

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            print(f"[VigilEye] No temporal checkpoint at {ckpt_path}; using rules only.")
            return

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.mean = np.asarray(ckpt["mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["std"], dtype=np.float32)
        self.window = int(ckpt.get("window", self.window))
        self.buffer = deque(maxlen=self.window)
        self.classes = ckpt.get("classes", CLASSES)

        self.model = build_model(
            n_features=int(ckpt["n_features"]),
            hidden=int(ckpt.get("hidden", 64)),
            layers=int(ckpt.get("layers", 1)),
            n_classes=len(self.classes),
            cell=ckpt.get("cell", "lstm"),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.available = True

    def reset(self) -> None:
        self.buffer.clear()

    def update(self, feature_vec: np.ndarray) -> Optional[Dict[str, float]]:
        """Push one frame; return class probabilities once the window is full."""
        if not self.available:
            return None
        self.buffer.append(np.asarray(feature_vec, dtype=np.float32))
        if len(self.buffer) < self.window:
            return None

        arr = np.stack(self.buffer, axis=0)
        arr = (arr - self.mean) / (self.std + 1e-6)
        with self.torch.no_grad():
            x = self.torch.from_numpy(arr[None, ...]).float().to(self.device)
            probs = self.torch.softmax(self.model(x), dim=-1)[0].cpu().numpy()
        return {c: float(p) for c, p in zip(self.classes, probs)}


def save_checkpoint(path, model, mean, std, window, classes, extra=None) -> None:
    torch = _torch()
    payload = {
        "state_dict": model.state_dict(),
        "n_features": int(model.n_features),
        "hidden": int(model.hidden),
        "layers": int(model.layers),
        "cell": model.cell,
        "mean": np.asarray(mean).tolist(),
        "std": np.asarray(std).tolist(),
        "window": int(window),
        "classes": list(classes),
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
