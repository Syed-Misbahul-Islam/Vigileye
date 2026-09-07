#!/usr/bin/env python3
"""Train the v2 temporal fusion model (LSTM/GRU) on extracted features.

    python scripts/train_fusion.py --features data/features.csv --epochs 40

Key methodology choices
-----------------------
* **Subject-wise splitting.** Windows from one person never appear in both
  train and test. Frame-level shuffling produces near-perfect scores that
  collapse the moment a new driver sits down.
* **Windows never cross video boundaries.** A window that spans the end of
  one clip and the start of another is a fabricated sample.
* **Class weighting.** Real drowsiness datasets are heavily skewed towards
  ALERT; unweighted training yields a model that predicts ALERT always and
  reports 78% accuracy.
* **Selection on macro-F1, not accuracy.** Missing a micro-sleep is far
  worse than a false alarm, and accuracy hides that entirely.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vigileye.fusion import FEATURE_NAMES                       # noqa: E402
from vigileye.temporal_model import build_model, save_checkpoint  # noqa: E402

CLASSES = ["ALERT", "DROWSY", "DISTRACTED"]


def make_windows(df: pd.DataFrame, window: int, stride: int):
    """Build (X, y, subject) windows without ever crossing a video boundary."""
    X, y, groups = [], [], []
    for (video, subject), g in df.groupby(["video", "subject"], sort=False):
        g = g.sort_values("frame")
        feats = g[FEATURE_NAMES].to_numpy(dtype=np.float32)
        labels = g["label_id"].to_numpy()
        if len(g) < window:
            continue
        for start in range(0, len(g) - window + 1, stride):
            X.append(feats[start : start + window])
            # The window's label is its majority label; with clip-level
            # ground truth this is simply the clip label.
            y.append(np.bincount(labels[start : start + window], minlength=3).argmax())
            groups.append(subject)
    return np.stack(X), np.asarray(y), np.asarray(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--out", default="models/vigileye_lstm.pt")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.25, help="fraction of SUBJECTS held out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.features)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise SystemExit(f"Feature CSV is missing columns: {missing}")
    print(f"Loaded {len(df)} frames, {df['subject'].nunique()} subjects")

    X, y, groups = make_windows(df, args.window, args.stride)
    print(f"Built {len(X)} windows of shape {X.shape[1:]}")
    print("Class balance:", dict(zip(CLASSES, np.bincount(y, minlength=3).tolist())))

    # -- subject-wise split ------------------------------------------------
    subjects = np.unique(groups)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * args.val_frac))
    val_subjects = set(subjects[:n_val].tolist())
    val_mask = np.array([g in val_subjects for g in groups])
    if val_mask.all() or not val_mask.any():
        raise SystemExit("Split failed - you need at least 2 distinct subjects.")

    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xva, yva = X[val_mask], y[val_mask]
    print(f"Train {len(Xtr)} windows / Val {len(Xva)} windows "
          f"(held-out subjects: {sorted(val_subjects)})")

    # -- normalisation (fit on TRAIN only) ---------------------------------
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long()
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xva).float(), torch.from_numpy(yva).long()
    )
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)

    model = build_model(
        n_features=len(FEATURE_NAMES), hidden=args.hidden, layers=args.layers,
        n_classes=3, dropout=args.dropout, cell=args.cell,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.cell.upper()} hidden={args.hidden} layers={args.layers} "
          f"({n_params:,} params)")

    counts = np.bincount(ytr, minlength=3).astype(np.float32)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32).to(device)
    print("Class weights:", weights.cpu().numpy().round(3).tolist())

    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total += loss.item() * len(xb)
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds.append(model(xb.to(device)).argmax(dim=-1).cpu().numpy())
                trues.append(yb.numpy())
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
        acc = (preds == trues).mean()

        print(f"epoch {epoch:3d}  loss {total/len(train_ds):.4f}  "
              f"val_acc {acc:.4f}  val_macroF1 {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            preds.append(model(xb.to(device)).argmax(dim=-1).cpu().numpy())
            trues.append(yb.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)

    print(f"\nBest macro-F1: {best_f1:.4f}\n")
    print(classification_report(trues, preds, target_names=CLASSES, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(trues, preds))

    save_checkpoint(
        args.out, model, mean, std, args.window, CLASSES,
        extra={"val_macro_f1": float(best_f1), "features": FEATURE_NAMES},
    )
    print(f"\nSaved -> {args.out}")
    print("Run live with:  python run.py --mode hybrid")


if __name__ == "__main__":
    main()
