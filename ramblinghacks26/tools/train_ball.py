"""Train a ball/not-ball patch classifier from hand labels.

Usage:
    python tools/train_ball.py \
        --labels labels/clip1_ball.json \
        --input  clips/clip1.mp4 \
        --out    models/ball_clf.pkl

Features per 24x24 patch:
    HOG  (winSize=24, blockSize=12, blockStride=6, cellSize=6, nbins=9)
         → 324-dim, L2-normalised
    HSV 3×3×3 histogram, flattened, L1-normalised → 27-dim
    Concat → 351-dim

Positives  : labeled center + 4 jitter copies (±2 px) ≈ 5 per frame
Negatives  : 100 random 24x24 patches per frame whose center is ≥40 px from ball

Classifier : LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Optional

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

PATCH = 24
HOG_BLOCK = 12
HOG_STRIDE = 6
HOG_CELL = 6
HOG_BINS = 9

# Expected feature dimension: computed at import time for reference.
# HOG: ((24-12)//6+1)^2 * (12//6)^2 * 9 = 3*3 * 4 * 9 = 324
# HSV hist: 3*3*3 = 27   total = 351

NEG_MIN_DIST = 40     # minimum distance from ball for a negative patch center
N_NEGS_PER_FRAME = 100
N_JITTER = 4          # extra positive copies per label (random ±2 px)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="labels/clip1_ball.json")
    p.add_argument("--input",  default="clips/clip1.mp4")
    p.add_argument("--out",    default="models/ball_clf.pkl")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def _make_hog():
    return cv2.HOGDescriptor(
        (PATCH, PATCH),           # winSize
        (HOG_BLOCK, HOG_BLOCK),   # blockSize
        (HOG_STRIDE, HOG_STRIDE), # blockStride
        (HOG_CELL, HOG_CELL),     # cellSize
        HOG_BINS,
    )

HOG = _make_hog()


def extract_patch(frame_bgr: np.ndarray, cx: float, cy: float
                  ) -> Optional[np.ndarray]:
    """Return the 24x24 BGR patch centred on (cx,cy), or None if out of bounds."""
    h, w = frame_bgr.shape[:2]
    half = PATCH // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + PATCH, y0 + PATCH
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return frame_bgr[y0:y1, x0:x1].copy()


def patch_features(patch_bgr: np.ndarray) -> np.ndarray:
    """351-dim feature vector for a 24x24 BGR patch."""
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    hog_vec = HOG.compute(gray).flatten()                 # 324-dim
    norm = np.linalg.norm(hog_vec)
    if norm > 1e-6:
        hog_vec = hog_vec / norm

    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [3, 3, 3], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    s = hist.sum()
    if s > 1e-6:
        hist /= s

    return np.concatenate([hog_vec, hist])                # 351-dim


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    with open(args.labels) as f:
        data = json.load(f)
    labels_list = data["frames"]
    print(f"[train] {len(labels_list)} labeled frames loaded", file=sys.stderr)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.input}", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rng = np.random.default_rng(42)

    X: list[np.ndarray] = []
    y: list[int] = []

    half = PATCH // 2

    for entry in labels_list:
        fi = int(entry["f"])
        bx, by = float(entry["xy"][0]), float(entry["xy"][1])

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  [warn] could not read frame {fi}, skipping", file=sys.stderr)
            continue

        # --- positives: center + jitter ------------------------------------- #
        jitter_offsets = [(0, 0)] + [
            (rng.integers(-2, 3), rng.integers(-2, 3)) for _ in range(N_JITTER)
        ]
        for dx, dy in jitter_offsets:
            patch = extract_patch(frame, bx + dx, by + dy)
            if patch is not None:
                X.append(patch_features(patch))
                y.append(1)

        # --- negatives: random patches ≥40 px from ball --------------------- #
        n_neg = 0
        attempts = 0
        max_attempts = N_NEGS_PER_FRAME * 20
        while n_neg < N_NEGS_PER_FRAME and attempts < max_attempts:
            attempts += 1
            nx = rng.integers(half, W - half)
            ny = rng.integers(half, H - half)
            if np.hypot(nx - bx, ny - by) < NEG_MIN_DIST:
                continue
            patch = extract_patch(frame, nx, ny)
            if patch is None:
                continue
            X.append(patch_features(patch))
            y.append(0)
            n_neg += 1

    cap.release()

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    pos = (y_arr == 1).sum()
    neg = (y_arr == 0).sum()
    print(f"[train] dataset: {pos} positives, {neg} negatives, "
          f"{X_arr.shape[1]}-dim features", file=sys.stderr)

    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=2000, solver="lbfgs")
    cv_scores = cross_val_score(clf, X_arr, y_arr, cv=5, scoring="accuracy")
    print(f"[train] 5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
          file=sys.stderr)

    if cv_scores.mean() < 0.90:
        print("[train] STOP — CV < 0.90. Add more labels or inspect failures.",
              file=sys.stderr)
        sys.exit(1)

    clf.fit(X_arr, y_arr)
    train_acc = (clf.predict(X_arr) == y_arr).mean()
    print(f"[train] train accuracy: {train_acc:.3f}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    payload = {
        "clf": clf,
        "hog_config": {
            "win_size": (PATCH, PATCH),
            "block_size": (HOG_BLOCK, HOG_BLOCK),
            "block_stride": (HOG_STRIDE, HOG_STRIDE),
            "cell_size": (HOG_CELL, HOG_CELL),
            "nbins": HOG_BINS,
        },
        "patch_size": PATCH,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"[train] model saved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
