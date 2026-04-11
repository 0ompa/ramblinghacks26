"""Runtime ball detector backed by the trained LogisticRegression classifier.

Feature extraction mirrors tools/train_ball.py exactly:
    HOG  (winSize=24, blockSize=12, blockStride=6, cellSize=6, nbins=9)
         → 324-dim, L2-normalised
    HSV 3x3x3 histogram, flattened, L1-normalised → 27-dim
    Concat → 351-dim

Public API
----------
LearnedBallDetector(model_path)
    .score(frame_bgr, cx, cy)             -> float in [0, 1]
    .best_in_roi(frame_bgr, cx, cy,
                 radius=30, stride=3)     -> ((bx, by), best_score)
"""
from __future__ import annotations

import pickle
from typing import Tuple

import cv2
import numpy as np


class LearnedBallDetector:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        self._clf = payload["clf"]
        cfg = payload["hog_config"]
        self._patch = payload["patch_size"]          # 24
        self._hog = cv2.HOGDescriptor(
            cfg["win_size"],
            cfg["block_size"],
            cfg["block_stride"],
            cfg["cell_size"],
            cfg["nbins"],
        )

    # ---------------------------------------------------------------------- #
    # Feature extraction (must match train_ball.py exactly)
    # ---------------------------------------------------------------------- #

    def _features(self, frame_bgr: np.ndarray, cx: float, cy: float
                  ) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        half = self._patch // 2
        x0 = int(round(cx)) - half
        y0 = int(round(cy)) - half
        x1, y1 = x0 + self._patch, y0 + self._patch
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None

        patch = frame_bgr[y0:y1, x0:x1]

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        hog_vec = self._hog.compute(gray)
        if hog_vec is None:
            return None
        hog_vec = hog_vec.flatten().astype(np.float32)
        norm = np.linalg.norm(hog_vec)
        if norm > 1e-6:
            hog_vec /= norm

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None,
                            [3, 3, 3], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten().astype(np.float32)
        s = hist.sum()
        if s > 1e-6:
            hist /= s

        return np.concatenate([hog_vec, hist])       # 351-dim

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def score(self, frame_bgr: np.ndarray, cx: float, cy: float) -> float:
        """predict_proba[1] of the 24x24 patch centred on (cx, cy)."""
        feat = self._features(frame_bgr, cx, cy)
        if feat is None:
            return 0.0
        prob = self._clf.predict_proba(feat.reshape(1, -1))[0, 1]
        return float(prob)

    def best_in_roi(
        self,
        frame_bgr: np.ndarray,
        cx: float,
        cy: float,
        radius: int = 30,
        stride: int = 3,
    ) -> Tuple[Tuple[float, float], float]:
        """Dense sliding window over ROI; returns (best_xy, best_score)."""
        best_xy = (cx, cy)
        best_score = -1.0

        # Batch all candidate locations for a single predict_proba call
        # (much faster than one call per position on sklearn LR).
        candidates: list[tuple[float, float]] = []
        feats: list[np.ndarray] = []

        for dy in range(-radius, radius + 1, stride):
            for dx in range(-radius, radius + 1, stride):
                px, py = cx + dx, cy + dy
                f = self._features(frame_bgr, px, py)
                if f is not None:
                    candidates.append((px, py))
                    feats.append(f)

        if not feats:
            return best_xy, 0.0

        X = np.stack(feats)
        probs = self._clf.predict_proba(X)[:, 1]
        idx = int(np.argmax(probs))
        best_score = float(probs[idx])
        best_xy = candidates[idx]

        return best_xy, best_score
