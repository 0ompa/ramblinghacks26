"""ROI-scoped learned verifier for basketball tracking.

The LK tracker is physically grounded but has no notion of "is this patch
still a ball?". Once a feature point slips onto a nearby distractor (a court
line, a jersey seam) the median drags the whole track with it.

This verifier answers that one question: *given a candidate (x,y), does the
local image patch look like the ball we seeded from?* It is fit **once** on
the seed frame from a single positive (the user-supplied seed point, plus a
handful of pixel jitters) against a few hundred random negative patches
sampled from the same frame. Scoring is nearest-exemplar in HOG feature
space, normalized to [0,1]:

    score = d_neg_min / (d_pos_min + d_neg_min)

So score > 0.5 means "closer to a known-ball patch than to any random
negative". The tracker uses this to veto LK drift and, when LK fails, to
pick a better candidate from a small grid around the Kalman prediction.

No model is trained online — the exemplars never update, so the verifier
can't drift the way a template EMA or an online discriminative classifier
does. This is the same failure mode that killed the template and MIL
variants; keeping the exemplars frozen sidesteps it entirely.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class BallVerifier:
    def __init__(
        self,
        patch_size: int = 24,
        n_negatives: int = 200,
        min_neg_separation: int = 40,
        rng_seed: int = 0,
    ):
        self.patch_size = patch_size
        self.n_negatives = n_negatives
        self.min_neg_separation = min_neg_separation
        self._rng = np.random.default_rng(rng_seed)

        # HOG on a small patch: 24x24 window, 12x12 blocks at 6px stride,
        # 6x6 cells, 9 bins → 324-dim descriptor. Small enough that the
        # dense grid search stays cheap.
        self._hog = cv2.HOGDescriptor(
            _winSize=(patch_size, patch_size),
            _blockSize=(12, 12),
            _blockStride=(6, 6),
            _cellSize=(6, 6),
            _nbins=9,
        )

        self._pos: Optional[np.ndarray] = None   # (Np, D)
        self._neg: Optional[np.ndarray] = None   # (Nn, D)

    # --- feature extraction ---------------------------------------------------

    def _extract(self, gray: np.ndarray, cx: float, cy: float) -> Optional[np.ndarray]:
        h, w = gray.shape[:2]
        half = self.patch_size // 2
        x0 = int(round(cx)) - half
        y0 = int(round(cy)) - half
        x1 = x0 + self.patch_size
        y1 = y0 + self.patch_size
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        patch = gray[y0:y1, x0:x1]
        feat = self._hog.compute(patch)
        if feat is None:
            return None
        feat = feat.flatten().astype(np.float32)
        n = float(np.linalg.norm(feat))
        if n < 1e-6:
            return None
        return feat / n

    # --- fitting --------------------------------------------------------------

    def add_positive(self, gray: np.ndarray, xy: Tuple[float, float]) -> bool:
        """Add a temporal positive exemplar to the frozen set.

        Intended for the first few frames of tracking, where the LK tracker
        is known to be reliable and the ball's appearance naturally changes
        (motion blur, rotation during a dribble bounce). Extending the
        exemplar set with a handful of these early observations makes the
        nearest-exemplar score robust to that variation without letting the
        exemplars drift later once the tracker is fully online.
        """
        if self._pos is None:
            return False
        f = self._extract(gray, float(xy[0]), float(xy[1]))
        if f is None:
            return False
        self._pos = np.vstack([self._pos, f[None, :]])
        return True

    def fit(self, gray: np.ndarray, xy: Tuple[float, float]) -> bool:
        cx, cy = float(xy[0]), float(xy[1])

        # Positives: the seed point plus small pixel jitters. Gives a bit of
        # robustness to the exact click location without pulling in any
        # background.
        pos_feats = []
        for dx, dy in [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2),
                       (-2, -2), (2, 2), (-2, 2), (2, -2)]:
            f = self._extract(gray, cx + dx, cy + dy)
            if f is not None:
                pos_feats.append(f)
        if not pos_feats:
            return False
        self._pos = np.stack(pos_feats)

        # Negatives: random locations across the frame, forbidden within
        # min_neg_separation px of the positive (so we don't accidentally
        # label the ball itself as a negative).
        h, w = gray.shape[:2]
        half = self.patch_size // 2 + 2
        min_sep_sq = self.min_neg_separation ** 2
        neg_feats = []
        tries = 0
        max_tries = self.n_negatives * 20
        while len(neg_feats) < self.n_negatives and tries < max_tries:
            tries += 1
            nx = int(self._rng.integers(half, max(half + 1, w - half)))
            ny = int(self._rng.integers(half, max(half + 1, h - half)))
            if (nx - cx) ** 2 + (ny - cy) ** 2 < min_sep_sq:
                continue
            f = self._extract(gray, nx, ny)
            if f is not None:
                neg_feats.append(f)
        if neg_feats:
            self._neg = np.stack(neg_feats)
        return True

    @property
    def is_ready(self) -> bool:
        return self._pos is not None

    # --- scoring --------------------------------------------------------------

    def score(self, gray: np.ndarray, cx: float, cy: float) -> float:
        if self._pos is None:
            return 0.0
        f = self._extract(gray, cx, cy)
        if f is None:
            return 0.0
        d_pos = float(np.min(np.linalg.norm(self._pos - f, axis=1)))
        if self._neg is None:
            # No negatives → fall back to exemplar similarity.
            return float(np.exp(-d_pos))
        d_neg = float(np.min(np.linalg.norm(self._neg - f, axis=1)))
        denom = d_pos + d_neg
        if denom < 1e-8:
            return 0.5
        return d_neg / denom

    def best_in_roi(
        self,
        gray: np.ndarray,
        cx: float,
        cy: float,
        radius: int,
        stride: int = 3,
    ) -> Tuple[Tuple[float, float], float]:
        """Dense grid search around (cx, cy); returns ((bx, by), best_score)."""
        best_xy = (cx, cy)
        best_s = -1.0
        for dy in range(-radius, radius + 1, stride):
            for dx in range(-radius, radius + 1, stride):
                s = self.score(gray, cx + dx, cy + dy)
                if s > best_s:
                    best_s = s
                    best_xy = (cx + dx, cy + dy)
        return best_xy, max(0.0, best_s)
