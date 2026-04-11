"""Ball vs not-ball scoring.

A real CNN could slot in here; for the hackathon we use a fast heuristic that
captures the three signals that actually distinguish a basketball in broadcast
footage from other moving blobs (shoes, jerseys, referee arms):

- **Color**: basketballs live in a narrow orange-brown HSV band.
- **Circularity**: already pre-filtered, but we rescore it smoothly.
- **Interior texture**: basketballs have visible seam lines; pure-color jersey
  patches don't. A Laplacian variance on the grayscale patch picks this up.

Scores are combined multiplicatively so a candidate must be reasonable on
*all* axes — this is strictly better than additive when any single axis is a
hard disqualifier (e.g. bright white sock ≈ 0 color score).
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .motion_detection import Candidate


# HSV range tuned for orange/brown basketballs under gym lighting.
BALL_HSV_LO = np.array([2, 70, 60], dtype=np.uint8)
BALL_HSV_HI = np.array([22, 255, 255], dtype=np.uint8)


class BallClassifier(Protocol):
    def score(self, cand: Candidate) -> float: ...


class HeuristicBallClassifier:
    def __init__(self, color_weight: float = 1.0, texture_weight: float = 0.6):
        self.color_weight = color_weight
        self.texture_weight = texture_weight

    def _color_score(self, patch_bgr: np.ndarray) -> float:
        if patch_bgr.size == 0:
            return 0.0
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BALL_HSV_LO, BALL_HSV_HI)
        frac = float(mask.mean()) / 255.0
        # saturate: a ball patch rarely has >70% orange pixels (edges, shadow).
        return float(np.clip(frac / 0.35, 0.0, 1.0))

    def _texture_score(self, patch_bgr: np.ndarray) -> float:
        if patch_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        # seam detail sits roughly in [30, 400]; clamp and normalize.
        return float(np.clip((lap_var - 20.0) / 180.0, 0.0, 1.0))

    def _shape_score(self, cand: Candidate) -> float:
        # soft ramp starting from the motion detector's hard threshold.
        return float(np.clip((cand.circularity - 0.45) / 0.45, 0.0, 1.0))

    def score(self, cand: Candidate) -> float:
        s_color = self._color_score(cand.patch)
        s_tex = self._texture_score(cand.patch)
        s_shape = self._shape_score(cand)
        # Multiplicative on color+shape (both must hold), texture is a bonus.
        base = (s_color ** self.color_weight) * (s_shape ** 0.8)
        bonus = 0.5 + 0.5 * s_tex  # bounded in [0.5, 1.0] so texture can't zero it out
        return float(base * (bonus ** self.texture_weight))
