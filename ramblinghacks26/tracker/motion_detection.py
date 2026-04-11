"""Motion-based candidate detection.

Design notes
------------
A broadcast basketball is small (~10-25 px diameter at 720p) and usually moves
faster than the surrounding players. We combine two cheap signals:

1. MOG2 background subtraction — robust to lighting, gets moving foreground.
2. 3-frame differencing — catches the *currently* moving pixels (players stop
   often; the ball rarely does).

Intersecting both masks kills most stationary-player false positives. We then
extract connected components and filter by shape/size.

The filters are deliberately loose: downstream stages (classifier, Kalman
gating) make the final association, so we want recall > precision here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Candidate:
    x: float          # center x (pixels)
    y: float          # center y
    radius: float     # effective radius
    area: float
    circularity: float
    speed: float      # crude motion magnitude
    patch: np.ndarray  # small BGR crop, for classifier

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


class MotionDetector:
    def __init__(
        self,
        min_area: int = 15,
        max_area: int = 900,
        min_circularity: float = 0.45,
        history: int = 120,
        var_threshold: float = 25.0,
        patch_pad: int = 6,
    ):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.patch_pad = patch_pad
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_prev_gray: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _frame_diff_mask(self, gray: np.ndarray) -> Optional[np.ndarray]:
        if self._prev_gray is None or self._prev_prev_gray is None:
            return None
        d1 = cv2.absdiff(gray, self._prev_gray)
        d2 = cv2.absdiff(self._prev_gray, self._prev_prev_gray)
        motion = cv2.bitwise_and(d1, d2)
        _, m = cv2.threshold(motion, 12, 255, cv2.THRESH_BINARY)
        return m

    def detect(self, frame_bgr: np.ndarray) -> List[Candidate]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        fg = self.bg.apply(frame_bgr, learningRate=-1)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel, iterations=1)

        diff = self._frame_diff_mask(gray_blur)
        mask = cv2.bitwise_and(fg, diff) if diff is not None else fg
        mask = cv2.dilate(mask, self._kernel, iterations=1)

        self._prev_prev_gray = self._prev_gray
        self._prev_gray = gray_blur

        candidates: List[Candidate] = []
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h, w = frame_bgr.shape[:2]
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area or area > self.max_area:
                continue
            cx, cy = cents[i]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = bw / max(bh, 1)
            if aspect < 0.4 or aspect > 2.5:
                continue
            # circularity via contour perimeter
            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            perim = cv2.arcLength(contours[0], True)
            circularity = 4 * np.pi * area / (perim * perim + 1e-6)
            if circularity < self.min_circularity:
                continue
            radius = 0.5 * (bw + bh) * 0.5
            pad = self.patch_pad + int(radius)
            x0 = max(0, int(cx) - pad)
            y0 = max(0, int(cy) - pad)
            x1 = min(w, int(cx) + pad)
            y1 = min(h, int(cy) + pad)
            patch = frame_bgr[y0:y1, x0:x1].copy()
            candidates.append(
                Candidate(
                    x=float(cx), y=float(cy),
                    radius=float(radius), area=float(area),
                    circularity=float(circularity),
                    speed=float(area),  # placeholder; tracker uses its own velocity
                    patch=patch,
                )
            )
        return candidates
