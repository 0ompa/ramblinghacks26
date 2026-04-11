"""Basketball detector: YOLO class-32 + HSV orange color fallback.

Two-stage strategy
------------------
1. YOLO class 32 (sports ball): fast, works when ball is clear and large enough.
2. HSV orange mask + contour circularity: catches the ball when YOLO misses it
   (small, motion-blurred, partially occluded). Basketball is distinctively
   orange; no other large object on a hardwood court shares that hue.

Both stages feed into the same bounce scorer and temporal tracker so the
caller gets a consistent (cx, cy, confidence) interface regardless of which
stage fired.

Bounce scoring
--------------
A dribbled ball has frequent vertical velocity reversals. We track Y-position
history and count sign-changes in dy. A static crowd ball or scoreboard logo
has score 0; a live dribble produces ~1-2 reversals per 20 frames → score ~0.1.
We boost confidence by this signal so the caller can trust ball detections more
when they exhibit the bounce pattern.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


# HSV range for basketball orange (works for both light and shadow sides).
_HSV_LO1 = np.array([5,  130, 100], dtype=np.uint8)   # warm orange
_HSV_HI1 = np.array([22, 255, 255], dtype=np.uint8)
_HSV_LO2 = np.array([0,  130, 100], dtype=np.uint8)   # reddish-orange edge
_HSV_HI2 = np.array([5,  255, 255], dtype=np.uint8)


class BallDetector:
    def __init__(self, model: str = "yolov8n.pt", history_len: int = 20):
        from ultralytics import YOLO
        self._model = YOLO(model)
        self._history_len = history_len
        self._tracked: Optional[Tuple[float, float]] = None
        self._y_history: List[Optional[float]] = []

    # ------------------------------------------------------------------
    # Bounce scoring
    # ------------------------------------------------------------------
    def _bounce_score(self) -> float:
        ys = [y for y in self._y_history if y is not None]
        if len(ys) < 4:
            return 0.0
        dys = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        sign_changes = sum(
            1 for i in range(len(dys) - 1) if dys[i] * dys[i + 1] < 0
        )
        return sign_changes / max(len(dys) - 1, 1)

    # ------------------------------------------------------------------
    # Stage 1: YOLO class 32
    # ------------------------------------------------------------------
    def _yolo_candidates(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        results = self._model(frame_bgr, verbose=False)[0]
        boxes = results.boxes
        out = []
        if boxes is not None:
            for i in range(len(boxes)):
                if int(boxes.cls[i].item()) != 32:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i].item())
                out.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, conf))
        return out

    # ------------------------------------------------------------------
    # Stage 2: HSV orange + contour circularity
    # ------------------------------------------------------------------
    def _color_candidates(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, _HSV_LO1, _HSV_HI1),
            cv2.inRange(hsv, _HSV_LO2, _HSV_HI2),
        )
        # Small morphological close to merge broken circles from motion blur
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        H, W = frame_bgr.shape[:2]
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Basketball at broadcast distance: roughly 15–55 px radius
            if area < 700 or area > 10_000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.45:   # blurry ball is still somewhat round
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            # Skip anything in the top 15% of frame (scoreboard area)
            if cy < H * 0.15:
                continue
            # Confidence proxy: circularity (max 1.0 for a perfect circle)
            out.append((cx, cy, float(circularity * 0.8)))
        return out

    # ------------------------------------------------------------------
    # Pick the best candidate (closest to last tracked position)
    # ------------------------------------------------------------------
    def _pick(
        self, candidates: List[Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
        if self._tracked is not None:
            tx, ty = self._tracked
            candidates.sort(
                key=lambda c: (c[0] - tx) ** 2 + (c[1] - ty) ** 2
            )
        else:
            candidates.sort(key=lambda c: -c[2])
        return candidates[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Return (cx, cy, confidence) or None if ball not found."""
        candidates = self._yolo_candidates(frame_bgr)
        if not candidates:
            candidates = self._color_candidates(frame_bgr)

        if not candidates:
            self._y_history.append(None)
            if len(self._y_history) > self._history_len:
                self._y_history.pop(0)
            return None

        cx, cy, conf = self._pick(candidates)
        self._tracked = (cx, cy)
        self._y_history.append(cy)
        if len(self._y_history) > self._history_len:
            self._y_history.pop(0)

        bounce = self._bounce_score()
        final_conf = float(np.clip(conf + bounce * 0.4, 0.0, 1.0))
        return cx, cy, final_conf
