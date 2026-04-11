"""Action-centroid localizer.

Converts a list of player bounding boxes per frame into a single (cx, cy)
point representing the weighted centre of action. Moving players get more
weight than stationary ones via a simple IoU-based motion estimate.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class ActionLocalizer:
    """Converts per-frame player boxes -> single weighted action centroid.

    Algorithm
    ---------
    1. For each current box compute centroid (bx, by).
    2. Match each current box to the best previous box by IoU.
    3. motion = centroid displacement in px (0 if no match).
    4. weight = conf * (1 + motion / 30).
    5. Return weighted mean of centroids.
    6. If no boxes: return previous (cx, cy) or frame-centre on first call.
    """

    def __init__(self):
        self._prev_boxes: List[Tuple[float, float, float, float, float]] = []
        self._last_cx: Optional[float] = None
        self._last_cy: Optional[float] = None

    def update(
        self,
        boxes: List[Tuple[float, float, float, float, float]],
        frame_wh: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float]:
        """Return (cx, cy) action centroid for this frame.

        Parameters
        ----------
        boxes:      list of (x1,y1,x2,y2,conf) from PlayerDetector.detect()
        frame_wh:   (width, height) of the frame — used only when no boxes have
                    ever been seen (first-frame fallback).
        """
        if not boxes:
            if self._last_cx is not None:
                return self._last_cx, self._last_cy  # type: ignore[return-value]
            # Absolute first frame with no detections
            if frame_wh is not None:
                cx, cy = frame_wh[0] / 2.0, frame_wh[1] / 2.0
            else:
                cx, cy = 0.0, 0.0
            self._last_cx, self._last_cy = cx, cy
            self._prev_boxes = []
            return cx, cy

        # Compute centroid for each box
        centroids = [((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                     for x1, y1, x2, y2, _ in boxes]

        # Estimate motion via IoU matching against previous frame
        motions: List[float] = []
        for i, (x1, y1, x2, y2, _) in enumerate(boxes):
            best_iou = 0.0
            best_disp = 0.0
            for px1, py1, px2, py2, _ in self._prev_boxes:
                iou = _iou((x1, y1, x2, y2), (px1, py1, px2, py2))
                if iou > best_iou:
                    best_iou = iou
                    pcx = (px1 + px2) / 2.0
                    pcy = (py1 + py2) / 2.0
                    best_disp = float(np.hypot(centroids[i][0] - pcx,
                                               centroids[i][1] - pcy))
            motions.append(best_disp)

        # Score every player by motion weight
        scored = []
        for i, (_, _, _, _, conf) in enumerate(boxes):
            w = conf * (1.0 + motions[i] / 10.0)
            scored.append((w, centroids[i]))

        # Keep only the top-K by motion weight.
        # This drops the large stationary paint cluster: a ball handler moving
        # 40 px/frame (w≈4.5) will always rank above standing players (w≈0.9),
        # so the paint cluster gets culled to at most K-1 entries even when
        # there are 8 players there.
        K = 4
        scored.sort(key=lambda x: -x[0])
        scored = scored[:K]

        total_w = sum(w for w, _ in scored)
        if total_w == 0.0:
            cx = float(np.mean([c[0] for _, c in scored]))
            cy = float(np.mean([c[1] for _, c in scored]))
        else:
            cx = sum(w * c[0] for w, c in scored) / total_w
            cy = sum(w * c[1] for w, c in scored) / total_w

        self._prev_boxes = list(boxes)
        self._last_cx, self._last_cy = cx, cy
        return cx, cy
