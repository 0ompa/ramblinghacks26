"""Seed-point refinement: snap a user-clicked coordinate to the actual ball.

The user's `--seed-xy` is an eyeball read-off from the first frame and is
typically off by 10-40 pixels — enough that a tight feature-extraction box
around it will grab background pixels (gym floor, a player's hand) instead
of the ball itself. Every downstream component (LK Shi-Tomasi features, the
HOG verifier's exemplar set, the Kalman initial state) inherits that
offset, so tracking drifts within a handful of frames.

This module does one job: given the user's seed, search a local window for
the ball and return a refined coordinate.

Method: HoughCircles on a Gaussian-blurred grayscale ROI, restricted to
ball-sized radii (5-15 px in 1080p broadcast), picking the circle whose
centre is closest to the user's seed. HoughCircles on a full frame is a
known distractor magnet (player heads — see hybrid_tracker.py post-mortem
#4), but constrained to a tight window around a known seed it is
extremely reliable because the ball is by construction the only ball-sized
circle nearby.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def refine_seed(
    frame_bgr: np.ndarray,
    xy: Tuple[float, float],
    search_radius: int = 40,
    min_radius_px: int = 5,
    max_radius_px: int = 15,
) -> Tuple[Tuple[float, float], bool]:
    """Refine a user seed to the nearest ball-shaped circle.

    Returns ((x, y), refined) where `refined` is False if no circle was
    found and the original xy is returned unchanged.
    """
    h, w = frame_bgr.shape[:2]
    cx, cy = int(round(xy[0])), int(round(xy[1]))
    pad = max_radius_px + 2
    x0 = max(0, cx - search_radius)
    y0 = max(0, cy - search_radius)
    x1 = min(w, cx + search_radius)
    y1 = min(h, cy + search_radius)
    if x1 - x0 < 2 * pad or y1 - y0 < 2 * pad:
        return (float(cx), float(cy)), False

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(gray[y0:y1, x0:x1], (5, 5), 1.0)
    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=80,
        param2=15,
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )
    if circles is None:
        return (float(cx), float(cy)), False

    best = None
    best_d = float("inf")
    for (lx, ly, r) in circles[0]:
        fx = x0 + lx
        fy = y0 + ly
        d = float(np.hypot(fx - cx, fy - cy))
        if d < best_d:
            best_d = d
            best = (float(fx), float(fy))
    if best is None:
        return (float(cx), float(cy)), False
    return best, True
