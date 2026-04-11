"""Lucas-Kanade optical-flow tracker with Kalman smoothing.

Why LK and not MIL/Hough/blobs (a chronicle of what I tried first)
------------------------------------------------------------------
This repo has burned through four tracker variants on the same broadcast
clip. In order:

1. Whole-frame motion blobs + color classifier — gym floor is the same
   orange as the ball, seed selection picked a distractor.
2. Template matching in a Kalman ROI — 15-px ball has almost no texture;
   the EMA template morphed into a smear and latched onto static
   background (court lines, bench seams).
3. TrackerMIL (OpenCV built-in) — tracked for ~50 frames, then the online
   discriminative classifier drifted to the player's orange-accented shoes,
   which move smoothly with the player and so slip past any velocity gate.
4. HoughCircles in a ROI — the ball is a circle, yes, but so are all the
   players' heads. The tracker latched onto a head almost immediately.

The common failure mode across 1-4: every appearance-based heuristic picks
a nearby distractor that looks similar to the ball. None of them track the
actual physical ball pixels.

Sparse Lucas-Kanade optical flow does. It follows *specific image points*
from one frame to the next by local gradient matching — it is physically
grounded in the sense that a tracked point moves exactly where that pixel
moved in the scene. Points that can't be matched (occlusion, motion blur)
are rejected by the status flag, and we re-detect fresh Shi-Tomasi corners
inside the current bounding box whenever we run low.

Pipeline
--------
seed(frame, x, y):
    extract ~12 good corner features in a tight box around (x,y)
    seed Kalman state at (x,y)

step(frame):
    pred = KF.predict()
    flow prev_points → new_points via cv2.calcOpticalFlowPyrLK
    drop dead points (status=0), high-error points, and points whose
        displacement diverges >N px from the group median (removes the few
        features that snapped onto background during motion blur)
    ball_obs = median of surviving points
    if |ball_obs - pred| or |ball_obs - last| exceeds max_step → reject
    if accepted → KF.correct; re-seed features every `refeature_every`
        frames inside a tight ROI around the current position
    if too few points survive → coast on Kalman prediction and try to
        re-detect features around the prediction

This is the simplest approach that cleanly separates "what the ball did"
(LK, physically grounded) from "where the ball is going" (Kalman) and it
does not suffer from the "latch onto orange shoe" drift that destroyed the
MIL/template/Hough variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .learned_detector import LearnedBallDetector
from .kalman_tracker import TrackState


@dataclass
class HybridConfig:
    # Feature detection (Shi-Tomasi inside a box around the ball).
    # Box must be tight enough that corners land on the ball surface itself
    # rather than the adjacent hand / jersey edge — those background
    # gradients are stronger than the ball's seams and will drag LK off.
    feature_box_half: int = 5         # half-width of seeding ROI in px
    max_features: int = 12
    quality_level: float = 0.005
    min_distance: int = 2
    block_size: int = 5
    min_points_to_track: int = 3      # below this → re-detect
    # 0 disables periodic re-seeding. Re-seeding at the Kalman-predicted
    # position was the #1 source of drift in an earlier attempt — once
    # Kalman slides slightly off-ball, the fresh corners snap onto court
    # lines or jersey seams and the track never comes back.
    refeature_every: int = 0

    # LK pyramid parameters
    lk_win_size: Tuple[int, int] = (15, 15)
    lk_max_level: int = 3

    # Outlier rejection on point flow
    outlier_dist_px: float = 6.0      # drop points farther than this from median
    max_step_px: float = 55.0         # hard per-frame ball motion gate

    # Kalman
    process_noise_pos: float = 1.0
    process_noise_vel: float = 6.0
    meas_noise: float = 3.0

    # Confidence + coasting
    max_coast_frames: int = 45
    conf_decay: float = 0.9
    conf_gain: float = 0.35

    # Learned ball detector (LogisticRegression trained via tools/train_ball.py).
    detector_model_path: str = "models/ball_clf.pkl"
    det_high_threshold: float = 0.7   # accept det_xy unconditionally
    det_low_threshold: float = 0.5    # accept det_xy if also close to lk_xy
    det_lk_agree_px: float = 25.0     # max dist between det and lk to agree
    det_lk_residual_px: float = 40.0  # max Kalman residual to trust lk_xy
    det_search_radius: int = 30
    det_search_stride: int = 3


class HybridBallTracker:
    def __init__(self, config: Optional[HybridConfig] = None):
        self.cfg = config or HybridConfig()

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32,
        )
        self.kf.processNoiseCov = np.diag(
            [self.cfg.process_noise_pos, self.cfg.process_noise_pos,
             self.cfg.process_noise_vel, self.cfg.process_noise_vel]
        ).astype(np.float32)
        self.kf.measurementNoiseCov = (np.eye(2) * self.cfg.meas_noise).astype(np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 50.0

        self._initialized = False
        self._prev_gray: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None  # shape (N, 1, 2) float32
        self._coast = 0
        self._confidence = 0.0
        self._last_xy: Optional[Tuple[float, float]] = None
        self._since_refeature = 0
        self._detector: Optional[LearnedBallDetector] = None
        self._seed_frame_index: Optional[int] = None

    # --- seeding --------------------------------------------------------------

    def _detect_features(self, gray: np.ndarray, cx: float, cy: float
                         ) -> Optional[np.ndarray]:
        h, w = gray.shape[:2]
        half = self.cfg.feature_box_half
        x0 = max(0, int(round(cx)) - half)
        y0 = max(0, int(round(cy)) - half)
        x1 = min(w, int(round(cx)) + half)
        y1 = min(h, int(round(cy)) + half)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        mask = np.zeros_like(gray)
        mask[y0:y1, x0:x1] = 255
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.max_features,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance,
            mask=mask,
            blockSize=self.cfg.block_size,
        )
        return pts  # shape (N, 1, 2) or None

    def seed(self, frame_bgr: np.ndarray, xy: Tuple[float, float]):
        x, y = float(xy[0]), float(xy[1])
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        pts = self._detect_features(gray, x, y)
        if pts is None or len(pts) == 0:
            # Fallback: seed one synthetic point at the given location.
            pts = np.array([[[x, y]]], dtype=np.float32)

        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 20.0
        self._initialized = True
        self._prev_gray = gray
        self._points = pts.astype(np.float32)
        self._coast = 0
        self._confidence = 1.0
        self._last_xy = (x, y)
        self._since_refeature = 0

        # Load the trained detector once (subsequent reseeds reuse it).
        if self._detector is None:
            self._detector = LearnedBallDetector(self.cfg.detector_model_path)

    # --- per-frame step -------------------------------------------------------

    def _flow(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Run LK flow on self._points; return surviving Nx2 positions or None."""
        if self._points is None or len(self._points) == 0 or self._prev_gray is None:
            return None
        nxt, st, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._points, None,
            winSize=self.cfg.lk_win_size,
            maxLevel=self.cfg.lk_max_level,
        )
        if nxt is None:
            return None
        mask = st.flatten() == 1
        survivors = nxt[mask].reshape(-1, 2)
        return survivors if len(survivors) > 0 else None

    @staticmethod
    def _prune_outliers(points: np.ndarray, max_dist: float) -> np.ndarray:
        if len(points) <= 2:
            return points
        median = np.median(points, axis=0)
        d = np.linalg.norm(points - median, axis=1)
        keep = d < max_dist
        if keep.sum() < 2:
            return points  # don't strip everything — fall back
        return points[keep]

    def step(self, frame_index: int, frame_bgr: np.ndarray) -> TrackState:
        if not self._initialized:
            return TrackState(
                frame_index=frame_index,
                position=(0.0, 0.0),
                velocity=(0.0, 0.0),
                confidence=0.0,
                observed=False,
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        pred = self.kf.predict().flatten()

        survivors = self._flow(gray)

        raw: Optional[Tuple[float, float]] = None
        accepted = False
        lk_candidate: Optional[Tuple[float, float]] = None
        pruned_pts: Optional[np.ndarray] = None
        if survivors is not None and len(survivors) >= self.cfg.min_points_to_track:
            pruned_pts = self._prune_outliers(survivors, self.cfg.outlier_dist_px)
            cx, cy = map(float, np.median(pruned_pts, axis=0))

            # Velocity gate against last accepted position.
            step_ok = True
            if self._last_xy is not None:
                step = float(np.hypot(cx - self._last_xy[0], cy - self._last_xy[1]))
                gate = self.cfg.max_step_px * max(1, self._coast + 1)
                if step > gate:
                    step_ok = False

            if step_ok:
                lk_candidate = (cx, cy)

        # Learned-detector decision logic.
        #
        # Run the detector in a ROI around the Kalman prediction every frame.
        # Combine with the LK candidate using a priority cascade:
        #   1. det_score >= 0.7               → use det_xy (high confidence)
        #   2. det_score >= 0.5 AND det close
        #      to lk_xy (< 25 px)             → use det_xy (detector + LK agree)
        #   3. LK inliers >= 3 AND Kalman
        #      residual < 40 px               → use lk_xy (LK looks trustworthy)
        #   4. else                            → coast on Kalman prediction
        meas_xy: Optional[Tuple[float, float]] = None
        meas_source = "coast"

        det_xy: Optional[Tuple[float, float]] = None
        det_score = 0.0
        if self._detector is not None:
            (det_xy, det_score) = self._detector.best_in_roi(
                frame_bgr,
                float(pred[0]), float(pred[1]),
                radius=self.cfg.det_search_radius,
                stride=self.cfg.det_search_stride,
            )

        lk_inliers = len(pruned_pts) if pruned_pts is not None else 0
        kalman_residual = (
            float(np.hypot(lk_candidate[0] - pred[0], lk_candidate[1] - pred[1]))
            if lk_candidate is not None else float("inf")
        )

        if det_xy is not None and det_score >= self.cfg.det_high_threshold:
            meas_xy = det_xy
            meas_source = "det_high"
            pruned_pts = None  # re-seed LK features at new location
        elif (det_xy is not None
              and det_score >= self.cfg.det_low_threshold
              and lk_candidate is not None
              and float(np.hypot(det_xy[0] - lk_candidate[0],
                                 det_xy[1] - lk_candidate[1]))
                  < self.cfg.det_lk_agree_px):
            meas_xy = det_xy
            meas_source = "det_agree"
            pruned_pts = None
        elif (lk_candidate is not None
              and lk_inliers >= self.cfg.min_points_to_track
              and kalman_residual < self.cfg.det_lk_residual_px):
            meas_xy = lk_candidate
            meas_source = "lk"

        if meas_xy is not None:
            mx, my = meas_xy
            meas = np.array([[mx], [my]], dtype=np.float32)
            self.kf.correct(meas)
            self._coast = 0
            self._confidence = min(1.0, self._confidence + self.cfg.conf_gain)
            raw = (mx, my)
            state = self.kf.statePost.flatten()
            self._last_xy = (float(state[0]), float(state[1]))
            if meas_source == "lk" and pruned_pts is not None:
                # LK observation: keep the surviving flow points.
                self._points = pruned_pts.reshape(-1, 1, 2).astype(np.float32)
            else:
                # Detector observation: re-seed Shi-Tomasi features tightly
                # around the confirmed ball location so LK tracks the right
                # pixels next frame (this is the root-cause fix for LK drift).
                new_pts = self._detect_features(gray, mx, my)
                self._points = (new_pts.astype(np.float32)
                                if new_pts is not None and len(new_pts) >= 2
                                else None)
                self._since_refeature = 0
            accepted = True

        if not accepted:
            self._coast += 1
            self._confidence *= self.cfg.conf_decay
            state = pred.copy()
            state[2] *= 0.92
            state[3] *= 0.92
            self.kf.statePost = state.reshape(4, 1).astype(np.float32)
            if self._coast > self.cfg.max_coast_frames:
                self._confidence = 0.0

        # Re-seed only when we're running low on points. Re-detect around
        # the LK *observation* (if we had one) rather than the Kalman state —
        # a stale Kalman state drags the new corners onto background.
        self._since_refeature += 1
        need_refeature = (
            self._points is None
            or len(self._points) < self.cfg.min_points_to_track
            or (self.cfg.refeature_every > 0
                and self._since_refeature >= self.cfg.refeature_every)
        )
        if need_refeature:
            anchor_x, anchor_y = (raw if raw is not None
                                  else (float(state[0]), float(state[1])))
            new_pts = self._detect_features(gray, anchor_x, anchor_y)
            if new_pts is not None and len(new_pts) >= 2:
                self._points = new_pts.astype(np.float32)
                self._since_refeature = 0

        self._prev_gray = gray

        return TrackState(
            frame_index=frame_index,
            position=(float(state[0]), float(state[1])),
            velocity=(float(state[2]), float(state[3])),
            confidence=float(self._confidence),
            observed=raw is not None,
            raw_position=raw,
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized
