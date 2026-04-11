"""Single-target Kalman tracker for the basketball.

State: [x, y, vx, vy]  (pixels, pixels/frame)
Observation: [x, y]

Why a single track?
  The pipeline downstream only wants *one* point to crop around. Multi-object
  tracking would be wasted work — we just need to stay locked on the ball and
  coast through occlusions.

Association strategy:
  Each frame we receive a list of scored candidates. We gate by Mahalanobis
  distance on the position covariance, then pick the candidate with the
  highest combined score:
      track_score(c) = classifier_score(c) * exp(-0.5 * d^2 / gate^2)
  This lets a low-classifier-score but spatially consistent candidate still
  win over a high-classifier-score but distant one — essential during motion
  blur when the ball looks nothing like a ball.

Coasting:
  When no candidate passes the gate, we just run the prediction step and
  decay confidence. After `max_coast` coasted frames we consider the track
  lost and re-seed from the best scoring candidate anywhere in the frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .classifier import BallClassifier
from .motion_detection import Candidate


@dataclass
class TrackState:
    frame_index: int
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    confidence: float
    observed: bool  # True if a detection was associated this frame
    raw_position: Optional[Tuple[float, float]] = None  # pre-Kalman observation


class BallKalmanTracker:
    def __init__(
        self,
        gate_radius: float = 60.0,
        process_noise_pos: float = 1.0,
        process_noise_vel: float = 5.0,
        meas_noise: float = 4.0,
        max_coast_frames: int = 25,
        min_score_to_seed: float = 0.25,
        conf_decay: float = 0.85,
        conf_gain: float = 0.35,
    ):
        self.gate_radius = gate_radius
        self.max_coast_frames = max_coast_frames
        self.min_score_to_seed = min_score_to_seed
        self.conf_decay = conf_decay
        self.conf_gain = conf_gain

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
        q = np.diag([process_noise_pos, process_noise_pos,
                     process_noise_vel, process_noise_vel]).astype(np.float32)
        self.kf.processNoiseCov = q
        self.kf.measurementNoiseCov = (np.eye(2) * meas_noise).astype(np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 500.0

        self._initialized = False
        self._coast = 0
        self._confidence = 0.0

    def _seed(self, cand: Candidate):
        self.kf.statePost = np.array(
            [[cand.x], [cand.y], [0.0], [0.0]], dtype=np.float32,
        )
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 50.0
        self._initialized = True
        self._coast = 0
        self._confidence = 0.6

    def _pick(
        self,
        candidates: List[Candidate],
        scores: List[float],
        predicted: np.ndarray,
    ) -> Tuple[Optional[int], float]:
        best_i, best_score = None, 0.0
        for i, (c, s) in enumerate(zip(candidates, scores)):
            d = float(np.hypot(c.x - predicted[0], c.y - predicted[1]))
            if d > self.gate_radius:
                continue
            # Gaussian falloff so distance-within-gate still matters.
            gated = s * float(np.exp(-0.5 * (d / (self.gate_radius * 0.5)) ** 2))
            if gated > best_score:
                best_score = gated
                best_i = i
        return best_i, best_score

    def step(
        self,
        frame_index: int,
        candidates: List[Candidate],
        classifier: BallClassifier,
    ) -> TrackState:
        # Score all candidates up front — cheap, and we reuse them on seed.
        scores = [classifier.score(c) for c in candidates] if candidates else []

        if not self._initialized:
            if candidates:
                i = int(np.argmax(scores))
                if scores[i] >= self.min_score_to_seed:
                    self._seed(candidates[i])
                    c = candidates[i]
                    return TrackState(
                        frame_index=frame_index,
                        position=(c.x, c.y),
                        velocity=(0.0, 0.0),
                        confidence=self._confidence,
                        observed=True,
                        raw_position=(c.x, c.y),
                    )
            return TrackState(
                frame_index=frame_index,
                position=(0.0, 0.0),
                velocity=(0.0, 0.0),
                confidence=0.0,
                observed=False,
            )

        pred = self.kf.predict().flatten()  # [x, y, vx, vy]
        pick_i, pick_score = self._pick(candidates, scores, pred)

        raw: Optional[Tuple[float, float]] = None
        if pick_i is not None:
            c = candidates[pick_i]
            meas = np.array([[c.x], [c.y]], dtype=np.float32)
            self.kf.correct(meas)
            self._coast = 0
            self._confidence = min(1.0, self._confidence + self.conf_gain * pick_score)
            raw = (c.x, c.y)
            state = self.kf.statePost.flatten()
        else:
            # Coasting: trust prediction, decay confidence, and dampen velocity
            # slightly so we don't drift to infinity during long occlusions.
            self._coast += 1
            self._confidence *= self.conf_decay
            state = pred.copy()
            state[2] *= 0.9
            state[3] *= 0.9
            self.kf.statePost = state.reshape(4, 1).astype(np.float32)

            if self._coast > self.max_coast_frames:
                # Drop the lock — next frame we'll try to re-seed from scratch.
                self._initialized = False
                self._confidence = 0.0

        return TrackState(
            frame_index=frame_index,
            position=(float(state[0]), float(state[1])),
            velocity=(float(state[2]), float(state[3])),
            confidence=float(self._confidence),
            observed=pick_i is not None,
            raw_position=raw,
        )


class TrajectorySmoother:
    """Second-stage smoothing on top of the Kalman output.

    The Kalman filter already smooths, but its output still jitters when the
    associated detection jumps between nearby blobs. An EMA with a separate,
    slower time constant gives the crop window the cinematic feel the spec
    asks for, without introducing the lag a plain moving-average would.

    We run it on position *and* velocity so the look-ahead bias in the crop
    generator stays stable too.
    """

    def __init__(self, alpha_pos: float = 0.35, alpha_vel: float = 0.2):
        self.alpha_pos = alpha_pos
        self.alpha_vel = alpha_vel
        self._pos: Optional[np.ndarray] = None
        self._vel: Optional[np.ndarray] = None

    def reset(self):
        self._pos = None
        self._vel = None

    def update(self, pos: Tuple[float, float], vel: Tuple[float, float]
               ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        p = np.array(pos, dtype=np.float32)
        v = np.array(vel, dtype=np.float32)
        if self._pos is None:
            self._pos = p
            self._vel = v
        else:
            self._pos = self.alpha_pos * p + (1 - self.alpha_pos) * self._pos
            self._vel = self.alpha_vel * v + (1 - self.alpha_vel) * self._vel
        return ((float(self._pos[0]), float(self._pos[1])),
                (float(self._vel[0]), float(self._vel[1])))
