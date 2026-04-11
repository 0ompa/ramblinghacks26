"""9:16 portrait crop window driven by the tracked ball.

Two rules make this feel cinematic rather than reactive:

1. **Look-ahead bias.** We anchor the crop slightly ahead of the ball along
   its velocity vector — the viewer sees where the play is going, not where
   it just was. The bias is proportional to velocity but capped so a wild
   prediction during coasting can't rip the frame off-screen.

2. **Asymmetric velocity-dependent smoothing.** When the ball is moving
   slowly the crop lerps gently (low alpha → stable); when it accelerates
   hard the crop loosens up and catches up faster. This mimics a human
   operator panning a camera.

The crop is clamped to frame bounds as a final step so it never produces
invalid coordinates for the downstream encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CropWindow:
    x: int  # top-left
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class PortraitCropGenerator:
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        aspect_w: int = 9,
        aspect_h: int = 16,
        lookahead_frames: float = 6.0,
        max_lookahead_px: float = 120.0,
        alpha_slow: float = 0.08,
        alpha_fast: float = 0.25,
        speed_for_fast: float = 18.0,  # px/frame at which we switch to alpha_fast
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        # Fit the tallest 9:16 rectangle that fits inside the source frame.
        if frame_w * aspect_h >= frame_h * aspect_w:
            self.crop_h = frame_h
            self.crop_w = int(round(frame_h * aspect_w / aspect_h))
        else:
            self.crop_w = frame_w
            self.crop_h = int(round(frame_w * aspect_h / aspect_w))

        self.lookahead_frames = lookahead_frames
        self.max_lookahead_px = max_lookahead_px
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        self.speed_for_fast = speed_for_fast
        self._center: np.ndarray | None = None

    def _target(self, pos: Tuple[float, float], vel: Tuple[float, float]) -> np.ndarray:
        v = np.array(vel, dtype=np.float32)
        lead = v * self.lookahead_frames
        n = float(np.linalg.norm(lead))
        if n > self.max_lookahead_px:
            lead *= self.max_lookahead_px / n
        return np.array(pos, dtype=np.float32) + lead

    def update(
        self,
        pos: Tuple[float, float],
        vel: Tuple[float, float],
        confidence: float = 1.0,
    ) -> CropWindow:
        target = self._target(pos, vel)
        speed = float(np.linalg.norm(vel))
        t = float(np.clip(speed / self.speed_for_fast, 0.0, 1.0))
        alpha = (1 - t) * self.alpha_slow + t * self.alpha_fast
        # Low confidence → slow down even further; we'd rather drift than jerk.
        alpha *= float(np.clip(0.3 + 0.7 * confidence, 0.3, 1.0))

        if self._center is None:
            self._center = target
        else:
            self._center = alpha * target + (1 - alpha) * self._center

        cx, cy = float(self._center[0]), float(self._center[1])
        x = int(round(cx - self.crop_w / 2))
        y = int(round(cy - self.crop_h / 2))
        x = int(np.clip(x, 0, self.frame_w - self.crop_w))
        y = int(np.clip(y, 0, self.frame_h - self.crop_h))
        return CropWindow(x=x, y=y, w=self.crop_w, h=self.crop_h)

    def reset(self):
        self._center = None
