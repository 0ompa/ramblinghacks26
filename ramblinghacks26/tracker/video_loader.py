"""Video I/O. Yields frames with metadata; keeps ownership of the VideoCapture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int


@dataclass
class Frame:
    index: int
    image: np.ndarray  # BGR uint8
    timestamp: float   # seconds


class VideoLoader:
    def __init__(self, path: str, start: int = 0, end: Optional[int] = None):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        self.info = VideoInfo(
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0,
            n_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        self.start = start
        self.end = end if end is not None else self.info.n_frames
        if start > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    def __iter__(self) -> Iterator[Frame]:
        idx = self.start
        while idx < self.end:
            ok, img = self.cap.read()
            if not ok:
                break
            yield Frame(index=idx, image=img, timestamp=idx / self.info.fps)
            idx += 1

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
