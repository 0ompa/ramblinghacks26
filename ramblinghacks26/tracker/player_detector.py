"""YOLOv8-based person detector.

Downloads yolov8n.pt on first use via ultralytics.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


class PlayerDetector:
    """Detect players (person class) in a BGR frame using YOLOv8n.

    Returns boxes as (x1, y1, x2, y2, conf) filtered by:
      - class == 0 (person)
      - box height >= min_height_px  (drops distant crowd / bench figures)
    """

    def __init__(self, model: str = "yolov8n.pt", min_height_px: int = 80):
        from ultralytics import YOLO  # lazy import so the rest of the module loads fast
        self._model = YOLO(model)
        self.min_height_px = min_height_px

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Run inference on one BGR frame.

        Returns list of (x1, y1, x2, y2, conf) for every person box
        whose pixel height is >= self.min_height_px.
        """
        results = self._model(frame_bgr, verbose=False)[0]
        boxes = results.boxes
        out: List[Tuple[float, float, float, float, float]] = []
        if boxes is None or len(boxes) == 0:
            return out

        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            if cls != 0:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i].item())
            if (y2 - y1) < self.min_height_px:
                continue
            out.append((x1, y1, x2, y2, conf))
        return out
