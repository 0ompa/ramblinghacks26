"""
Stage 4 — Focus Point Computation

Produces a single (cx, cy) per frame — the point the crop centers on.

Priority hierarchy:
    1. Ball detected with conf > threshold  → ball centroid
    2. Ball track interpolated (Kalman)     → predicted centroid, weighted less
    3. Ball track lost entirely             → centroid of most-active players
    4. No tracks at all                     → frame center

Input:  frame_data dict from detect.py  {"ball": (cx,cy)|None, "players": [(x1,y1,x2,y2),...]}
Output: {"cx": float, "cy": float, "source": str}

PLACEHOLDER — Person 2 implements this.
"""

import cv2
import numpy as np
from utils import get_box_center


def compute_focus_point(frame_data, frame, prev_frame=None,
                        ball_conf_threshold=0.4, frame_width=1920, frame_height=1080):
    """Compute focus point for a single frame.

    Args:
        frame_data:  dict with "ball" and "players" from detect.py
        frame:       current BGR frame (for optical flow fallback)
        prev_frame:  previous BGR frame (for optical flow)
        ball_conf_threshold: minimum confidence to trust ball detection
        frame_width:  source frame width
        frame_height: source frame height

    Returns:
        dict: {"cx": float, "cy": float, "source": str}
    """
    if frame_data["ball"] is not None:
        cx, cy = frame_data["ball"]
        return {"cx": cx, "cy": cy, "source": "ball"}

    if frame_data["players"]:
        centers = [get_box_center(b) for b in frame_data["players"]]
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)
        return {"cx": cx, "cy": cy, "source": "player_cluster"}

    return {"cx": frame_width / 2, "cy": frame_height / 2, "source": "frame_center"}
