"""Stage 5 — Crop Window Smoothing"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_and_crop(focus_points, meta,
                    sigma=15.0, lookahead_weight=0.3, velocity_window=3):
    """Smooth focus trajectory and compute crop windows for 9:16 output."""
    if not focus_points:
        return []

    src_w = meta["width"]
    src_h = meta["height"]

    raw_cx = np.array([fp["cx"] for fp in focus_points], dtype=np.float64)
    raw_cy = np.array([fp["cy"] for fp in focus_points], dtype=np.float64)

    # Pass 1 — Gaussian low-pass filter
    smoothed_cx = gaussian_filter1d(raw_cx, sigma=sigma)
    smoothed_cy = gaussian_filter1d(raw_cy, sigma=sigma)

    # Pass 2 — Velocity-based look-ahead bias
    crop_cx = np.copy(smoothed_cx)
    crop_cy = np.copy(smoothed_cy)
    for i in range(velocity_window, len(smoothed_cx)):
        vx = smoothed_cx[i] - smoothed_cx[i - velocity_window]
        vy = smoothed_cy[i] - smoothed_cy[i - velocity_window]
        crop_cx[i] = smoothed_cx[i] + lookahead_weight * vx
        crop_cy[i] = smoothed_cy[i] + lookahead_weight * vy

    # 9:16 portrait crop from 16:9 source: full height, slice width
    crop_w = int(src_h * 9 / 16)
    crop_h = src_h

    crop_windows = []
    for i in range(len(focus_points)):
        x1 = int(crop_cx[i] - crop_w // 2)
        y1 = 0
        x1 = max(0, min(x1, src_w - crop_w))
        crop_windows.append({
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_w": crop_w,
            "crop_h": crop_h,
        })

    return crop_windows
