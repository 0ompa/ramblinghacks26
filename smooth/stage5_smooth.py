import numpy as np
from scipy.ndimage import gaussian_filter1d


SIGMA = 18
LOOKAHEAD_WEIGHT = 0.28
LOOKAHEAD_FRAMES = 4


def smooth_focus_trajectory(
    focus_x: np.ndarray,
    focus_y: np.ndarray,
    sigma: float = SIGMA,
    lookahead_weight: float = LOOKAHEAD_WEIGHT,
    lookahead_frames: int = LOOKAHEAD_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    smooth_x = gaussian_filter1d(focus_x.astype(float), sigma=sigma)
    smooth_y = gaussian_filter1d(focus_y.astype(float), sigma=sigma)

    N = len(smooth_x)
    crop_x = smooth_x.copy()
    crop_y = smooth_y.copy()

    for i in range(N):
        j = min(i + lookahead_frames, N - 1)
        vel_x = smooth_x[j] - smooth_x[i]
        vel_y = smooth_y[j] - smooth_y[i]
        crop_x[i] += lookahead_weight * vel_x
        crop_y[i] += lookahead_weight * vel_y

    return crop_x, crop_y


def compute_crop_x1(
    crop_cx: np.ndarray,
    crop_w: int,
    frame_w: int,
) -> np.ndarray:
    x1 = crop_cx - crop_w / 2
    x1 = np.clip(x1, 0, frame_w - crop_w).astype(int)
    return x1
