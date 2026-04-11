"""Portrait crop pipeline using the Roboflow basketball model.

Usage:
    python run.py --input ramblinghacks26/clips/clip5.mp4 --output out/clip5_portrait.mp4
    python run.py --input ramblinghacks26/clips/  --output out/   # batch

Two-pass design
---------------
Pass 1  Run the model on every frame and collect raw (x, y) ball positions.
        Frames where no ball is detected are stored as None.

Pass 2  Smooth the position track, then re-read the video and write crops.

Why two passes?
  The model fires on false positives (scoreboard numbers, jersey print, crowd)
  that are spatially close enough to defeat simple distance gating.  A
  per-frame fix just moves the problem around.  Processing all detections
  together lets us apply a median filter that collapses isolated outlier
  frames to Nothing, then fill genuine gaps with linear interpolation so the
  crop glides rather than freezing forever.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

API_KEY  = os.environ["ROBOFLOW_API_KEY"]
MODEL_ID = "basketball-xil7x/1"

# Reject any ball detection that jumps more than this many px from the last
# accepted position (per frame elapsed).  Anything larger is a false positive.
MAX_SPEED_PX_PER_FRAME = 55   # generous: ~55 px/frame ≈ fast pass at 1080p/25fps

# Median window for outlier rejection (must be odd).  A value of 5 means a
# single bad frame is outvoted 4-to-1 and collapses to its neighbours.
MEDIAN_WIN = 7

# After outlier rejection, fill gaps and then apply a light Gaussian blur so
# the crop centre moves smoothly rather than linearly.
SMOOTH_SIGMA_FRAMES = 4       # Gaussian sigma in frames


# ---------------------------------------------------------------------------
# 9:16 crop rectangle (no internal smoothing — we smooth the positions before)
# ---------------------------------------------------------------------------

class PortraitCropper:
    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        if frame_w * 16 >= frame_h * 9:
            self.crop_h = frame_h
            self.crop_w = int(round(frame_h * 9 / 16))
        else:
            self.crop_w = frame_w
            self.crop_h = int(round(frame_w * 16 / 9))

    def crop_rect(self, cx: float, cy: float) -> tuple[int, int, int, int]:
        x = int(np.clip(round(cx - self.crop_w / 2), 0, self.frame_w - self.crop_w))
        y = int(np.clip(round(cy - self.crop_h / 2), 0, self.frame_h - self.crop_h))
        return x, y, self.crop_w, self.crop_h


# ---------------------------------------------------------------------------
# Pass 1 — run inference, return raw per-frame ball positions
# ---------------------------------------------------------------------------

def collect_detections(
    model,
    video_path: str,
) -> tuple[list[tuple[float, float] | None], float, int, int]:
    """
    Returns (positions, fps, W, H).
    positions[i] is (cx, cy) or None for frame i.
    """
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  pass 1 / inference  ({n_tot} frames) …", flush=True)

    positions: list[tuple[float, float] | None] = []
    last_accepted: tuple[float, float] | None = None
    frames_since  = 0

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.infer(frame)[0]
        balls  = [p for p in result.predictions if p.class_name == "ball"]

        best: tuple[float, float] | None = None
        if balls:
            if last_accepted is not None:
                gate = MAX_SPEED_PX_PER_FRAME * (frames_since + 1)
                # Filter to in-gate candidates only.
                ok = [b for b in balls
                      if ((b.x - last_accepted[0])**2 +
                          (b.y - last_accepted[1])**2) ** 0.5 <= gate]
                if ok:
                    ok.sort(key=lambda b: (b.x - last_accepted[0])**2 +
                                          (b.y - last_accepted[1])**2)
                    best = (ok[0].x, ok[0].y)
                # else: all detections too far → treat as no-ball this frame
            else:
                # No prior position — accept highest-confidence detection.
                balls.sort(key=lambda b: -b.confidence)
                best = (balls[0].x, balls[0].y)

        if best is not None:
            last_accepted = best
            frames_since  = 0
        else:
            frames_since += 1

        positions.append(best)
        i += 1

    cap.release()
    print(f"    detected {sum(p is not None for p in positions)}/{len(positions)} frames")
    return positions, fps, W, H


# ---------------------------------------------------------------------------
# Smooth the raw positions to eliminate remaining outliers and fill gaps
# ---------------------------------------------------------------------------

def smooth_positions(
    raw: list[tuple[float, float] | None],
    W: int,
    H: int,
) -> list[tuple[float, float]]:
    """
    1. Outlier rejection via per-axis median filter on detected frames.
       A false positive that slipped through the gate will be surrounded by
       very different values; the median collapses it to the neighbourhood
       median, which we then threshold-compare against the raw value to
       decide whether to keep it or blank it.
    2. Linear interpolation across blanked/missing frames.
    3. Light Gaussian blur for smooth panning motion.
    """
    n = len(raw)
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    for i, p in enumerate(raw):
        if p is not None:
            xs[i], ys[i] = p

    # --- Step 1: median-filter on detected frames to find outliers ----------
    # Work only where we have detections; interpolate NaNs temporarily for
    # the filter, then restore them.
    def interp_nans(arr: np.ndarray) -> np.ndarray:
        nans = np.isnan(arr)
        if nans.all():
            return np.full_like(arr, (W if arr is xs else H) / 2.0)
        idx = np.arange(n)
        arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
        return arr

    xs_filled = interp_nans(xs.copy())
    ys_filled = interp_nans(ys.copy())

    # Uniform (box) filter approximates median cheaply and is differentiable.
    # For proper outlier removal we use a real median via a sliding window.
    half = MEDIAN_WIN // 2

    def running_median(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            out[i] = np.median(arr[lo:hi])
        return out

    xs_med = running_median(xs_filled)
    ys_med = running_median(ys_filled)

    # Blank detections that deviate too far from their neighbourhood median.
    OUTLIER_THRESH = MAX_SPEED_PX_PER_FRAME * 2
    for i in range(n):
        if raw[i] is not None:
            dev = ((xs[i] - xs_med[i])**2 + (ys[i] - ys_med[i])**2) ** 0.5
            if dev > OUTLIER_THRESH:
                xs[i] = np.nan
                ys[i] = np.nan

    # --- Step 2: interpolate across all NaN gaps ---------------------------
    xs_filled2 = interp_nans(xs.copy())
    ys_filled2 = interp_nans(ys.copy())

    # --- Step 3: Gaussian blur for smooth panning --------------------------
    sigma = SMOOTH_SIGMA_FRAMES
    # scipy uniform_filter is a box filter, but a repeated application
    # approximates Gaussian.  Two passes of window ≈ sigma*sqrt(3) is enough.
    win = max(3, int(sigma * 2.5) | 1)   # odd window
    for _ in range(2):
        xs_filled2 = uniform_filter1d(xs_filled2, size=win, mode="nearest")
        ys_filled2 = uniform_filter1d(ys_filled2, size=win, mode="nearest")

    return [(float(xs_filled2[i]), float(ys_filled2[i])) for i in range(n)]


# ---------------------------------------------------------------------------
# Pass 2 — re-read video and write portrait crops using smoothed centres
# ---------------------------------------------------------------------------

def write_crops(
    video_path: str,
    output_path: str,
    centres: list[tuple[float, float]],
    fps: float,
    W: int,
    H: int,
) -> None:
    cropper = PortraitCropper(W, H)
    cap     = cv2.VideoCapture(video_path)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer  = cv2.VideoWriter(output_path, fourcc, fps,
                              (cropper.crop_w, cropper.crop_h))

    print(f"  pass 2 / writing  crop={cropper.crop_w}x{cropper.crop_h} …",
          flush=True)

    for i, (cx, cy) in enumerate(centres):
        ret, frame = cap.read()
        if not ret:
            break
        x, y, cw, ch = cropper.crop_rect(cx, cy)
        crop = np.ascontiguousarray(frame[y:y + ch, x:x + cw])
        if crop.shape[1] != cw or crop.shape[0] != ch:
            crop = cv2.resize(crop, (cw, ch))
        writer.write(crop)

    cap.release()
    writer.release()
    print(f"  -> {output_path}  ({len(centres)} frames)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_video(model, input_path: str, output_path: str) -> None:
    name = os.path.basename(input_path)
    print(f"\n[{name}]")

    raw, fps, W, H = collect_detections(model, input_path)
    centres        = smooth_positions(raw, W, H)
    write_crops(input_path, output_path, centres, fps, W, H)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading Roboflow model …", end=" ", flush=True)
    from inference import get_model
    model = get_model(MODEL_ID, api_key=API_KEY)
    print("OK")

    if os.path.isdir(args.input):
        clips = sorted(f for f in os.listdir(args.input)
                       if f.lower().endswith(".mp4"))
        os.makedirs(args.output, exist_ok=True)
        for clip in clips:
            inp = os.path.join(args.input, clip)
            out = os.path.join(args.output,
                               os.path.splitext(clip)[0] + "_portrait.mp4")
            process_video(model, inp, out)
    else:
        process_video(model, args.input, args.output)


if __name__ == "__main__":
    main()
