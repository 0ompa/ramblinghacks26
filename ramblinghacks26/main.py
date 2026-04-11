"""Basketball portrait-crop pipeline.

Usage:
    python main.py --input clips/clip1.mp4 \
        --out out/tracked.mp4 --crop-out out/crop.mp4 --jsonl out/track.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Optional

import cv2
import numpy as np

from tracker import PortraitCropGenerator, TrajectorySmoother, VideoLoader
from tracker.player_detector import PlayerDetector
from tracker.action_localizer import ActionLocalizer
from tracker.ball_detector import BallDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to input video")
    p.add_argument("--out", default=None, help="annotated mp4 output")
    p.add_argument("--crop-out", default=None, help="portrait 9:16 mp4 output")
    p.add_argument("--jsonl", default=None, help="per-frame track state jsonl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    return p.parse_args()


def ensure_dir(path: Optional[str]):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def draw_overlay(img, boxes, focus_x, focus_y, ball_pos, crop):
    """Draw cyan player boxes, yellow ball marker, magenta focus crosshair + crop rect."""
    out = img.copy()

    # Cyan bounding boxes for detected players
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      (255, 255, 0), 1)

    # Yellow circle at detected ball position
    if ball_pos is not None:
        bx, by, bconf = ball_pos
        cv2.circle(out, (int(bx), int(by)), 18, (0, 255, 255), 2)

    # Magenta crosshair at final focus point
    cv2.drawMarker(out, (int(focus_x), int(focus_y)), (255, 0, 255),
                   cv2.MARKER_CROSS, 24, 2)

    # Magenta crop-window rectangle
    cv2.rectangle(out, (crop.x, crop.y),
                  (crop.x + crop.w, crop.y + crop.h), (255, 0, 255), 2)

    ball_str = f"ball=({ball_pos[0]:.0f},{ball_pos[1]:.0f}) " if ball_pos else "ball=none "
    txt = ball_str + f"players={len(boxes)}"
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 4)
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    return out


def main():
    args = parse_args()
    for path in (args.out, args.crop_out, args.jsonl):
        ensure_dir(path)

    loader = VideoLoader(args.input, start=args.start, end=args.end)
    info = loader.info
    print(f"[video] {info.width}x{info.height} @ {info.fps:.2f} fps, "
          f"{info.n_frames} frames", file=sys.stderr)

    player_detector = PlayerDetector(model="yolov8n.pt", min_height_px=80)
    ball_detector = BallDetector(model="yolov8n.pt", history_len=20)
    localizer = ActionLocalizer()
    smoother = TrajectorySmoother(alpha_pos=0.1, alpha_vel=0.08)
    cropper = PortraitCropGenerator(info.width, info.height)

    writer: Optional[cv2.VideoWriter] = None
    crop_writer: Optional[cv2.VideoWriter] = None
    jsonl_fp = open(args.jsonl, "w") if args.jsonl else None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    prev_sm_x: Optional[float] = None
    prev_sm_y: Optional[float] = None
    # Median filter: kills frame-to-frame centroid oscillation without
    # introducing the lag that a plain moving average would.
    _cx_buf: collections.deque = collections.deque(maxlen=15)
    _cy_buf: collections.deque = collections.deque(maxlen=15)

    try:
        for frame in loader:
            # --- Player action centroid (fallback) ---
            boxes = player_detector.detect(frame.image)
            player_cx, player_cy = localizer.update(
                boxes, frame_wh=(info.width, info.height)
            )

            # --- Ball detection (primary) ---
            ball_pos = ball_detector.detect(frame.image)

            if ball_pos is not None:
                bx, by, bconf = ball_pos
                # Go straight to the ball — don't blend with player centroid.
                # The player centroid is a fallback only; mixing it in just
                # pulls the crop away from a ball we can already see.
                focus_x, focus_y = bx, by
                # Flush the median buffer so it doesn't fight the ball position.
                # Without this, 15 stale player-centroid frames hold the median
                # in the wrong place even when the ball is clearly detected.
                _cx_buf.clear()
                _cy_buf.clear()
                _cx_buf.append(bx)
                _cy_buf.append(by)
            else:
                focus_x, focus_y = player_cx, player_cy
                # --- Median filter (player centroid only) ---
                # Kills top-K swap jitter. Only applied when ball is lost so
                # the buffer doesn't accumulate stale values that resist snapping
                # to the ball when it reappears.
                _cx_buf.append(focus_x)
                _cy_buf.append(focus_y)
                focus_x = float(np.median(_cx_buf))
                focus_y = float(np.median(_cy_buf))

            # Smooth the focus point (no raw velocity — keeps crop in alpha_slow).
            sm_pos, _ = smoother.update((focus_x, focus_y), (0.0, 0.0))

            # Velocity from the SMOOTHED position is clean enough for look-ahead.
            sm_vx = sm_pos[0] - prev_sm_x if prev_sm_x is not None else 0.0
            sm_vy = sm_pos[1] - prev_sm_y if prev_sm_y is not None else 0.0
            prev_sm_x, prev_sm_y = sm_pos

            crop = cropper.update(sm_pos, (sm_vx, sm_vy), confidence=1.0)

            if jsonl_fp is not None:
                jsonl_fp.write(json.dumps({
                    "frame_index": frame.index,
                    "action_xy": [focus_x, focus_y],
                    "ball_xy": [ball_pos[0], ball_pos[1]] if ball_pos else None,
                    "ball_conf": ball_pos[2] if ball_pos else None,
                    "n_players": len(boxes),
                    "crop": list(crop.as_tuple()),
                }) + "\n")

            if args.out:
                overlay = draw_overlay(frame.image, boxes, focus_x, focus_y,
                                       ball_pos, crop)
                if writer is None:
                    writer = cv2.VideoWriter(args.out, fourcc, info.fps,
                                             (info.width, info.height))
                writer.write(overlay)

            if args.crop_out:
                sub = frame.image[crop.y:crop.y + crop.h,
                                  crop.x:crop.x + crop.w]
                if crop_writer is None:
                    crop_writer = cv2.VideoWriter(args.crop_out, fourcc, info.fps,
                                                  (crop.w, crop.h))
                crop_writer.write(sub)

            if frame.index % 50 == 0:
                ball_str = f"ball=({ball_pos[0]:.0f},{ball_pos[1]:.0f},c={ball_pos[2]:.2f})" \
                    if ball_pos else "ball=none"
                print(f"  frame {frame.index}  players={len(boxes)}"
                      f"  {ball_str}"
                      f"  focus=({focus_x:.0f},{focus_y:.0f})", file=sys.stderr)
    finally:
        loader.close()
        if writer is not None:
            writer.release()
        if crop_writer is not None:
            crop_writer.release()
        if jsonl_fp is not None:
            jsonl_fp.close()


if __name__ == "__main__":
    main()
