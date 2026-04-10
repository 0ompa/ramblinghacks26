"""Full pipeline integration"""

import argparse
import os
import sys

from ingest import get_metadata, load_all_frames
from detect import detect_frames
from focus import compute_focus_point
from smoothing import smooth_and_crop
from encode import encode_portrait
from config import CONF_THRESHOLD, YOLO_WEIGHTS


def run_pipeline(video_path,
                 sigma=15.0, lookahead_weight=0.3, velocity_window=3,
                 ball_conf_threshold=0.4, every_n=1,
                 output_path=None):
    """Run the full crop pipeline on a single video."""
    # Stage 1: Ingest
    meta, frames, indices = load_all_frames(video_path, every_n=every_n)
    print(f"[pipeline] {meta['width']}x{meta['height']} @ {meta['fps']}fps, "
          f"{len(frames)} frames loaded")

    # Stage 2: Detection (uses existing detect.py)
    frame_data = detect_frames(video_path, weights=YOLO_WEIGHTS, conf=CONF_THRESHOLD)
    print(f"[pipeline] detected {len(frame_data)} frames")

    # If we sampled every_n > 1, trim frame_data to match
    if every_n > 1:
        frame_data = frame_data[::every_n][:len(frames)]

    # Stage 4: Focus points
    focus_points = []
    for i, (frame, fd) in enumerate(zip(frames, frame_data)):
        prev_frame = frames[i - 1] if i > 0 else None
        fp = compute_focus_point(
            fd, frame, prev_frame,
            ball_conf_threshold=ball_conf_threshold,
            frame_width=meta["width"],
            frame_height=meta["height"],
        )
        focus_points.append(fp)

    # Stage 5: Smoothing + crop windows
    crop_windows = smooth_and_crop(
        focus_points, meta,
        sigma=sigma,
        lookahead_weight=lookahead_weight,
        velocity_window=velocity_window,
    )

    # Stage 6: Encode (optional)
    if output_path:
        encode_portrait(frames, crop_windows, meta, output_path, video_path)
        print(f"[pipeline] output written to {output_path}")

    return crop_windows, frame_data, meta


def main():
    parser = argparse.ArgumentParser(
        description="16:9 → 9:16 Basketball Crop Pipeline")
    parser.add_argument("video", help="Path to source MP4")
    parser.add_argument("-o", "--output", default=None, help="Output MP4 path")
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--lookahead", type=float, default=0.3)
    parser.add_argument("--velocity-window", type=int, default=3)
    parser.add_argument("--ball-conf", type=float, default=0.4)
    parser.add_argument("--every-n", type=int, default=1)
    args = parser.parse_args()

    if not args.output:
        base = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join("..", "output", f"{base}_portrait.mp4")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    run_pipeline(
        video_path=args.video,
        sigma=args.sigma,
        lookahead_weight=args.lookahead,
        velocity_window=args.velocity_window,
        ball_conf_threshold=args.ball_conf,
        output_path=args.output,
        every_n=args.every_n,
    )


if __name__ == "__main__":
    main()
