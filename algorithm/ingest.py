import os
import cv2
import numpy as np


def get_metadata(video_path):
    """Extract resolution, FPS, frame count, duration without decoding."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        meta = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        meta["duration_s"] = meta["frame_count"] / meta["fps"] if meta["fps"] > 0 else 0.0
        return meta
    finally:
        cap.release()


def iter_frames(video_path, every_n=1, max_frames=None):
    """Yield (frame_index, frame) tuples without loading full video into RAM."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    yielded = 0
    idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every_n == 0:
                yield idx, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def load_all_frames(video_path, every_n=1, max_frames=None):
    """Load all frames into memory for two-pass processing."""
    meta = get_metadata(video_path)
    frames = []
    indices = []

    for idx, frame in iter_frames(video_path, every_n=every_n, max_frames=max_frames):
        frames.append(frame)
        indices.append(idx)

    return meta, frames, indices
