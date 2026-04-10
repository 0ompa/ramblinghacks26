"""
Stage 6 — Frame Crop & Encode

Applies crop windows to frames and writes the final 9:16 portrait MP4.

For each frame:
    1. Slice the crop region from the source frame
    2. Resize to 1080x1920
    3. Write to output via cv2.VideoWriter

Optionally remuxes audio from the original via FFmpeg.
"""

import os
import subprocess
import cv2
import numpy as np


def encode_portrait(frames, crop_windows, meta, output_path,
                    source_video_path=None):
    """Crop, resize, and encode all frames to a portrait MP4.

    Args:
        frames:            list of BGR uint8 numpy arrays
        crop_windows:      list of {"crop_x1", "crop_y1", "crop_w", "crop_h"}
        meta:              video metadata dict from ingest
        output_path:       where to write the portrait MP4
        source_video_path: original video (for audio remux, optional)

    Returns:
        output_path
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    out_w, out_h = 1080, 1920
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, meta["fps"], (out_w, out_h))

    for frame, cw in zip(frames, crop_windows):
        x1 = cw["crop_x1"]
        y1 = cw["crop_y1"]
        w = cw["crop_w"]
        h = cw["crop_h"]
        cropped = frame[y1:y1+h, x1:x1+w]
        resized = cv2.resize(cropped, (out_w, out_h))
        writer.write(resized)

    writer.release()

    # Remux audio from original if available
    if source_video_path and os.path.isfile(source_video_path):
        tmp = output_path + ".tmp.mp4"
        os.rename(output_path, tmp)
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", tmp,
                "-i", source_video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                output_path,
            ], capture_output=True, check=True)
            os.remove(tmp)
        except (subprocess.CalledProcessError, FileNotFoundError):
            os.rename(tmp, output_path)

    return output_path
