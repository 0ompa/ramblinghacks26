import cv2
import numpy as np
import subprocess
import os
import tempfile


def apply_crop(frame: np.ndarray, x1: int, crop_w: int) -> np.ndarray:
    cropped = frame[:, x1 : x1 + crop_w]
    portrait = cv2.resize(cropped, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
    return portrait


def encode_clip(
    input_path: str,
    output_path: str,
    crop_x1_array: np.ndarray,
    crop_w: int,
    fps: float = 25.0,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    tmp_dir = tempfile.mkdtemp(prefix="portrait_frames_")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx < len(crop_x1_array):
            x1 = int(crop_x1_array[idx])
        else:
            x1 = int(crop_x1_array[-1])

        portrait = apply_crop(frame, x1, crop_w)
        out_file = os.path.join(tmp_dir, f"{idx:06d}.jpg")
        cv2.imwrite(out_file, portrait, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1

    cap.release()

    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", os.path.join(tmp_dir, "%06d.jpg"),
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v", "-map", "1:a",
        "-shortest",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

    return output_path
