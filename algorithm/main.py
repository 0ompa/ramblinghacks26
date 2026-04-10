"""
main.py: this is the entry point for the basketball clip importance scoring pipeline.
run in terminal: python main.py path/to/clip.mp4
"""

import sys
import cv2

from detect import detect_frames
from importance import compute_importance
from config import YOLO_WEIGHTS, CONF_THRESHOLD, EMA_ALPHA


def get_frame_height(video_path):
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return height


def main(video_path):
    print(f"[1/3] Loading video: {video_path}")
    frame_height = get_frame_height(video_path)
    print(f"Frame height: {frame_height}px")

    print(f"[2/3] Running YOLO detection...")
    frame_data = detect_frames(video_path, weights=YOLO_WEIGHTS, conf=CONF_THRESHOLD)
    print(f"Detected {len(frame_data)} frames")

    print(f"[3/3] Computing importance scores...")
    scores = compute_importance(frame_data, frame_height, alpha=EMA_ALPHA)

    print("\n--- Importance Scores (first 20 frames) ---")
    for i, score in enumerate(scores[:20]):
        bar = "█" * int(score * 40)
        print(f"Frame {i:3d}: {score:.4f}  {bar}")

    print(f"\nMax score : {max(scores):.4f} at frame {scores.index(max(scores))}")
    print(f"Mean score: {sum(scores)/len(scores):.4f}")

    return scores


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])