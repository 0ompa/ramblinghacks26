"""Basketball tracking pipeline.

Stages:
    video_loader       → frame iterator
    kalman_tracker     → 2nd-stage trajectory smoothing
    crop_generator     → 9:16 cinematic crop window
    player_detector    → YOLOv8 person detection
    action_localizer   → weighted action centroid from player boxes
"""
from .video_loader import VideoLoader, Frame, VideoInfo
from .kalman_tracker import TrajectorySmoother
from .crop_generator import PortraitCropGenerator, CropWindow

__all__ = [
    "VideoLoader", "Frame", "VideoInfo",
    "TrajectorySmoother",
    "PortraitCropGenerator", "CropWindow",
]
