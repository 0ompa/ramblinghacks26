"""Run repo-root ``run.py`` portrait pipeline from Django (Roboflow + OpenCV)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from django.conf import settings


class PipelineError(Exception):
    """User-facing pipeline failure."""


def _load_run_module():
    path = Path(settings.REPO_ROOT) / "run.py"
    if not path.is_file():
        raise PipelineError(f"Missing pipeline script: {path}")
    spec = importlib.util.spec_from_file_location("highlight_run", path)
    if spec is None or spec.loader is None:
        raise PipelineError("Could not load run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["highlight_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def process_highlight_to_portrait(landscape_mp4: str, portrait_out_mp4: str) -> None:
    """
    Run the same two-pass flow as ``python run.py --input ... --output ...``.
    Writes portrait MP4 (OpenCV ``mp4v``) to ``portrait_out_mp4``.
    """
    try:
        run_mod = _load_run_module()
    except KeyError as e:
        key = getattr(e, "args", [""])[0]
        if key == "ROBOFLOW_API_KEY":
            raise PipelineError(
                "ROBOFLOW_API_KEY is not set. Add it to a .env file in the repo root."
            ) from e
        raise PipelineError(f"Missing environment variable: {key}") from e
    except Exception as e:
        raise PipelineError(f"Could not load pipeline: {e}") from e

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise PipelineError("ROBOFLOW_API_KEY is not set.")

    try:
        print("Roboflow hosted API …", flush=True)
        model = run_mod.load_roboflow_model(api_key=api_key)
        print("OK", flush=True)
        run_mod.process_video(model, landscape_mp4, portrait_out_mp4)
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(f"Pipeline failed: {e}") from e


def try_transcode_portrait_h264(src_mp4: str, dst_mp4: str) -> bool:
    """Re-encode to H.264 + faststart for HTML5 video. Returns True if ffmpeg succeeded."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        r = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                src_mp4,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                dst_mp4,
            ],
            capture_output=True,
            timeout=3600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def ffprobe_has_audio(path: str) -> bool:
    """True if the container has at least one audio stream."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not os.path.isfile(path):
        return False
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if r.returncode != 0:
        return False
    try:
        data = json.loads(r.stdout or "{}")
        streams = data.get("streams") or []
        return len(streams) > 0
    except json.JSONDecodeError:
        return False


def try_export_youtube_shorts(
    portrait_path: str,
    landscape_path: str,
    dst_mp4: str,
) -> bool:
    """
    Portrait 9:16 scaled/padded to 1080x1920, H.264 + yuv420p + faststart for YouTube Shorts.
    Audio: landscape track if present (typical game audio), else portrait, else silent video.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    vf = (
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p"
    )
    v_args = [
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "medium",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]

    try:
        if ffprobe_has_audio(landscape_path):
            r = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    portrait_path,
                    "-i",
                    landscape_path,
                    "-filter_complex",
                    f"[0:v]{vf}[v]",
                    "-map",
                    "[v]",
                    "-map",
                    "1:a:0",
                    *v_args,
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-shortest",
                    dst_mp4,
                ],
                capture_output=True,
                timeout=7200,
                check=False,
            )
        elif ffprobe_has_audio(portrait_path):
            r = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    portrait_path,
                    "-vf",
                    vf,
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    *v_args,
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "48000",
                    dst_mp4,
                ],
                capture_output=True,
                timeout=7200,
                check=False,
            )
        else:
            r = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    portrait_path,
                    "-vf",
                    vf,
                    "-an",
                    *v_args,
                    dst_mp4,
                ],
                capture_output=True,
                timeout=7200,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0
