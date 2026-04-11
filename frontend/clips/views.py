import os
import re
import tempfile
from pathlib import Path

import shutil

from django.conf import settings
from django.contrib import messages
from django.http import Http404, HttpResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from .forms import HighlightUploadForm
from .pipeline import (
    PipelineError,
    process_highlight_to_portrait,
    try_export_youtube_shorts,
    try_transcode_portrait_h264,
)


_CLIP_LANDSCAPE = re.compile(r"^clip(\d+)_landscape\.mp4$", re.IGNORECASE)
_CLIP_ID = re.compile(r"^clip(\d+)$", re.IGNORECASE)


def _portrait_basename(videos_dir: Path, n_str: str) -> str | None:
    webm = videos_dir / f"clip{n_str}_portrait.webm"
    mp4 = videos_dir / f"clip{n_str}_portrait.mp4"
    if webm.is_file():
        return webm.name
    if mp4.is_file():
        return mp4.name
    return None


def next_clip_index(videos_dir: Path) -> int:
    if not videos_dir.is_dir():
        return 1
    highest = 0
    for p in videos_dir.iterdir():
        if not p.is_file():
            continue
        m = _CLIP_LANDSCAPE.match(p.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def discover_clips(videos_dir: Path) -> list[dict]:
    """Pair clipN_landscape.mp4 with clipN_portrait.webm or clipN_portrait.mp4."""
    if not videos_dir.is_dir():
        return []
    rows: list[tuple[int, str, str, str]] = []
    for f in videos_dir.iterdir():
        if not f.is_file():
            continue
        m = _CLIP_LANDSCAPE.match(f.name)
        if not m:
            continue
        n_str = m.group(1)
        port_name = _portrait_basename(videos_dir, n_str)
        if not port_name:
            continue
        rows.append((int(n_str), n_str, f.name, port_name))
    rows.sort(key=lambda r: r[0])
    clips: list[dict] = []
    for num, n_str, land_name, port_name in rows:
        clips.append(
            {
                "id": f"clip{n_str}",
                "title": f"Clip {num}",
                "play_type": "Highlight",
                "original_video_url": f"/videos/{land_name}",
                "portrait_video_url": f"/videos/{port_name}",
            }
        )
    return clips


def _resolve_clip_pair(clip_id: str) -> tuple[Path, Path] | None:
    """Return (portrait_path, landscape_path) under VIDEOS_DIR, or None."""
    m = _CLIP_ID.fullmatch((clip_id or "").strip())
    if not m:
        return None
    n_str = m.group(1)
    videos_dir = Path(settings.VIDEOS_DIR)
    land = videos_dir / f"clip{n_str}_landscape.mp4"
    port_name = _portrait_basename(videos_dir, n_str)
    if not land.is_file() or not port_name:
        return None
    port = videos_dir / port_name
    if not port.is_file():
        return None
    return port, land


@require_http_methods(["GET"])
def export_youtube_shorts(request, clip_id: str):
    """
    Download a YouTube Shorts–friendly MP4 (1080x1920 H.264 + AAC when available).
    Requires ffmpeg on the server PATH.
    """
    pair = _resolve_clip_pair(clip_id)
    if not pair:
        raise Http404("Clip not found")

    portrait_path, landscape_path = pair

    if not shutil.which("ffmpeg"):
        return HttpResponse(
            "ffmpeg is not installed or not on PATH. Install ffmpeg to export for YouTube.",
            status=503,
            content_type="text/plain; charset=utf-8",
        )

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    ok = try_export_youtube_shorts(
        str(portrait_path),
        str(landscape_path),
        tmp_path,
    )
    if not ok:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return HttpResponse(
            "Export failed (ffmpeg). The clip may be corrupt or unsupported.",
            status=500,
            content_type="text/plain; charset=utf-8",
        )

    m_id = _CLIP_ID.fullmatch(clip_id.strip())
    safe_name = f"clip{m_id.group(1)}_youtube_shorts.mp4" if m_id else "youtube_shorts.mp4"

    def stream_and_remove():
        try:
            with open(tmp_path, "rb") as fh:
                while True:
                    chunk = fh.read(65536)
                    if not chunk:
                        break
                    yield chunk
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    resp = StreamingHttpResponse(
        stream_and_remove(),
        content_type="video/mp4",
    )
    resp["Content-Disposition"] = f'attachment; filename="{safe_name}"'
    return resp


def index(request):
    clips = discover_clips(settings.VIDEOS_DIR)
    return render(request, "clips/index.html", {"clips": clips})


@require_http_methods(["POST"])
def upload_highlight(request):
    form = HighlightUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        messages.error(request, form.errors.as_text())
        return redirect("index")

    videos_dir = Path(settings.VIDEOS_DIR)
    videos_dir.mkdir(parents=True, exist_ok=True)
    n = next_clip_index(videos_dir)
    n_str = str(n)
    land_path = videos_dir / f"clip{n_str}_landscape.mp4"
    raw_portrait = videos_dir / f"clip{n_str}_portrait_raw.mp4"
    final_portrait = videos_dir / f"clip{n_str}_portrait.mp4"

    uploaded = form.cleaned_data["video"]
    tmp_in = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_in = tmp.name
            for chunk in uploaded.chunks():
                tmp.write(chunk)
        shutil.copyfile(tmp_in, land_path)

        process_highlight_to_portrait(str(land_path), str(raw_portrait))

        if try_transcode_portrait_h264(str(raw_portrait), str(final_portrait)):
            raw_portrait.unlink(missing_ok=True)
            messages.success(
                request,
                f"Clip {n} added (portrait re-encoded with ffmpeg for the browser).",
            )
        else:
            os.replace(str(raw_portrait), str(final_portrait))
            messages.success(
                request,
                f"Clip {n} added. Install ffmpeg with libx264 for more reliable browser playback.",
            )
    except PipelineError as e:
        land_path.unlink(missing_ok=True)
        raw_portrait.unlink(missing_ok=True)
        final_portrait.unlink(missing_ok=True)
        messages.error(request, str(e))
    except Exception as e:
        land_path.unlink(missing_ok=True)
        raw_portrait.unlink(missing_ok=True)
        final_portrait.unlink(missing_ok=True)
        messages.error(request, f"Upload failed: {e}")
    finally:
        if tmp_in and os.path.isfile(tmp_in):
            os.unlink(tmp_in)

    return redirect("index")
