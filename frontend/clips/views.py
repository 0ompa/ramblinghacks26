import re
from pathlib import Path

from django.conf import settings
from django.shortcuts import render


_CLIP_LANDSCAPE = re.compile(r"^clip(\d+)_landscape\.mp4$", re.IGNORECASE)


def discover_clips(videos_dir: Path) -> list[dict]:
    """Pair clipN_landscape.mp4 with clipN_portrait.webm in the same folder."""
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
        portrait = videos_dir / f"clip{n_str}_portrait.webm"
        if not portrait.is_file():
            continue
        rows.append((int(n_str), n_str, f.name, portrait.name))
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


def index(request):
    clips = discover_clips(settings.VIDEOS_DIR)
    return render(request, "clips/index.html", {"clips": clips})
