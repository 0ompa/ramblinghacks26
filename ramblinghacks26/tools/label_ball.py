"""Interactive ball labeling tool (matplotlib backend).

Usage:
    python tools/label_ball.py --input clips/clip1.mp4 --out labels/clip1_ball.json

Controls (in the plot window):
    Left-click  — mark ball center, auto-advance to next frame
    n           — confirm click and advance (same as clicking) / skip if no click
    s           — skip this frame (no label saved)
    q           — quit and save whatever has been labeled so far

Output: labels/clip1_ball.json
    {"frames": [{"f": <frame_index>, "xy": [x, y]}, ...]}
    Only frames where you clicked are written (skipped frames are omitted).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

N_SAMPLES = 25


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="clips/clip1.mp4")
    p.add_argument("--out", default="labels/clip1_ball.json")
    p.add_argument("--n", type=int, default=N_SAMPLES,
                   help="number of evenly-spaced frames to label")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None,
                   help="last frame index (inclusive); default=last frame")
    return p.parse_args()


def read_frame(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    return img if ok else None


def main():
    args = parse_args()

    import matplotlib
    matplotlib.use("WXAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.input}", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[video] {W}x{H} @ {fps:.2f} fps, {total} frames", file=sys.stderr)

    end = args.end if args.end is not None else total - 1
    end = min(end, total - 1)

    n = args.n
    indices = (
        [args.start]
        if n == 1
        else [int(round(args.start + i * (end - args.start) / (n - 1))) for i in range(n)]
    )
    seen: set[int] = set()
    unique: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    indices = unique

    print(f"[labeler] will visit {len(indices)} frames: {indices}", file=sys.stderr)
    print("[labeler] LEFT-CLICK=mark ball  n=next  s=skip  q=quit", file=sys.stderr)

    labels: list[dict] = []

    # Shared mutable state between callbacks and the main loop.
    state = {"click": None, "action": None}  # action: None | "advance" | "skip" | "quit"

    # Scale down for display so the window fits on screen.
    scale = min(1.0, 1280 / W)
    disp_w = int(W * scale)
    disp_h = int(H * scale)

    fig, ax = plt.subplots(figsize=(disp_w / 96, disp_h / 96))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_axis_off()
    im_obj = ax.imshow(np.zeros((disp_h, disp_w, 3), dtype=np.uint8))
    title_obj = ax.set_title("", fontsize=10, color="white",
                              backgroundcolor="black", pad=4)
    dot = ax.plot([], [], "go", markersize=12, markeredgecolor="lime",
                  markeredgewidth=2)[0]
    cross_h = ax.plot([], [], "g-", linewidth=1.5)[0]
    cross_v = ax.plot([], [], "g-", linewidth=1.5)[0]

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return
        state["click"] = (int(round(mx)), int(round(my)))
        state["action"] = "advance"
        fig.canvas.draw_idle()

    def on_key(event):
        k = event.key
        if k == "n":
            state["action"] = "advance"
        elif k == "s":
            state["action"] = "skip"
        elif k == "q":
            state["action"] = "quit"

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def render(frame_idx, i_frame, img_bgr):
        rgb = cv2.cvtColor(
            cv2.resize(img_bgr, (disp_w, disp_h)), cv2.COLOR_BGR2RGB
        )
        im_obj.set_data(rgb)
        im_obj.set_extent([0, disp_w, disp_h, 0])  # origin='upper'
        ax.set_xlim(0, disp_w)
        ax.set_ylim(disp_h, 0)

        if state["click"] is not None:
            mx, my = state["click"]
            dot.set_data([mx], [my])
            r = 12
            cross_h.set_data([mx - r, mx + r], [my, my])
            cross_v.set_data([mx, mx], [my - r, my + r])
            rx = int(round(mx / scale))
            ry = int(round(my / scale))
            title = (f"[{i_frame+1}/{len(indices)}] f={frame_idx}  "
                     f"ball=({rx},{ry})  — click again to adjust | n=confirm | s=skip | q=quit")
        else:
            dot.set_data([], [])
            cross_h.set_data([], [])
            cross_v.set_data([], [])
            title = (f"[{i_frame+1}/{len(indices)}] f={frame_idx}  "
                     f"— LEFT-CLICK ball | s=skip | q=quit")
        title_obj.set_text(title)
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ion()
    plt.show(block=False)

    i = 0
    while i < len(indices):
        frame_idx = indices[i]
        img = read_frame(cap, frame_idx)
        if img is None:
            print(f"  [skip] could not read frame {frame_idx}", file=sys.stderr)
            i += 1
            continue

        # Reset per-frame state.
        state["click"] = None
        state["action"] = None

        render(frame_idx, i, img)

        while state["action"] is None:
            plt.pause(0.05)
            # Re-render to show updated click marker.
            render(frame_idx, i, img)

        action = state["action"]
        state["action"] = None

        if action == "advance":
            if state["click"] is not None:
                mx, my = state["click"]
                rx = int(round(mx / scale))
                ry = int(round(my / scale))
                labels.append({"f": frame_idx, "xy": [rx, ry]})
                print(f"  labeled f={frame_idx} -> ({rx},{ry})", file=sys.stderr)
            else:
                print(f"  skipped f={frame_idx} (n with no click)", file=sys.stderr)
            i += 1
        elif action == "skip":
            print(f"  skipped f={frame_idx}", file=sys.stderr)
            i += 1
        elif action == "quit":
            break

    plt.ioff()
    plt.close(fig)
    cap.release()

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"frames": labels}, f, indent=2)

    print(f"[done] saved {len(labels)} labels to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
