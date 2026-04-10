"""
run_sweep.py — Parameter Sweep Runner

Runs the full pipeline across a grid of hyperparameters, scores each
configuration, and outputs a ranked leaderboard + CSV/JSON.

Usage:
    python run_sweep.py ../clips/game1.mp4
    python run_sweep.py ../clips/ --top 5

On a 4070 with a 30-second clip at 25fps, expect ~15-25s per config.
A 25-config grid takes ~8-10 minutes.
"""

import argparse
import glob
import os
import sys

from pipeline import run_pipeline
from eval_harness import (
    parameter_sweep,
    save_results_csv,
    save_results_json,
    print_leaderboard,
)
PARAM_GRID = {
    "sigma":              [8.0, 12.0, 15.0, 20.0, 25.0],
    "lookahead_weight":   [0.1, 0.2, 0.3, 0.5],
    "ball_conf_threshold": [0.3, 0.4, 0.5],
}


def find_clips(paths):
    """Resolve file paths and directories into a flat list of MP4s."""
    clips = []
    for p in paths:
        if os.path.isdir(p):
            clips.extend(sorted(glob.glob(os.path.join(p, "*.mp4"))))
        elif os.path.isfile(p) and p.lower().endswith(".mp4"):
            clips.append(p)
        else:
            print(f"[warn] skipping: {p}")
    return clips


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for the crop pipeline")
    parser.add_argument("clips", nargs="+", help="MP4 files or directories")
    parser.add_argument("--top", type=int, default=10, help="Show top N results")
    parser.add_argument("--out-dir", default="../output",
                        help="Directory for result files")
    parser.add_argument("--every-n", type=int, default=1,
                        help="Frame sampling rate (higher = faster)")
    args = parser.parse_args()

    clips = find_clips(args.clips)
    if not clips:
        print("No MP4 clips found.")
        sys.exit(1)

    total_configs = 1
    for v in PARAM_GRID.values():
        total_configs *= len(v)

    print(f"[sweep] {len(clips)} clip(s), {total_configs} configs per clip")

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []

    for clip_path in clips:
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        print(f"\n{'='*60}")
        print(f"  CLIP: {clip_name}")
        print(f"{'='*60}\n")

        def pipeline_fn(video_path, **params):
            return run_pipeline(video_path, every_n=args.every_n, **params)

        results = parameter_sweep(
            run_pipeline_fn=pipeline_fn,
            video_path=clip_path,
            param_grid=PARAM_GRID,
            clip_name=clip_name,
        )
        all_results.extend(results)

        csv_path = os.path.join(args.out_dir, f"{clip_name}_sweep.csv")
        save_results_csv(results, csv_path)

    all_results.sort(key=lambda r: r["composite"], reverse=True)
    print_leaderboard(all_results, top_n=args.top)

    save_results_json(all_results, os.path.join(args.out_dir, "sweep_all.json"))
    save_results_csv(all_results, os.path.join(args.out_dir, "sweep_all.csv"))

    if all_results:
        best = all_results[0]
        print(f"[sweep] BEST CONFIG: composite={best['composite']:.1f}")
        for k, v in best["params"].items():
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
