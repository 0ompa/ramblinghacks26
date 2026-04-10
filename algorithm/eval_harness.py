"""
Stage 7 — Evaluation Harness

Automated, quantitative scoring of crop quality so the team can iterate
on hyperparameters (sigma, lookahead_weight, ball_conf_threshold) without
eyeballing every clip.

Composite Score (0–100), weighted sum of four sub-metrics:

    Ball Retention (40%)  — fraction of frames where ball centroid is
                            inside the crop window
    Smoothness (25%)      — inverse of mean frame-to-frame crop jitter
    Centering (20%)       — how horizontally centered the ball is in crop
    Coverage (15%)        — fraction of player bboxes >=50% inside crop

Usage:
    from eval_harness import evaluate_clip, parameter_sweep, print_leaderboard

    result = evaluate_clip(crop_windows, frame_data, meta)
    results = parameter_sweep(pipeline_fn, video_path, param_grid)
    print_leaderboard(results)
"""

import csv
import itertools
import json
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# Metric weights — edit these to change what matters most
# ---------------------------------------------------------------------------
METRIC_WEIGHTS = {
    "ball_retention": 0.40,
    "smoothness":     0.25,
    "centering":      0.20,
    "coverage":       0.15,
}


# ---------------------------------------------------------------------------
# Sub-metric implementations
# ---------------------------------------------------------------------------

def _ball_retention(crop_windows, frame_data):
    """Fraction of ball-visible frames where ball centroid is inside crop."""
    visible = 0
    retained = 0
    for cw, fd in zip(crop_windows, frame_data):
        if fd["ball"] is None:
            continue
        visible += 1
        bx, by = fd["ball"]
        x1 = cw["crop_x1"]
        y1 = cw["crop_y1"]
        x2 = x1 + cw["crop_w"]
        y2 = y1 + cw["crop_h"]
        if x1 <= bx <= x2 and y1 <= by <= y2:
            retained += 1
    return retained / visible if visible > 0 else 1.0


def _smoothness(crop_windows):
    """Score 0–1 based on crop-center jitter between consecutive frames.

    Jitter is normalized by crop width so the metric is resolution-independent.
    0% jitter → 1.0,  >=5% jitter per frame → 0.0.
    """
    if len(crop_windows) < 2:
        return 1.0

    centers = np.array([
        (cw["crop_x1"] + cw["crop_w"] / 2.0, cw["crop_y1"] + cw["crop_h"] / 2.0)
        for cw in crop_windows
    ])
    displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)

    crop_w = crop_windows[0]["crop_w"]
    norm_jitter = np.mean(displacements) / crop_w

    MAX_JITTER = 0.05
    return float(np.clip(1.0 - norm_jitter / MAX_JITTER, 0.0, 1.0))


def _centering(crop_windows, frame_data):
    """How well-centered the ball is horizontally within the crop.

    1.0 = perfectly centered, 0.0 = at crop edge.
    Only scored on frames where ball is visible and inside crop.
    """
    scores = []
    for cw, fd in zip(crop_windows, frame_data):
        if fd["ball"] is None:
            continue
        bx, _ = fd["ball"]
        cx = cw["crop_x1"] + cw["crop_w"] / 2.0
        half_w = cw["crop_w"] / 2.0
        if half_w == 0:
            continue
        offset = abs(bx - cx) / half_w
        scores.append(float(np.clip(1.0 - offset, 0.0, 1.0)))
    return float(np.mean(scores)) if scores else 1.0


def _coverage(crop_windows, frame_data, overlap_threshold=0.5):
    """Fraction of player bboxes that are >=overlap_threshold inside crop."""
    frame_scores = []
    for cw, fd in zip(crop_windows, frame_data):
        bboxes = fd["players"]
        if not bboxes:
            continue
        cx1 = cw["crop_x1"]
        cy1 = cw["crop_y1"]
        cx2 = cx1 + cw["crop_w"]
        cy2 = cy1 + cw["crop_h"]

        covered = 0
        for bbox in bboxes:
            px1, py1, px2, py2 = bbox
            ix1 = max(cx1, px1)
            iy1 = max(cy1, py1)
            ix2 = min(cx2, px2)
            iy2 = min(cy2, py2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            player_area = max(1, (px2 - px1) * (py2 - py1))
            if inter / player_area >= overlap_threshold:
                covered += 1
        frame_scores.append(covered / len(bboxes))

    return float(np.mean(frame_scores)) if frame_scores else 1.0


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def evaluate_clip(crop_windows, frame_data, meta,
                  clip_name="unnamed", params=None, elapsed_s=0.0):
    """Score a single clip's crop output.

    Args:
        crop_windows:  list of {"crop_x1", "crop_y1", "crop_w", "crop_h"}
        frame_data:    list of {"ball": (cx,cy)|None, "players": [bbox,...]}
                       — the same format detect.py already produces
        meta:          video metadata dict from ingest.get_metadata
        clip_name:     label for reporting
        params:        hyperparameters that produced this crop
        elapsed_s:     pipeline runtime for this config

    Returns:
        dict with sub-metrics + composite score
    """
    br = _ball_retention(crop_windows, frame_data)
    sm = _smoothness(crop_windows)
    cn = _centering(crop_windows, frame_data)
    cv = _coverage(crop_windows, frame_data)

    composite = 100.0 * (
        METRIC_WEIGHTS["ball_retention"] * br
        + METRIC_WEIGHTS["smoothness"] * sm
        + METRIC_WEIGHTS["centering"] * cn
        + METRIC_WEIGHTS["coverage"] * cv
    )

    return {
        "clip": clip_name,
        "params": params or {},
        "ball_retention": br,
        "smoothness": sm,
        "centering": cn,
        "coverage": cv,
        "composite": composite,
        "elapsed_s": elapsed_s,
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def parameter_sweep(run_pipeline_fn, video_path, param_grid, clip_name="unnamed"):
    """Run the pipeline across a grid of hyperparameters, score each config.

    Args:
        run_pipeline_fn: callable(video_path, **params)
            Must return (crop_windows, frame_data, meta).
        video_path:  path to source clip
        param_grid:  dict mapping param names to lists of values
                     e.g. {"sigma": [8, 12, 15], "lookahead_weight": [0.1, 0.3]}
        clip_name:   label for reporting

    Returns:
        list of result dicts, sorted by composite score descending
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    results = []
    total = len(combos)

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"[sweep] config {i}/{total}: {params}")

        t0 = time.perf_counter()
        try:
            crop_windows, frame_data, meta = run_pipeline_fn(video_path, **params)
        except Exception as e:
            print(f"  !! pipeline failed: {e}")
            continue
        elapsed = time.perf_counter() - t0

        result = evaluate_clip(
            crop_windows, frame_data, meta,
            clip_name=clip_name, params=params, elapsed_s=elapsed,
        )
        print(f"  composite={result['composite']:.1f}  "
              f"ball_ret={result['ball_retention']:.2f}  "
              f"smooth={result['smoothness']:.2f}  "
              f"center={result['centering']:.2f}  "
              f"coverage={result['coverage']:.2f}  "
              f"({elapsed:.1f}s)")
        results.append(result)

    results.sort(key=lambda r: r["composite"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _flat_result(r):
    """Flatten a result dict for CSV/JSON export."""
    flat = {"clip": r["clip"]}
    for k, v in r["params"].items():
        flat[f"param_{k}"] = v
    for key in ("ball_retention", "smoothness", "centering", "coverage", "composite", "elapsed_s"):
        flat[key] = round(r[key], 4)
    return flat


def save_results_csv(results, path):
    """Write sweep results to CSV."""
    if not results:
        return
    rows = [_flat_result(r) for r in results]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] saved {len(rows)} results to {path}")


def save_results_json(results, path):
    """Write sweep results to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump([_flat_result(r) for r in results], f, indent=2)
    print(f"[eval] saved {len(results)} results to {path}")


def print_leaderboard(results, top_n=10):
    """Pretty-print the top N configurations."""
    n = min(top_n, len(results))
    print(f"\n{'='*80}")
    print(f"  TOP {n} CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Composite':<12}{'Ball Ret':<10}{'Smooth':<10}"
          f"{'Center':<10}{'Coverage':<10}{'Params'}")
    print(f"{'-'*80}")
    for i, r in enumerate(results[:n], 1):
        param_str = "  ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"{i:<6}{r['composite']:<12.1f}{r['ball_retention']:<10.3f}"
              f"{r['smoothness']:<10.3f}{r['centering']:<10.3f}"
              f"{r['coverage']:<10.3f}{param_str}")
    print()
