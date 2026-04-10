"""Smoke test for the eval harness using synthetic data."""

from eval_harness import evaluate_clip, print_leaderboard

FAKE_META = {"width": 1920, "height": 1080, "fps": 25.0,
             "frame_count": 250, "duration_s": 10.0}
CROP_W = int(1080 * 9 / 16)  # 607


def test_perfect_tracking():
    """Ball always centered in crop → should score close to 100."""
    n = 250
    crop_windows = [{"crop_x1": 656, "crop_y1": 0,
                     "crop_w": CROP_W, "crop_h": 1080}] * n
    ball_cx = 656 + CROP_W // 2
    ball_cy = 540
    frame_data = [
        {"ball": (ball_cx, ball_cy),
         "players": [(700, 400, 800, 800), (850, 400, 950, 800)]}
    ] * n

    result = evaluate_clip(crop_windows, frame_data, FAKE_META,
                           clip_name="perfect", params={"sigma": 15.0})
    print(f"Perfect tracking:  composite={result['composite']:.1f}  "
          f"(ball_ret={result['ball_retention']:.3f}, "
          f"smooth={result['smoothness']:.3f}, "
          f"center={result['centering']:.3f}, "
          f"coverage={result['coverage']:.3f})")
    assert result["composite"] > 90, f"Expected >90, got {result['composite']}"
    return result


def test_ball_lost():
    """Ball always outside crop → ball_retention should be 0."""
    n = 250
    crop_windows = [{"crop_x1": 0, "crop_y1": 0,
                     "crop_w": CROP_W, "crop_h": 1080}] * n
    frame_data = [
        {"ball": (1800.0, 540.0),
         "players": [(100, 400, 200, 800)]}
    ] * n

    result = evaluate_clip(crop_windows, frame_data, FAKE_META,
                           clip_name="ball_lost", params={"sigma": 15.0})
    print(f"Ball lost:         composite={result['composite']:.1f}  "
          f"(ball_ret={result['ball_retention']:.3f})")
    assert result["ball_retention"] == 0.0
    return result


def test_jittery_crop():
    """Crop alternates left/right → smoothness should be ~0."""
    n = 250
    crop_windows = []
    for i in range(n):
        x = 0 if i % 2 == 0 else 1920 - CROP_W
        crop_windows.append({"crop_x1": x, "crop_y1": 0,
                             "crop_w": CROP_W, "crop_h": 1080})
    frame_data = [{"ball": None, "players": []}] * n

    result = evaluate_clip(crop_windows, frame_data, FAKE_META,
                           clip_name="jittery", params={"sigma": 2.0})
    print(f"Jittery crop:      composite={result['composite']:.1f}  "
          f"(smooth={result['smoothness']:.3f})")
    assert result["smoothness"] < 0.1
    return result


if __name__ == "__main__":
    results = [
        test_perfect_tracking(),
        test_ball_lost(),
        test_jittery_crop(),
    ]
    print()
    print_leaderboard(results, top_n=3)
    print("All tests passed.")
