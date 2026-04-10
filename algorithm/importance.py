from utils import distance, get_box_center


def compute_importance(frame_data, frame_height, alpha=0.3):
    """
    compute an importance score per frame using motion and position heuristics.

    Args:
        frame_data:   list of dicts with keys "ball" and "players"
        frame_height: pixel height of the video frame
        alpha:        EMA smoothing factor (0 = max smooth, 1 = no smooth)

    Returns:
        importance_scores: list of floats, one per frame (smoothed)
    """
    raw_scores = []

    for i, frame in enumerate(frame_data):
        prev = frame_data[i - 1] if i > 0 else None

        ball_speed = 0.0
        if prev is not None and frame["ball"] is not None and prev["ball"] is not None:
            ball_speed = distance(frame["ball"], prev["ball"])

        avg_player_speed = 0.0
        if prev is not None and frame["players"] and prev["players"]:
            curr_centers = [get_box_center(b) for b in frame["players"]]
            prev_centers = [get_box_center(b) for b in prev["players"]]

            n = min(len(curr_centers), len(prev_centers))
            if n > 0:
                total = sum(distance(curr_centers[j], prev_centers[j]) for j in range(n))
                avg_player_speed = total / n

        position_score = 0.0
        if frame["ball"] is not None:
            _, ball_y = frame["ball"]
            # normalize
            position_score = 1.0 - (ball_y / frame_height)
            position_score = max(0.0, min(1.0, position_score))

        # normalize speeds to a 0–1 range using a soft cap ---
        # *cap at 200px movement to avoid extreme outliers dominating
        SPEED_CAP = 200.0
        ball_speed_norm = min(ball_speed, SPEED_CAP) / SPEED_CAP
        player_speed_norm = min(avg_player_speed, SPEED_CAP) / SPEED_CAP

        score = (
            0.6 * ball_speed_norm +
            0.3 * player_speed_norm +
            0.1 * position_score
        )
        raw_scores.append(score)

    # exponential smoothing
    smoothed = []
    prev_smooth = raw_scores[0] if raw_scores else 0.0
    for score in raw_scores:
        s = alpha * score + (1 - alpha) * prev_smooth
        smoothed.append(s)
        prev_smooth = s

    return smoothed