"""
file runs YOLOv8 on a video and extracts ball and player positions per frame.
"""

from ultralytics import YOLO

# YOLO class IDs (COCO):
#   0  = person
#   32 = sports ball
PERSON_CLASS = 0
BALL_CLASS = 32

_model = None

def _load_model(weights="yolov8n.pt"):
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model


def detect_frames(video_path, weights="yolov8n.pt", conf=0.3):
    """
    runs YOLO on every frame
    returns frame_data: list of per-frame dicts
    """
    import cv2

    model = _load_model(weights)
    cap = cv2.VideoCapture(video_path)
    frame_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)[0]

        ball = None
        players = []

        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if cls == BALL_CLASS:
                ball = ((x1 + x2) / 2, (y1 + y2) / 2)
            elif cls == PERSON_CLASS:
                players.append((x1, y1, x2, y2))

        frame_data.append({"ball": ball, "players": players})

    cap.release()
    return frame_data