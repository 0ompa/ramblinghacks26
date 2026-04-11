import cv2
from inference import get_model

rf = Roboflow(api_key="R6tRPGl4GmNaGmoDBkWZ")

# pick any basketball model from universe
model = rf.workspace().project("basketball-xil7x/1n").version(1).model

cap = cv2.VideoCapture("your_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.predict(frame, confidence=40, overlap=30).json()

    for pred in result["predictions"]:
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        cv2.rectangle(frame, (x-w//2, y-h//2), (x+w//2, y+h//2), (0,255,0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()