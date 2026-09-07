import cv2
from vigileye.cnn_inference import CNNDrowsinessPredictor

cnn = CNNDrowsinessPredictor("models/drowsiness_model.pth")

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("CNN live test started")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Use the center region as a temporary face crop
    h, w = frame.shape[:2]

    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    y1 = int(h * 0.15)
    y2 = int(h * 0.85)

    face_crop = frame[y1:y2, x1:x2]

    label, confidence, drowsy_prob = cnn.predict(face_crop)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.putText(
        frame,
        f"Prediction: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Drowsy probability: {drowsy_prob:.2f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("VigilEye CNN Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()