import cv2
import torch
import numpy as np
from ultralytics import YOLO
from extract_pose_features import compute_pose_features
from train_lstm import ViolenceLSTM

# Load YOLO Pose Model
pose_model = YOLO("yolo11n-pose.pt")

# Load LSTM Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViolenceLSTM().to(device)
model.load_state_dict(torch.load("violence_lstm.pth", map_location=device))
model.eval()

SEQ_LEN = 32
buffer = []

cap = cv2.VideoCapture(0)   # Use your webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose detection
    result = pose_model(frame)[0]

    if len(result.keypoints) > 0:
        k = result.keypoints.xy[0].cpu().numpy()
    else:
        k = None

    # Convert keypoints → features
    f = compute_pose_features(k)

    buffer.append(f)
    if len(buffer) > SEQ_LEN:
        buffer.pop(0)

    # Once buffer is full, make prediction
    if len(buffer) == SEQ_LEN:
        x = torch.tensor([buffer], dtype=torch.float32).to(device)
        y = model(x)
        cls = torch.argmax(y).item()

        label = "VIOLENCE" if cls == 1 else "SAFE"
        color = (0, 0, 255) if cls == 1 else (0, 255, 0)

        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3)

    cv2.imshow("Real-Time Violence Detection (YOLO Pose + LSTM)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
