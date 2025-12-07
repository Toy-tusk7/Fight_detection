import cv2
import torch
import numpy as np
from ultralytics import YOLO
from extract_pose_features import compute_pose_features
from train_lstm import ViolenceLSTM


# ============================
# Load YOLO Pose Model
# ============================
pose_model = YOLO("yolo11n-pose.pt")

# ============================
# Load LSTM Model
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = ViolenceLSTM().to(device)
model.load_state_dict(torch.load("violence_lstm.pth", map_location=device))
model.eval()


# ============================
# Tracking Buffer
# ============================
SEQ_LEN = 32
buffer = []

# Colors
SAFE_COLOR = (0, 255, 0)
VIO_COLOR = (0, 0, 255)
SKELETON_COLOR = (255, 255, 0)

# Body keypoint connections (COCO format)
skeleton_pairs = [
    (5, 7), (7, 9),        # Left arm
    (6, 8), (8, 10),       # Right arm
    (11, 13), (13, 15),    # Left leg
    (12, 14), (14, 16),    # Right leg
    (5, 6),                # Shoulders
    (11, 12),              # Hips
    (5, 11), (6, 12)       # Torso
]


# ============================
# Draw Skeleton Function
# ============================
def draw_skeleton(frame, keypoints):
    if keypoints is None:
        return frame

    # Draw points
    for (x, y) in keypoints:
        cv2.circle(frame, (int(x), int(y)), 4, SKELETON_COLOR, -1)

    # Draw limbs
    for a, b in skeleton_pairs:
        if a < len(keypoints) and b < len(keypoints):
            x1, y1 = keypoints[a]
            x2, y2 = keypoints[b]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                     SKELETON_COLOR, 2)

    return frame


# ============================
# Webcam
# ============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Pose detection
    result = pose_model(frame)[0]

    if len(result.keypoints) > 0:
        keypoints = result.keypoints.xy[0].cpu().numpy()
    else:
        keypoints = None

    # Draw pose
    draw_skeleton(frame, keypoints)

    # 2) Extract features and add to buffer
    features = compute_pose_features(keypoints)
    buffer.append(features)

    if len(buffer) > SEQ_LEN:
        buffer.pop(0)

    # 3) Predict once enough frames collected
    if len(buffer) == SEQ_LEN:
        x = torch.tensor([buffer], dtype=torch.float32).to(device)
        output = model(x)
        cls = torch.argmax(output).item()

        if cls == 1:
            label_text = "VIOLENCE DETECTED"
            color = VIO_COLOR
        else:
            label_text = "SAFE"
            color = SAFE_COLOR

        # Display result
        cv2.putText(frame, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Show GUI
    cv2.imshow("Violence Detection (GUI + 2D Pose)", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
