import cv2
import torch
import numpy as np
from ultralytics import YOLO
from extract_pose_features import compute_pose_features
from train_lstm import ViolenceLSTM

# =========================
# SETTINGS
# =========================
SEQ_LEN = 32
VIOLENCE_THRESHOLD = 0.70
PRED_HISTORY_SIZE = 10

buffer = []
pred_history = []

print("[SYSTEM] Loading models...")

# Load pose model
pose_model = YOLO("yolo11n-pose.pt")

# Load LSTM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViolenceLSTM().to(device)
model.load_state_dict(torch.load("violence_lstm.pth", map_location=device))
model.eval()

print("[SYSTEM] Models ready.")


# =========================
# POSE DRAWING
# =========================
def draw_skeleton(frame, keypoints):
    if keypoints is None:
        return frame
    for (x, y) in keypoints:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
    return frame


# =========================
# READ FROM VIDEO FILE
# =========================
VIDEO_PATH = "fi136.mp4"  # <<<< CHANGE THIS

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Failed to open video file!")
    exit()

print(f"[SYSTEM] Processing video: {VIDEO_PATH}")

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video.")
        break

    # --------------------------------------------------------
    # YOLO Pose
    # --------------------------------------------------------
    results = pose_model(frame, imgsz=256, verbose=False)[0]

    if len(results.keypoints) > 0:
        kp = results.keypoints.xy[0].cpu().numpy()
    else:
        kp = None

    draw_skeleton(frame, kp)

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------
    features = compute_pose_features(kp)
    buffer.append(features)

    if len(buffer) > SEQ_LEN:
        buffer.pop(0)

    label = "SAFE"
    color = (0, 255, 0)
    violence_prob = 0.0

    # --------------------------------------------------------
    # LSTM INFERENCE
    # --------------------------------------------------------
    if len(buffer) == SEQ_LEN:

        np_seq = np.array(buffer, dtype=np.float32)
        x = torch.from_numpy(np_seq).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)

        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
        violence_prob = float(prob[1])

        # Smoothing
        pred_history.append(violence_prob)
        if len(pred_history) > PRED_HISTORY_SIZE:
            pred_history.pop(0)

        avg_prob = sum(pred_history) / len(pred_history)

        if avg_prob > VIOLENCE_THRESHOLD:
            label = "VIOLENCE"
            color = (0, 0, 255)
        else:
            label = "SAFE"
            color = (0, 255, 0)

    # --------------------------------------------------------
    # DRAW LABELS
    # --------------------------------------------------------
    cv2.putText(frame, f"STATUS: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, f"Violence Prob: {violence_prob:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Video Violence Detection", frame)

    if cv2.waitKey(25) == ord('q'):  # press Q to quit
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Done.")
