import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from extract_pose_features import compute_pose_features
from train_lstm import ViolenceLSTM
import os

# ===========================
# CONFIG
# ===========================
SEQ_LEN = 32
VIOLENCE_THRESHOLD = 0.70
PRED_HISTORY_SIZE = 10
VIOLENCE_HOLD_TIME = 10      # seconds
face_vio_folder = "face_vio"

os.makedirs(face_vio_folder, exist_ok=True)

buffer = []
pred_history = []
violence_active_start = None


# ===========================
# LOAD MODELS
# ===========================
print("[SYSTEM] Loading YOLO...")
pose_model = YOLO("yolo11n-pose.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViolenceLSTM().to(device)
model.load_state_dict(torch.load("violence_lstm.pth", map_location=device))
model.eval()

print("[SYSTEM] Models loaded.")


# ===========================
# CAMERA SETUP
# ===========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 320)
cap.set(4, 240)

if not cap.isOpened():
    print("[ERROR] Camera failed.")
    exit()


# ===========================
# DRAW SKELETON
# ===========================
def draw_skeleton(frame, keypoints):
    if keypoints is None:
        return frame
    for (x, y) in keypoints:
        cv2.circle(frame, (int(x), int(y)), 3, (0,255,255), -1)
    return frame


# ===========================
# MAIN LOOP
# ===========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------------------
    # YOLO POSE
    # -----------------------------------------
    result = pose_model(frame, imgsz=256, verbose=False)[0]

    if len(result.keypoints) > 0:
        kp = result.keypoints.xy[0].cpu().numpy()
    else:
        kp = None

    draw_skeleton(frame, kp)

    # -----------------------------------------
    # FEATURES
    # -----------------------------------------
    features = compute_pose_features(kp)
    buffer.append(features)
    if len(buffer) > SEQ_LEN:
        buffer.pop(0)

    label = "SAFE"
    color = (0,255,0)
    violence_prob = 0.0

    # -----------------------------------------
    # LSTM PREDICTION
    # -----------------------------------------
    if len(buffer) == SEQ_LEN:

        np_seq = np.array(buffer, dtype=np.float32)
        x = torch.from_numpy(np_seq).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)

        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
        violence_prob = float(prob[1])

        # Smooth predictions
        pred_history.append(violence_prob)
        if len(pred_history) > PRED_HISTORY_SIZE:
            pred_history.pop(0)

        avg_prob = sum(pred_history) / len(pred_history)

        if avg_prob > VIOLENCE_THRESHOLD:
            label = "VIOLENCE"
            color = (0,0,255)

        else:
            label = "SAFE"
            color = (0,255,0)
            violence_active_start = None  # reset timer when safe

    # -----------------------------------------
    # VIOLENCE TIMER LOGIC
    # -----------------------------------------
    current_time = time.time()

    if label == "VIOLENCE":
        if violence_active_start is None:
            violence_active_start = current_time
        else:
            elapsed = current_time - violence_active_start

            # 10 SEC REACHED → TAKE SCREENSHOT
            if elapsed >= VIOLENCE_HOLD_TIME:
                filename = os.path.join(face_vio_folder, f"vio_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[ALERT] Violence for 10 seconds! Saved: {filename}")
                violence_active_start = None  # reset
    else:
        violence_active_start = None

    # -----------------------------------------
    # DISPLAY
    # -----------------------------------------
    cv2.putText(frame, f"STATUS: {label}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, f"Prob: {violence_prob:.2f}", (10,75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if violence_active_start:
        cv2.putText(frame, f"Recording.. {int(current_time - violence_active_start)}s",
                    (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Realtime Violence Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Stopped.")
