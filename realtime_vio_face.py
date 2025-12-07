import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from extract_pose_features import compute_pose_features
from train_lstm import ViolenceLSTM
from face_recog_helper import load_known_faces, recognize_faces_in_frame
import os
import csv

# ===============================
# CONFIG
# ===============================
SEQ_LEN = 32
VIOLENCE_THRESHOLD = 0.70
PRED_HISTORY_SIZE = 10
VIOLENCE_HOLD_TIME = 10

face_vio_folder = "face_vio"
face_crop_folder = "face_vio_faces"
log_file = "logs/vio_log.csv"

os.makedirs(face_vio_folder, exist_ok=True)
os.makedirs(face_crop_folder, exist_ok=True)
os.makedirs("logs", exist_ok=True)

buffer = []
pred_history = []
violence_active_start = None


# ===============================
# LOAD MODELS
# ===============================
print("[SYSTEM] Loading YOLO...")
pose_model = YOLO("yolo11n-pose.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViolenceLSTM().to(device)
model.load_state_dict(torch.load("violence_lstm.pth", map_location=device))
model.eval()

print("[SYSTEM] YOLO + LSTM Ready.")

known_encodings, known_names = load_known_faces()


# ===============================
# CAMERA SETUP
# ===============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 320)
cap.set(4, 240)

if not cap.isOpened():
    print("[ERROR] Camera failed.")
    exit()


# ===============================
# UTILITIES
# ===============================
def draw_skeleton(frame, keypoints):
    if keypoints is None:
        return frame
    for (x, y) in keypoints:
        cv2.circle(frame, (int(x), int(y)), 3, (0,255,255), -1)
    return frame


def save_log(name, img_path):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), name, img_path])
    print(f"[LOG] Saved: {name} -> {img_path}")


# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------- YOLO -----------------------------
    result = pose_model(frame, imgsz=256, verbose=False)[0]
    kp = result.keypoints.xy[0].cpu().numpy() if len(result.keypoints) > 0 else None
    draw_skeleton(frame, kp)

    # ----------------------------- FEATURES -----------------------------
    features = compute_pose_features(kp)
    buffer.append(features)
    if len(buffer) > SEQ_LEN:
        buffer.pop(0)

    label = "SAFE"
    color = (0,255,0)
    violence_prob = 0.0

    # ----------------------------- LSTM -----------------------------
    if len(buffer) == SEQ_LEN:
        np_seq = np.array(buffer, dtype=np.float32)
        x = torch.from_numpy(np_seq).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)

        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
        violence_prob = float(prob[1])

        pred_history.append(violence_prob)
        if len(pred_history) > PRED_HISTORY_SIZE:
            pred_history.pop(0)

        avg_prob = sum(pred_history) / len(pred_history)

        if avg_prob > VIOLENCE_THRESHOLD:
            label = "VIOLENCE"
            color = (0, 0, 255)
        else:
            violence_active_start = None

    # ----------------------------- TIMER -----------------------------
    current_time = time.time()

    if label == "VIOLENCE":
        if violence_active_start is None:
            violence_active_start = current_time
        else:
            elapsed = current_time - violence_active_start

            if elapsed >= VIOLENCE_HOLD_TIME:
                filename = os.path.join(face_vio_folder, f"vio_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)

                print(f"[ALERT] Violence >10s: saved screenshot {filename}")

                # ------------------ FACE RECOGNITION ------------------
                faces = recognize_faces_in_frame(frame, known_encodings, known_names)

                for i, (name, (top, right, bottom, left), face_img) in enumerate(faces):
                    face_path = os.path.join(face_crop_folder, f"{name}_{int(time.time())}_{i}.jpg")
                    cv2.imwrite(face_path, face_img)

                    print(f"[FACE] {name} detected → {face_path}")

                    save_log(name, face_path)

                violence_active_start = None

    # ----------------------------- DISPLAY -----------------------------
    cv2.putText(frame, f"STATUS: {label}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Realtime Violence + Face Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("[SYSTEM] Shutdown.")
