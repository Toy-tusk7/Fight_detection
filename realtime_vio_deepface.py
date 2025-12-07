import os
import time
import csv

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

from extract_pose_features import compute_pose_features
from train_lstm import ViolenceLSTM

# ==============================
# CONFIG
# ==============================
SEQ_LEN = 32
VIOLENCE_THRESHOLD = 0.70
PRED_HISTORY_SIZE = 10
VIOLENCE_HOLD_TIME = 10  # seconds of continuous violence before taking screenshot

# Folders
FACE_VIO_DIR = "face_vio"
FACE_VIO_FACES_DIR = "face_vio_faces"
LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "vio_log.csv")

# DeepFace DB
DB_PATH = "known_faces"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"  # fast & simple

os.makedirs(FACE_VIO_DIR, exist_ok=True)
os.makedirs(FACE_VIO_FACES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================
# STATE
# ==============================
buffer = []
pred_history = []
violence_active_start = None  # time when violence started

# ==============================
# LOAD MODELS
# ==============================
print("[SYSTEM] Loading YOLO pose model...")
pose_model = YOLO("yolo11n-pose.pt")
print("[SYSTEM] YOLO loaded.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[SYSTEM] Using device:", device)

print("[SYSTEM] Loading LSTM model...")
lstm_model = ViolenceLSTM().to(device)
lstm_model.load_state_dict(torch.load("violence_lstm.pth", map_location=device))
lstm_model.eval()
print("[SYSTEM] LSTM loaded.")

print("[SYSTEM] Warming up DeepFace model (first time may take a few seconds)...")
# Dummy call to load model into memory
_ = DeepFace.find(
    img_path=np.zeros((100, 100, 3), dtype=np.uint8),
    db_path=DB_PATH,
    model_name=MODEL_NAME,
    detector_backend=DETECTOR_BACKEND,
    enforce_detection=False,
    silent=True
)
print("[SYSTEM] DeepFace ready.")

# ==============================
# UTILS
# ==============================
def draw_skeleton(frame, keypoints):
    if keypoints is None:
        return frame
    for (x, y) in keypoints:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
    return frame


def log_event(name, screenshot_path, face_path=None):
    """Append a row to vio_log.csv"""
    header_needed = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["timestamp", "person_name", "screenshot_path", "face_path"])
        writer.writerow([time.time(), name, screenshot_path, face_path or ""])
    print(f"[LOG] {name} | screenshot={screenshot_path} | face={face_path}")


def recognize_faces_from_frame(frame, screenshot_path):
    """
    Use DeepFace to:
      1) Detect faces in frame
      2) For each face, identify from DB_PATH
      3) Save cropped face images
      4) Log results
    """
    # 1) Extract face regions
    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
    except Exception as e:
        print(f"[DEEPFACE] extract_faces error: {e}")
        return

    if len(faces) == 0:
        print("[DEEPFACE] No faces found in violent frame.")
        log_event("NoFace", screenshot_path, None)
        return

    print(f"[DEEPFACE] Found {len(faces)} face(s) in screenshot.")

    for idx, face_info in enumerate(faces):
        facial_area = face_info.get("facial_area", None)
        if not facial_area:
            continue

        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

        # Ensure bounding box is within frame bounds
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)

        face_crop = frame[y1:y2, x1:x2]

        # 2) Recognize face using DeepFace.find on the crop
        try:
            df_list = DeepFace.find(
                img_path=face_crop,
                db_path=DB_PATH,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True
            )
        except Exception as e:
            print(f"[DEEPFACE] find() error on face {idx}: {e}")
            continue

        name = "Unknown"

        if isinstance(df_list, list) and len(df_list) > 0 and not df_list[0].empty:
            row = df_list[0].iloc[0]
            identity_path = row["identity"]
            # Name can be folder name or file name, depending on your known_faces structure
            # Here assume: known_faces/<person_name>/image.jpg
            person_dir = os.path.dirname(identity_path)
            name = os.path.basename(person_dir) or os.path.splitext(os.path.basename(identity_path))[0]

        # 3) Save face crop
        face_filename = os.path.join(
            FACE_VIO_FACES_DIR,
            f"{name}_{int(time.time())}_{idx}.jpg"
        )
        cv2.imwrite(face_filename, face_crop)
        print(f"[DEEPFACE] Face {idx}: {name} -> {face_filename}")

        # 4) Log event
        log_event(name, screenshot_path, face_filename)


# ==============================
# CAMERA LOOP
# ==============================
print("[SYSTEM] Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 320)
cap.set(4, 240)

if not cap.isOpened():
    print("[ERROR] Camera failed to open.")
    raise SystemExit

print("[SYSTEM] Camera ready. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    label = "SAFE"
    color = (0, 255, 0)
    violence_prob = 0.0

    # ---------- YOLO pose ----------
    result = pose_model(frame, imgsz=256, verbose=False)[0]
    kp = result.keypoints.xy[0].cpu().numpy() if len(result.keypoints) > 0 else None
    draw_skeleton(frame, kp)

    # ---------- Pose -> features ----------
    features = compute_pose_features(kp)
    buffer.append(features)
    if len(buffer) > SEQ_LEN:
        buffer.pop(0)

    # ---------- LSTM inference ----------
    if len(buffer) == SEQ_LEN:
        np_seq = np.array(buffer, dtype=np.float32)
        x = torch.from_numpy(np_seq).unsqueeze(0).to(device)

        with torch.no_grad():
            out = lstm_model(x)

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
            # If not violence, reset timer
            violence_active_start = None

    # ---------- Violence timer logic ----------
    now = time.time()

    if label == "VIOLENCE":
        if violence_active_start is None:
            violence_active_start = now
        else:
            elapsed = now - violence_active_start
            cv2.putText(frame, f"VIOLENCE TIMER: {int(elapsed)}s",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if elapsed >= VIOLENCE_HOLD_TIME:
                # Take screenshot
                screenshot_path = os.path.join(
                    FACE_VIO_DIR,
                    f"vio_{int(time.time())}.jpg"
                )
                cv2.imwrite(screenshot_path, frame)
                print(f"[ALERT] Violence > {VIOLENCE_HOLD_TIME}s, saved: {screenshot_path}")

                # Run DeepFace recognition on this frame
                recognize_faces_from_frame(frame, screenshot_path)

                # Reset timer so it doesn't trigger every frame
                violence_active_start = None
    else:
        violence_active_start = None

    # ---------- UI ----------
    cv2.putText(frame, f"STATUS: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(frame, f"Violence Prob: {violence_prob:.2f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Violence + DeepFace", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Stopped.")
