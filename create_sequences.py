import cv2
import numpy as np
import os
from ultralytics import YOLO
from extract_pose_features import compute_pose_features

# 🟢 IMPORTANT: Set your input folders here
INPUT_DIR_VIO = "dataset/raw_videos/vio"
INPUT_DIR_NONVIO = "dataset/raw_videos/nonvio_aug"

# 🟢 Output folder for saved sequences
OUT_DIR = "dataset/sequences"
os.makedirs(OUT_DIR, exist_ok=True)

# 🟢 Load YOLO-Pose model
model = YOLO("yolo11n-pose.pt")  # You can switch to yolo11s-pose or yolo11m-pose

SEQ_LEN = 32  # number of frames per LSTM sequence window


def process_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO pose estimation
        result = model(frame)[0]

        if len(result.keypoints):
            keyp = result.keypoints.xy[0].cpu().numpy()
        else:
            keyp = None

        # Convert keypoints → feature vector
        f = compute_pose_features(keyp)
        frames.append(f)

    cap.release()

    # if video is too short → skip
    if len(frames) < SEQ_LEN:
        return

    # sliding window (step=4)
    for i in range(0, len(frames) - SEQ_LEN, 4):
        seq = np.stack(frames[i:i + SEQ_LEN], axis=0)

        base = os.path.basename(video_path).split(".")[0]
        out_path = f"{OUT_DIR}/{base}_{i}.npz"

        np.savez_compressed(out_path, x=seq, y=label)

        print(f"✔ Saved: {out_path}")


def process_folder(folder_path, label):
    for f in os.listdir(folder_path):
        if f.lower().endswith(".mp4"):
            full_path = os.path.join(folder_path, f)
            print(f"\n📌 Processing: {full_path}")
            process_video(full_path, label)


print("=== Processing VIOLENCE videos ===")
process_folder(INPUT_DIR_VIO, 1)

print("\n=== Processing NON-VIOLENCE videos ===")
process_folder(INPUT_DIR_NONVIO, 0)

print("\n🎉 ALL SEQUENCES CREATED SUCCESSFULLY!")
