import cv2
import os
import numpy as np
import random

INPUT_DIR = "dataset/raw_videos/non-vio"
OUTPUT_DIR = "dataset/raw_videos/nonvio_aug"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_gaussian_noise(frame):
    noise = np.random.normal(0, 15, frame.shape).astype(np.int16)
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def adjust_brightness(frame):
    factor = random.uniform(0.6, 1.4)
    return cv2.convertScaleAbs(frame, alpha=factor, beta=0)

def motion_blur(frame):
    size = random.choice([3,5,7])
    kernel = np.zeros((size, size))
    kernel[size//2] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(frame, -1, kernel)

def speed_change(frames, factor):
    idx = np.arange(0, len(frames), factor).astype(int)
    idx = np.clip(idx, 0, len(frames)-1)
    return [frames[i] for i in idx]

def horizontal_flip(frame):
    return cv2.flip(frame, 1)

def rotate_small(frame):
    angle = random.uniform(-5, 5)
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(frame, M, (w, h))

def crop(frame):
    h, w = frame.shape[:2]
    x = random.randint(0, w//20)
    y = random.randint(0, h//20)
    cropped = frame[y:h-y, x:w-x]
    return cv2.resize(cropped, (w, h))

AUG_FUNCS = [
    add_gaussian_noise,
    adjust_brightness,
    motion_blur,
    horizontal_flip,
    rotate_small,
    crop
]

def augment_video(video_path, out_name):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    # Choose 1–3 random augmentations
    funcs = random.sample(AUG_FUNCS, random.randint(1,3))

    aug_frames = frames.copy()
    for f in funcs:
        aug_frames = [f(frm) for frm in aug_frames]

    # Speed augmentation
    if random.random() < 0.5:
        factor = random.choice([0.7, 0.8, 1.2, 1.3])
        aug_frames = speed_change(aug_frames, factor)

    # Save video
    h, w = aug_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{OUTPUT_DIR}/{out_name}", fourcc, 20, (w,h))

    for frm in aug_frames:
        out.write(frm)
    out.release()

    print("✔ Augmented:", out_name)


# Generate 500+ augmented videos
count = 0
for file in os.listdir(INPUT_DIR):
    if file.endswith(".mp4"):
        for i in range(5):   # 5 augmentations per video
            augment_video(os.path.join(INPUT_DIR, file), f"aug_{i}_{file}")
            count += 1

print(f"\n🎉 Generated {count} augmented non-violent videos!")
