import os

OUT_DIR = "dataset/raw_videos/rwf"
os.makedirs(OUT_DIR, exist_ok=True)

print("📥 Cloning GitHub dataset...")

# Clone directly into your dataset folder
os.system(f"git clone https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git {OUT_DIR}")

print("🎉 RWF (AIRTLab) dataset cloned successfully!")
print("📂 Files placed in:", OUT_DIR)
