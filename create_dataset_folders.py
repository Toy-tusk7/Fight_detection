import os

# Base directory for dataset
BASE = "dataset"

# Folder structure
folders = [
    f"{BASE}",
    f"{BASE}/raw_videos",
    f"{BASE}/raw_videos/rwf",
    f"{BASE}/raw_videos/rlvs",
    f"{BASE}/raw_videos/hockey",
    f"{BASE}/raw_videos/movies_fight",
    f"{BASE}/raw_videos/ucf101",
    f"{BASE}/raw_videos/smoking",
    f"{BASE}/raw_videos/custom",

    f"{BASE}/annotations",

    f"{BASE}/clips",
    f"{BASE}/clips/normal",
    f"{BASE}/clips/punching",
    f"{BASE}/clips/slapping",
    f"{BASE}/clips/kicking",
    f"{BASE}/clips/smoking",
    f"{BASE}/clips/aggressive",

    f"{BASE}/sequences",
]

# Create directories
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

print("\n🎉 All dataset folders created successfully!")

