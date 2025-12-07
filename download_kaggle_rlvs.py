import os
import zipfile

# Make sure kaggle is installed: pip install kaggle
# And kaggle.json is placed in: C:/Users/<user>/.kaggle/ (Windows)
# or ~/.kaggle/ (Linux/Mac)

DATASET = "mohamedmustafa/real-life-violence-situations-dataset"
OUT_DIR = "dataset/raw_videos/rlvs"

os.makedirs(OUT_DIR, exist_ok=True)

print("📥 Downloading Kaggle dataset...")
os.system(f"kaggle datasets download -d {DATASET} -p {OUT_DIR}")

# Unzip downloaded files
for file in os.listdir(OUT_DIR):
    if file.endswith(".zip"):
        zip_path = os.path.join(OUT_DIR, file)
        print("📦 Unzipping", zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(OUT_DIR)
        os.remove(zip_path)

print("🎉 Kaggle RLVS dataset downloaded and extracted successfully!")
