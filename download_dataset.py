"""
Download dataset Grape Disease Original dari Kaggle.

Cara penggunaan:
  1. Install kaggle API: pip install kaggle
  2. Upload kaggle.json ke ~/.kaggle/kaggle.json
  3. Jalankan: python download_dataset.py

Atau download manual dari:
  https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original

Setelah download, extract ke folder data/train/ dengan struktur:
  data/train/
    ├── Black_Rot/
    ├── ESCA/
    ├── Leaf_Blight/
    └── Healthy/
"""

import subprocess
import zipfile
from pathlib import Path

DATASET = 'rm1000/grape-disease-dataset-original'
DATA_DIR = Path('data')


def download_via_kaggle():
    print(f"Downloading {DATASET}...")
    result = subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', DATASET],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error:", result.stderr)
        return False
    print("Download complete.")
    return True


def extract_dataset(zip_name):
    print(f"Extracting {zip_name}...")
    with zipfile.ZipFile(zip_name, 'r') as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {DATA_DIR / 'train'}")


if __name__ == '__main__':
    if not (DATA_DIR / 'train').exists():
        (DATA_DIR / 'train').mkdir(parents=True, exist_ok=True)
    zip_path = Path(f'{DATASET.split("/")[-1]}.zip')
    if not zip_path.exists():
        if not download_via_kaggle():
            print("\nGagal download via Kaggle API.")
            print("Download manual dari:")
            print("  https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original")
            print(f"Kemudian extract ke folder {DATA_DIR / 'train'}/")
            exit(1)
    extract_dataset(zip_path)
    print("\nDataset siap! Struktur folder:")
    for cls_dir in sorted((DATA_DIR / 'train').iterdir()):
        if cls_dir.is_dir():
            n_files = len(list(cls_dir.glob('*')))
            print(f"  {cls_dir.name}/ ({n_files} files)")
