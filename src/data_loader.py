# data_loader.py
import os
import glob
import zipfile
from pathlib import Path

def mount_drive():
    """
    Mount Google Drive (Colab only).
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        print("Google Drive mount: Not running on Colab, skipping mount.")

def download_kaggle_dataset(dataset, dest_path="data"):
    """
    Download and unzip a Kaggle dataset using Kaggle API.
    'dataset' should be in form 'username/dataset-name'.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        dest_path = Path(dest_path)
        dest_path.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset, path=str(dest_path), unzip=True)
        print(f"Downloaded and extracted dataset to {dest_path.resolve()}")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Ensure Kaggle API is configured and running in an environment with internet access.")

def unzip_dataset(zip_path, extract_to):
    """
    Unzip a zip file to the specified directory.
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"Unzipped {zip_path} to {extract_to}")

def count_images(folder, extensions=['.jpg', '.jpeg', '.png']):
    """
    Count image files in a folder (recursively) with given extensions.
    """
    count = 0
    folder = Path(folder)
    for ext in extensions:
        count += len(list(folder.rglob(f'*{ext}')))
    return count

def list_images(folder, extensions=['.jpg', '.jpeg', '.png']):
    """
    List image files in a folder (recursively) with given extensions.
    """
    images = []
    folder = Path(folder)
    for ext in extensions:
        images.extend(list(folder.rglob(f'*{ext}')))
    return sorted(images)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare data")
    parser.add_argument("--mount", action="store_true", help="Mount Google Drive (Colab)")
    parser.add_argument("--dataset", type=str, default="pkdarabi/cardetection", help="Kaggle dataset (username/dataset)")
    parser.add_argument("--dest", type=str, default="data", help="Destination directory for dataset")
    args = parser.parse_args()

    if args.mount:
        mount_drive()

    # Download and prepare dataset
    download_kaggle_dataset(args.dataset, args.dest)

    # Example: List images in train/valid/test folders if they exist
    car_dir = Path(args.dest) / "cardetection"
    if not car_dir.exists():
        car_dir = Path(args.dest)  # maybe dataset was extracted to dest directly
    for split in ['train', 'valid', 'test']:
        img_folder = car_dir / split / "images"
        if img_folder.exists():
            num_imgs = count_images(img_folder)
            print(f"{split.capitalize()} images in '{img_folder}': {num_imgs}")
        else:
            print(f"Folder {img_folder} does not exist.")
