"""
Step 1: Setup & Dataset Download

"""

import subprocess
import sys

# Install dependencies (run this cell first in Colab)
def install_dependencies():
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "torch-geometric",
        "rdflib",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "networkx",
        "pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv "
        "-f https://data.pyg.org/whl/torch-2.3.0+cu121.html",
    ]
    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install"] + pkg.split(), check=True)
    print("All packages installed.")


# Download AIFB dataset 
import os
import urllib.request
import zipfile

DATA_DIR = "./data"
DATASET_URL = "https://data.dgl.ai/dataset/rdf/aifb-hetero.zip"
ZIP_PATH = os.path.join(DATA_DIR, "aifb-hetero.zip")


def download_aifb():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print("Downloading AIFB dataset...")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print(f" Downloaded to {ZIP_PATH}")
    else:
        print("Dataset already downloaded.")

    # NEW
    extract_dir = DATA_DIR
    # NEW
    print("Dataset already extracted.")

    return extract_dir


if __name__ == "__main__":
    
    dataset_dir = download_aifb()
    print(f"\nDataset directory: {dataset_dir}")
    print("Files:", os.listdir(dataset_dir))