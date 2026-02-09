"""
Script to download optional large FITS test files for SunXCorr.
Run this script to fetch test data if you want to run the full test suite.
"""
import os
from pathlib import Path
import urllib.request

# List of URLs for test FITS files (replace with your actual links)
TEST_DATA_URLS = [
    # Example:
    # "https://your-server.org/data/FSI_174_2024-10-17T014055.208.fits",
]

TARGET_DIR = Path(__file__).parent / "fits_files"
TARGET_DIR.mkdir(exist_ok=True)

def download_file(url, target_dir):
    filename = url.split("/")[-1]
    dest = target_dir / filename
    if dest.exists():
        print(f"{filename} already exists, skipping.")
        return
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")

if __name__ == "__main__":
    if not TEST_DATA_URLS:
        print("No test data URLs provided. Please edit download_test_data.py and add your links.")
    for url in TEST_DATA_URLS:
        download_file(url, TARGET_DIR)
