"""Bit 1 of the URL-classifier pipeline: pull the raw labeled URL dataset from Kaggle.

Requires a free Kaggle account and API token. One-time setup:
  1. https://www.kaggle.com/settings -> "Create New Token" -> downloads kaggle.json
  2. Place it at ~/.kaggle/kaggle.json (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)

No key is ever passed on the command line or read by anything other than the
official `kaggle` package, which reads that file (or KAGGLE_USERNAME/KAGGLE_KEY
env vars) itself.

Usage:
    python -m ml.training.download_dataset
"""

import os
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = os.environ.get("KAGGLE_DATASET_SLUG", "sid321axn/malicious-urls-dataset")
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _credentials_available() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def main() -> None:
    if not _credentials_available():
        print(
            "No Kaggle API credentials found.\n\n"
            "Setup (one-time):\n"
            "  1. Create a free account at https://www.kaggle.com if you don't have one.\n"
            "  2. Go to https://www.kaggle.com/settings -> API -> \"Create New Token\".\n"
            "     This downloads a kaggle.json file.\n"
            f"  3. Move it to: {Path.home() / '.kaggle' / 'kaggle.json'}\n\n"
            "Then re-run: python -m ml.training.download_dataset"
        )
        sys.exit(1)

    # imported lazily so the credential check above can run first and print a
    # clean message instead of the library's own import-time error.
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET_SLUG} to {RAW_DIR} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(RAW_DIR), unzip=False)

    zip_files = list(RAW_DIR.glob("*.zip"))
    for zip_path in zip_files:
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV found in {RAW_DIR} after download - check the dataset slug/contents.")
        sys.exit(1)

    print(f"Done. Files in {RAW_DIR}:")
    for f in csv_files:
        print(f"  {f.name} ({f.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
