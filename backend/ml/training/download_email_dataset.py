"""Bit 0 of the email-classifier pipeline: pull the raw labeled email dataset
from Kaggle. Mirrors ml/training/download_dataset.py's approach (same
credential handling, same "no key ever touches the command line" guarantee).

This downloads ALL of the dataset's per-source CSVs (Enron, CEAS_08, Ling,
Nazario, Nigerian_Fraud, SpamAssasin) plus its pre-merged phishing_email.csv.
build_email_features.py only uses the per-source files, deliberately -- the
merged file collapses provenance and drops the sender/receiver/date columns,
which is exactly what let 83.5% generic spam get relabeled "phishing" in the
merge (see README_EMAIL.md section 1). Keeping the raw per-source files lets
that be audited instead of trusted.

Usage:
    python -m ml.training.download_email_dataset
"""

import sys
from pathlib import Path

DATASET_SLUG = "naserabdullahalam/phishing-email-dataset"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_email"


def _credentials_available() -> bool:
    import os

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def main() -> None:
    if not _credentials_available():
        print(
            "No Kaggle API credentials found.\n\n"
            "Setup (one-time, same as ml/training/download_dataset.py):\n"
            "  1. https://www.kaggle.com/settings -> API -> \"Create New Token\"\n"
            f"  2. Move the downloaded kaggle.json to: {Path.home() / '.kaggle' / 'kaggle.json'}\n\n"
            "Then re-run: python -m ml.training.download_email_dataset"
        )
        sys.exit(1)

    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET_SLUG} to {RAW_DIR} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(RAW_DIR), unzip=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV found in {RAW_DIR} after download - check the dataset slug/contents.")
        sys.exit(1)

    print(f"Done. Files in {RAW_DIR}:")
    for f in csv_files:
        print(f"  {f.name} ({f.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
