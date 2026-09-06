"""Bit 1 of the email-classifier pipeline (build order phase 2's NLP component):
turn the raw Kaggle "phishing-email-dataset" sources into a clean, honestly-scoped
train/val/test set for TF-IDF and (later) BERT training.

Why this doesn't just use the dataset's own "phishing_email.csv"/label column:
inspecting actual email content (not just labels) found the marketed "phishing"
class (42,891 rows) is 83.5% generic commercial spam (pharma ads, replica
watches, weight-loss pills from CEAS_08 and an injected-spam portion of
"Enron.csv" that isn't real Enron correspondence at all), not phishing. Per an
external review of this project's URL pipeline: "Never treat spam labels as
phishing labels without an explicit, supported mapping." Verified by reading
samples from every source before deciding scope -- see README_EMAIL.md section
1 for the full audit.

Scope actually used here (user-confirmed choice, "strict phishing only"):
  - phishing:   Nazario (1,565) + Nigerian_Fraud (3,332) -- both individually
    confirmed by content to be genuinely deceptive fraud/phishing, not generic
    spam. Nigerian_Fraud is advance-fee ("419") scam, a phishing subtype.
  - legitimate: the "ham" (label=0) rows of Enron, CEAS_08, SpamAssassin, and
    Ling -- confirmed genuine correspondence (corporate, mailing-list,
    academic registers) by reading samples, not just trusting the label.
  - excluded entirely: CEAS_08/SpamAssassin/Ling's "spam" (label=1) rows, and
    Enron's injected-spam label=1 rows -- generic spam, not phishing, and
    deliberately not included as either class (they're off-topic for this
    binary task, not "hard negatives").

This makes the resulting dataset small and heavily imbalanced (~11% positive)
compared to the marketed 82k/52%-balanced version -- a real cost of scoping
honestly, not a mistake. See README_EMAIL.md for the tradeoff discussion.

Usage:
    python -m ml.training.build_email_features
"""

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline.preprocessing import preprocess_email_text, strip_email_headers  # noqa: E402

RAW_DIR = BACKEND_ROOT / "ml" / "data" / "raw_email"
PROCESSED_DIR = BACKEND_ROOT / "ml" / "data" / "processed_email"

PHISHING_SOURCES = ["Nazario", "Nigerian_Fraud"]
LEGITIMATE_HAM_SOURCES = ["Enron", "CEAS_08", "SpamAssasin", "Ling"]

RANDOM_STATE = 42

# Some Nazario rows are raw mbox artifacts, not captured emails at all: either
# a mail-client-generated placeholder ("This text is part of the internal
# format of your mail folder, and is not a real message" -- literally not an
# email), or a full leaked RFC822 header block starting on line 1 (so
# strip_email_headers() below can't help -- it only recognizes headers that
# start immediately, and these rows have prose before the headers). Found by
# reading actual content, not inferred -- see README_EMAIL.md section 1.
_MBOX_ARTIFACT_MARKERS = (
    "is not a real message",
    "Return-Path:",
    "X-Original-To:",
    "Delivered-To:",
)


def _combine_text(df: pd.DataFrame) -> pd.Series:
    return (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()


def _is_mbox_artifact(text: str) -> bool:
    return any(marker in text for marker in _MBOX_ARTIFACT_MARKERS)


def _subject_group_key(subject: str) -> str:
    """Normalized subject line, used to group-split so template-reused
    campaign emails (confirmed: 145/1565 Nazario and 780/3332 Nigerian_Fraud
    rows share a subject with another row in the same source) can't have one
    copy in train and a near-duplicate in test."""
    s = str(subject).strip().lower()
    s = re.sub(r"^(re|fwd?)\s*:\s*", "", s)  # strip reply/forward prefixes
    s = re.sub(r"\s+", " ", s)
    return s or "__blank_subject__"


def _load_raw() -> pd.DataFrame:
    if not RAW_DIR.exists():
        print(f"No raw email data at {RAW_DIR}. Run: python -m ml.training.download_email_dataset")
        sys.exit(1)

    frames = []
    for source in PHISHING_SOURCES:
        df = pd.read_csv(RAW_DIR / f"{source}.csv")
        frames.append(pd.DataFrame({
            "text": _combine_text(df),
            "subject": df["subject"],
            "label": "phishing",
            "source": source,
        }))
    for source in LEGITIMATE_HAM_SOURCES:
        df = pd.read_csv(RAW_DIR / f"{source}.csv")
        ham = df[df["label"] == 0]
        frames.append(pd.DataFrame({
            "text": _combine_text(ham),
            "subject": ham["subject"],
            "label": "legitimate",
            "source": source,
        }))
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} rows: {combined['label'].value_counts().to_dict()}")
    print(f"By source: {combined['source'].value_counts().to_dict()}")
    return combined


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_raw()

    df = df[df["text"].str.len() > 0].copy()

    artifact_mask = df["text"].apply(_is_mbox_artifact)
    if artifact_mask.any():
        print(f"Dropping {artifact_mask.sum()} raw mbox-artifact rows (mail-client placeholders / leaked headers)")
        df = df[~artifact_mask]

    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    if before != len(df):
        print(f"Dropped {before - len(df)} exact-duplicate email texts")

    print("Stripping any leaked RFC822 headers, then tokenizing (lemmatize + stopword removal, "
          "same preprocess_email_text() the live app uses) ...")
    df["body_for_nlp"] = df["text"].apply(strip_email_headers)
    df["tokens"] = df["body_for_nlp"].apply(lambda t: " ".join(preprocess_email_text(t)))
    empty_tokens = df["tokens"].str.len() == 0
    if empty_tokens.any():
        print(f"Dropping {empty_tokens.sum()} rows with no tokens left after preprocessing")
        df = df[~empty_tokens]

    groups = df["subject"].apply(_subject_group_key)
    train_idx, temp_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE).split(df, groups=groups)
    )
    train_df, temp_df = df.iloc[train_idx].copy(), df.iloc[temp_idx].copy()
    temp_groups = groups.iloc[temp_idx]
    val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE).split(temp_df, groups=temp_groups)
    )
    val_df, test_df = temp_df.iloc[val_idx].copy(), temp_df.iloc[test_idx].copy()

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        print(f"  {name}: {len(split_df)} rows, label balance {split_df['label'].value_counts().to_dict()}")

    # Subject-group leakage check (should be 0 by construction). Bug fixed
    # here: val_idx/test_idx are positional relative to temp_df (the val+test
    # pool), not the full df/groups -- indexing `groups` (full-df-positional)
    # with them picked essentially arbitrary rows and reported a meaningless
    # "overlap" count. Must index temp_groups (temp_df-positional) instead.
    train_groups = set(groups.iloc[train_idx])
    val_groups = set(temp_groups.iloc[val_idx])
    test_groups = set(temp_groups.iloc[test_idx])
    print(
        f"Subject-group overlap: train/test={len(train_groups & test_groups)}, "
        f"train/val={len(train_groups & val_groups)}, val/test={len(val_groups & test_groups)} (expect 0, 0, 0)"
    )

    out_path = PROCESSED_DIR / "email_features.parquet"
    full_df[["text", "tokens", "label", "source", "split"]].to_parquet(out_path, index=False)
    print(f"Wrote {len(full_df)} rows to {out_path}")

    test_path = PROCESSED_DIR / "test_emails.csv"
    test_df[["text", "label"]].to_csv(test_path, index=False)
    print(f"Wrote held-out test emails (for live API evaluation) to {test_path}")


if __name__ == "__main__":
    main()
