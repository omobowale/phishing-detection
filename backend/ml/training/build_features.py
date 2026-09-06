"""Bit 2 of the URL-classifier pipeline: turn raw labeled URLs into the exact
feature set /detect computes at inference time.

Reuses app.pipeline.feature_extraction_url.extract_url_features() directly -
training and serving must compute features identically, or the model learns
on one distribution and scores on another.

Usage:
    python -m ml.training.build_features
"""

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit, train_test_split

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline.feature_extraction_url import extract_url_features  # noqa: E402
from ml.training.known_legitimate_domains import NO_PHISHING_POSSIBLE_DOMAINS  # noqa: E402

RAW_DIR = BACKEND_ROOT / "ml" / "data" / "raw"
PROCESSED_DIR = BACKEND_ROOT / "ml" / "data" / "processed"

# suffix_list_urls=() forces the bundled Public Suffix List snapshot, never a
# live network fetch -- a training script silently hitting the network for an
# updated PSL would make the grouped split non-reproducible run to run (the
# PSL does change over time). Update the pinned `tldextract` version in
# requirements.txt to get a newer snapshot deliberately, not implicitly.
_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

# Kept modest by default so a first end-to-end run finishes in a couple of
# minutes on a laptop; raise via env var once the pipeline is proven out.
MAX_ROWS_PER_CLASS = int(os.environ.get("MAX_ROWS_PER_CLASS", "50000"))

# "grouped" (default): train/val/test split by registrable domain, so no host
# can appear in more than one split -- the methodologically correct choice for
# claiming generalization to unseen domains (a second external review found
# the previous per-URL random split let 5,026/10,000 test rows share a
# hostname with something in train). "random" reproduces the old per-URL
# random split, kept only so the two can be trained and compared side by side
# -- see ml/README.md section 10.
SPLIT_STRATEGY = os.environ.get("SPLIT_STRATEGY", "grouped")

# Set to skip the known-legitimate-host phishing-label cleaning step entirely,
# to train and compare with/without it -- see ml/README.md section 10.
SKIP_DOMAIN_CLEANING = os.environ.get("SKIP_DOMAIN_CLEANING", "") not in ("", "0")

LABEL_MAP = {"benign": "legitimate", "phishing": "phishing"}


def _find_raw_csv() -> Path:
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV found in {RAW_DIR}. Run: python -m ml.training.download_dataset")
        sys.exit(1)
    return csv_files[0]


def _host_of(url: str) -> str:
    try:
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
            url = "http://" + url
        # .hostname, not .netloc.split(":")[0] -- same userinfo-confusion bug
        # fixed in app/pipeline/preprocessing.py's get_domain(); matters here
        # too since a wrong host would silently skip real mislabeled rows or
        # wrongly match unrelated ones during the cleaning step below.
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _is_provably_mislabeled_phishing(host: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in NO_PHISHING_POSSIBLE_DOMAINS)


def _registrable_domain(host: str) -> str:
    """Public Suffix List-aware registrable domain, e.g. "en.wikipedia.org" ->
    "wikipedia.org", and critically "example.co.uk" -> "example.co.uk" (not
    "co.uk"). An earlier version used a naive last-two-labels heuristic (still
    used, with the same known limitation, by app/pipeline/feature_extraction_url.py's
    subdomain_count/brand_keyword_outside_domain -- fixing that too is a
    separate change, since it affects live inference features and would need
    its own retrain/revalidation cycle, not folded into this split-grouping fix).
    An external review measured that heuristic collapsing 1,768 unrelated
    *.co.uk hosts into one fake "co.uk" group here, which is exactly the kind
    of unnecessary grouping/distribution shift a domain-grouped split is
    supposed to avoid, not introduce.

    Falls back to the host itself for IPs and other suffix-less hosts (each
    such host is its own group) -- tldextract's PSL lookup returns an empty
    string for those, which would otherwise wrongly lump every IP address and
    "localhost" into one giant fake group."""
    return _TLD_EXTRACT(host).top_domain_under_public_suffix or host


def _normalize_for_conflict_check(url: str) -> str | None:
    """Looser than _host_of: normalizes scheme default, host case, an empty/root
    path, and drops the fragment, so "EXAMPLE.com" and "example.com/" and
    "http://example.com" are treated as the same URL for conflict-detection
    purposes. This intentionally catches more than exact-string matching (an
    external review found 3,940-4,043 such groups with conflicting labels,
    depending on normalization details, vs. only 6 exact-string conflicts)."""
    try:
        u = str(url).strip()
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
            u = "http://" + u
        parsed = urlparse(u)
        netloc = parsed.netloc.lower()
        path = parsed.path if parsed.path not in ("", "/") else "/"
        query = f"?{parsed.query}" if parsed.query else ""
        return f"http://{netloc}{path}{query}"
    except Exception:
        return None


def _load_labeled_urls() -> pd.DataFrame:
    raw_path = _find_raw_csv()
    print(f"Loading {raw_path.name} ...")
    df = pd.read_csv(raw_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "url" not in df.columns or "type" not in df.columns:
        print(f"Expected 'url' and 'type' columns, found: {list(df.columns)}")
        sys.exit(1)

    df["type"] = df["type"].str.strip().str.lower()
    df = df[df["type"].isin(LABEL_MAP.keys())].copy()
    df["label"] = df["type"].map(LABEL_MAP)

    # Ordering matters here and this is deliberately conflict-detection FIRST,
    # domain-based cleaning SECOND. An external review caught a real bug in
    # the previous order (domain cleaning first): if a URL had conflicting
    # benign/phishing labels and the phishing copy happened to be on a
    # known-legitimate host, domain cleaning would remove that copy before the
    # conflict check ever ran -- silently "resolving" the disagreement by
    # domain reputation alone, exactly the kind of unexamined authority the
    # conflict-quarantine step exists to avoid. Detecting conflicts on the
    # original binary pool first, then applying domain-based cleaning to
    # whatever's left, keeps the two decisions independent and auditable.

    # 1. Quarantine (drop + record, don't silently resolve) rows whose ground
    # truth is disputed at two levels:
    #  a. exact URL string appears with BOTH labels (6 cases in the raw CSV,
    #     e.g. "en.wikipedia.org/wiki/E-book" labeled both benign and phishing)
    #  b. same URL *after normalizing* scheme/case/empty-path/fragment appears
    #     with both labels (thousands of cases -- an external review found
    #     3,940-4,043 such groups depending on normalization details; a
    #     model can't learn a consistent boundary from contradictory examples
    #     regardless of how similar or different two exact strings are)
    # Quarantined rows are saved to quarantined_conflicts.csv rather than just
    # discarded, so the decision is auditable and reversible.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    exact_conflict = df.groupby("url")["label"].transform("nunique") > 1
    norm = df["url"].astype(str).apply(_normalize_for_conflict_check)
    valid_norm = norm.notna()
    norm_conflict = pd.Series(False, index=df.index)
    if valid_norm.any():
        norm_label_counts = df[valid_norm].groupby(norm[valid_norm])["label"].transform("nunique")
        norm_conflict.loc[valid_norm] = norm_label_counts > 1

    quarantined = exact_conflict | norm_conflict
    if quarantined.any():
        df.loc[quarantined, ["url", "type", "label"]].to_csv(
            PROCESSED_DIR / "quarantined_conflicts.csv", index=False
        )
        print(
            f"Quarantined {quarantined.sum()} rows with disputed (exact- or "
            f"normalized-URL) conflicting labels -> {PROCESSED_DIR / 'quarantined_conflicts.csv'}"
        )
        df = df[~quarantined]

    # 2. THEN drop rows where "phishing" on a known-legitimate host is very
    # strong evidence of a labeling error, not proof in the mathematical sense
    # a second external review correctly pushed back on ("a familiar domain
    # alone does not prove a phishing label is wrong") -- see
    # ml/training/known_legitimate_domains.py's docstring for the exact
    # rationale and ml/README.md section 3.4 for the supporting evidence (e.g.
    # 74.7% of this raw dataset's microsoft.com rows and 90.9% of its
    # github.com rows are labeled "phishing", and every sampled example across
    # all ~30 domains on the list was manually read and found to be an
    # ordinary real page). This does NOT touch rows on domains with a genuine
    # free-hosting/user-content surface (Google Forms, GitHub user pages,
    # etc.), where trusted-domain-abuse phishing is a real, documented pattern
    # worth keeping. Rows removed here are saved to
    # domain_cleaned_phishing_labels.csv, not just silently discarded, and this
    # step can be disabled (SKIP_DOMAIN_CLEANING=1) to train and compare
    # with/without -- see ml/README.md section 10.
    if not SKIP_DOMAIN_CLEANING:
        host = df["url"].astype(str).apply(_host_of)
        mislabeled = (df["label"] == "phishing") & host.apply(_is_provably_mislabeled_phishing)
        if mislabeled.any():
            df.loc[mislabeled, ["url", "type", "label"]].to_csv(
                PROCESSED_DIR / "domain_cleaned_phishing_labels.csv", index=False
            )
            print(
                f"Dropping {mislabeled.sum()} rows with 'phishing' on a known-legitimate host "
                f"-> logged to domain_cleaned_phishing_labels.csv"
            )
            df = df[~mislabeled]
    else:
        print("SKIP_DOMAIN_CLEANING set: keeping rows the known-legitimate-host filter would otherwise drop")

    # Drop exact-duplicate URLs (same url, same label) before splitting. A
    # duplicate that ends up in both train and test lets the model "recognize"
    # a memorized training row at test time rather than genuinely generalizing,
    # inflating held-out metrics. keep="first" is arbitrary but deterministic
    # (stable given a fixed raw CSV) -- fine now that conflicting labels are
    # already removed above, so "first" and "any other copy" agree on the label.
    before = len(df)
    df = df.drop_duplicates(subset="url", keep="first")
    if before != len(df):
        print(f"Dropped {before - len(df)} exact-duplicate URL rows before splitting")

    # cap and balance per class so one class doesn't swamp the other
    parts = [
        group.sample(n=min(len(group), MAX_ROWS_PER_CLASS), random_state=42)
        for _, group in df.groupby("label")
    ]
    df = pd.concat(parts, ignore_index=True)
    print(f"Using {len(df)} rows: {df['label'].value_counts().to_dict()}")
    return df[["url", "label"]]


def _build_feature_matrix(urls_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skipped = 0
    for url, label in zip(urls_df["url"], urls_df["label"]):
        try:
            features = extract_url_features(str(url))
        except Exception:
            skipped += 1
            continue
        # sklearn/xgboost want numeric dtypes, not Python bool
        features = {k: (int(v) if isinstance(v, bool) else v) for k, v in features.items()}
        features["url"] = url
        features["label"] = label
        rows.append(features)

    if skipped:
        print(f"Skipped {skipped} URLs that failed feature extraction.")
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    urls_df = _load_labeled_urls()
    features_df = _build_feature_matrix(urls_df)

    if SPLIT_STRATEGY == "grouped":
        print("Split strategy: grouped by registrable domain (no domain appears in more than one split)")
        groups = features_df["url"].astype(str).apply(lambda u: _registrable_domain(_host_of(u)))
        train_idx, temp_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(features_df, groups=groups)
        )
        train_df, temp_df = features_df.iloc[train_idx].copy(), features_df.iloc[temp_idx].copy()
        temp_groups = groups.iloc[temp_idx]
        val_idx, test_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(temp_df, groups=temp_groups)
        )
        val_df, test_df = temp_df.iloc[val_idx].copy(), temp_df.iloc[test_idx].copy()
        # GroupShuffleSplit doesn't stratify by label -- a domain-grouped split
        # can't simultaneously guarantee zero domain leakage AND exact 50/50
        # balance in every split, since some domains skew heavily to one label.
        # Report the resulting balance rather than assume it stayed 50/50.
        for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
            print(f"  {name} label balance: {split_df['label'].value_counts().to_dict()}")
    else:
        print("Split strategy: random per-URL (old methodology, kept only for comparison -- see ml/README.md section 10)")
        train_df, temp_df = train_test_split(
            features_df, test_size=0.2, stratify=features_df["label"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
        )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Host overlap: for "grouped" this should print 0 by construction (a
    # confirmation the grouping worked), for "random" it reproduces the
    # leakage the external review measured (32.6%) -- see ml/README.md
    # section 10 for the side-by-side comparison this enables.
    train_hosts = set(train_df["url"].astype(str).apply(_host_of))
    test_hosts = set(test_df["url"].astype(str).apply(_host_of))
    val_hosts = set(val_df["url"].astype(str).apply(_host_of))
    train_test_host_overlap = len(train_hosts & test_hosts)
    print(
        f"Host overlap: {train_test_host_overlap}/{len(test_hosts)} test hosts "
        f"({train_test_host_overlap / len(test_hosts):.1%}) also appear in train; "
        f"{len(train_hosts & val_hosts)}/{len(val_hosts)} val hosts also appear in train"
    )

    suffix = "" if SPLIT_STRATEGY == "grouped" else "_random_split"
    if SKIP_DOMAIN_CLEANING:
        suffix += "_no_domain_cleaning"
    out_path = PROCESSED_DIR / f"url_features{suffix}.parquet"
    full_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(full_df)} rows to {out_path}")
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    test_urls_path = PROCESSED_DIR / f"test_urls{suffix}.csv"
    test_df[["url", "label"]].to_csv(test_urls_path, index=False)
    print(f"Wrote held-out test URLs (for live API evaluation) to {test_urls_path}")


if __name__ == "__main__":
    main()
