"""Bit 3 of the URL-classifier pipeline (build order phase 2): train Random
Forest and XGBoost on the feature matrix from build_features.py, evaluate both
against the spec's performance targets (section 9), and serialize whichever
scores higher on the validation set to /ml/saved_models for classifiers.py to
load at inference time.

Usage:
    python -m ml.training.train_url_classifier
"""

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BACKEND_ROOT / "ml" / "data" / "processed"
SAVED_MODELS_DIR = BACKEND_ROOT / "ml" / "saved_models"
# Comparison/experiment runs (ARTIFACT_SUFFIX set) never write into
# SAVED_MODELS_DIR directly -- they go in their own subdirectory, so there is
# no filename pattern for a cleanup step to get wrong. A prior version tried
# to keep the primary model safe by excluding "*_random_split*" from a glob
# cleanup, which correctly protected that one comparison run but silently
# deleted a different one ("*_no_domain_cleaning*") the exclusion didn't know
# about -- a real, confirmed loss of a saved model. Physical separation is
# the fix, not a smarter exclusion list.
EXPERIMENTS_DIR = SAVED_MODELS_DIR / "experiments"

# Which feature matrix to train on -- "url_features.parquet" (default, built
# with SPLIT_STRATEGY=grouped) is the primary artifact classifiers.py loads.
# Set FEATURES_FILE=url_features_random_split.parquet + ARTIFACT_SUFFIX=_random_split
# to train against the old per-URL random split instead, purely for the
# side-by-side comparison in ml/README.md section 10.
FEATURES_FILE = os.environ.get("FEATURES_FILE", "url_features.parquet")
ARTIFACT_SUFFIX = os.environ.get("ARTIFACT_SUFFIX", "")

# subset of app.pipeline.feature_extraction_url.extract_url_features()'s keys --
# classifiers.py builds inference vectors from this same list (saved alongside
# the model, so this is the only place the column list needs to be correct).
#
# "has_https" is deliberately excluded: over 90% of this dataset's raw URLs
# have no scheme at all (an external review measured 91.7% for benign, 73.6%
# for phishing -- corrected from an earlier draft of this comment, which
# imprecisely said "no scheme" when it meant "not literally https://"; ~8.3%
# of benign / ~26.4% of phishing rows DO have an explicit scheme, just mostly
# "http://" not "https://"). Whatever the exact split, has_https's nonzero
# values turned out to mostly encode which subset of the dataset happened to
# be stored with an explicit "https://" prefix, not actual TLS usage -- it
# dominated feature importance (~51%) while making the model confidently misclassify ordinary
# bare domains (see train run notes). Revisit if a less skewed dataset is used.
FEATURE_COLUMNS = [
    "url_length",
    "host_length",
    "subdomain_count",
    "is_ip_address",
    "special_char_count",
    "digit_count",
    "path_length",
    "query_length",
    "brand_keyword_count",
    "brand_keyword_outside_domain",
    "suspicious_tld",
    "has_at_symbol",
]

# XGBoost monotonic constraints, same order as FEATURE_COLUMNS: 1 means "raising
# this feature can only raise phishing-probability, never lower it", 0 means
# unconstrained. Only set where domain logic is unambiguous at every value the
# feature can take -- an IP-as-host, a brand name outside the registrable
# domain, a known-bad TLD, or an "@" in the URL are never *more* legitimate no
# matter what.
#
# subdomain_count is deliberately NOT constrained, despite looking like an
# obvious candidate: going from 0 to 1 subdomain (bare domain -> "www.example.com")
# is the single most common, totally benign pattern on the entire web, while
# going from 1 to 4+ is the genuinely suspicious case. A monotonic constraint
# can't express that distinction -- it forces every step to point the same
# direction -- and empirically this was directly responsible for misclassifying
# github.com-shaped legitimate URLs (they all carry a "www."/"docs."-style
# single subdomain) as phishing. See ml/README.md for the full trace.
MONOTONE_CONSTRAINTS = (
    0,  # url_length
    0,  # host_length
    0,  # subdomain_count
    1,  # is_ip_address
    0,  # special_char_count
    0,  # digit_count
    0,  # path_length
    0,  # query_length
    0,  # brand_keyword_count
    1,  # brand_keyword_outside_domain
    1,  # suspicious_tld
    1,  # has_at_symbol
)

# spec section 9 engineering targets
TARGET_F1 = 0.90
TARGET_PRECISION = 0.91
TARGET_RECALL = 0.90

# Sanity check: plain, single-domain URLs with no suspicious features at all.
# These are the single most common shape of real-world input to this app (a
# user checking a bare domain), so a model that can't get these right is not
# fit to ship even if it clears the aggregate targets above -- that's exactly
# what happened on the first two training runs (see project memory), where the
# model flagged nearly every bare domain as phishing due to dataset artifacts.
SANITY_LEGITIMATE_URLS = [
    "google.com",
    "example.com",
    "github.com",
    "wikipedia.org",
    "microsoft.com",
    "amazon.com",
    "apple.com",
    "cnn.com",
    # realistic full URLs (scheme + "www."-style subdomain + real path) -- these
    # exposed two further bugs this bare-domain-only list originally missed:
    # url_length/etc. being scheme-sensitive, and an overly-blunt monotonic
    # constraint on subdomain_count penalizing an ordinary "www." (see
    # ml/README.md).
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.python.org/downloads/",
    "https://github.com/anthropics/claude-code",
    "https://www.nytimes.com/section/technology",
]
SANITY_PHISHING_URLS = [
    "http://192.168.1.10/wp-admin/login.php",
    "http://paypal.com.verify-account.security-check.top/login",
    "http://secure-appleid-verify.com/login@apple.com",
    "http://faceb00k-account-recovery.tk/verify",
]


def _load_split(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
    split_df = df[df["split"] == name]
    X = split_df[FEATURE_COLUMNS]
    y = (split_df["label"] == "phishing").astype(int)
    return X, y


def _evaluate(model, X, y) -> dict:
    # model.predict() applies the standard 0.5 probability threshold, which is
    # what app.pipeline.classifiers.TrainedURLClassifier now also defaults to
    # for the URL-only case (see _combine()'s docstring) -- these numbers and
    # the live API's decisions are evaluated the same way.
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def _baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Trivial always-predict-the-majority-class baseline. Any real classifier
    that doesn't clearly beat this isn't learning anything the raw class
    balance didn't already give away for free.

    The majority class must come from TRAINING labels, not test labels --
    an earlier version of this function computed it from y_test itself, which
    means the "baseline" was quietly using knowledge of the test set's answer
    key to pick its constant prediction (a second external review caught
    this). A real naive baseline only gets to see training data.
    """
    majority = int(round(y_train.mean()))
    y_pred = pd.Series(majority, index=y_test.index)
    return {
        "majority_class": "phishing" if majority == 1 else "legitimate",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def _sanity_check(model) -> dict:
    """See SANITY_LEGITIMATE_URLS/SANITY_PHISHING_URLS above -- aggregate metrics
    on the held-out split alone were not enough to catch a model that flags every
    bare domain as phishing, so this runs a small, hand-picked check every time.

    Known limitation, confirmed empirically (not a bug to keep chasing): bare,
    short, unbranded domains are structurally identical whether legitimate or
    malicious -- url_length/subdomain_count/etc. carry no brand-recognition
    signal, and this dataset's own bare-short-domain slice is ~65% phishing, so
    a well-fit model leans "phishing" here by design, not by defect. This is
    exactly the gap the spec's admin-managed Whitelist exists to cover (known
    trusted domains should bypass the ML pipeline entirely) -- expect the
    legitimate_pass rate to stay low until a differently-sourced dataset with a
    realistic real-world prior for this URL shape is used."""
    from app.pipeline.feature_extraction_url import extract_url_features

    def _predict(url: str) -> str:
        f = extract_url_features(url)
        vector = [[f.get(col, 0) for col in FEATURE_COLUMNS]]
        return "phishing" if model.predict(vector)[0] == 1 else "legitimate"

    legit_correct = sum(1 for u in SANITY_LEGITIMATE_URLS if _predict(u) == "legitimate")
    phish_correct = sum(1 for u in SANITY_PHISHING_URLS if _predict(u) == "phishing")
    return {
        "legitimate_pass": f"{legit_correct}/{len(SANITY_LEGITIMATE_URLS)}",
        "phishing_pass": f"{phish_correct}/{len(SANITY_PHISHING_URLS)}",
        "all_passed": legit_correct == len(SANITY_LEGITIMATE_URLS)
        and phish_correct == len(SANITY_PHISHING_URLS),
    }


def main() -> None:
    features_path = PROCESSED_DIR / FEATURES_FILE
    if not features_path.exists():
        print(f"No feature matrix at {features_path}. Run: python -m ml.training.build_features")
        sys.exit(1)
    print(f"Training on {features_path}")

    df = pd.read_parquet(features_path)
    X_train, y_train = _load_split(df, "train")
    X_val, y_val = _load_split(df, "val")
    X_test, y_test = _load_split(df, "test")

    candidates = {
        # NOTE: heavier regularization (shallower trees, larger min leaf/child
        # weight) was tried here and made both the aggregate metrics AND the
        # bare-domain sanity check worse -- the sanity-check failure isn't
        # overfitting, it's an information ceiling (see SANITY_LEGITIMATE_URLS
        # comment + project memory), so these settings intentionally stay close
        # to sklearn/xgboost defaults. monotone_constraints is kept below since
        # it encodes real domain expertise and is free either way.
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            monotone_constraints=MONOTONE_CONSTRAINTS,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
    }

    val_results = {}
    for name, model in candidates.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        val_metrics = _evaluate(model, X_val, y_val)
        val_results[name] = {"model": model, "val_metrics": val_metrics}
        print(f"  val: {val_metrics}")

    best_name = max(val_results, key=lambda n: val_results[n]["val_metrics"]["f1"])
    best_model = val_results[best_name]["model"]
    test_metrics = _evaluate(best_model, X_test, y_test)
    baseline_metrics = _baseline_metrics(y_train, y_test)
    sanity = _sanity_check(best_model)

    print(f"\nSelected {best_name} (highest val F1)")
    print(f"Baseline (always predict '{baseline_metrics['majority_class']}'): {baseline_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Bare-domain sanity check: {sanity}")

    meets_targets = (
        test_metrics["f1"] >= TARGET_F1
        and test_metrics["precision"] >= TARGET_PRECISION
        and test_metrics["recall"] >= TARGET_RECALL
    )
    print(
        f"Meets spec targets (F1>={TARGET_F1}, precision>={TARGET_PRECISION}, "
        f"recall>={TARGET_RECALL}): {meets_targets}"
    )

    if not sanity["all_passed"]:
        print(
            "\nWARNING: model failed the bare-domain sanity check. Confirmed this is "
            "a feature-set information ceiling, not overfitting -- see _sanity_check "
            "docstring. This is exactly what the admin-managed Whitelist is for: seed "
            "it with known trusted domains rather than expecting the URL-structure "
            "model to recognize them. NOT recommended to switch URL_CLASSIFIER_BACKEND "
            "to 'trained' unless you've reviewed that tradeoff.\n"
        )

    if ARTIFACT_SUFFIX:
        # comparison/experiment run: lives entirely under experiments/, never
        # touches the primary model file or its directory.
        out_dir = EXPERIMENTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"url_classifier_{best_name}{ARTIFACT_SUFFIX}.joblib"
        metrics_path = out_dir / f"url_classifier_metrics{ARTIFACT_SUFFIX}.json"
    else:
        # primary run: always the same fixed filename, algorithm-agnostic, so
        # saving it is a plain overwrite -- there is nothing else to clean up.
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SAVED_MODELS_DIR / "url_classifier.joblib"
        metrics_path = SAVED_MODELS_DIR / "url_classifier_metrics.json"

    joblib.dump({"model": best_model, "feature_columns": FEATURE_COLUMNS, "model_name": best_name}, out_path)
    print(f"Saved {best_name} to {out_path}")

    metrics_path.write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "val_metrics": {n: r["val_metrics"] for n, r in val_results.items()},
                "test_metrics": test_metrics,
                "baseline_metrics": baseline_metrics,
                "meets_spec_targets": meets_targets,
                "bare_domain_sanity_check": sanity,
            },
            indent=2,
        )
    )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
