"""Bit 2 of the email-classifier pipeline (build order phase 2's NLP component,
"classical models" half of spec section 2's "TF-IDF for classical models, BERT
embeddings for transformer model"): TF-IDF + Logistic Regression / Random
Forest / Complement Naive Bayes on the honestly-scoped dataset from
build_email_features.py.

This is a genuinely imbalanced problem (~11% phishing) by construction -- see
build_email_features.py's docstring for why: the phishing class is scoped to
content-verified fraud (Nazario + Nigerian_Fraud) rather than the much larger
but spam-conflated marketed dataset. Accuracy is close to meaningless here (a
trivial always-legitimate baseline scores ~89%); precision/recall/F1 and
average precision (area under the precision-recall curve) are what matter.

Usage:
    python -m ml.training.train_email_classifier
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import ComplementNB

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BACKEND_ROOT / "ml" / "data" / "processed_email"
SAVED_MODELS_DIR = BACKEND_ROOT / "ml" / "saved_models"
EXPERIMENTS_DIR = SAVED_MODELS_DIR / "experiments"

RANDOM_STATE = 42
MAX_FEATURES = 20_000

# Same targets as the URL classifier (spec section 9 doesn't give email-specific
# numbers, so this reuses the project-wide thresholds) -- kept for comparability,
# not because they're calibrated to this task's realistic class balance.
TARGET_F1 = 0.90
TARGET_PRECISION = 0.91
TARGET_RECALL = 0.90

# Sanity check: hand-written, unambiguous examples. Loosely mirrors
# train_url_classifier.py's bare-domain sanity check -- aggregate metrics
# alone were not enough to catch that project's dataset-artifact failures.
SANITY_PHISHING = [
    "URGENT: Your account has been suspended. Click here to verify your identity "
    "immediately or your account will be permanently closed within 24 hours.",
    "Dear Customer, we detected unusual activity on your PayPal account. Please "
    "confirm your password and credit card details at the link below to restore access.",
    "Dearest Friend, I am the widow of a former minister and I need your urgent "
    "assistance to transfer $10,500,000 USD out of the country. Please reply with your bank details.",
]
SANITY_LEGITIMATE = [
    "Hi team, attached are the meeting notes from this afternoon's planning session. Let me know if I missed anything.",
    "Thanks for your email. I'll review the attached proposal and get back to you by Friday.",
    "Reminder: the quarterly report is due next Monday. Please send your section drafts by end of week.",
]


def _load_split(df: pd.DataFrame, name: str) -> tuple[pd.Series, pd.Series]:
    split_df = df[df["split"] == name]
    return split_df["tokens"], (split_df["label"] == "phishing").astype(int)


def _evaluate(model, vectorizer, tokens, y) -> dict:
    X = vectorizer.transform(tokens)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "average_precision": average_precision_score(y, y_proba),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def _baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> dict:
    majority = int(round(y_train.mean()))
    y_pred = pd.Series(majority, index=y_test.index)
    return {
        "majority_class": "phishing" if majority == 1 else "legitimate",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def _sanity_check(model, vectorizer) -> dict:
    from app.pipeline.preprocessing import preprocess_email_text

    def _predict(text: str) -> str:
        tokens = " ".join(preprocess_email_text(text))
        return "phishing" if model.predict(vectorizer.transform([tokens]))[0] == 1 else "legitimate"

    phishing_correct = sum(1 for t in SANITY_PHISHING if _predict(t) == "phishing")
    legit_correct = sum(1 for t in SANITY_LEGITIMATE if _predict(t) == "legitimate")
    return {
        "phishing_pass": f"{phishing_correct}/{len(SANITY_PHISHING)}",
        "legitimate_pass": f"{legit_correct}/{len(SANITY_LEGITIMATE)}",
        "all_passed": phishing_correct == len(SANITY_PHISHING) and legit_correct == len(SANITY_LEGITIMATE),
    }


def main() -> None:
    features_path = PROCESSED_DIR / "email_features.parquet"
    if not features_path.exists():
        print(f"No feature matrix at {features_path}. Run: python -m ml.training.build_email_features")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    train_tokens, y_train = _load_split(df, "train")
    val_tokens, y_val = _load_split(df, "val")
    test_tokens, y_test = _load_split(df, "test")

    print(f"Vectorizing (TF-IDF, max_features={MAX_FEATURES}) ...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_tokens)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "complement_nb": ComplementNB(),
    }

    val_results = {}
    for name, model in candidates.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        val_metrics = _evaluate(model, vectorizer, val_tokens, y_val)
        val_results[name] = {"model": model, "val_metrics": val_metrics}
        print(f"  val: {val_metrics}")

    best_name = max(val_results, key=lambda n: val_results[n]["val_metrics"]["f1"])
    best_model = val_results[best_name]["model"]
    test_metrics = _evaluate(best_model, vectorizer, test_tokens, y_test)
    baseline_metrics = _baseline_metrics(y_train, y_test)
    sanity = _sanity_check(best_model, vectorizer)

    print(f"\nSelected {best_name} (highest val F1)")
    print(f"Baseline (always predict '{baseline_metrics['majority_class']}'): {baseline_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Sanity check: {sanity}")

    meets_targets = (
        test_metrics["f1"] >= TARGET_F1
        and test_metrics["precision"] >= TARGET_PRECISION
        and test_metrics["recall"] >= TARGET_RECALL
    )
    print(
        f"Meets spec-wide targets (F1>={TARGET_F1}, precision>={TARGET_PRECISION}, "
        f"recall>={TARGET_RECALL}): {meets_targets}"
    )

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVED_MODELS_DIR / "email_classifier.joblib"
    joblib.dump(
        {"model": best_model, "vectorizer": vectorizer, "model_name": best_name},
        out_path,
    )
    print(f"Saved {best_name} to {out_path}")

    metrics_path = SAVED_MODELS_DIR / "email_classifier_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "val_metrics": {n: r["val_metrics"] for n, r in val_results.items()},
                "test_metrics": test_metrics,
                "baseline_metrics": baseline_metrics,
                "meets_spec_targets": meets_targets,
                "sanity_check": sanity,
                "train_size": len(y_train),
                "val_size": len(y_val),
                "test_size": len(y_test),
                "positive_rate_train": float(y_train.mean()),
            },
            indent=2,
        )
    )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
