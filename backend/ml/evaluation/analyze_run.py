"""Verify trained API/offline parity and summarize errors without changing labels.

    python -m ml.evaluation.analyze_run ml/evaluation/runs/<run>/results.json
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from app.pipeline.feature_extraction_url import extract_url_features
from app.pipeline.feature_extraction_email import extract_email_features
from ml.evaluation.run_evaluation import BACKEND_ROOT, classification_metrics, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    predictions_path = args.report.with_suffix(".predictions.jsonl")
    is_email = report.get("input_type", "url") == "email_text"
    model_path = BACKEND_ROOT / ("ml/saved_models/email_classifier.joblib" if is_email else "ml/saved_models/url_classifier.joblib")
    hash_key = "email_model_sha256" if is_email else "model_sha256"
    if report["runtime"][hash_key] != file_hash(model_path):
        raise RuntimeError("Current model is not the evaluated model")
    if report["predictions_sha256"] != file_hash(predictions_path):
        raise RuntimeError("Prediction evidence changed")
    records = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    bundle = joblib.load(model_path)
    field = "email_text" if is_email else "url"
    if is_email:
        tokens = [extract_email_features(r[field].strip())["tokens_text"] for r in records]
        vectors = bundle["vectorizer"].transform(tokens)
    else:
        features = pd.DataFrame([extract_url_features(r[field].strip()) for r in records])
        vectors = features[bundle["feature_columns"]]
    probabilities = bundle["model"].predict_proba(vectors)[:, 1]
    predicted = bundle["model"].predict(vectors)
    mismatches = [r[field] for r, p in zip(records, predicted)
                  if r["classification"] != ("phishing" if p else "legitimate")]
    confidence_mismatches = sum(
        abs(r["confidence_score"] - round(float(p if r["classification"] == "phishing" else 1 - p), 4)) > .0001
        for r, p in zip(records, probabilities))
    slices = {}
    slice_masks = {"all_email": [True] * len(records)} if is_email else {
        "root_or_no_path": features.path_length == 0, "non_root_path": features.path_length > 0}
    for name, mask in slice_masks.items():
        subset = [r for r, include in zip(records, mask) if include]
        slices[name] = {"rows": len(subset), **classification_metrics(subset)} if subset else {"rows": 0}
    wrong = [r for r in records if r["classification"] != r["true_label"]]
    analysis = {
        "report_sha256": file_hash(args.report), "model_sha256": file_hash(model_path),
        "evaluated_count": len(records), "offline_api_prediction_mismatches": len(mismatches),
        "offline_api_confidence_mismatches": int(confidence_mismatches),
        "mismatch_inputs": mismatches, "slices": slices,
        "false_positive_examples": [r for r in wrong if r["true_label"] == "legitimate"][:20],
        "false_negative_examples": [r for r in wrong if r["true_label"] == "phishing"][:20],
        "interpretation": "Source labels are retained as supplied; examples are errors relative to that ground truth, not independently adjudicated labels.",
    }
    output = args.report.with_suffix(".analysis.json")
    if output.exists():
        raise RuntimeError("Analysis already exists; preserve prior evidence")
    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps({k: analysis[k] for k in ("evaluated_count", "offline_api_prediction_mismatches", "offline_api_confidence_mismatches", "slices")}, indent=2))
    print(f"Saved {output}")
    if mismatches or confidence_mismatches:
        raise SystemExit("API/offline parity failed; inspect the analysis")


if __name__ == "__main__":
    main()
