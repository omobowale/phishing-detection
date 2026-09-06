"""Evaluate a running API; use managed_run for an isolated server and fresh database."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import time

import httpx
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

BACKEND_ROOT = Path(__file__).resolve().parents[2]
TEST_URLS_PATH = BACKEND_ROOT / "ml/data/processed/test_urls.csv"
API_PREFIX = "/api/v1"


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def latency_summary(values):
    ordered = sorted(values)
    def percentile(q):
        return ordered[max(0, math.ceil(len(ordered) * q) - 1)] if ordered else None
    return {"mean": sum(ordered) / len(ordered) if ordered else None,
            "p50": percentile(.50), "p95": percentile(.95), "p99": percentile(.99)}


def classification_metrics(results):
    if not results:
        raise RuntimeError("No successful detections; classification metrics are undefined")
    truth = [r["true_label"] == "phishing" for r in results]
    predicted = [r["classification"] == "phishing" for r in results]
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[False, True]).ravel()
    return {"accuracy": accuracy_score(truth, predicted),
            "precision": precision_score(truth, predicted, zero_division=0),
            "recall": recall_score(truth, predicted, zero_division=0),
            "f1_score": f1_score(truth, predicted, zero_division=0),
            "confusion_matrix": dict(zip(["true_negative", "false_positive", "false_negative", "true_positive"],
                                         map(int, [tn, fp, fn, tp])))}


def _run_detection_pass(base_url, urls_df):
    results, rejected = [], []
    with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client:
        for i, row in enumerate(urls_df.itertuples(), 1):
            start = time.perf_counter()
            record = {"url": row.url, "true_label": row.label}
            try:
                response = client.post(f"{API_PREFIX}/detect", json={"url": row.url})
                record["round_trip_ms"] = (time.perf_counter() - start) * 1000
                if response.status_code == 200:
                    results.append({**record, **response.json()})
                else:
                    rejected.append({**record, "status_code": response.status_code, "error": response.text[:1000]})
            except httpx.HTTPError as exc:
                rejected.append({**record, "round_trip_ms": (time.perf_counter() - start) * 1000,
                                 "status_code": None, "error": str(exc)})
            if i % 1000 == 0:
                print(f"  {i}/{len(urls_df)} requests completed", flush=True)
    return results, rejected


def _throughput_probe(base_url, sample_urls, concurrency=40):
    """Measure valid-input capacity, retaining HTTP statuses and round-trip times."""
    if not sample_urls:
        raise ValueError("Throughput probe needs valid inputs")
    with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client:
        def one(url):
            start = time.perf_counter()
            try:
                response = client.post(f"{API_PREFIX}/detect", json={"url": url})
                status = str(response.status_code)
            except httpx.HTTPError:
                status = "transport_error"
            return status, (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(pool.map(one, sample_urls))
        elapsed = time.perf_counter() - start
    statuses = Counter(s for s, _ in outcomes)
    successes = statuses.get("200", 0)
    return {"requests": len(outcomes), "successful_requests": successes,
            "failed_requests": len(outcomes) - successes, "status_counts": dict(statuses),
            "error_rate": 1 - successes / len(outcomes), "concurrency": concurrency,
            "elapsed_seconds": elapsed, "throughput_rps": successes / elapsed,
            "attempted_rps": len(outcomes) / elapsed,
            "client_observed_round_trip_ms": latency_summary([ms for _, ms in outcomes])}


def _label_logs(db_path, results):
    """Only label an exact, fresh-run match; validate before making any updates."""
    with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=rw", uri=True) as db:
        logs = db.execute("SELECT id,input_data,prediction,actual_label FROM detection_logs").fetchall()
        by_input = {f"url: {r['url'].strip()}": r for r in results}
        if len(by_input) != len(results) or len(logs) != len(results):
            raise RuntimeError("Evaluation requires unique inputs and an otherwise empty log database")
        updates, seen = [], set()
        for log_id, input_data, prediction, label in logs:
            result = by_input.get(input_data)
            if result is None or input_data in seen or label is not None or result["classification"] != prediction:
                raise RuntimeError("Database rows do not match this evaluation's responses")
            seen.add(input_data)
            updates.append((result["true_label"], log_id))
        db.executemany("UPDATE detection_logs SET actual_label=? WHERE id=?", updates)
        db.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8123")
    parser.add_argument("--admin-email", required=True)
    parser.add_argument("--admin-password", default=os.getenv("EVALUATION_PASSWORD"))
    parser.add_argument("--classifier-backend", required=True, choices=["trained", "rule_based"])
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.admin_password:
        parser.error("Set EVALUATION_PASSWORD or --admin-password")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    predictions_path = args.output.with_suffix(".predictions.jsonl")
    if args.output.exists() or predictions_path.exists():
        parser.error("Output exists; choose a new run path to preserve prior evidence")
    test_hash = file_hash(TEST_URLS_PATH)
    urls = pd.read_csv(TEST_URLS_PATH)
    if args.limit:
        urls = urls.head(args.limit)
    with httpx.Client(base_url=args.base_url, timeout=30, trust_env=False) as client:
        response = client.post(f"{API_PREFIX}/auth/login",
                               data={"username": args.admin_email, "password": args.admin_password})
        response.raise_for_status()
        client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
        def get(path):
            response = client.get(f"{API_PREFIX}/{path}")
            response.raise_for_status()
            return response.json()
        runtime = get("metrics/runtime")  # Load model before timing warm inference.
        expected_db = sha256(str(args.db_path.resolve()).encode()).hexdigest()
        if runtime["classifier_backend"] != args.classifier_backend or runtime["database_identity"] != expected_db:
            raise RuntimeError("Server backend/database does not match this evaluation")
        if get("metrics")["total_requests"] or get("whitelist"):
            raise RuntimeError("Evaluation requires a fresh log database and empty allowlist")
        print(f"Verified runtime: {runtime}; evaluating {len(urls)} URLs", flush=True)
        start = time.perf_counter()
        results, rejected = _run_detection_pass(args.base_url, urls)
        elapsed = time.perf_counter() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("x", encoding="utf-8") as stream:
            for record in results:
                stream.write(json.dumps(record) + "\n")
        metrics = classification_metrics(results)
        _label_logs(args.db_path, results)
        logged_metrics = get("metrics")
        if logged_metrics["labeled_sample_size"] != len(results) or any(
            not math.isclose(logged_metrics[k], metrics[k], abs_tol=1e-12)
            for k in ("accuracy", "precision", "recall", "f1_score")
        ):
            raise RuntimeError("API metrics disagree with independently scored HTTP responses")
        valid_urls = pd.Series([r["url"] for r in results]).sample(n=min(500, len(results)), random_state=42).tolist()
        print("Running valid-input concurrent capacity probe", flush=True)
        throughput = _throughput_probe(args.base_url, valid_urls)
        if get("metrics/runtime") != runtime or file_hash(TEST_URLS_PATH) != test_hash:
            raise RuntimeError("Runtime or test data changed during evaluation")
    failures = [r for r in rejected if r["status_code"] not in (400, 422)]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classifier_backend_evaluated": args.classifier_backend, "runtime": runtime,
        "test_set_sha256": test_hash, "predictions_sha256": file_hash(predictions_path),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {p: version(p) for p in ("fastapi", "uvicorn", "scikit-learn", "xgboost", "numpy", "pandas", "httpx")}},
        "source_sha256": {str(p.relative_to(BACKEND_ROOT)): file_hash(p)
                          for root in (BACKEND_ROOT / "app", BACKEND_ROOT / "ml/evaluation")
                          for p in sorted(root.rglob("*.py"))},
        "test_set_size": len(urls), "evaluated_count": len(results), "rejected_count": len(rejected),
        "validation_rejection_count": len(rejected) - len(failures), "unexpected_failure_count": len(failures),
        "rejected_by_label": dict(Counter(r["true_label"] for r in rejected)), "rejected_samples": rejected,
        "classification_metrics": metrics, "metrics_endpoint_response": logged_metrics,
        "sequential_pass_seconds": elapsed, "sequential_req_per_sec": len(results) / elapsed,
        "client_observed_round_trip_ms": latency_summary([r["round_trip_ms"] for r in results]),
        "throughput_probe": throughput,
        "meets_throughput_target": throughput["throughput_rps"] >= 20 and throughput["failed_requests"] == 0,
        "timing_scope": "Warm model; successful-request HTTP round trips. Concurrent latency separate. SQLite, local loopback.",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("evaluated_count", "rejected_count", "classification_metrics", "client_observed_round_trip_ms", "throughput_probe")}, indent=2))
    print(f"Saved {args.output}", flush=True)
    if failures or throughput["failed_requests"]:
        raise SystemExit("Evaluation recorded unexpected failures; inspect the saved report")


if __name__ == "__main__":
    main()
