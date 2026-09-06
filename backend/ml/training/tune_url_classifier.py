"""Proper hyperparameter search for the URL classifier, on top of the current
clean, domain-grouped dataset (build_features.py's default SPLIT_STRATEGY=grouped).

Why this is a separate script from train_url_classifier.py: the ad-hoc
regularization experiments recorded in ml/README.md section 5 were run BEFORE
the dataset fixes in sections 3.4-3.5 and 10 (label cleaning, conflict
quarantine, domain-grouped splitting) -- "tuning didn't help" was a real
finding on that older, leakier data, but it doesn't automatically carry over
to the current, cleaner dataset. This runs an actual search (not a few
hand-picked configs) to check.

Methodology: train+val rows are pooled into one cross-validation set (more
data for the search than a single train/val split point-estimate), split into
folds with StratifiedGroupKFold -- grouped by registrable domain (so no domain
crosses a fold boundary, consistent with build_features.py's SPLIT_STRATEGY=grouped
rationale) and approximately stratified by label within that constraint. The
held-out test split is touched exactly once, at the end, to report the final
number -- never during the search itself.

Usage:
    python -m ml.training.tune_url_classifier
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.training.build_features import _host_of, _registrable_domain  # noqa: E402
from ml.training.train_url_classifier import (  # noqa: E402
    FEATURE_COLUMNS,
    MONOTONE_CONSTRAINTS,
    TARGET_F1,
    TARGET_PRECISION,
    TARGET_RECALL,
    _baseline_metrics,
    _evaluate,
    _sanity_check,
)

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BACKEND_ROOT / "ml" / "data" / "processed"
# tuning is inherently a comparison/experiment run -- never touches the
# primary saved_models/url_classifier.joblib, always lives in experiments/
# (see train_url_classifier.py's EXPERIMENTS_DIR comment for why).
EXPERIMENTS_DIR = BACKEND_ROOT / "ml" / "saved_models" / "experiments"

N_ITER = 25
N_SPLITS = 4
RANDOM_STATE = 42

# _registrable_domain/_host_of imported from build_features.py rather than
# redefined here -- an earlier version had its own copies using the old naive
# last-two-labels heuristic, which could disagree with the PSL-aware grouping
# build_features.py's split now uses, undermining the whole point of
# cross-validating with domain-disjoint folds.


def main() -> None:
    features_path = PROCESSED_DIR / "url_features.parquet"
    if not features_path.exists():
        print(f"No feature matrix at {features_path}. Run: python -m ml.training.build_features")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    cv_pool = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_cv = cv_pool[FEATURE_COLUMNS]
    y_cv = (cv_pool["label"] == "phishing").astype(int)
    groups = cv_pool["url"].astype(str).apply(lambda u: _registrable_domain(_host_of(u)))

    X_test = test_df[FEATURE_COLUMNS]
    y_test = (test_df["label"] == "phishing").astype(int)

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print(f"CV pool: {len(cv_pool)} rows across {groups.nunique()} distinct registrable domains")
    print(f"Searching {N_ITER} configs x {N_SPLITS} domain-disjoint folds per model ...\n")

    # n_jobs=-1 on the outer RandomizedSearchCV parallelizes across the
    # n_iter*cv fold fits; leaving n_jobs=-1 on the estimators too would
    # oversubscribe CPU cores (each of many parallel fits also trying to spawn
    # its own thread pool) and slow the whole search down, not speed it up.
    searches = {
        "random_forest": RandomizedSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            param_distributions={
                "n_estimators": randint(150, 600),
                "max_depth": [None, 6, 10, 16, 24, 32],
                "min_samples_leaf": randint(1, 15),
                "min_samples_split": randint(2, 20),
                "max_features": ["sqrt", "log2", None],
                "class_weight": [None, "balanced"],
            },
            n_iter=N_ITER,
            scoring="f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        ),
        "xgboost": RandomizedSearchCV(
            XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
                eval_metric="logloss",
                monotone_constraints=MONOTONE_CONSTRAINTS,
            ),
            param_distributions={
                "n_estimators": randint(150, 600),
                "max_depth": randint(3, 12),
                "learning_rate": uniform(0.01, 0.29),
                "min_child_weight": randint(1, 15),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "reg_lambda": uniform(0.1, 5.0),
                "gamma": uniform(0.0, 2.0),
            },
            n_iter=N_ITER,
            scoring="f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        ),
    }

    results = {}
    for name, search in searches.items():
        print(f"Tuning {name} ...")
        search.fit(X_cv, y_cv, groups=groups)
        print(f"  best CV F1: {search.best_score_:.4f}")
        print(f"  best params: {search.best_params_}")
        results[name] = search

    best_name = max(results, key=lambda n: results[n].best_score_)
    best_model = results[best_name].best_estimator_
    best_cv_f1 = results[best_name].best_score_

    test_metrics = _evaluate(best_model, X_test, y_test)
    baseline_metrics = _baseline_metrics(y_cv, y_test)
    sanity = _sanity_check(best_model)

    print(f"\nSelected {best_name} (best CV F1 {best_cv_f1:.4f} across domain-disjoint folds)")
    print(f"Best params: {results[best_name].best_params_}")
    print(f"Baseline (always predict '{baseline_metrics['majority_class']}'): {baseline_metrics}")
    print(f"Test metrics (held out, touched once): {test_metrics}")
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

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENTS_DIR / f"url_classifier_{best_name}_tuned.joblib"
    joblib.dump({"model": best_model, "feature_columns": FEATURE_COLUMNS, "model_name": best_name}, out_path)
    print(f"\nSaved (comparison artifact, NOT the primary model) to {out_path}")

    metrics_path = EXPERIMENTS_DIR / "url_classifier_metrics_tuned.json"
    metrics_path.write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                                 for k, v in results[best_name].best_params_.items()},
                "best_cv_f1": best_cv_f1,
                "cv_folds": N_SPLITS,
                "search_iterations": N_ITER,
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
