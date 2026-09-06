# Phishing Detection Backend

FastAPI backend implementing the pipeline described in `../phishing-detection-dev-spec.md`.

## Status

End-to-end skeleton: `/detect`, `/whitelist`, `/logs`, `/metrics`, and JWT auth are all
wired up and tested. Phases 1 (data) and 2 (model development) have been run, plus four
rounds of external security/validation/evaluation/dataset-rigor review and fix passes
(2026-09-06) — see `ml/README.md` section 14.5 for the one consolidated results table to
cite, and the rest of the document for the full audit trail. A real hyperparameter search
(not just hand-picked configs) was also run and did not improve on the default config
(§14.4). Short version: **neither classifier backend currently meets the spec's detection
targets, but the performance NFRs (latency, throughput) are met.** The live default,
`rule_based` (`app/pipeline/classifiers.py`), measured **1.69% recall** in a formal
held-out evaluation (`ml/evaluation/results.json`) — it does not function as a phishing
detector in any meaningful sense. A trained URL model exists at
`ml/saved_models/url_classifier.joblib`; evaluated properly (domain-grouped train/test
split via a Public Suffix List-aware grouping, so no domain is shared between them), it
measures **F1 0.839, precision 0.841, recall 0.837**, still below the spec's F1≥0.90
target, and still fails a bare-domain/well-known-site sanity check. This is now an open
problem requiring either a better dataset or a narrower thesis scope (see `ml/README.md`
section 8) — not something the codebase alone can resolve. On the brighter side: the
same fresh evaluation confirms average latency ~13ms (p99 ~39ms) and **41.5 successful
req/s under concurrent load, clearing the spec's 20 req/s target** — an earlier
benchmark bug had this failing at 5.51 req/s, since corrected (§12.5/§12.7). Swap
classifiers by adding a new `BaseClassifier` subclass and pointing `CLASSIFIER_BACKEND`
at it; no API code needs to change — but read `ml/README.md` first.

## Deployment configuration

- **`SECRET_KEY`**: `app/main.py` refuses to start with `ENVIRONMENT=production` if this
  is still the insecure code default. Generate a real one:
  `python -c "import secrets; print(secrets.token_hex(32))"`.
- **`ENVIRONMENT`**: `development` (default, just warns about the default secret) or
  `production` (hard-fails startup instead).
- **`CLASSIFIER_BACKEND`**: `rule_based` or `trained` — see Status above before changing this.
- **NLTK data**: run `python -m scripts.download_nltk_data` once during setup/deploy.
  The app deliberately never downloads NLTK corpora at request time (it used to, adding
  1.4-2.4s to a request's first email-preprocessing call) — without this step, email
  preprocessing falls back to a smaller built-in stop-word list instead of failing.
- **Raw email storage**: `DetectionLog.input_data` stores the full raw `email_text` a
  caller submits, indefinitely, in plaintext, with no redaction or retention policy.
  Acceptable for local dev/thesis demo data; revisit before this ever holds real users'
  email content.
- SQLite (WAL mode, enabled in `app/db/base.py`) currently meets the spec's throughput
  target (41.5 successful req/s at 40 concurrent clients — see `ml/README.md` section
  12.7). An earlier benchmark bug made this look like a real SQLite bottleneck (5.51
  req/s); that was the benchmark, not the database. Still switch to Postgres before a
  real multi-user deployment — SQLite's single-writer model will eventually become a
  real ceiling at higher concurrency than was tested here.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env
python -m scripts.download_nltk_data   # one-time; see "Deployment configuration" above
```

`requirements.txt` includes the full stack from the spec (torch/transformers/xgboost
included for the model-training phase). If you only need to run the API skeleton
right now, installing everything except those three is enough.

For the full ML pipeline (dataset download, feature build, training, live-API
evaluation) as a reproducible command sequence, see `ml/README.md` section 15.

## Run

```bash
uvicorn app.main:app --reload
```

API docs at `http://localhost:8000/docs`. Endpoints are under `/api/v1`.

## Test

```bash
pytest
```

## Notes / deviations from the spec's ERD

- `User` gained a `hashed_password` column — required for JWT auth, not in the
  original ERD.
- `Detection_Log` gained a nullable `actual_label` column — `GET /metrics` needs a
  ground-truth label to compute accuracy/precision/recall/F1; without it there's
  nothing to compare predictions against. Populate it during the phase-5 evaluation
  run against a labeled test set.
- Dev DB tables are created automatically on startup (`Base.metadata.create_all`).
  Switch to Alembic migrations before this touches a shared/production database.
