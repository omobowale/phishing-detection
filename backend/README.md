# Phishing Detection Backend

FastAPI backend implementing the pipeline described in `../phishing-detection-dev-spec.md`.

## Status

End-to-end skeleton: `/detect`, `/whitelist`, `/logs`, `/metrics`, and JWT auth are all
wired up and tested against a **rule-based placeholder classifier**
(`app/pipeline/classifiers.py`). No trained models exist yet — that's phase 2 of the
build order (see spec section 8). Swap in Random Forest / XGBoost / BERT by adding a
new `BaseClassifier` subclass and pointing `CLASSIFIER_BACKEND` at it; no API code
needs to change.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` includes the full stack from the spec (torch/transformers/xgboost
included for the model-training phase). If you only need to run the API skeleton
right now, installing everything except those three is enough.

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
