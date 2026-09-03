# AI-Powered Phishing Detection Framework — Developer Handoff Spec

**Project:** Real-time phishing detection using hybrid ML (Random Forest / XGBoost) + NLP (fine-tuned BERT)
**Author:** Omobowale Sikiru Otuyiga — MIT thesis, MIVA Open University
**Purpose of this doc:** enough detail for a developer to start building the backend (Python/FastAPI) and frontend (React) without re-reading the full thesis.

---

## 1. What we're building

A system that takes a URL and/or a block of email text and returns a phishing/legitimate classification with a confidence score, in under 500ms per request. It combines:

- **Structured feature classifiers** (Random Forest, XGBoost) on URL and email-header features
- **A fine-tuned BERT model** on email body text for semantic/contextual signals
- **An admin-managed whitelist** that short-circuits the pipeline for trusted domains
- **A logging layer** that records every detection for later evaluation (accuracy, precision, recall, F1, latency)

There are two deliverables: a Python backend (FastAPI) that does the actual classification, and a React frontend that lets a user submit inputs and lets an admin manage the whitelist and view logs.

---

## 2. Architecture (6 layers)

```
Input Layer (FastAPI endpoints)
   → whitelist pre-check (bypasses pipeline if domain matches)
Preprocessing Module (URL normalization / text tokenization, stop-word removal, lemmatization)
Feature Extraction Module
   → URL features (length, subdomain count, IP-as-domain, special chars, brand keywords)
   → Email header features (SPF/DKIM/DMARC, reply-to mismatch, routing anomalies)
   → NLP features (TF-IDF for classical models, BERT embeddings for transformer model)
ML Classification Engine (Random Forest / XGBoost / fine-tuned BERT)
Real-Time Inference Engine (FastAPI async handlers)
Output Layer (classification + confidence score + log entry, JSON response)
```

---

## 3. Tech stack

**Backend**
- Python 3.10+
- FastAPI (async) + Uvicorn
- scikit-learn (Random Forest baseline, evaluation utilities)
- XGBoost
- Hugging Face Transformers (BERT fine-tuning / inference) — consider DistilBERT for latency-sensitive paths
- NLTK (tokenization, stop-words, lemmatization)
- Pandas / NumPy (data handling)
- A relational DB for logs + whitelist (Postgres recommended; SQLite fine for dev)
- SQLAlchemy or similar ORM

**Frontend**
- React (Vite or Next.js — Next.js if you want SSR/admin routing out of the box)
- A form for submitting URL / email text for classification
- An admin dashboard: whitelist CRUD, detection log viewer, basic metrics view (accuracy/precision/recall/F1/latency once available)
- Charting library (recharts or similar) for the metrics dashboard

---

## 4. Data model (from the ERD)

**User**
| field | type | notes |
|---|---|---|
| user_id | PK | |
| name | string | |
| email | string | |
| role | enum | `admin` \| `end_user` |

**Detection_Log**
| field | type | notes |
|---|---|---|
| id | PK | |
| user_id | FK → User | |
| input_type | enum | `url` \| `email_text` \| `both` |
| input_data | text | raw submitted input |
| prediction | enum | `phishing` \| `legitimate` |
| confidence_score | float | 0–1 |
| processing_time | float | ms — primary latency measurement field |
| timestamp | datetime | |

**Whitelist**
| field | type | notes |
|---|---|---|
| id | PK | |
| domain | string | |
| added_by | FK → User | admin only |
| date_added | datetime | |

---

## 5. API endpoints

All under a `/api/v1` prefix, JSON in/out.

### Detection
- `POST /detect`
  - Body: `{ "url"?: string, "email_text"?: string }` — at least one required
  - Flow: validate input → whitelist check on URL domain (if present) → if whitelisted, return immediately → else preprocess → extract features → classify → log → return
  - Response:
    ```json
    {
      "classification": "phishing" | "legitimate",
      "confidence_score": 0.962,
      "processing_time_ms": 312,
      "whitelisted": false
    }
    ```

### Whitelist (admin-only — needs auth/role check)
- `GET /whitelist` — list trusted domains
- `POST /whitelist` — body `{ "domain": string }`
- `DELETE /whitelist/{id}`

### Logs
- `GET /logs` — paginated, filterable by date range / classification / user
- `GET /logs/{id}`

### Evaluation / metrics
- `GET /metrics` — returns accuracy, precision, recall, F1, average latency, throughput computed from logged (labeled) test data. This endpoint is what the results chapter and the frontend metrics dashboard both read from — don't hardcode numbers here.

### Auth
- Basic auth or JWT is fine for a thesis-scope project. Admin-only endpoints must check `role == admin`. Don't skip this — it's called out as a non-functional requirement (whitelist tampering = full pipeline bypass for an attacker).

---

## 6. Error handling requirements

- Malformed/empty input → `400` with a clear message, never an unhandled exception (this maps to NFR7 in the thesis).
- Whitelist admin endpoints → `403` for non-admin callers.
- Model/pipeline failures → `500` with a logged error, not a silent failure — every request should still produce a log entry or a clear error, since the logs are the evaluation data source.

---

## 7. Suggested repo structure

```
/backend
  /app
    /api          # FastAPI routers: detect.py, whitelist.py, logs.py, metrics.py
    /core          # config, auth
    /pipeline
      preprocessing.py
      feature_extraction_url.py
      feature_extraction_email.py
      classifiers.py       # loads trained RF/XGBoost/BERT models
    /models        # SQLAlchemy models: user.py, detection_log.py, whitelist.py
    /schemas       # Pydantic request/response models
    main.py
  /ml
    /training      # scripts for training RF, XGBoost, fine-tuning BERT
    /notebooks     # Jupyter experimentation
    /saved_models  # serialized trained models (.pkl / HF checkpoint dirs)
  /tests
  requirements.txt

/frontend
  /src
    /pages (or /app if Next.js)   # detect page, admin dashboard, logs view
    /components
    /api                          # fetch wrappers for backend endpoints
  package.json
```

---

## 8. Build order (maps to the thesis's 5-phase Agile plan)

1. **Data collection & preprocessing** — get PhishTank, UCI Phishing Websites, Enron, Kaggle datasets loading and cleaned; build the shared preprocessing pipeline.
2. **Model development** — train/evaluate Random Forest and XGBoost on structured features; fine-tune BERT on email text. Keep these as separate, swappable components behind a common classifier interface.
3. **Backend integration** — wire the trained models into the FastAPI pipeline described above; get `/detect` working end-to-end with whitelist bypass and logging.
4. **Frontend** — detection form first, then admin dashboard (whitelist CRUD + log viewer), then metrics view once `/metrics` has real data behind it.
5. **Evaluation** — run the held-out test set through the deployed pipeline, populate `/metrics`, and use that (real) output for the thesis results chapter — not before.

---

## 9. Performance targets to build against

- F1-score ≥ 0.90
- Precision ≥ 0.91
- Recall ≥ 0.90
- Average inference latency ≤ 500ms per request
- Throughput ≥ 20 requests/second via the API

These are the numbers the pipeline needs to be capable of hitting — treat them as engineering targets, not as anything to hardcode into a demo response.
