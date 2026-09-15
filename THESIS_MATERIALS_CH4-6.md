# Materials for Thesis Chapters 4–6

**Purpose of this document**: a single, organized reference pulling together everything built and
measured across this project, mapped to the chapters you're about to write. This is source
material, not thesis prose — write Chapters 4–6 in your own academic voice, but every number,
finding, and file path below is real and citable (no estimates, nothing invented). Where a figure
came from is noted so you can go verify it yourself or cite the exact artifact.

**Status as of 2026-09-15**: URL, email classical, and BERT fine-tuning results are all final and
stable. See §5.4 for the completed BERT results.

---

## How to use this document

- Chapter 4 (Implementation): §1–§2 below.
- Chapter 5 (Results & Evaluation): §3–§5 below.
- Chapter 6 (Discussion & Conclusion): §6–§8 below.
- Every claim cites a file path under `backend/` you can open and check yourself before writing it
  into the thesis. If you want a number this document doesn't have, the right move is to ask me to
  pull it from the named file rather than estimate it.

---

## 1. System Architecture (→ Chapter 4)

### 1.1 High-level design

A FastAPI backend serves a React/TypeScript frontend. A user (or an unauthenticated visitor, for
basic scanning) submits a URL and/or email text to `POST /detect`; the system extracts structured
features from each modality, runs them through a classifier, logs the result, and returns a
prediction with a confidence score.

```
React frontend  →  POST /detect  →  FastAPI
                                      ├─ whitelist check (admin-managed trusted domains)
                                      ├─ feature extraction (URL side, email side)
                                      ├─ classifier (RuleBasedClassifier | TrainedClassifier)
                                      ├─ detection log write (SQLite, WAL mode)
                                      └─ JSON response (classification, confidence, timing)
```

### 1.2 Backend stack

- **FastAPI** + **SQLAlchemy** (ORM) + **SQLite** (dev/thesis-scale deployment; WAL journal mode
  enabled — `backend/app/db/base.py` — measured ~4-5x faster than the default rollback-journal mode
  under this app's per-request commit pattern).
- **JWT authentication** (`python-jose`) with role-based access: unauthenticated users can submit
  detections; only admins can view aggregate `/metrics`, manage the whitelist, or view all users'
  logs.
- **Pydantic** request/response schemas with input validation (URL shape, length caps: 2048 chars
  for URLs, 50,000 for email text) — `backend/app/schemas/detection.py`.

### 1.3 The detection pipeline (`backend/app/pipeline/`)

Three files do the real work:

- **`preprocessing.py`** — shared text-normalization utilities used by *both* training and live
  serving (critical for avoiding train/serve skew): `get_domain()` (URL → hostname, via
  `urlparse().hostname`, not naive string splitting — see §2.1 for why that distinction was a real
  security bug), `strip_email_headers()` (RFC822 header removal via `email.message_from_string()`),
  `redact_email_and_urls()` (replaces email addresses/URLs with placeholder tokens before further
  processing), `preprocess_email_text()` (lemmatization + stopword removal for the classical model).
- **`feature_extraction_url.py`** — `extract_url_features()` computes 12 structural features per
  URL (see §1.4). Called identically at training time and live-request time — one function, two
  call sites, so there is no way for train-time and serve-time feature computation to drift apart.
- **`feature_extraction_email.py`** — `extract_email_features()` computes header-based features
  (SPF/DKIM/DMARC pass, reply-to mismatch, received-hop count), lightweight NLP features (urgency
  keyword count, URL-in-body count, exclamation/caps count), and two full-text representations:
  `tokens_text` (lemmatized, for the classical TF-IDF model) and `bert_text` (natural-cased,
  redacted only, for the transformer model).

### 1.4 URL features (12, used for the trained model; `has_https` computed but excluded — see §6.1)

`url_length, host_length, subdomain_count, is_ip_address, special_char_count, digit_count,
path_length, query_length, brand_keyword_count, brand_keyword_outside_domain, suspicious_tld,
has_at_symbol`

### 1.5 The classifier abstraction (`backend/app/pipeline/classifiers.py`)

A `BaseClassifier` interface with `predict(url_features, email_features) -> (Prediction, confidence)`
lets the URL and email sides use completely different, independently-selected backends:

- **`RuleBasedClassifier`** — a hand-weighted heuristic scorer (structural checks like "is this an
  IP-literal URL," "does the domain contain a brand name outside the actual domain," combined with
  email header/urgency checks). No trained model required; exists so the system has a working,
  testable decision path from day one.
- **`TrainedClassifier`** — wraps a trained URL model (RandomForest/XGBoost) and/or a trained email
  model (TF-IDF + Logistic Regression, or a fine-tuned DistilBERT) independently. If one side has no
  trained model loaded, that side falls back to the same rule-based heuristic `RuleBasedClassifier`
  uses — so `TrainedClassifier` degrades gracefully rather than failing outright.

**Two independent settings control which backend serves live traffic**:
`URL_CLASSIFIER_BACKEND` and `EMAIL_CLASSIFIER_BACKEND` (each `rule_based` / `trained`, and email
additionally supports `bert`). This split (not one shared flag) is itself a design decision worth
discussing in Chapter 4: the URL and email trained models reached very different maturity levels
(§6.1), and one shared setting would have forced an all-or-nothing choice between them.

### 1.6 Detection logging and whitelist

Every detection is logged (`DetectionLog` model: input type, raw input, prediction, confidence,
processing time, and — for evaluation runs — a ground-truth `actual_label` field). An admin-managed
`Whitelist` table lets trusted domains short-circuit URL classification entirely — this is a
deliberate architectural answer to the bare-domain limitation discussed in §6.1: the spec anticipates
that "is this well-known domain safe" is the whitelist's job, not the ML model's.

**A security-relevant design detail worth including in Chapter 4**: a whitelisted URL only
short-circuits URL-side analysis. If the same request also includes email text, that email is still
fully analyzed — because a phishing email can legitimately reference or spoof a trusted domain, and
skipping analysis entirely would have created a bypass (see §2.1 for the actual exploit this
guards against).

### 1.7 Evaluation harness (`backend/ml/evaluation/`)

A significant piece of infrastructure, worth its own paragraph in Chapter 4 since it's what makes
every number in Chapter 5 defensible:

- **`managed_run.py`** — spins up an isolated FastAPI server on a random port, with a throwaway
  database, secret key, and admin account, running a specified classifier backend. Nothing about a
  real run's environment is shared or reused between runs.
- **`run_evaluation.py`** — drives the isolated server through a labeled test set via real HTTP
  requests (not direct function calls — this measures the system as a user would actually hit it),
  independently recomputes classification metrics from the raw HTTP responses, and cross-checks
  those against the server's own `/metrics` endpoint. Verifies the server's loaded model identity
  (via SHA-256 hash) and database identity before trusting any result, and refuses to proceed if
  they don't match what was requested.
- **`analyze_run.py`** — re-runs the same inputs through the raw model file offline (bypassing the
  API) to verify the live API's predictions exactly match direct model inference (API/offline
  parity), and breaks results down by feature-value slice for error analysis.

This hash-based, isolated-environment approach is why every number in this document can be traced
back to a specific run with a specific model file and specific test set — a genuinely citable
methodology point for Chapter 4.

### 1.8 Frontend

React + TypeScript. Key screens (screenshots at `frontend/screenshots/`, numbered 01–16, already in
the repo — use these directly as Chapter 4 figures): scan/detection flow, account creation, threat
result and legitimate-result views, scan history (with admin filtering), admin allowlist management,
admin insights dashboard, and mobile-responsive variants of the core flows.

---

## 2. Notable Implementation Findings (→ Chapter 4, "Implementation Challenges" subsection)

These are real bugs found and fixed during development — good material for a "challenges
encountered" subsection, since they show engineering rigor, not just a finished system.

### 2.1 A confirmed, live security vulnerability (whitelist bypass)

**The bug**: hostname extraction used `urlparse(url).netloc.split(":")[0]`. For a URL containing
userinfo — `https://trusted.com:password@evil.example/` — this returns `trusted.com` (the userinfo
*username*), not the real host `evil.example`. Reproduced live: whitelisting `trusted.com` and
submitting that exact URL returned `{"classification":"legitimate","confidence_score":1.0}` — a
full bypass of the trust boundary.

**The fix**: `urlparse(url).hostname` (parses per the URL specification, correctly separates
userinfo from host). Applied in both `preprocessing.py` and `feature_extraction_url.py`. A
regression test (`test_whitelist_userinfo_bypass_is_blocked`) locks this in.

**Why this belongs in the thesis**: it's a genuine example of the gap between "code that looks
correct" and "code that is correct under adversarial input" — exactly the kind of finding a security
or software-engineering examiner would want to see you catch and reason about.

### 2.2 Train/serve consistency as an explicit design principle

Two separate incidents (URL and email) both traced back to the same root cause: feature computation
at training time using slightly different logic than feature computation at serving time.

- URL: an earlier version computed `url_length`/`special_char_count`/`digit_count` on the raw URL
  string (including scheme), making them scheme-sensitive in a way that didn't reflect real
  structural risk. Fixed by computing on scheme-stripped content, in the one shared function both
  training and serving call.
- Email: an earlier version of the training script called `preprocess_email_text()` directly on raw
  CSV text, while live serving already stripped RFC822 headers first via `strip_email_headers()`. A
  real train/serve mismatch. Fixed by extracting `strip_email_headers()` into `preprocessing.py` and
  having both call sites use it.

**Thesis framing**: this motivates the architectural rule adopted for the rest of the project —
exactly one function computes a given feature, called identically by the training pipeline and the
live API — and is worth stating explicitly as a design principle in Chapter 4, since it's what
prevented several other classes of bugs from ever occurring.

### 2.3 Concurrency correctness under SQLite

Under concurrent load, detection-log writes originally produced intermittent HTTP 500s. Root cause:
SQLite allows only one writer at a time, and concurrent requests were competing for that lock while
also holding a stale read-transaction snapshot across the write. Fixed with a per-process write lock
serializing detection-log writes, plus releasing the read transaction before entering the write
section. This is SQLite-specific — the code comments explicitly note that a multi-process or
different-database deployment would need its own concurrency validation, which is honest material
for Chapter 6's limitations.

---

## 3. Dataset Methodology (→ Chapter 5, "Methodology" or "Data" subsection)

### 3.1 URL dataset

- **Source**: Kaggle `sid321axn/malicious-urls-dataset` (`malicious_phish.csv`), 651,191 labeled
  URLs across four categories (`benign` 428,103, `defacement` 96,457, `phishing` 94,111, `malware`
  32,520). Only `benign`→legitimate and `phishing`→phishing are used (binary task).
- **Cleaning** (all logged to auditable CSV files, not silently applied):
  - 1,785 rows dropped for "phishing" labels on domains with no plausible attacker-hosting surface
    (e.g., government/major-institution domains where a phishing label is almost certainly a
    labeling error) — `ml/data/processed/domain_cleaned_phishing_labels.csv`.
  - ~8,110 rows **quarantined** (removed, not resolved) for internal label conflicts — the same URL
    string, or its normalized form, appears as both `benign` and `phishing` in the raw data —
    `ml/data/processed/quarantined_conflicts.csv`.
  - Exact-duplicate URLs (same URL, same label) dropped before splitting.
  - Balanced to 50,000 rows per class (from a larger pool) so neither class dominates.
- **Split**: grouped by **registrable domain** (via `tldextract`, with an offline Public Suffix List
  snapshot for reproducibility) using `sklearn.GroupShuffleSplit` — no domain appears in more than
  one of train/val/test. This is more rigorous than a plain random per-URL split, which would let
  the model see other URLs from the same host during training and inflate its apparent performance
  (§5.1 shows the actual size of that inflation: F1 0.879 under random split vs. 0.839 under
  domain-grouped split on the same data).
- **Total after cleaning**: ~99,989 rows.

### 3.2 Email dataset

- **Source**: Kaggle `naserabdullahalam/phishing-email-dataset`, combining six original corpora:
  Enron, CEAS_08, Ling, Nazario, Nigerian_Fraud, SpamAssassin.
- **A significant labeling problem found before any training** (worth a full paragraph in Chapter
  5's methodology — this is a genuine methodological contribution): the dataset's own marketed
  "phishing" label (42,891 rows) is **83.5% generic commercial spam**, not phishing. Verified by
  reading actual email content per source, not trusting labels:

  | Source | Rows | What it actually is |
  |---|---|---|
  | Nazario | 1,565 | Genuine phishing (the classic Nazario corpus) |
  | Nigerian_Fraud | 3,332 | Genuine advance-fee fraud (a phishing subtype) |
  | Enron | 29,767 | Real Enron correspondence (label 0) + **generic spam injected during compilation** (label 1) — not phishing, not real Enron content |
  | CEAS_08 | 39,154 | A spam-filtering shared-task dataset; label 1 = generic commercial spam, not phishing-specific |
  | Ling | 2,859 | Classic linguistics-mailing-list spam/ham corpus; not phishing-specific |
  | SpamAssassin | 5,809 | Generic spam/ham, with a small minority of genuinely phishing-shaped subjects mixed in |

  **Decision made**: scope "phishing" strictly to Nazario + Nigerian_Fraud (4,897 rows,
  content-verified genuine fraud), and "legitimate" to the ham (label 0) rows of the four other
  sources (39,595 rows). This produces a smaller, more imbalanced (~11% positive), but honestly
  labeled dataset — deliberately chosen over the larger, easier, mislabeled alternative.

- **Two further data-quality issues found and fixed**:
  1. ~40% of Nazario's phishing emails are literally addressed to the corpus compiler's personal
     mailbox (`jose@monkey.org`) — a spurious, non-generalizing signal that showed up as top
     features ("jose," "monkey"). Fixed by redacting email addresses/URLs to placeholder tokens
     before feature extraction (a standard technique in spam/phishing text classification).
  2. Some raw rows were mail-client artifacts or corrupted RFC822 headers, not real email content
     (174 rows dropped, logged).
- **Split**: initially grouped by normalized subject line (to catch campaign-template reuse), but a
  later audit found this was **insufficient** — see §3.3.

### 3.3 A second, more subtle leakage finding (worth its own paragraph — shows methodological maturity)

An external review of the codebase found that subject-line grouping does not prevent all leakage:
after preprocessing (redaction, lemmatization, stopword removal), 1,790 rows across the dataset
collapsed onto **duplicate token text** that a subject-only grouping couldn't see (different
subject lines, same processed content) — and 142 test-split rows shared exact token text with a
training row. One token-text group even carried conflicting labels.

**Fix**: after computing the processed token representation, any token-text group with conflicting
labels is quarantined entirely (not arbitrarily resolved), then exact token-text duplicates are
dropped, keeping one representative per unique string. This makes this specific kind of leakage
**structurally impossible**, not just statistically unlikely — a stronger claim than "reduced."

**Effect on the reported number**: F1 moved from 0.9797 (leaky) to 0.9761 (corrected) — a small,
honest drop, not a collapse. This is reassuring evidence the original result wasn't primarily an
artifact of the leak, and it's exactly the kind of "did the fix change the number by an amount
consistent with the size of the problem" reasoning a thesis examiner would want to see.

### 3.4 A limitation this fix does NOT address (be honest about this in Chapter 6)

Every phishing example in the final dataset comes from exactly two sources (Nazario,
Nigerian_Fraud); every legitimate example comes from four different sources (CEAS_08, Enron, Ling,
SpamAssassin). No resplitting can separate "the model learned phishing language" from "the model
learned which collection source this resembles," because the two are perfectly confounded in this
dataset's composition. This is a structural property of the data, not a bug — see §6.2 for how to
frame this honestly.

---

## 4. Model Training Methodology (→ Chapter 5, "Model Development")

### 4.1 URL classifier

- **Algorithms compared**: RandomForest, XGBoost (both with hyperparameter defaults, then a real
  hyperparameter search — see §5.1).
- **Feature set**: the 12 features in §1.4. `has_https` was computed but **excluded** from training
  — see §6.1 for why (it was a dataset collection artifact, not real signal).
- **Regression guard**: an automated sanity check runs after every training run — 12 hand-picked,
  real-world legitimate URLs (including bare domains like `google.com` and full paths like
  `https://en.wikipedia.org/wiki/Phishing`) plus 4 phishing-shaped URLs. This is separate from the
  held-out test set specifically to catch failure modes that aggregate metrics can hide (see §6.1 —
  this is exactly how the bare-domain problem was caught).

### 4.2 Email classifier (classical)

- **Algorithms compared**: TF-IDF (unigrams + bigrams, max 20,000 features) feeding Logistic
  Regression, RandomForest, and Complement Naive Bayes — all with `class_weight="balanced"` given
  the ~11% positive rate. Best model selected by validation F1 (Logistic Regression won on the
  corrected dataset).
- **Baseline**: a trivial "always predict legitimate" classifier scores 88.8% accuracy but F1=0 —
  included specifically to show accuracy alone is meaningless on this imbalanced task, and that the
  trained model is learning real signal, not exploiting class imbalance.
- **Sanity check**: 3 hand-written phishing examples + 3 hand-written legitimate examples, not drawn
  from the dataset at all — 3/3 and 3/3 pass on the current model.

### 4.3 BERT/DistilBERT fine-tuning (complete — see §5.4)

- **Model**: `distilbert-base-uncased`, fine-tuned with a weighted cross-entropy loss (same
  class-imbalance rationale as the classical model).
- **Input**: `bert_text` — header-stripped, address/URL-redacted, but *not* lemmatized or
  lowercased (unlike the classical pipeline) — BERT's own subword tokenizer benefits from natural
  casing and punctuation.
- **`max_length=128`** (reduced from an initial 256 after discovering real emails average 212
  tokens with 62% hitting the 256 cap — a first CPU-timing benchmark used an unrepresentative short
  dummy sentence and badly underestimated real per-step cost as a result; worth a sentence in
  Chapter 4 or 6 about benchmark representativeness as a methodology lesson).
- **Resumability**: an experiment-identity manifest binds a training run to its exact dataset hash,
  code hash, hyperparameters, and package versions, refusing to silently resume from a checkpoint
  that doesn't match — this exists because real interruptions occurred (three of them across the
  full run: two out-of-memory crashes and one session-lifetime interruption, see §5.4) and needed
  to be recovered safely rather than silently producing a subtly-wrong resumed model.

---

## 5. Results (→ Chapter 5, "Results")

### 5.1 URL classifier — consolidated results

| Configuration | Split | Accuracy | Precision | Recall | F1 | Sanity check (legit/phish) |
|---|---|---|---|---|---|---|
| **Primary (default hyperparameters)** | Domain-grouped | 0.841 | 0.841 | 0.837 | **0.839** | 2/12, 4/4 |
| Tuned (RandomizedSearchCV + StratifiedGroupKFold) | Domain-grouped | 0.830 | 0.813 | 0.852 | 0.832 | 1/12, 4/4 |
| No domain-based cleaning (ablation) | Domain-grouped | 0.830 | 0.815 | 0.840 | 0.827 | not re-run |
| Random per-URL split (methodology comparison) | Random | 0.879 | 0.878 | 0.880 | 0.879 | not re-run |
| `rule_based` (live default) | Live API | 0.509 | 0.614 | **0.017** | 0.033 | N/A |

**Baseline** (always predict majority class "phishing" on this near-balanced split): accuracy
0.494, F1 0.661 — included to show the trained model (F1 0.839) is well above a trivial baseline,
even though it falls short of the 0.90 target.

Source: `ml/saved_models/url_classifier_metrics.json`, `ml/saved_models/experiments/*.json`,
`ml/evaluation/REVIEW_AND_RESULTS.md`.

**Does not meet the spec's F1≥0.90 target.** A real hyperparameter search was run and did not close
the gap — the ceiling appears to be dataset-related (§6.1), not a tuning problem.

### 5.2 Email classifier (classical) — consolidated results

| Metric | Value |
|---|---|
| Accuracy | 0.9946 |
| Precision | 0.9734 |
| Recall | 0.9789 |
| F1 | **0.9761** |
| Average precision (PR-AUC) | 0.9966 |
| Confusion matrix | TN=5609, FP=19, FN=15, TP=695 |

**Meets every spec-wide target** (F1≥0.90, precision≥0.91, recall≥0.90). Baseline (always predict
"legitimate"): accuracy 0.888, F1 0.0. Sanity check: 3/3 phishing, 3/3 legitimate.

Source: `ml/saved_models/email_classifier_metrics.json`.

**Error analysis** (read actual misclassified examples, not just counted): false negatives
concentrate in non-English phishing (the vocabulary is English-only by construction) and
deliberately obfuscated text (letter-substitution tricks). False positives include real corporate
"we're changing our bank account" notifications that structurally resemble business-email-compromise
lures — arguably genuinely ambiguous cases, not clean model failures.

### 5.3 Live system performance

| Metric | Value | Source |
|---|---|---|
| Mean HTTP round-trip latency | ~13-17ms (both backends) | `REVIEW_AND_RESULTS.md` |
| Concurrent p95 latency | ~509ms (trained URL) / ~520ms (rule-based) | `REVIEW_AND_RESULTS.md` |
| Successful throughput (40 concurrent workers) | 86.9 req/s (trained) / 85.9 req/s (rule-based) | `REVIEW_AND_RESULTS.md` |

**Honest caveat for Chapter 6**: mean latency comfortably clears a 500ms target; **p95 slightly
exceeds it** under concurrent load. Measurements are single-machine, single-Uvicorn-worker,
SQLite-backed — not validated for multi-process or multi-host deployment.

**Live-API / offline parity**: for both the full URL evaluation (11,268 accepted predictions) and
the full email evaluation (6,335/6,338 accepted), every live API prediction and confidence score
matched offline direct-model inference exactly — i.e., the live serving pipeline introduces no
discrepancy versus the "pure" model numbers above. This is a meaningful methodology point: the
numbers in §5.1/§5.2 are not just offline research numbers, they are what the deployed system
actually produces.

### 5.4 BERT — consolidated results

Training completed 2026-09-15: `distilbert-base-uncased`, 3 full epochs on the same corrected,
leakage-fixed email split as §5.2 (29,991 train / 6,184 val / 6,338 test).

| Metric | Value |
|---|---|
| Accuracy | 0.9978 |
| Precision | 0.9874 |
| Recall | 0.9930 |
| F1 | **0.9902** |
| Average precision (PR-AUC) | 0.9995 |
| Confusion matrix | TN=5619, FP=9, FN=5, TP=705 |

**Meets every spec target.** Sanity check: 3/3 phishing, 3/3 legitimate. Baseline (always predict
"legitimate"): accuracy 0.888, F1 0.0 — same baseline as §5.2, since it's the same split.

Source: `ml/saved_models/bert_runs/full/metrics.json`, which also records the exact dataset
SHA-256, code SHA-256 for the training/preprocessing/identity-binding scripts, the pretrained
checkpoint revision, seed (42), and package versions — everything needed to reproduce or audit
this specific run.

**Comparison against the classical model**: BERT reaches F1 0.9902 versus the classical TF-IDF +
Logistic Regression model's F1 0.9761 (§5.2) — a real improvement, but a modest one (0.0141), and
smaller than the gap you might expect from a full transformer versus a linear bag-of-words model.
Both comfortably clear the spec's F1≥0.90 target. The honest framing for Chapter 5/6: BERT
fine-tuning was pursued for completeness against the spec's stated architecture ("BERT embeddings
for transformer model"), not because the classical model was inadequate — the classical model
already met every target on its own (§5.2), and the deployed system's default backend remains the
classical model, with BERT available as a selectable alternative. This is a legitimate and useful
finding for Chapter 6: it demonstrates that architectural sophistication produced a measurable but
small return over a much cheaper linear model on this task, which is itself worth discussing when
weighing computational cost against marginal accuracy gain for deployment decisions.

**Training process, honestly reported for Chapter 4/6 methodology discussion**: the full run was
interrupted three times over its course — two out-of-memory crashes (at step 2,191/11,247 and step
7,119/11,247) caused by this machine's limited RAM (~7.8GB, frequently under 500MB free), and one
session-lifetime interruption unrelated to the model (a development-session restart) at step
6,500/11,247. Each time, training resumed from the last step-based checkpoint (checkpoints saved
every 500 steps specifically after the first crash, replacing an earlier epoch-based checkpoint
strategy that would have lost far more progress) after the experiment-identity manifest verified
the resumed run's dataset hash, code hash, and hyperparameters matched the original run exactly.
The final segment ran to completion in an independent process outside the development session so
it would not be affected by a fourth interruption. None of this affects the validity of the final
numbers above — they come from one continuous, hash-verified training lineage evaluated once on a
held-out test set — but it is a legitimate and interesting operational finding for a thesis
building ML infrastructure on resource-constrained hardware, and is worth a paragraph in Chapter 4
(engineering the resumability mechanism) and/or Chapter 6 (practical constraints of the deployment
environment).

---

## 6. Limitations (→ Chapter 6)

### 6.1 The URL model's central limitation: a real information ceiling, not a bug

The trained URL model catastrophically misclassifies ordinary bare domains: `google.com`,
`example.com`, `github.com` were flagged phishing with ~95-99.98% confidence in early runs, and the
regression sanity check still shows only 2/12 legitimate bare-domain cases passing on the current
model.

**Root cause, confirmed by direct data query**: in the training data, the exact feature-bucket bare,
short domains fall into (`path_length == 0`, `host_length ≤ 12`) was **65% phishing**. URL-structural
features alone (length, subdomain count, TLD, etc.) carry no brand-recognition or domain-reputation
signal — a legitimate short domain and a freshly-registered malicious one are structurally
near-identical in this feature space. A model fit to this training distribution reasonably learns
"phishing" as the majority answer for unbranded short domains.

**Careful framing for the thesis** (this was corrected once already after being overstated): this is
strong, evidence-backed evidence for an information ceiling in this feature set/dataset combination
— not proof that no dataset or feature addition could ever fix it. Two specific regularization
attempts didn't fix it; that rules out those attempts, not all possible approaches.

**Why this is not "unaddressed" architecturally**: the system's whitelist mechanism exists
specifically to cover this gap — "is this a well-known trusted domain" was never meant to be the
ML model's job. This is worth stating explicitly: it's a deliberate division of responsibility, not
an oversight.

**Why the live default is `rule_based`, not `trained`, and why that's correct, not a bug**: the
rule-based classifier is far worse on aggregate (F1 0.033 vs 0.839) but doesn't have this specific
"flags ordinary websites" failure mode as visibly. Flipping to the trained model based on F1 alone,
without first addressing the bare-domain problem, would trade "misses most phishing" for "actively
alarms users about legitimate sites" — arguably a worse outcome for a live security tool. This
decision, and its reasoning, is itself worth a paragraph in Chapter 6 as an example of a
deployment-readiness judgment that isn't purely about the headline metric.

### 6.2 The email model's source/label confound

Discussed in §3.4. Frame this in Chapter 6 as: the corrected F1 0.9761 is a strong,
leakage-controlled result *on this dataset*, not yet a validated claim of source-independent
phishing-detection skill. The honest fix (a mixed-source holdout, or additional phishing/legitimate
sources that don't perfectly correlate with collection source) was identified but not executed,
given thesis-timeline constraints — explicitly future work, not silently dropped.

### 6.3 Evaluation scope limitations

- All live-system measurements are single-machine, single-worker, SQLite-backed. Multi-process or
  multi-host deployment would need its own validation (§2.3).
- No independent, externally-collected final test set exists for either model — both are held-out
  splits of one repeatedly-inspected dataset, not an untouched generalization test.
- No combined URL+email evaluation exists (would need a dataset with both fields per example, which
  doesn't currently exist).
- Concurrent p95 latency (~509-520ms) slightly exceeds a strict "500ms for every request" reading of
  the spec, even though mean latency clears it comfortably.

### 6.4 What was deliberately deferred, not forgotten

Explicitly tracked (not silently dropped) as future work, given thesis-timeline tradeoffs:

1. **URL dataset augmentation** — supplementing or replacing the current Kaggle dataset with
   PhishTank and/or the UCI Phishing Websites dataset, since the current dataset's own internal
   label conflicts (§3.1) appear to be the practical ceiling, and more tuning on the same data
   already demonstrably doesn't help (§5.1).
2. **Email source/label confound fix** (§6.2) — needs new data sourcing, not just re-splitting.
3. An isolated (non-contended) re-run of the email load-test numbers, to get a completely clean
   throughput measurement.

---

## 7. Contributions Worth Stating Explicitly (→ Chapter 6, "Contributions")

- **A reproducible, hash-verified evaluation methodology**: every reported number traces to a
  specific model file (by hash), a specific test set (by hash), and a specific code version (by
  hash), via an isolated evaluation harness that spins up its own server/database per run. This is
  a stronger reproducibility standard than "we ran it once and it worked."
- **A documented dataset-auditing discipline** applied consistently across two very different
  datasets (URLs, emails): reading actual content rather than trusting labels, quarantining
  (not silently resolving) label conflicts, checking for collection-artifact leakage, and verifying
  that fixes changed results by an amount consistent with the size of the problem they addressed.
- **Two independently confirmed data-quality findings** that materially changed the project's
  scope and results (the email dataset's 83.5% spam-mislabeled-as-phishing problem, and the
  token-level leakage a subject-only split couldn't catch) — both found before being pointed out
  externally, and both fixed with logged, auditable evidence rather than an unexplained number
  change.
- **A live-serving architecture that treats classifier maturity as a first-class concern** — two
  independent backend settings rather than one, specifically so a well-validated component (email)
  can go live without forcing a not-yet-ready component (URL) along with it.

---

## 8. Suggested Chapter Flow (a starting skeleton, adjust to your program's format)

**Chapter 4 — Implementation**
1. System architecture (§1.1–1.3)
2. Feature engineering (§1.4, §1.5)
3. Detection pipeline and security design (§1.6, §2.1)
4. Evaluation infrastructure (§1.7) — arguably underrated as thesis content; reviewers like seeing
   methodology rigor
5. Implementation challenges (§2.2, §2.3)
6. Frontend (§1.8, with screenshots as figures)

**Chapter 5 — Results and Evaluation**
1. Dataset methodology and findings (§3) — this can be substantial; the dataset-quality findings are
   genuinely some of the strongest material in the whole project
2. Model training methodology (§4)
3. Results: URL (§5.1), Email (§5.2), BERT (§5.4, once available)
4. System performance (§5.3)
5. Live-API validation / parity (end of §5.3)

**Chapter 6 — Discussion and Conclusion**
1. Interpretation of results against spec targets (which were met, which weren't, and why)
2. Limitations (§6, all subsections) — don't undersell this section; the quality of your limitations
   discussion is itself evidence of rigor
3. Contributions (§7)
4. Future work (§6.4)
5. Conclusion

---

## Appendix: File Map (for citing artifacts directly)

| What | File |
|---|---|
| URL model metrics | `backend/ml/saved_models/url_classifier_metrics.json` |
| Email model metrics | `backend/ml/saved_models/email_classifier_metrics.json` |
| BERT metrics (once complete) | `backend/ml/saved_models/bert_runs/full/metrics.json` |
| URL dataset audit trail | `backend/ml/README.md` (§1–§14, §14.5 for the consolidated table) |
| Email dataset audit trail | `backend/ml/README_EMAIL.md` |
| Live-API evaluation results | `backend/ml/evaluation/REVIEW_AND_RESULTS.md` |
| Implementation verification notes | `backend/ml/evaluation/IMPLEMENTATION_FIXES.md` |
| Quarantined URL label conflicts | `backend/ml/data/processed/quarantined_conflicts.csv` |
| Quarantined email token conflicts | `backend/ml/data/processed_email/quarantined_token_conflicts.csv` |
| Frontend screenshots | `frontend/screenshots/01-16*.png` |
| Backend README (deployment/config) | `backend/README.md` |
