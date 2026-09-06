# Email/NLP Pipeline — Dataset, Training, and Findings

This documents the email/NLP half of build-order phase 2 (`../../phishing-detection-dev-spec.md`
section 8's "TF-IDF for classical models, BERT embeddings for transformer model"). Written to be
citable directly in the thesis, same convention as `ml/README.md` (the URL classifier's
equivalent document) — every number below came from an actual run.

**Current state: classical TF-IDF classifier trained and meets spec-wide targets. BERT fine-tuning
not yet done — CPU-only compute on this machine (no CUDA GPU), so the time budget for that needs a
decision before committing to it. Not yet wired into `classifiers.py`/live serving.**

---

## 1. Dataset: sourcing and a serious labeling problem found before training anything

**Source**: Kaggle `naserabdullahalam/phishing-email-dataset` ("Phish No More"), combining Enron,
CEAS_08, Ling, Nazario, Nigerian_Fraud, and SpamAssassin. Downloaded via
`ml/training/download_email_dataset.py` (mirrors the URL pipeline's `download_dataset.py`).

**The marketed label is not trustworthy as "phishing vs legitimate."** Before writing any training
code, every source's actual email content was read (not just labels), following the same
discipline that caught the URL project's dataset issues. Finding, per source:

| Source | Rows | Label meaning (verified by reading content) |
|---|---|---|
| Nazario | 1,565 | Genuine phishing (100% positive; the classic Nazario phishing corpus) |
| Nigerian_Fraud | 3,332 | Genuine advance-fee ("419") fraud (100% positive; a phishing subtype) |
| Enron | 29,767 | This is the **Enron-Spam** corpus: label 0 = real Enron correspondence, label 1 = **generic spam injected during compilation** (pharma ads, replica watches, weight-loss pills) — confirmed by reading label=1 rows, none were phishing, none were real Enron email |
| CEAS_08 | 39,154 | A spam-filtering shared-task dataset; label 1 = generic commercial spam ("Replica Watches," "Befriend Jenna Jameson," dating/pharma ads), not phishing-specific |
| Ling | 2,859 | Ling-Spam, a classic linguistics-mailing-list spam/ham corpus; not phishing-specific |
| SpamAssassin | 5,809 | Generic spam/ham; label=1 is mostly commercial spam with a small minority of genuine phishing-shaped subjects ("RE: Your Bank Account Information") mixed in |

**Consequence**: the dataset's own marketed totals (42,891 "phishing" / 39,595 "legitimate",
reproduced exactly from these source files) are **83.5% generic spam mislabeled as phishing**
(CEAS_08's 21,842 + Enron's injected 13,976 = 35,818 of 42,891 "phishing" rows). This is precisely
the "never treat spam labels as phishing labels without an explicit, supported mapping" trap an
external review of the URL pipeline warned about — confirmed here to be a real, not hypothetical,
risk.

### Decision (user-confirmed, "strict phishing only")

Three options were presented (strict/broad/hybrid); **strict** was chosen:

- **Phishing** = Nazario + Nigerian_Fraud only (4,897 rows) — both individually content-verified
  as genuinely deceptive fraud, not generic spam.
- **Legitimate** = the "ham" (label=0) rows of all four ham-bearing sources (39,595 rows) —
  content-verified as genuine correspondence across diverse registers: Enron (corporate),
  CEAS_08/SpamAssassin (mailing-list/technical), Ling (academic/linguistics).
- **Excluded entirely**: all "spam" (label=1) rows from CEAS_08/SpamAssassin/Ling and Enron's
  injected-spam rows. Not used as either class — generic spam is off-topic for a phishing
  classifier, not a useful "hard negative."

**Cost of this choice, stated plainly**: the resulting dataset is far smaller and far more
imbalanced (44,492 rows, ~11% positive) than the marketed version (82,486 rows, ~52% positive). A
broader-scope model would likely report higher raw numbers while actually being a "spam detector,"
not a "phishing detector" — the honest, smaller dataset was chosen deliberately over the easier,
larger, mislabeled one.

---

## 2. Two more data-quality issues found and fixed (same discipline as the URL pipeline)

### 2.1 A collector's own mailbox leaked into the "phishing" class as a spurious signal

A first trained model's top TF-IDF features included **"jose", "monkey", "monkey org"** —
fragments of `jose@monkey.org`. Investigating: ~40% of Nazario rows are phishing emails literally
addressed to Jose Nazario personally (`"Dear jose@monkey.org, your account..."`), since he
personally collected them into this corpus. This is genuine phishing *content* (not a metadata
artifact), but it's a spurious, non-generalizing signal — a classifier that partly keys on "does
this mention this one specific recipient" won't generalize to phishing addressed to anyone else.

**Fix**: `app/pipeline/preprocessing.py`'s `preprocess_email_text()` now redacts email addresses
and URLs to placeholder tokens (`emailaddresstoken`, `urltoken`) *before* word-tokenization — the
previous tokenizer (`_WORD_RE`, alphabetic-only) fragmented `jose@monkey.org` into meaningless
word-pieces instead of treating it as one unit. This is standard practice in spam/phishing text
classification generally, not a one-off patch for this dataset, and it's in the **shared**
preprocessing function both training and live serving call, so this isn't a train/serve
inconsistency.

**Effect on metrics: negligible** (test F1 0.9824 before this fix -> 0.9839 after, precision
0.9955 -> 0.9970) — reassuring evidence the strong result isn't primarily an artifact of this leak;
if anything, removing the spurious signal correlated slightly with a small improvement.

### 2.2 Raw mbox/RFC822 artifacts, and a real train/serve inconsistency

Some Nazario rows are not real captured emails at all: literal mail-client placeholder text
(`"This text is part of the internal format of your mail folder, and is not a real message"`) or
raw leaked SMTP headers (`Return-Path:`, `X-Original-To:`, `Delivered-To:`) sitting directly in the
`body` CSV field.

Separately, a real bug: `app/pipeline/feature_extraction_email.py`'s `extract_email_features()`
(live serving) already stripped RFC822 headers via `email.message_from_string()` before calling
`preprocess_email_text()`, but the first version of `build_email_features.py` (training) called
`preprocess_email_text()` directly on raw CSV text — a train/serve mismatch, the same category of
bug the URL pipeline was built around avoiding (`extract_url_features()` being the single shared
source of truth for both).

**Fixes**:
- Extracted the header-stripping into a shared `strip_email_headers()` in `preprocessing.py`,
  used by both `extract_email_features()` and `build_email_features.py`.
- Rows matching known mbox-corruption markers (the placeholder text, or `Return-Path:`/
  `X-Original-To:`/`Delivered-To:` — these have prose *before* the header block, so
  `email.message_from_string()` can't parse them as headers and strip them) are dropped entirely:
  **174 rows removed**, logged as a count (not silently).

**Effect on metrics**: small, honest dip (test F1 0.9839 -> 0.9797, precision 0.9970 -> 0.9900,
recall essentially unchanged at 0.9697) — the expected cost of removing 174 contaminated rows and
cleaning header noise, not a red flag. See §3 for the final numbers this produced.

---

## 3. Training and results (classical TF-IDF baseline)

`ml/training/train_email_classifier.py`: TF-IDF (unigrams+bigrams, max 20,000 features, fit on
train only) into Logistic Regression / Random Forest / Complement Naive Bayes, `class_weight`
balanced given the ~11% positive rate. Random Forest selected (highest validation F1).

| Split | Rows | Phishing | Legitimate |
|---|---|---|---|
| Train | 30,857 | 3,241 | 27,616 |
| Val | 6,690 | 718 | 5,972 |
| Test | 6,757 | 923 | 5,834 |

Split by subject-line group (normalized, reply/forward-prefix stripped) — not a plain random
split — since 145/1565 Nazario and 780/3332 Nigerian_Fraud rows share a subject with another row in
the same source (campaign template reuse), the email-equivalent of the URL pipeline's domain-
grouped split. Verified 0 subject-group overlap between every split pair.

**Test metrics** (current, after both fixes in §2):

| Metric | Value |
|---|---|
| Accuracy | 0.9945 |
| Precision | 0.9900 |
| Recall | 0.9697 |
| F1 | 0.9797 |
| Average precision (PR-AUC) | 0.9939 |
| Confusion matrix | TN=5825, FP=9, FN=28, TP=895 |

**Meets every spec-wide target** (F1≥0.90, precision≥0.91, recall≥0.90 — the spec doesn't give
email-specific numbers, so this reuses the project-wide thresholds for comparability). Baseline
(always predict the training majority, "legitimate"): accuracy 0.863, F1 0.0 — the trained model
is clearly learning real signal, not just exploiting class imbalance.

**Sanity check** (hand-written, novel examples not from the dataset — same discipline as the URL
pipeline's bare-domain check): 3/3 phishing examples correctly flagged, 3/3 legitimate examples
correctly passed.

**Error analysis** (read actual misclassified examples, not just counted them): false negatives
are concentrated in genuinely hard cases — non-English phishing (German-language Apple ID phishing
was missed; the TF-IDF vocabulary is English-only by construction), and deliberately obfuscated
text evading filters (e.g. `"W##number3##e##number1##..."` letter-substitution tricks). False
positives include real corporate "we're changing our bank account" and "update your account
information" notifications (Enron, CEAS_08 ham) that structurally resemble classic phishing/BEC
lures — arguably genuinely ambiguous cases, not model failures, since real-world business-email-
compromise scams use exactly this template.

**Top predictive features** (confirmed, post-fix): `account, dear, bank, million, money, email,
reply, fund, click, contact, security, urgent, dollar, transfer, mr, country, assistance,
emailaddresstoken, kindly, ...` — genuine advance-fee-fraud/phishing vocabulary, no source-specific
artifacts remaining. `emailaddresstoken` ranking as a real (not source-specific) signal is a good
outcome: the model learned "an email address is mentioned" generalizes, not "which one."

---

## 4. What's NOT done yet

- **BERT/DistilBERT fine-tuning** — spec section 3 explicitly names this ("Hugging Face
  Transformers... consider DistilBERT for latency-sensitive paths"). Not started. This machine has
  no CUDA GPU (`torch.cuda.is_available()` is `False`; integrated Intel UHD 620 only), so this
  would be a CPU-only fine-tune — needs a realistic time budget agreed before committing to it, not
  assumed.
- **Not integrated into `app/pipeline/classifiers.py`** — `RuleBasedClassifier`'s hand-weighted
  email heuristic is still what's live. Wiring in the trained classifier (a new `EmailClassifier`
  implementing `BaseClassifier`, analogous to `TrainedURLClassifier`) is straightforward given the
  URL pipeline's existing pattern, but deliberately deferred until the BERT-vs-classical-only
  scope question above is settled, so it isn't redone twice.
- **No combined URL+email evaluation** — needs a labeled dataset with both a URL and email body
  per example, which doesn't exist yet; out of scope for this pass.
- **No independent second email test set** — same caveat the URL pipeline's methodology carries:
  this is one dataset's held-out split, not an independently-collected final test set.

---

## 5. Reproducing this from scratch

```bash
cd backend
python -m ml.training.download_email_dataset   # needs ~/.kaggle/kaggle.json
python -m scripts.download_nltk_data            # one-time, if not already done

python -m ml.training.build_email_features      # writes ml/data/processed_email/email_features.parquet
python -m ml.training.train_email_classifier    # writes ml/saved_models/email_classifier.joblib
```

`random_state=42` throughout. Raw data lives under `ml/data/raw_email/` (gitignored, regenerate via
the download script); the per-source CSVs are used, not the dataset's own pre-merged
`phishing_email.csv` (which drops provenance and is what let the spam-mislabeling go unnoticed in
the first place).
