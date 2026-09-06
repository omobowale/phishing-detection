# Email/NLP Pipeline — Dataset, Training, and Findings

> Review update (fixed): a review found the saved 97.97% F1 was reproduced on pre-deduplication
> artifacts with 142 test rows whose token text appeared in training and one conflicting
> token-text group — subject grouping alone doesn't prevent this. **Fixed and retrained**: see
> section 3.1 for the corrected result (F1 0.9761, still meets every spec target) and section 6
> for the full review. The dataset's source/label confound (every source contributes only one
> class) is a separate, structural limitation that resplitting cannot fix — still applies, see 3.1.


This documents the email/NLP half of build-order phase 2 (`../../phishing-detection-dev-spec.md`
section 8's "TF-IDF for classical models, BERT embeddings for transformer model"). Written to be
citable directly in the thesis, same convention as `ml/README.md` (the URL classifier's
equivalent document) — every number below came from an actual run.

**Current state: classical TF-IDF classifier trained, meets spec-wide targets, and is wired into
live serving (`EMAIL_CLASSIFIER_BACKEND=trained`, see section 4 — independent of the URL model's
own `URL_CLASSIFIER_BACKEND` setting, since the two have very different maturity; see `ml/README.md`
§16). BERT fine-tuning is in progress: a full 3-epoch run on the corrected dataset (§3.1) started
2026-09-06, CPU-only (no CUDA GPU on this machine), checkpointed per epoch and resumable across
interruptions (`ml/training/experiment_identity.py`). Not complete as of this writing — see section
7 once it is.**

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
emailaddresstoken, kindly, ...` — genuine advance-fee-fraud/phishing vocabulary, no obvious collector-address fragments among those inspected features. This does not rule out other source-specific cues. `emailaddresstoken` ranking as a real (not source-specific) signal is a good
outcome: the model learned "an email address is mentioned" generalizes, not "which one."

### 3.1 Token-level leakage fix (2026-09-06) — corrected results

§6's review found that subject-line grouping does not prevent leakage: after redaction/
lemmatization/stopword-removal, 1,790 rows across the dataset collapsed onto duplicate token text,
and 142 test rows had token text identical to a training row — real leakage a subject-only split
can't see, since it only groups by the raw subject line, not by what the text becomes after
preprocessing. One token-text group also carried both labels (a genuine annotation conflict, not
just a duplicate).

**Fix** in `build_email_features.py`, applied in the same order the URL pipeline's own conflict-
quarantine bug taught (quarantine conflicts *before* other cleaning, never let a later step
silently resolve a disagreement it can't see): after computing `tokens`, any token-text group
whose rows disagree on label is quarantined entirely (3 rows, 1 group — logged to
`quarantined_token_conflicts.csv`), then the remaining exact token-text duplicates are dropped,
keeping one representative per unique token string (1,788 rows). This isn't a heuristic — dropping
to one row per unique token string makes cross-split leakage of this kind structurally impossible,
not just less likely.

Rebuilt and retrained (`Logistic Regression` now edges out Random Forest on validation F1 by a
hair — both are close):

| Split | Rows | Phishing | Legitimate |
|---|---|---|---|
| Train | 29,991 | 3,325 | 26,666 |
| Val | 6,184 | 715 | 5,469 |
| Test | 6,338 | 710 | 5,628 |

| Metric | Before fix | After fix |
|---|---|---|
| Accuracy | 0.9945 | 0.9946 |
| Precision | 0.9900 | 0.9734 |
| Recall | 0.9697 | 0.9789 |
| F1 | 0.9797 | 0.9761 |
| Average precision (PR-AUC) | 0.9939 | 0.9966 |
| Confusion matrix | TN=5825, FP=9, FN=28, TP=895 | TN=5609, FP=19, FN=15, TP=695 |

**Still meets every spec-wide target** (F1 0.9761 ≥ 0.90, precision 0.9734 ≥ 0.91, recall 0.9789 ≥
0.90), sanity check still 3/3 and 3/3. The number moved by about half a point of F1, not
collapsed — reassuring evidence the original result wasn't primarily an artifact of this specific
leakage, though see the caveat below, which this fix does not address.

**What this fix does NOT address** (per §6, and worth restating plainly rather than letting the
corrected numbers imply more than they show): every source in this dataset contributes only one
class — Nazario/Nigerian_Fraud are 100% phishing, the four ham sources are 100% legitimate. No
resplitting can separate "the model learned phishing language" from "the model learned which
collection/source this came from" when every example of one class comes from sources the other
class never touches. This is a structural property of the dataset's composition, not a splitting
bug — the honest fix would be sourcing phishing and legitimate examples from more overlapping
collection pipelines, which isn't available here. Treat the corrected F1 0.9761 as a strong,
now leakage-controlled result on *this* dataset, not a settled claim that the model has learned
source-independent phishing language.

---

## 4. What's NOT done yet

- **BERT/DistilBERT fine-tuning** — spec section 3 explicitly names this ("Hugging Face
  Transformers... consider DistilBERT for latency-sensitive paths"). Not started. This machine has
  no CUDA GPU (`torch.cuda.is_available()` is `False`; integrated Intel UHD 620 only), so this
  would be a CPU-only fine-tune — needs a realistic time budget agreed before committing to it, not
  assumed.
- **No combined URL+email evaluation** — needs a labeled dataset with both a URL and email body
  per example, which doesn't exist yet; out of scope for this pass.
- **No independent second email test set** — same caveat the URL pipeline's methodology carries:
  this is one dataset's held-out split, not an independently-collected final test set.

**Now integrated into `app/pipeline/classifiers.py`**: `TrainedClassifier` (formerly
`TrainedURLClassifier`, generalized) loads `email_classifier.joblib` when `EMAIL_CLASSIFIER_
BACKEND=trained`, and/or `url_classifier.joblib` when `URL_CLASSIFIER_BACKEND=trained` --
these are two independent settings (`ml/README.md` §16), not one shared flag, specifically so the
email model (trained; further validation required) can go live without also re-enabling the URL model's known bare-domain false-
positive problem (not ready, `ml/README.md` §5/§6). Either side uses its rule-based heuristic when configured as `rule_based`. A component configured as `trained` fails explicitly if its artifact is missing. `extract_email_
features()` now also returns `tokens_text` -- the exact same `strip_email_headers()` ->
`preprocess_email_text()` output `build_email_features.py` used to build its `tokens` column --
so the live TF-IDF vectorizer sees identical input to training, not a re-derived approximation.

Verified end-to-end against the real running FastAPI server (not just direct model calls), with
the actual deployed settings (`EMAIL_CLASSIFIER_BACKEND=trained`, `URL_CLASSIFIER_BACKEND=
rule_based`): a real phishing email and a real legitimate email both classify correctly via the
trained email model, and `google.com` correctly classifies as legitimate via the URL side's
rule-based heuristic -- confirming the backend split actually prevents the bare-domain regression
in the live default configuration, not just in a hypothetical one. Separately, forcing
`URL_CLASSIFIER_BACKEND=trained` for a one-off check reproduces the already-documented bare-domain
false positive on `example.com`/`google.com` (`ml/README.md` §5) -- the existing, accepted URL
model limitation, not a new integration bug, and exactly why it isn't the live default.

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


## 6. Integration and dataset review (2026-09-06)

The previous saved Random Forest metrics were independently reproduced from all 6,757
held-out token sequences: F1 0.979748, precision 0.990044, recall 0.969664. A sample
of 100 raw test emails produced the same tokens through live feature extraction.
The test CSV matches the parquet test split.

The following findings describe the historical artifacts; see the latest rebuild
verification below for resolved items:

- 1,790 duplicate token rows remain after exact raw-text deduplication. Across
  training and test, 39 token strings overlap, affecting 142 test rows.
- One token-text group contains both labels. Detect conflicts after preprocessing
  before keeping an arbitrary raw-text copy or assigning splits.
- Every source contributes only one class: Nazario/Nigerian_Fraud are positive;
  CEAS_08/Enron/Ling/SpamAssasin are negative. A subject-grouped split retains this
  association, so it cannot separate phishing cues from collection-source cues.
- Six held-out emails exceed or otherwise fail the current API input schema.
  API evaluation reports rejection coverage separately from classification metrics.

Preserve this run as a development baseline. For a stronger experiment, quarantine
conflicting postprocessed groups, assign connected subject/token groups to one
split, and evaluate on separately sourced positive and negative messages. Do not
claim the current source-label association was eliminated by address redaction.
Evidence counts are saved in `evaluation/email_dataset_review.json`.

A confirmed serving regression was fixed: enabling a trained model for the unused
modality previously moved heuristic-only requests from threshold 0.4 to 0.5.
The threshold now remains 0.4 unless a trained model actually contributes to the
submitted input. Regression tests cover both URL-only and email-only fallbacks.

Reproduce email API evaluation with an isolated database/server:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluation.managed_run --backend trained --input-type email_text
```

For a short integration check add `--limit 20`. A short check is not a full test-set
performance result. Trained email and URL-plus-email fusion remain separate claims;
this email-only benchmark does not validate the combined classifier or replace
BERT/DistilBERT work required by the original specification.

**Follow-up: done.** `build_email_features.py` now quarantines token-text label conflicts and
deduplicates on token text before splitting (see §3.1); features were rebuilt and the classical
model retrained on the corrected data. Corrected result: F1 0.9761 (was 0.9797 on the leaky
split), still meets every spec target. The source/label confound noted above is unchanged by this
fix and remains an open, structural limitation — not something a live-API re-evaluation would
reveal either, since it's a property of the training data's composition, not of serving.


### DistilBERT work observed during this review

`training/train_email_bert.py` and smoke artifacts now exist. The observed
`bert_email_classifier_metrics.json` covers 40 training, 16 validation, and 16
test examples with one epoch (test F1 0.3158); it is a pipeline smoke check, not
a full-dataset BERT result. The runtime classifier currently loads only the
classical joblib email model; no DistilBERT serving path is wired in yet.

The latest updates remove exact token overlap and seed before `from_pretrained`
initializes the classification head. Source bias still needs evaluation. Bind resume
checkpoints to a dataset/split/configuration fingerprint: the script still resumes
`checkpoint-*` under a shared directory without verifying those fingerprints.
Preserve smoke metrics separately from final experiment results.


### Follow-up builder change

The builder now quarantines conflicting token groups and deduplicates token text
before splitting. This resolves the identified identical-token overlap in the
implementation. The audit counts above refer to the saved pre-change dataset
(SHA-256 recorded in `evaluation/email_dataset_review.json`), not a claim that
this revised builder has already been run. Rebuild features and retrain into a
separate experiment before replacing the primary model or citing updated metrics.
Source/label association remains a separate limitation after token deduplication.


### Latest rebuild verification

The new saved dataset has 42,513 rows: 29,991 train, 6,184 validation, and 6,338
test. A fresh artifact audit confirms zero duplicate token rows, zero conflicting
token groups, and zero test rows with training token text. The selected model is
now Logistic Regression, with saved test F1 0.976124. Thus the rebuild/retraining
requirement above has now been completed; the historical Random Forest results
and smoke timings do not describe the new model. Source-label association remains.
See `evaluation/email_dataset_review_updated.json` for the new artifact hashes
and `evaluation/REVIEW_AND_RESULTS.md` for consolidated results.

---

## 7. BERT fine-tuning results (placeholder — fill in once training completes)

Started 2026-09-06: `ml.training.train_email_bert` (`ml/saved_models/bert_runs/full/`),
DistilBERT-base, 3 epochs, `max_length=128` (reduced from an initial 256 after discovering the
original CPU-time benchmark used an unrepresentative ~20-token dummy sentence — real emails average
212 tokens, 62% hit a 256-token cap, so the benchmark badly underestimated real per-step cost; see
git history for `ml/training/train_email_bert.py` around 2026-09-06 for the full account). Runs on
the same corrected, leakage-fixed split as §3.1/§6 (29,991 train / 6,184 val / 6,338 test),
resumable via `ml/training/experiment_identity.py`'s manifest binding.

**Do not cite any BERT number until this section is filled in with a real completed run's results**
(test accuracy/precision/recall/F1/average precision, confusion matrix, sanity check, and a
comparison against the classical model's F1 0.9761). The only BERT run completed before this one
was a 40/16/16-example smoke check (test F1 0, `IMPLEMENTATION_FIXES.md`) that verifies the
pipeline executes, not that a useful model was trained.
