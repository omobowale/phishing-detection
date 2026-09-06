# API evaluation and update review - 2026-09-06

The trained URL pipeline was evaluated end to end, its SQLite concurrency failure
was fixed, and the rule-based baseline was re-evaluated. The later email integration
was reviewed and a cross-modality threshold regression was fixed. Source code,
models, prediction files, and test data are identified by hashes inside each run.

## Full URL results

Both runs used the same test CSV and exactly the same 11,268 accepted URL/label
pairs (verified). Of 11,296 submitted inputs, 28 were rejected by validation:
15 legitimate and 13 phishing. There were no unexpected sequential failures.

| Backend | Accuracy | Precision | Recall | F1 | Mean HTTP round trip (ms) | Successful req/s | Load failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trained XGBoost | 0.8420 | 0.8421 | 0.8370 | 0.8396 | 16.83 | 86.90 | 0 |
| Rule-based baseline | 0.5093 | 0.6144 | 0.0169 | 0.0329 | 15.77 | 85.89 | 0 |

- [Trained report](runs/20260906T093924Z_trained_5bbd80/results.json)
- [Trained API/offline parity and error slices](runs/20260906T093924Z_trained_5bbd80/results.analysis.json)
- [Rule-based report](runs/20260906T140806Z_url_rule_based_421789/results.json)

These are development evaluations on the current repeatedly inspected dataset.
The primary model remains below the research targets (precision >=0.91,
recall >=0.90, F1 >=0.90). Rejection exclusions explain the small difference
between API F1 0.83955 and full offline F1 0.83904; the model was not retrained.
All 11,268 accepted API predictions and confidence scores matched offline inference.

The capacity probe used the same seeded sample of 500 valid inputs, 40 concurrent
client workers, a warm model, one local Uvicorn worker, and SQLite. Both mean
sequential round-trip latency and successful throughput meet the specified targets
in these runs. Concurrent p95 latency was about 509 ms trained and 520 ms rule-based;
these measurements are not a guarantee that every request finishes within 500 ms.
Runs occurred at different times on this workstation, so small speed differences
are not evidence that one classifier is intrinsically faster. The trained run
predates the email backend refactor; current URL inference uses the same model,
features, and decision threshold, and the latest integration has separate tests.

## Fixes and evidence

- SQLite log writes originally produced three HTTP 500s and two transport failures
  under concurrent load. A first fix removed the expired-row refresh; residual
  insert contention remained. The final implementation releases the read transaction
  and queues detection-log writes per process for SQLite. Other databases do not use
  this queue. Multi-process/multi-host deployments require their own load validation.
- [Original failed full run](runs/20260906T093200Z_trained_13b90c/results.json) is
  preserved rather than overwritten. The final full load probe had zero failures.
- The evaluator verifies runtime backend/model/database identities, independently
  scores responses, validates every log before labeling, and checks API metrics.
- Each managed run creates its own DB, temporary admin, logs, predictions, and
  report, then stops only its own server. Existing data and results are preserved.
- The evaluator now supports `--input-type email_text`, separates status codes and
  transport errors, samples only valid inputs for capacity, and rejects a run if
  source or test files change during measurement.
- Enabling a trained email model previously changed URL-only heuristic decisions
  from threshold 0.4 to 0.5. The fallback now retains 0.4 unless a trained component
  actually contributes. Regression tests cover both modalities.

## New email/BERT review

The latest rebuilt email dataset contains 42,513 rows (29,991 train / 6,184
validation / 6,338 test). The follow-up audit confirms zero duplicate token rows,
zero conflicting token groups, and zero test rows sharing training token text.
The selected model has changed from Random Forest to Logistic Regression; its
test F1 is 0.976124, independently reproduced across all 6,338 test rows;
the test CSV exactly matches the parquet test split. The earlier 0.979748 Random Forest score and leakage
counts describe the previous artifacts only. See [updated audit](email_dataset_review_updated.json)
and [historical audit](email_dataset_review.json), each with artifact hashes.

Every source still contributes only one class, so collection-source cues remain a
possible contributor to the high score. Deduplication fixes exact-token overlap;
it does not establish generalization to independent sources or remove all near-duplicates.

The earlier Random Forest email HTTP smoke check used 20 examples only. It passed response/log consistency,
but had mean round-trip time about 1.06 seconds including first-use preprocessing;
its 20-request capacity probe was about 4.13 req/s. These small, cold-start-influenced
measurements do not establish full-set email performance. A local microbenchmark
found single-threaded Random Forest inference faster than `n_jobs=-1` for one email,
with identical probabilities in that check; no model artifact was changed.

The [latest Logistic Regression email smoke check](runs/20260906T141737Z_email_text_trained_69687a/results.json)
completed with 20 accepted inputs and 20/20 successful load requests. Sequential
median latency was 15.44 ms; mean 487.53 ms includes a 9.45-second first request.
The small warm capacity probe measured 73.54 req/s. All 20 examples were phishing,
so this smoke sample cannot establish balanced classification performance or
full-set serving capacity. It verifies that the newly saved model is served.

The initial DistilBERT review found a missing serving backend and unverified
checkpoint reuse. Both implementation issues are now addressed; see
[implementation fixes](IMPLEMENTATION_FIXES.md). Smoke experiments remain distinct
from full training and do not establish BERT research performance.

All 46 backend tests passed after the integration changes. The final URL smoke
check also completed with 20/20 successful requests in its capacity probe.

## Reproduce

From `backend`, using the existing environment and saved model:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluation.managed_run --backend trained
.\.venv\Scripts\python.exe -m ml.evaluation.managed_run --backend rule_based
.\.venv\Scripts\python.exe -m ml.evaluation.managed_run --backend trained --input-type email_text --limit 20
.\.venv\Scripts\python.exe -m pytest -q
```

The full classical email API evaluation has now been run; see the implementation
fixes report for its results and startup changes. Preserve the current artifacts as baselines before rebuilding.
The next research milestone is a leakage-controlled email experiment and independent
external URL/email evaluation, followed by full BERT and mixed-input evaluation if
retaining the original thesis scope.
