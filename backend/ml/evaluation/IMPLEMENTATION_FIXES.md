# Implementation fixes - 2026-09-06

The API now warms preprocessing and the selected models before accepting traffic.
A configured missing model fails startup rather than the first detection request.
The classical model artifacts, dataset labels, and production backend selection
were not changed by these implementation fixes.

## Completed changes

- Added an explicit `EMAIL_CLASSIFIER_BACKEND=bert` serving path, independent of
  the URL backend. It loads local files only, verifies the exported preprocessing
  and label contract, uses evaluation/inference mode, and bounds concurrent CPU
  tensor allocations with an inference lock. Runtime reports identify its files
  with a directory hash. Existing classical/heuristic fusion behavior is retained.
- BERT receives header-stripped, address/URL-redacted natural text through the
  same cleaning functions used in training. Classical TF-IDF still receives its
  existing lemmatized token text.
- Training manifests bind resumes to dataset, code, package versions, pretrained
  revision, seed, and hyperparameters. Unidentified/incompatible checkpoints fail
  explicitly. Dataset or code changes during training prevent final export.
- Smoke and full experiments have separate directories and marked metrics.
  Smoke samples include both classes. Training checks for conflicting token labels
  and token text crossing splits before fitting. Completed experiments are not
  overwritten by another invocation.
- API evaluation and offline parity analysis now support BERT exports. Added the
  missing `accelerate` Trainer dependency and regression coverage for checkpoint
  identity, the BERT text contract, and incomplete/invalid exports.

## Verification

All 49 backend tests passed. The full classical email HTTP run accepted 6,335 of
6,338 inputs (three validation rejections), with F1 **0.976124**, precision
**0.973389**, and recall **0.978873**. Every accepted prediction and confidence
score matched offline inference. Its 500-request load probe had zero failures.
The run overlapped a BERT smoke job, so its 19.17 ms mean and 48.58 req/s capacity
are measurements under competing work, not isolated capacity estimates.

- [Full email HTTP evidence](runs/20260906T142809Z_email_text_trained_5be2d6/results.json)
- [Full email parity](runs/20260906T142809Z_email_text_trained_5be2d6/results.analysis.json)
- [BERT HTTP integration](runs/20260906T143226Z_email_text_bert_9fa3c0/results.json)
- [BERT parity](runs/20260906T143226Z_email_text_bert_9fa3c0/results.analysis.json)
- [BERT smoke training](runs/bert_implementation_check_20260906/metrics.json)

The BERT implementation check trained on 40 examples and evaluated on 16 validation
and 16 test examples. Its test F1 was zero; these artifacts demonstrate execution,
not a useful trained detector. Four live API inputs and their load probe completed
without errors, with zero offline/API prediction or confidence mismatches. CPU
inference was slow (seconds per request). A small thread-count comparison did not
establish that thread tuning would meet the latency target; no speculative override
was applied. The serving backend is implemented, but a fully trained, validated
BERT model is still needed before selecting it for normal use.

## Commands

From `backend`, after installing requirements:

```powershell
# Separate implementation check; use a new run directory if this one is complete.
.\.venv\Scripts\python.exe -m ml.training.train_email_bert --smoke --local-files-only

# Full experiment on the existing split; this can take many CPU hours.
.\.venv\Scripts\python.exe -m ml.training.train_email_bert --local-files-only

# Evaluate the completed full BERT export in an isolated server/database.
.\.venv\Scripts\python.exe -m ml.evaluation.managed_run --backend bert --input-type email_text

# Evaluate another export explicitly (including an intentionally selected smoke export).
.\.venv\Scripts\python.exe -m ml.evaluation.managed_run --backend bert --input-type email_text --bert-model-path PATH_TO_MODEL --limit 20

.\.venv\Scripts\python.exe -m pytest -q
```

`--local-files-only` requires the pretrained model in the local Hugging Face cache.
Normal serving uses `EMAIL_CLASSIFIER_BACKEND=bert` and optionally
`EMAIL_BERT_MODEL_PATH`; the default path is `ml/saved_models/bert_runs/full/model`.
No default was changed to use a smoke model. Legacy checkpoints without a matching
manifest must remain separate; do not fabricate a manifest to adopt them.

## Deferred research work

Dataset source bias, independent-source holdouts, near-duplicate grouping, final
URL feature/model selection, and conclusions about generalization require the
evaluation-design stage. Full BERT training and a research comparison remain
outstanding; this implementation pass did not run a multi-hour full experiment.
The URL model's F1 remains 0.8396. Neither that target shortfall nor the email
source-label association can be honestly declared fixed by these code changes.
