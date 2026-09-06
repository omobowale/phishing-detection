# ML Pipeline — Dataset, Training, and Findings

This documents build-order phases 1 and 2 (`../../phishing-detection-dev-spec.md` section 8):
data collection/preprocessing and model development for the **URL classifier**. Written to be
citable directly in the thesis (methodology, results, and limitations/future-work chapters) —
every number below came from an actual run, not an estimate.

**The email/NLP classifier (TF-IDF + planned BERT) has its own document: `README_EMAIL.md`** — a
separate dataset, separate pipeline, separate findings. Read that one for anything email-related;
this document is URL-only throughout.

**Skip straight to §14.5 for the one consolidated results table to actually cite.** Everything
below it is the audit trail of how those numbers were arrived at, across four rounds of external
review — useful for methodology/limitations chapters, but §14.5 is the source of truth for numbers.

**Current state: `URL_CLASSIFIER_BACKEND` is `rule_based` in production, not `trained`** (this is
now a setting independent of the email classifier's — see `app/core/config.py` and `README.md`'s
Status section). A trained model exists at `saved_models/url_classifier.joblib` and is fully wired
into `app/pipeline/classifiers.py` (`TrainedClassifier`, loaded whenever `URL_CLASSIFIER_BACKEND=
trained`), but is deliberately not the default. Four rounds of external review plus this project's
own follow-up work converge on the same conclusion: neither backend is fit to ship as-is, and the
dataset needed considerably more cleaning and methodology correction than any single pass caught
before its numbers should be presented as a final benchmark.

- **Phase 5's live-API evaluation of `rule_based`**, re-run under the corrected evaluation script
  (`ml/evaluation/results.json`, §12.7): **1.69% recall**, accuracy 0.509, precision 0.614, F1
  0.033, on 11,268 evaluated test rows (28 rejected as malformed input, see §12.1/§12.7) — it
  misses nearly all real phishing URLs. One genuine good-news correction from this re-run:
  **concurrent throughput now measures 41.5 successful req/s, clearing the spec's 20 req/s
  target** — the previously-reported 5.51 req/s was a benchmark artifact (§7.4/§12.5), not a real
  server limitation.
- **The trained URL model, evaluated properly (domain-grouped split via a Public Suffix
  List-aware grouping, no host shared between train/val/test): F1 0.839, precision 0.841, recall
  0.837.** This is the current, most-corrected number — see §12.2 for exactly what changed since
  the F1 0.827 figure the second review round produced (short version: a `.co.uk`-grouping bug and
  a majority-baseline bug, both fixed, both explained there). Evaluated the old, easier way (random
  per-URL split, letting the model see other URLs from the same host during training) it reads F1
  0.879 — see §10.2 for the comparison and why it should be read as "a different evaluation
  protocol," not simply "worse because of leakage."

This document went through four rounds of external review (§7, §10, §12, §14) that each corrected
real overreach in earlier versions of it — including this document's own prior claims about what
had been "ruled out" or "proven." Read §14.5 before citing any number from §3-§6 or §10; those
sections are kept for the audit trail, not as final results.

---

## 1. Data collection (phase 1)

- **Source:** Kaggle dataset `sid321axn/malicious-urls-dataset` (`malicious_phish.csv`), pulled
  via `training/download_dataset.py` using the official `kaggle` package (reads
  `~/.kaggle/kaggle.json`; no credentials are ever hardcoded or passed on the command line).
- **Raw composition:** 651,191 labeled URLs across four categories: `benign` (428,103),
  `defacement` (96,457), `phishing` (94,111), `malware` (32,520).
- **Used subset:** only `benign` → `legitimate` and `phishing` → `phishing`, since the spec's
  `Prediction` enum is binary (phishing/legitimate). `defacement` and `malware` rows are dropped.
- **Cleaning, current pipeline (see §10 for full methodology and the with/without comparisons)**:
  1,785 rows dropped for "phishing" labels on domains with no plausible attacker-hosting surface
  (§3.4, `known_legitimate_domains.py` — logged to `domain_cleaned_phishing_labels.csv`, not
  silently discarded, and can be disabled via `SKIP_DOMAIN_CLEANING=1` to compare with/without);
  ~8,110 rows **quarantined** (dropped and logged to `quarantined_conflicts.csv`) for disputed
  ground truth — either the exact URL string or its scheme/case/path-normalized form carries both
  `benign` and `phishing` labels in the raw CSV; then ~36-41 exact-duplicate URLs (same URL, same
  label) dropped before splitting.
- **Balancing:** capped and randomly sampled to 50,000 rows per class (`MAX_ROWS_PER_CLASS` env
  var, default 50000) so one class can't swamp the other — 100,000 rows before feature
  extraction; ~8-11 failed feature extraction and were skipped, leaving **~99,989 rows** (exact
  count varies slightly run to run because the balancing step randomly samples from a pool whose
  size depends on which cleaning steps ran).
- **Split (default, primary): grouped by registrable domain** — `SPLIT_STRATEGY=grouped` (default)
  in `build_features.py`, using `sklearn.GroupShuffleSplit` so no domain appears in more than one
  of train/val/test. `random_state=42`, frozen. Because grouping doesn't stratify by label, the
  resulting class balance drifts from 50/50 per split (reported by the script each run) — this is
  expected and disclosed, not a bug. Written to `data/processed/url_features.parquet`; held-out
  test URLs alone to `data/processed/test_urls.csv` (used by the phase-5 live-API evaluation run).
  `SPLIT_STRATEGY=random` reproduces the old per-URL random split for comparison, written to
  `*_random_split.parquet`/`.csv` instead — see §10 for why the grouped split is what should
  actually be trusted, and by how much the numbers differ.
- **Assumption worth flagging for the thesis:** this dataset's `benign` class is standing in for
  "legitimate URL" generally. It was not curated specifically for a phishing-detection thesis, and
  as section 3 and §10 below show, its collection methodology introduced real artifacts and
  substantial internal label inconsistency.

### Data pipeline engineering notes (not thesis-relevant, but real fixes)

- `build_features.py`'s per-class balancing originally used
  `df.groupby("label", group_keys=False).apply(lambda g: g.sample(...))`. Under pandas 3.0.5
  (the version resolved into `backend/.venv`), this silently **drops the grouping column** from
  the result — a real pandas 2.2→3.0 behavior change, not a typo. Fixed with an explicit
  `[group.sample(...) for _, group in df.groupby("label")]` + `pd.concat`.
- `requirements.txt` gained `kaggle>=1.6`, `joblib>=1.4`, `pyarrow>=16.0` (parquet I/O).
  `backend/.gitignore` gained `/ml/data/` (raw + processed data is regenerable, not committed).

---

## 2. Feature engineering

13 structured URL features are computed by `app/pipeline/feature_extraction_url.py`'s
`extract_url_features()` — the **same function** is called at training time
(`build_features.py`) and at live inference time (`app/api/detect.py`), so train/serve feature
computation cannot drift:

`url_length, host_length, subdomain_count, is_ip_address, has_https, special_char_count,
digit_count, path_length, query_length, brand_keyword_count, brand_keyword_outside_domain,
suspicious_tld, has_at_symbol`

`has_https` is computed but **excluded from the ML model's training columns** — see the
data-quality findings below for why.

---

## 3. Data quality findings (important for a limitations chapter)

Two real dataset artifacts were found and fixed by testing the first trained model against plain,
ordinary URLs (`google.com`, `example.com`) rather than trusting held-out-set metrics alone.

### 3.1 `path_length` artifact (fixed)

In the raw dataset, **99.9% of `legitimate` URLs have `path_length ≥ 1`** (i.e. they were stored
with at least a trailing `/`), while **23% of `phishing` URLs have `path_length == 0`** (stored as
a bare domain, no path at all). This is a collection/formatting artifact of how the dataset's two
classes were assembled, not a real behavioral difference between phishing and legitimate URLs —
but a tree-based model finds it a near-perfect (and spurious) split.

**Fix (permanent, kept):** `feature_extraction_url.py` now computes
`path_length = len(parsed.path.rstrip("/"))`, so a bare root path and no path score identically.
This is a defensible feature-engineering choice independent of the dataset issue (a lone trailing
slash isn't meaningfully "having a path") and should stay regardless of what dataset is used.

### 3.2 `has_https` artifact (feature dropped from training)

**Corrected by a later external review** (§10): an earlier version of this section said "99.5% of
raw URLs have no scheme at all," which conflated "not literally `https://`" with "no scheme at
all" — those are different things (`http://` is a scheme too). The precise numbers: **8.3% of
benign / 26.4% of phishing rows have some explicit scheme** (mostly `http://`); narrowing to
`https://` specifically, it's **0.46% of benign / 7.4% of phishing** — which is what `has_https`
actually measures, and where the original finding still holds. Within that `https://`-carrying
minority, `phishing` outnumbered `legitimate` roughly 16:1 — but this reflects which subset of the
dataset happened to be stored with a `https://` prefix, not actual TLS usage. Despite its low
variance, this feature carried **~51% of total feature importance** in the first trained model and
was the single largest cause of it misclassifying ordinary HTTPS URLs.

**Fix (kept):** dropped from `train_url_classifier.py`'s `FEATURE_COLUMNS`. Still computed by
`extract_url_features()` and still used by `RuleBasedClassifier`'s heuristic (a general "no HTTPS
is somewhat suspicious" rule is reasonable there; it's specifically *training an ML model on this
dataset's skewed version of it* that was the problem).

### 3.3 Label noise (documented, not fixed — inherent to the dataset)

Found by manual inspection while spot-checking the processed features: the URL
`id0498372885938paypal.com.au.account.verification-re5983202k.com.au.mubashar786.com/...` — a
textbook PayPal-verification phishing pattern — is labeled `legitimate` in the source Kaggle
dataset. This caps the ceiling on achievable precision/recall regardless of model quality, and is
worth citing directly as a dataset-limitation in the thesis (public phishing-URL datasets
routinely carry some fraction of mislabeled rows).

### 3.4 Systematic label corruption on well-known tech domains (found via external review, fixed)

An external code review of this project reported that the trained model flagged ordinary,
realistic URLs on well-known domains — `github.com/anthropics/claude-code`,
`en.wikipedia.org/wiki/Phishing`, `docs.python.org/...` — as phishing with 88-99% confidence, far
beyond just the bare-domain case in §5. Tracing this to the source data revealed **systematic
label corruption concentrated on specific major domains**, not a model-generalization problem:

| Host | Rows | Labeled "phishing" |
|---|---|---|
| `microsoft.com` | 490 | **366 (74.7%)** |
| `github.com` | 33 | **30 (90.9%)** |
| `apple.com` | 615 | 79 (12.8%) |
| `wikipedia.org` | 13,426 | 550 (4.1%) |

Reading the actual URLs confirmed these are wrong labels, not a defensible "trusted-domain-abuse"
pattern (which *is* real for hosting platforms like Google Forms — see below): `office.microsoft.com/en-us/visio`,
`www.microsoft.com/careers/`, `research.microsoft.com/~cmbishop/`, `github.com/joyent/node/wiki`
are all labeled "phishing" and are unambiguously ordinary pages.

**Fix (`ml/training/known_legitimate_domains.py`, kept):** rows labeled "phishing" are dropped
when the host is an exact match or subdomain of one of ~30 curated domains that have **no possible
attacker-hosting surface** — pure corporate/editorial/government sites with no free-hosting or
user-generated-content feature (this is a stronger claim than "it's a well-known brand", which
alone wouldn't justify removal). **Deliberately excluded**: Google (Forms/Sites/Docs),
GitHub (user pages), WordPress.com, Blogspot, and similar platforms, where trusted-domain-abuse
phishing is a real, documented technique — stripping "phishing" labels there would remove genuine
signal, not just noise. The list and its exclusions were audited by reading 3+ sample dropped URLs
from *every* domain on the list (not just the two largest), confirming ~1,785 rows worth of
corrected labels; the one caveat found (`salesforce.com`'s hits are on `*.my.salesforce.com`,
a per-customer CRM subdomain — not free public hosting, but not purely static content either) is
documented in that file rather than hidden.

This fix alone (before the scheme-invariance fix in §3.5) measurably improved held-out metrics —
see run 6 in §4's table — but did **not** fix the github.com/wikipedia.org-with-a-path
misclassifications, which turned out to have a different, second cause.

### 3.5 Scheme-sensitive features + an overly-blunt monotonic constraint (found via external review, fixed)

After §3.4's fix, `en.wikipedia.org/wiki/Phishing` (with a real path) was *still* misclassified,
and tracing individual feature values found two more real, permanent bugs:

1. **`url_length`, `special_char_count`, and `digit_count` were computed on the raw,
   un-normalized URL string.** `has_https` was already excluded from training for exactly this
   class of reason (§3.2), but these three features silently reintroduced the same
   scheme-presence sensitivity: since ~99.5% of this dataset's raw URLs have no scheme at all,
   typing `https://` in front of an otherwise-identical URL added 8 characters that shifted these
   features enough to flip predictions. Verified directly:
   `en.wikipedia.org/wiki/Phishing` (no scheme) scored 0.80 phishing-probability;
   `https://en.wikipedia.org/wiki/Phishing` (identical otherwise) scored 0.9987. **Fix (permanent,
   kept):** these three features are now computed on the scheme-stripped normalized URL, so a
   caller typing `https://`, `http://`, or nothing produces identical feature values.
2. **The `subdomain_count` monotonic constraint (added in run 5, §4) was too blunt.** After fix
   #1, `github.com`-shaped URLs (bare, `subdomain_count=0`) classified correctly, but
   `www.`/`en.`-prefixed URLs (`subdomain_count=1`) — the single most common, most benign pattern
   on the entire web — still didn't. The constraint forced "more subdomains -> only more
   phishing-like" across the *entire* range, when in reality 0->1 subdomain is overwhelmingly
   benign and only 1->4+ is the genuinely suspicious jump; a monotonic constraint cannot express
   that distinction. **Fix (kept):** removed the constraint on `subdomain_count` specifically,
   keeping it only on features that are unambiguous at *every* value (`is_ip_address`,
   `brand_keyword_outside_domain`, `suspicious_tld`, `has_at_symbol`).

**Net effect on aggregate metrics: negative.** Both fixes are individually correct and permanent,
but removing them cost more shortcut-driven predictive power than genuine signal was recovered —
see run 7 (final) in §4. This is itself a finding: on this dataset, a meaningful share of the
"good" earlier numbers came from artifacts the model was exploiting, not real phishing signal.
`train_url_classifier.py`'s `SANITY_LEGITIMATE_URLS` now includes these exact realistic-URL cases
permanently, alongside the original bare-domain list, so this class of regression can't silently
reappear.

---

## 4. Model training and evaluation (phase 2)

**Superseded by §10 — every row in the table below (including run 8) was evaluated on the old
per-URL random split, which a second external review found let over half the test set share a
hostname with training data. Run 8's numbers are NOT what's currently saved; §10.2 has the
domain-grouped re-evaluation (F1 0.827) that actually is.** Kept below as the audit trail showing
how the model evolved through the first review round.

`training/train_url_classifier.py` trains Random Forest and XGBoost on the processed features,
picks whichever has the higher validation F1, evaluates it on the untouched test split, and
serializes it to `saved_models/`. Spec section 9 targets: **F1 ≥ 0.90, precision ≥ 0.91,
recall ≥ 0.90**.

| Run | Config | Test accuracy | Test precision | Test recall | Test F1 | Meets targets |
|---|---|---|---|---|---|---|
| 1 (initial) | default hyperparameters, `has_https` included, buggy `path_length` | 0.913 | 0.913 | 0.913 | 0.913 | **Yes** — but see §5, this model was broken |
| 2 (after §3.1-3.2 fixes) | default hyperparameters | 0.912 | 0.907 | 0.918 | 0.912 | Precision narrowly missed |
| 3 (heavy regularization) | shallow trees, high min-leaf/child-weight, subsample/colsample, monotonic constraints | 0.857 | 0.840 | 0.882 | 0.860 | No — worse across the board |
| 4 (moderate regularization) | mid-depth trees, moderate min-leaf/child-weight | 0.895 | 0.885 | 0.908 | 0.897 | No |
| 5 | default hyperparameters + monotonic constraints (incl. `subdomain_count`) | 0.909 | 0.905 | 0.915 | 0.910 | Precision narrowly missed |
| 6 (after §3.4 label cleaning + dedup) | same as run 5 | 0.918 | 0.913 | 0.925 | 0.919 | **Yes** — but see §3.5, still broken on realistic branded URLs |
| 7 (after §3.5 fixes) | scheme-invariant features, `subdomain_count` unconstrained | 0.864 | 0.858 | 0.874 | 0.866 | No |
| **8 (final, after §7.3's conflicting-label fix — this is what's saved)** | same as run 7 | **0.870** | **0.868** | **0.874** | **0.871** | **No** |

XGBoost beat Random Forest on validation F1 in runs 1-6; **Random Forest was selected in runs 7-8**
(marginally higher validation F1 once XGBoost's monotonic-constraint advantage on the removed
shortcut features went away). Confusion matrix for run 8's test set: TN=4334, FP=666, FN=629,
TP=4370 (`ml/saved_models/url_classifier_metrics.json` has the exact numbers plus a trivial
majority-class baseline — accuracy 0.50, F1 0.0 — for comparison; the model clearly beats that
baseline, just not the spec's target).

**Run 2/5/6's precision gap (0.905-0.913) is most plausibly explained in part by the label noise
in §3.3** — there is a hard ceiling on precision when a nontrivial fraction of "legitimate" ground
truth is actually mislabeled phishing. **Runs 7-8's larger F1 drop (0.919 -> 0.871) is different:
it's the cost of removing two real shortcuts the model was exploiting** (§3.5) rather than data
noise — an honest, lower number is what's left over once those are gone.

---

## 5. The bare-domain finding (central limitation — read this before touching the model again)

### 5.1 What was found

Runs 1 and 2 both looked good on paper (met or nearly met every spec target) but **catastrophically
misclassified ordinary bare domains**: `google.com`, `example.com`, `github.com`, and similar
inputs were flagged phishing with ~95–99.98% confidence. This is not a rare edge case — a bare
domain is arguably the single most common input shape a real user would submit to this app.

The failure was traced to feature-value granularity, not just the artifacts above: after both
fixes, `nhptv.org` (`url_length=9`, all other features zero) was scored phishing (proba 0.98),
while `nhptv.org/` — identical in every feature except `url_length=10` — was scored legitimate
(proba 0.08). A single character was flipping the verdict.

### 5.2 Hypothesis testing: is it overfitting?

**Language corrected by a later external review** (§10): this section originally claimed the
regularization experiments below "rule out overfitting as the cause" — too strong a claim for what
was actually shown. What the experiments demonstrate is narrower: two specific regularization
configurations (runs 3 and 4 in the table above: shallower trees, much larger
`min_samples_leaf`/`min_child_weight`, subsampling) did not fix the bare-domain sanity check, and
made aggregate metrics worse too. That is evidence against *those specific configurations* helping,
not proof that no form of overfitting is involved or that other regularization approaches would
fail the same way. Treat §5.3 below as a strong, well-evidenced explanation, not an established
mathematical fact.

### 5.3 A likely explanation: an information ceiling, not a bug

Direct query against the training data for the exact feature-bucket that `google.com`/`example.com`
fall into (`path_length == 0`, `host_length ≤ 12` — bare, short domains):

```
label
phishing      1392   (65.4%)
legitimate     736   (34.6%)
```

**This bucket was 65% phishing in the training data used at the time.** URL-structure-only
features (length, subdomain count, special characters, TLD, etc.) carry no brand-recognition or
domain-reputation signal — `google.com` and an anonymous freshly-registered short malicious domain
are structurally near-identical to this feature set. A model fit to this training distribution
would reasonably lean "phishing" for unbranded short domains, since in this slice of the data
that's the majority label. **This is a strong, evidence-backed explanation for the bare-domain
failure, not a proven "unavoidable ceiling"** — a second external review correctly pushed back on
that framing (§10): it's possible a differently-composed or larger dataset, additional features
(brand/reputation signal), or other approaches not yet tried could still improve this. What's
established is that the specific fixes tried so far didn't, not that none could.

This is precisely the gap the spec's architecture (section 2) already anticipates: *"An admin-
managed whitelist that short-circuits the pipeline for trusted domains."* The intended mitigation
for "is this well-known domain safe" was never meant to be the ML model's job — it's the
whitelist's.

### 5.4 Built-in regression guard

`train_url_classifier.py` now runs a `_sanity_check()` after every training run: 8 known-legitimate
bare domains (`google.com`, `example.com`, `github.com`, `wikipedia.org`, `microsoft.com`,
`amazon.com`, `apple.com`, `cnn.com`) plus 4 realistic full URLs added after §3.5's investigation
(`https://en.wikipedia.org/wiki/Phishing`, `https://www.python.org/downloads/`,
`https://github.com/anthropics/claude-code`, `https://www.nytimes.com/section/technology`), and 4
phishing-shaped URLs (an IP-address login path, a paypal-subdomain + suspicious-TLD combination,
an apple-verify domain with an `@`, a facebook typosquat on `.tk`). Final model (run 7): **1/12
legitimate cases correct, 4/4 phishing-shaped URLs correct.** The script prints an explicit warning
and a recommendation against switching `URL_CLASSIFIER_BACKEND` to `trained` if this check fails —
future retraining attempts (e.g. after sourcing a different dataset) get this check automatically;
don't remove it without a good reason.

---

## 6. Decision and alternatives considered

**Decision (2026-09-05): keep `CLASSIFIER_BACKEND=rule_based` in production. Document this
limitation and proceed to phase 5 (evaluation) rather than invest further engineering time here.**
This was a deliberate scope call, not a default — five alternatives were identified and weighed:

| # | Alternative | Cost | Why not chosen (this round) |
|---|---|---|---|
| 1 | Bundle a static top-domains popularity list (e.g. Tranco / Cisco Umbrella top 1M) as an automatic pre-check distinct from the admin whitelist | Low — no retraining, no network calls, offline set lookup | Not needed yet; deferred as future work |
| 2 | Add real domain-reputation features (WHOIS registration age, DNS/certificate issuance recency, hosting ASN reputation) | High — needs live network lookups per request, risks the spec's ≤500ms latency target unless cached | Out of scope for this pass; the most "production-realistic" fix per anti-phishing literature |
| 3 | Expand the brand-keyword list (currently 15 entries) to hundreds of known brands | Low | Only closes part of the gap — doesn't help unbranded legitimate domains like `sfwa.org` |
| 4 | Calibrate model confidence / add an "uncertain" middle zone instead of forcing a binary call | Medium | Doesn't fix accuracy, only the appearance of false confidence; worth doing eventually |
| **5** | **Document as a known limitation, ship `rule_based`, move on to phase 5** | **Lowest** | **Chosen** — time/scope tradeoff for thesis timeline; the rule-based classifier is transparent, already tested, and the trained model's failure mode is now well-understood and written up rather than hidden |

The trained model and its metrics remain at `saved_models/` (currently `url_classifier_random_forest.joblib`
— §4's run 7 selected Random Forest, not XGBoost; the filename tracks whichever model actually won)
/ `url_classifier_metrics.json` as an artifact for future work — not deleted, just not wired in as
the default. **This decision predates §7's external review and §3.4-3.5's follow-up fixes** —
revisit §8 for where things stand now.

---

## 7. External code review (2026-09-06): security, validation, and evaluation-methodology fixes

A external review of the running system (not just the ML pipeline) found six categories of issues,
all independently verified against the live code/API before being fixed (not taken on faith).
Full detail lives in code comments at each fix site; this is the index.

### 7.1 Critical: admin-whitelist bypass (security)

`app/pipeline/preprocessing.py`'s `get_domain()` and `app/pipeline/feature_extraction_url.py`'s
host extraction both did `urlparse(url).netloc.split(":")[0]` to get the hostname. For a URL with
userinfo, e.g. `https://trusted.com:password@evil.example/`, this takes `trusted.com` — the
userinfo *username* — instead of the real host `evil.example`. **Reproduced live**: whitelisting
`trusted.com` and submitting that exact URL returned `{"classification":"legitimate",
"confidence_score":1.0,"whitelisted":true}`, a full bypass. **Fixed** by using
`urlparse(url).hostname` (which parses per the URL spec and handles userinfo correctly) in both
places. Regression test: `tests/test_detect.py::test_whitelist_userinfo_bypass_is_blocked`.

Related: a whitelisted URL used to skip analysis of an accompanying `email_text` entirely (a
phishing email can legitimately reference or spoof a trusted domain). **Fixed** in
`app/api/detect.py`: the whitelist fast-path (skip classification, return legitimate/1.0
immediately) now only fires for URL-only submissions; a mixed submission still runs the email
through the classifier, while still reporting `whitelisted: true` for the URL portion. Regression
test: `test_mixed_submission_still_analyzes_email_when_url_whitelisted`.

### 7.2 Input validation

Confirmed live before fixing: a whitespace-only URL or email returned a normal `legitimate`
response instead of a 400; `"not a url"` was accepted; `"http://["` raised an unhandled exception
(500) from deep inside `urlparse`. `app/schemas/detection.py`'s `DetectRequest` now: trims
whitespace (so a whitespace-only value behaves as "not provided"), validates the URL against a
permissive hostname-shape regex, enforces `max_length` caps (2048 for `url`, 50,000 for
`email_text`), and lets `urlparse`'s own `ValueError` on structurally invalid input surface as a
normal pydantic validation error (→ 400) instead of an unhandled exception. `app/api/detect.py`
also now separates parsing/feature-extraction failures (`ValueError` → 400, the client's fault)
from classifier failures (any other exception → 500, per spec section 6) — previously both were
lumped into one handler that always returned 500.

### 7.3 Reproducibility

- **Deduplication**: `build_features.py` now drops exact-duplicate URLs before splitting (41
  dropped in the final run) — previously 6 duplicate URLs (12 rows) survived into the processed
  dataset, including one shared between train and test. **Also found during this pass** (not in
  the original review, found independently while re-checking the raw data): 6 URLs across the raw
  CSV carry **conflicting ground truth** — the identical URL string labeled both `benign` and
  `phishing`. `drop_duplicates(keep="first")` alone would have silently and arbitrarily resolved
  these by file order; fixed to detect and drop every row for a conflicting URL instead (§1).
- **Host overlap**: now measured and printed by `build_features.py` (32.3% of test hosts also
  appear in train) rather than left unmeasured — see §1.
- **Label-cleaning audit**: broadened from a couple of examples to reading 3+ sample dropped rows
  from every domain in `known_legitimate_domains.py` (~30 domains, not just the two largest) — see
  §3.4.
- **Frozen artifacts**: `random_state=42` throughout (`build_features.py`'s balancing and split,
  `train_url_classifier.py`'s model init); `MAX_ROWS_PER_CLASS=50000` (env-overridable, default
  documented in code); the exact raw dataset is `sid321axn/malicious-urls-dataset` via
  `download_dataset.py`. Reproduction command sequence is in §9.
- **Confusion matrix / baseline / FP-FN**: `train_url_classifier.py`'s `_evaluate()` now reports a
  full confusion matrix, and `_baseline_metrics()` reports a trivial majority-class baseline for
  comparison — both saved in `url_classifier_metrics.json` (see §4).
- **Separate URL/email/combined results**: not done — there is no trained email model to report
  separate results for (see §8.1). The `_combine()` blending logic (URL model score + rule-based
  email heuristic score) is unit-tested in `tests/test_classifiers.py`, but its output was never
  independently evaluated against a labeled email dataset.

### 7.4 Evaluation and inference correctness

- **Train/serve threshold mismatch**: `train_url_classifier.py`'s offline evaluation used
  `model.predict()` (standard 0.5 probability threshold), while the live
  `TrainedURLClassifier`/`_combine()` used a 0.4 threshold copied from `RuleBasedClassifier`'s
  unrelated hand-tuned heuristic scoring. **Fixed**: `_combine()` now defaults to 0.5 (matching
  offline evaluation); `RuleBasedClassifier` explicitly opts into its own 0.4.
- **Model reloaded from disk per request**: `get_classifier()` deserialized the joblib model file
  on every `/detect` call when `CLASSIFIER_BACKEND=trained`. **Fixed**: cached via `lru_cache`,
  keyed on the backend string (not a single global cache) so tests can still switch backends
  within one process.
- **`processing_time_ms` excluded the DB commit**: elapsed time was measured before `db.commit()`.
  **Fixed**: the log row is now written with a placeholder, then patched with the real elapsed
  time (measured after the first commit) in a second, cheap single-field update — see
  `app/api/detect.py`'s comment for why a row can't contain its own commit duration in one pass.
  The full HTTP response transmission time is still not included; that's a client-side/network
  measurement the handler has no visibility into.
- **Concurrency probe confound**: `ml/evaluation/run_evaluation.py`'s throughput probe opened a
  brand-new `httpx.Client` (and therefore a new TCP connection) per request and never checked
  response status. **Fixed**: a single shared client across all worker threads, plus explicit
  failure counting. This means the previously-reported 5.51 req/s concurrent-throughput number is
  **not yet re-validated** with the corrected benchmark — don't cite it as a clean measurement of
  SQLite's ceiling until it's re-run (see §8.2).
- **Hardcoded backend label in eval output**: `results.json` hardcoded the string `"rule_based"`
  regardless of what was actually live. **Fixed**: `--classifier-backend` is now a required CLI
  argument, used verbatim in the output.
- **NLTK downloading at request time**: `_load_nltk()` called `nltk.download()` synchronously
  inside a request on first use (measured 1.4-2.4s for a single whitespace-only email request).
  **Fixed**: request-time code only calls `nltk.data.find()` (no network access) and falls back to
  a minimal built-in stop-word list if corpora aren't present; `scripts/download_nltk_data.py` is
  the one-time setup step to run during deploy, outside any request path.

### 7.5 Tests and documentation

- Added `tests/test_classifiers.py` (caching behavior, threshold semantics, trained-backend
  inference), `tests/test_metrics.py` (admin gating, hand-verified precision/recall/F1/accuracy
  against seeded log rows), `tests/test_logs.py` (ownership: own logs visible, other users' logs
  hidden, admin sees all including anonymous requests, 403/404 on specific-log access), and
  expanded `tests/test_detect.py` with the validation/bypass/mixed-submission regression tests
  above plus a real pipeline-failure test (mocked classifier raises → 500, not a crash).
  `test_detect_email_text` no longer accepts either classification as a pass — it's split into a
  deterministic benign case and a deterministic phishing-shaped case (real header mismatch +
  urgency keywords), so it actually exercises detection quality.
- Frontend copy corrected: `DetectPage.tsx` claimed "AI-powered threat analysis" and "Hybrid ML and
  language analysis" while the live classifier is a hand-weighted heuristic with no ML or NLP
  model in the loop. Changed to describe what's actually running.
- `SECRET_KEY`: `app/main.py` now refuses to start when `ENVIRONMENT=production` and the secret is
  still the insecure code default, and logs a warning in development. No `.env` existed in this
  dev environment, so the default was live — this closes that gap without requiring a human to
  remember to check.
- **Raw email storage**: `DetectionLog.input_data` stores the full raw `email_text` a caller
  submits, indefinitely, in plaintext, with no redaction. This is a real privacy/PII consideration
  for anything beyond a local dev/thesis-demo database — not fixed in this pass (no retention
  policy or redaction was implemented), flagged here so it's not silently forgotten if this ever
  runs against real user-submitted email content.
- **Deployment configuration**: see the new "Deployment" section in `backend/README.md` for
  `ENVIRONMENT`/`SECRET_KEY`/`CLASSIFIER_BACKEND`/NLTK-setup instructions consolidated in one place.

---

## 8. Where this actually stands (updated 2026-09-06, superseded by §10's numbers)

§6's original decision (ship `rule_based`, document the bare-domain gap, move on) was made without
knowing two things later evaluation surfaced: (1) `rule_based`'s real held-out recall is **1.69%**
(§12.7's re-run under the corrected evaluation script; originally measured as 1.74% before dataset
and script fixes), not just "weaker than the trained model on bare domains" — it is not functioning
as a phishing
detector in any meaningful sense; (2) the trained model's real ceiling on *this* dataset, once
every identified bug is fixed, was F1 0.866-0.871 under this document's *first* evaluation
methodology — below spec, not 0.91. **§10 corrects this further: under a properly domain-grouped
split, it's F1 0.827.** Every number in this section predates that correction; the conclusion
(neither backend is ready) still holds, more strongly.

**Neither backend currently meets the spec's targets.** The options, in the reviewer's own words:
either invest further — a better/differently-composed dataset is the highest-leverage next step,
since every fix so far has been about *not corrupting* the signal that's there, not adding new
signal (brand recognition, domain reputation, a cleaner benign-class sample) — or **agree a
narrower URL-only thesis scope with the supervisor** and report this full investigation (including
the honest numbers and their evolution across two review rounds) as the methodology and results,
which is defensible, evidenced work even though it doesn't hit the original targets. Building a
trained email classifier (TF-IDF or fine-tuned BERT, per spec section 3) is a distinct, multi-day
undertaking — a new dataset (e.g. Enron), a new feature/training pipeline, and its own evaluation —
not something folded into this fix pass; `RuleBasedClassifier`'s weighted heuristic remains the
only email-side signal.

### 8.1 What "validate how URL and email predictions combine" actually has

`_combine()`'s weighted-blend logic (URL model score at a fixed weight vs. the email heuristic's
own weighted checks) is unit-tested for its threshold semantics
(`tests/test_classifiers.py::test_combine_default_threshold_matches_model_predict_semantics`) and
exercised end-to-end via the mixed-submission regression test (§7.1). What's still missing: an
actual accuracy evaluation of the *combined* score against labeled mixed url+email examples — there
is no such labeled dataset in this project, and building one is part of the same scope question as
the email classifier itself.

### 8.2 Immediate next step if evaluation work continues (done — see §12.7)

This used to say "re-run the concurrency probe before citing any throughput number." That re-run
happened (§12.7): concurrent throughput measures **41.5 successful req/s**, clearing the spec's 20
req/s target. The earlier 5.51 req/s was indeed the benchmark confound, not a real limitation.

---

## 10. Second external review (2026-09-06): dataset rigor, and what actually changed

A second, independent review went deeper into the raw dataset itself — inspecting all 651,191 raw
rows, the processed parquet, and the test CSV — and found the label-conflict and evaluation
problems were more extensive than this document had previously stated, plus caught two places
where earlier sections of this document overstated their own conclusions (now corrected at §3.2,
§5.2-5.3). Every claim below was independently re-derived from the raw CSV before acting on it, not
taken on faith — see the verification commands in each subsection.

### 10.1 Label conflicts are far more extensive than the 6-URL finding in §3.4

- **10,066 duplicate url+label rows** in the raw CSV (extra copies beyond the first occurrence) —
  a bigger number than this document's earlier "6 duplicate URLs" because that number came from
  *after* this project's own cleaning steps had already run; 10,066 is the raw, unfiltered count.
- **After normalizing** (default scheme when absent, lowercase host, empty path → `/`, fragment
  dropped — deliberately loose, so treat these as *candidates*, not confirmed duplicates of the
  same real-world page), **thousands of URL groups carry conflicting benign/phishing labels**: the
  review measured 3,940 groups / 7,909 rows; re-deriving the same idea independently (slightly
  different normalization choices — e.g. whether the query string is retained) gave 4,043 groups /
  8,116 rows. The exact count depends on normalization details the review itself flagged as
  judgment calls, but the phenomenon — thousands of near-duplicate URLs with contradictory ground
  truth, not just 6 — is real and reproducible.

**Fix**: `build_features.py` now quarantines (drops **and logs to
`quarantined_conflicts.csv`**, rather than silently keeping one label) any row whose URL, or its
normalized form, appears with more than one label anywhere in the raw+cleaned pool. This runs
*after* the domain-based cleaning (§3.4), so counts don't double up. Current run: **~8,110 rows
quarantined**. This is a stricter standard than exact-string matching alone (§3.4's original fix)
and removes an order of magnitude more disputed rows.

### 10.2 The test set didn't test generalization to unseen domains — fixed with a domain-grouped split

The review measured, on the dataset as it stood at the time: only 1 exact URL shared between train
and test, but **2,344 hostnames** shared, and **5,026 of 10,000 test rows** (just over half) had a
hostname that also appeared somewhere in train. A held-out score computed that way can't support a
claim about generalizing to domains the model has never seen — it's partly measuring "did this
model memorize this host's typical URL shape," which is a different (still possibly useful, but
different) question.

**Fix**: `build_features.py` now splits by **registrable domain** (`SPLIT_STRATEGY=grouped`,
the new default, via `sklearn.GroupShuffleSplit`) rather than per-URL. Verified: host overlap
between train and test is now **0/5,007 (0.0%)** — not reduced, eliminated by construction.
**Update (third review round):** the registrable-domain extraction originally used the same
simple last-two-labels heuristic as `feature_extraction_url.py`, which a review measured was
collapsing 1,768 unrelated `*.co.uk` hosts into one fake "co.uk" group -- unnecessary grouping the
domain-split was supposed to prevent, not cause. Fixed: now uses `tldextract` (Public Suffix
List-aware, pinned to its bundled offline snapshot via `suffix_list_urls=()` so the grouping is
reproducible run to run regardless of network access) for the split-grouping use specifically.
`feature_extraction_url.py`'s own use of the naive heuristic for live inference features
(`subdomain_count`, `brand_keyword_outside_domain`) is a separate, not-yet-fixed limitation --
changing that touches served features and would need its own retrain/revalidation cycle.

**The side-by-side comparison, same cleaned data, same code, same hyperparameters, only the split
strategy differs:**

| Split strategy | Host overlap (train∩test) | Test accuracy | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
| Random per-URL (old methodology) | 32.4% of test hosts | 0.879 | 0.878 | 0.880 | 0.879 |
| **Domain-grouped (new default)** | **0.0%** | **0.841** | **0.841** | **0.837** | **0.839** |

**The ~4-point F1 gap is an observed difference between evaluation protocols, not something to
attribute entirely to leakage** — a third review correctly pushed back on this document's earlier,
stronger claim ("the cost of the old methodology's leakage, not a real change in the model").
Host leakage is the mechanism that makes the two splits structurally different, but the two
protocols also produce different test populations and different class proportions (random-split
test was near-50/50; domain-grouped test skews differently because whole domains, not individual
rows, get assigned to a split) — both of those independently affect precision/recall/F1, separate
from whatever the model "learned" from seeing a host before. What's solid: the domain-grouped
number is the methodologically correct one for a generalization claim, and it's lower. What's not
established: precisely how much of the gap is leakage specifically versus these other confounds.
The domain-grouped number (F1 0.839, current pipeline after all fixes) is the one to cite as this
project's measured generalization performance; the random-split number (F1 0.879) is a comparison
point, not a competing "real" result. Reproduce both:
`python -m ml.training.build_features` (grouped, default) and
`SPLIT_STRATEGY=random python -m ml.training.build_features`, then
`python -m ml.training.train_url_classifier` and
`FEATURES_FILE=url_features_random_split.parquet ARTIFACT_SUFFIX=_random_split python -m ml.training.train_url_classifier`
respectively — outputs land in `saved_models/url_classifier_metrics.json` and
`url_classifier_metrics_random_split.json` without overwriting each other.

### 10.3 Formatting differs sharply between classes (confirms and extends §3.1-3.2)

| Property | Benign | Phishing |
|---|---|---|
| Explicit URL scheme present | 8.26% | 26.41% |
| No path at all | 0.066% | 23.14% |
| Root path or no path | 12.20% | 25.41% |

This is the same phenomenon §3.1 (path) and §3.2 (scheme) already found and fixed at the feature
level (`path_length` treats no-path/root-path identically; `url_length` etc. are now
scheme-invariant). The review's note that "scheme formatting still affects features such as URL
length" was accurate about the *original* bug (§3.5) — after §3.5's fix, scheme presence no longer
affects `url_length`/`special_char_count`/`digit_count`, verified directly:
`extract_url_features("example.com")` and `extract_url_features("https://example.com")` now
produce identical values for those three fields. What §3.5's fix does *not* and cannot address is
any *other* systematic formatting difference correlated with collection source (e.g. path/query
conventions) beyond scheme and trailing-slash — those would require a differently-sourced dataset
to fully rule out, not a further feature tweak.

### 10.4 Domain-based label cleaning: quarantine methodology adopted, with/without comparison run

The review's critique of §3.4 was fair: "a familiar domain alone does not prove a phishing label is
wrong" is correct, and this document's earlier language ("provably wrong," "impossible") claimed
more certainty than the evidence supports, even though the underlying reasoning (no plausible
attacker-hosting surface on a pure corporate/editorial domain) is still a strong basis for
suspicion. Response:

- Language throughout corrected from "provably wrong" to "very strong evidence of a labeling
  error" (see `build_features.py` and `known_legitimate_domains.py`).
- Dropped rows are now **logged to `domain_cleaned_phishing_labels.csv`** (previously just
  counted and discarded) — auditable and reversible.
- The step can be disabled entirely (`SKIP_DOMAIN_CLEANING=1`) to train and compare with/without.
  **Result of that comparison** (domain-grouped split both times, so this isolates the cleaning
  step's effect as cleanly as a domain-grouped split allows — see caveat below):

| | Test precision | Test recall | Test F1 |
|---|---|---|---|
| With domain-based cleaning (current default) | 0.828 | 0.825 | 0.827 |
| Without domain-based cleaning | 0.875 | 0.837 | 0.855 |

**This is counter to the naive expectation** that removing mislabeled "phishing" rows should only
improve precision. It didn't, in this comparison. The honest caveat: because `GroupShuffleSplit`'s
partitioning depends on which domains/rows are available, removing 1,785 rows changes the group
pool going into the split, so the two runs above don't share the *exact* same train/val/test
domain assignment — this comparison is informative but not a perfectly isolated ablation. A fully
controlled version (fix the domain-to-split assignment first, then apply/skip the label cleaning
within it) was not done here for time reasons. Read this as "the effect of domain-based cleaning on
this metric is unclear and possibly small or even negative," not as grounds to revert the cleaning
— the underlying evidence for those labels being wrong (§3.4) stands regardless of how it moves
aggregate numbers on a differently-imperfect dataset.

### 10.5 What a labeled email dataset would require (not addressed, scope boundary confirmed)

The raw dataset has URL strings and a class label only — no email bodies, subjects, headers, or
SPF/DKIM/DMARC observations, and no paired URL+email examples. This confirms §8's existing
conclusion: a trained email/BERT classifier needs its own labeled dataset (e.g. a phishing-email
corpus, not a generic spam corpus — the review correctly notes spam ≠ phishing) and is out of scope
for this fix pass.

### 10.6 Net effect on what to actually present

**Superseded by §14's consolidated table — this section's F1 0.827 was current as of the second
review round only; a third review round found and fixed a further grouping bug (§12.2), moving the
verified current number to F1 0.839.** Kept here for the audit trail. The general point still
holds: present the domain-grouped number as this project's honest, defensible number for the
trained URL model, not the higher figures from before §10; present the random-split number (F1
0.879) only as a methodology comparison, explicitly labeled as measuring something weaker than
true unseen-domain generalization. All of them are below the spec's F1≥0.90 target. Preserve the
raw dataset unmodified (already true — all cleaning happens in `build_features.py` at processing
time, never mutating `ml/data/raw/`); the recommendation to add an independent, more recent test
set with documented label provenance is noted as future work (§8), not attempted here.

---

## 12. Third external review (2026-09-06): pipeline bugs in the fixes themselves

A third review, run after §7-§10's fixes, found six more issues — this time mostly bugs *in the
fixes*, not new dataset findings. Each was independently verified before acting.

### 12.1 The API rejects some held-out test URLs (fixed)

`DetectRequest`'s validation (§7.2) rejected some legitimate held-out test rows outright, and
`ml/evaluation/run_evaluation.py` called `raise_for_status()` unconditionally, so a full evaluation
run would crash at the first rejection. Verified two distinct causes: (1) a handful of raw dataset
rows contain literal stray quote characters or other genuinely malformed content baked into the
URL string (e.g. one row is outright binary garbage) — these are correctly rejected, a raw-data
quality issue, not a validation bug; (2) the hostname regex disallowed underscores, incorrectly
rejecting real (if RFC-noncompliant) hostnames like `good_guild.w.interia.pl` that appear as
legitimate-labeled examples in this project's own dataset. Fixed: the regex now allows underscores
in hostname labels. `run_evaluation.py` no longer crashes on a rejection — it records each one
(URL, true label, status code, reason) in `rejected_samples` in the output and prints a
label-breakdown warning, rather than either crashing or silently dropping them (dropping silently
would risk skewing reported accuracy if rejections aren't evenly split between classes).

### 12.2 Registrable-domain grouping was still wrong for multi-part TLDs (fixed)

Covered above in §10.2's update — the last-two-labels heuristic used for the domain-grouped split
collapsed 1,768 unrelated `*.co.uk` rows into one fake "co.uk" group. Fixed with `tldextract`
(Public Suffix List-aware, offline snapshot pinned via `suffix_list_urls=()`). Rebuilding the
dataset and retraining with this fix (and the majority-baseline fix in §12.6) moved the primary
result from F1 0.827 to **F1 0.839** (precision 0.841, recall 0.837) — an improvement, since the
`.co.uk` bug was distorting the split's class balance and domain grouping, not helping it.

### 12.3 Domain-based cleaning ran before conflict detection (fixed)

A real ordering bug: `build_features.py` used to apply the known-legitimate-domain phishing-label
cleaning (§3.4) *before* checking for conflicting labels (§10.1). If a URL had conflicting
benign/phishing labels and the phishing copy happened to be on a known-legitimate host, domain
cleaning silently removed that copy first — resolving the disagreement by domain reputation alone
before the conflict-quarantine step ever got a chance to flag it as disputed. Fixed: conflict
detection and quarantine now run first, against the original binary (benign/phishing) pool; domain
cleaning runs second, against whatever remains. Effect on counts: quarantine now catches 8,116 rows
(essentially unchanged); domain cleaning subsequently finds 1,782 rows (down slightly from 1,785,
since a few were already removed by the conflict check) — the two steps no longer interact in an
unexamined way.

### 12.4 Comparison model artifacts weren't properly isolated (fixed, and real damage found)

A real, confirmed data-loss bug: the primary training run's cleanup logic deleted every
`url_classifier_*.joblib` file except ones ending in `_random_split` — which correctly protected
that one comparison run, but silently deleted the `_no_domain_cleaning` model artifact from §10.4's
comparison the next time the primary model was retrained (confirmed: only that run's metrics JSON
survived; the `.joblib` was gone). `classifiers.py`'s loader had the same class of bug — it
excluded only `_random_split`, leaving other comparison artifacts eligible to be picked up as "the"
trained model if the primary file were ever missing.

**Fix**: eliminated the whole bug class via physical separation rather than a smarter exclusion
list. The primary model now always saves to a single fixed path,
`saved_models/url_classifier.joblib` (algorithm-agnostic filename — which model won is recorded
inside the file, not in its name), which `classifiers.py` loads directly with no glob at all. Every
comparison/experiment run (`ARTIFACT_SUFFIX` set) writes under `saved_models/experiments/` instead
— a different directory, not just a different filename — so there is no pattern-matching step left
that could get the exclusion list wrong. The lost `_no_domain_cleaning` model was not
re-generated (its metrics are preserved and are what §10.4 cites; only the reusable model object
was lost, not the recorded result).

### 12.5 Performance benchmark: further corrections

`_throughput_probe()` now reports `throughput_rps` as successful requests per second specifically
(previously it counted failed requests in the numerator, which would let a mostly-erroring backend
still report a deceptively high "throughput"), alongside a separate `attempted_rps` and
`error_rate`. `_run_detection_pass()` now also records each request's client-observed wall-clock
round-trip time (previously only the server-reported `processing_time_ms` in the response body was
available), and `run_evaluation.py`'s output reports mean/p50/p95/p99 of that distribution.
**Not fixed, by design, not oversight**: `processing_time_ms` still excludes the second (metadata-patch)
commit's own duration — see `app/api/detect.py`'s comment on why a logged row structurally cannot
contain its own full commit time without infinite regress; this is a documented, accepted
limitation, not a bug. The evaluation was re-run against the corrected script after all of the
above fixes landed — see §12.7 for the fresh, current numbers.

### 12.6 Two research-claim corrections (fixed)

- **The majority-class baseline was computing its majority class from the test set's own
  labels**, not the training set's — meaning the "naive baseline" was quietly using the answer key
  it's supposed to not have access to. Confirmed: on this project's own current split, the training
  majority class is `phishing` while the primary reported baseline (before the fix) showed
  `legitimate`, i.e. the bug was live and changing which baseline was reported. Fixed:
  `_baseline_metrics()` now takes both `y_train` and `y_test`, computes the majority class from
  training labels only, and evaluates that fixed prediction against test labels.
- **The random-vs-grouped-split F1 gap attribution was overstated** — see §10.2's update above.

### 12.7 Fresh phase-5 evaluation, run after all of the above fixes

Re-ran `ml/evaluation/run_evaluation.py` against a clean DB and empty whitelist, `rule_based`
backend, the full current held-out test set (11,296 rows). Full output in
`ml/evaluation/results.json`.

- **28/11,296 rows rejected** by input validation (0.25%) — 16 legitimate, 12 phishing. Read
  through all 28: they're genuinely malformed (raw dataset rows with stray literal quote
  characters baked into the string, one row of outright binary garbage, a couple of edge cases like
  a URL ending in a bare `.`) rather than input the API should accept. Not silently dropped: listed
  individually with their true label in `results.json`'s `rejected_samples`.
- **Classification quality on the remaining 11,268 rows**: accuracy 0.509, precision 0.614,
  **recall 0.0169**, F1 0.0329. Consistent with (and now measured on the fully-corrected pipeline,
  superseding) the earlier 1.74% recall finding — `rule_based` is still not a functioning phishing
  detector.
- **Latency**: server-reported average 13.3ms; client-observed round-trip p50 17.0ms, p95 20.2ms,
  p99 39.4ms — comfortably under the spec's 500ms target.
- **Throughput — the one number that flipped**: concurrent probe (40 concurrency, 500 requests)
  now measures **41.5 successful req/s**, clearing the spec's 20 req/s target, with a 0.6% error
  rate. The previously-reported 5.51 req/s was a real measurement of a real confound (a new TCP
  connection per request in the old benchmark code), not evidence of an actual server-side
  concurrency ceiling — once that confound and the WAL journal-mode fix (§10, `app/db/base.py`)
  were both in place, the throughput NFR is met.

---

## 14. Fourth external review (2026-09-06): reproducibility gaps, and the tuning result

A fourth review, run after §12's fixes, independently re-verified 36/36 tests passing, the
99,989-row processed count, zero exact-duplicate/conflicting URLs, zero registrable-domain overlap
across all three splits, and that the saved model's metrics matched its JSON exactly. It found four
more issues, all documentation/reproducibility gaps rather than new correctness bugs, plus flagged
that the tuning experiment (§12, `tune_url_classifier.py`) hadn't produced a result yet at the time
of that review.

### 14.1 Reproduction commands used inconsistent databases (fixed)

Real bug in §13's instructions: the `uvicorn` command didn't set `DATABASE_URL`, so the server used
its code-default `phishing_detection.db`, while `run_evaluation.py` defaults `--db-path` to
`ml/evaluation/eval.db` — a different file. Following the documented commands literally would let
`/detect` write logs into one database while the evaluator tried to attach ground-truth labels to a
different, unrelated one, matching zero rows. **Fixed**: §13's commands now export one
`DATABASE_URL` and reuse it for the server, admin creation, and `--db-path` explicitly.
`_label_logs()` also now raises an error if the matched-row count doesn't equal the expected count,
instead of printing a soft "matched 0/11268" that a reader could miss — a silent 0%-match would
otherwise make `/metrics`' accuracy/precision/recall/F1 look computed when the labeled sample is
actually empty.

### 14.2 Comparison experiments regenerated; causal effect remains unmeasured

The no-domain-cleaning model and metrics have been regenerated with the current
PSL-aware pipeline. Its precision is 0.815 versus 0.841 for the primary model.
However, these experiments have different sampled records and domain assignments
(8,794 versus 11,296 test rows). This does not isolate the effect of cleaning, and
does not establish that the earlier reversal was caused by the grouping bug.
A controlled comparison must freeze a common evaluation population and domain
assignment before varying the training-data cleaning policy.

### 14.3 Reproducibility: exact dependency pins and a verified determinism claim (fixed)

- `tldextract>=5.1` allowed different installs to bundle different Public Suffix List snapshots,
  which can change the domain-grouped split's composition. Pinned to the exact installed version
  (`==5.3.2`) in `requirements.txt`, along with `pandas`, `numpy`, `scikit-learn`, and `xgboost` --
  the packages that actually affect the numbers in this document, not just API compatibility.
- This document previously hedged that "exact row/split counts vary by a handful... between runs
  of the same command," without ever having verified that. **Verified now**: ran
  `build_features.py` twice in a row with identical code/config -- both runs produced byte-identical
  row counts, split sizes, and per-split label balances. The earlier hedge was unnecessary and has
  been removed (§13).

### 14.4 Tuning result: confirms the earlier finding, doesn't beat it

`tune_url_classifier.py` (§12) finished: `RandomizedSearchCV` with `StratifiedGroupKFold`
(domain-disjoint folds), 25 configs x 4 folds, both Random Forest and XGBoost. Best: Random Forest,
CV F1 0.831, **test F1 0.832** (precision 0.813, recall 0.852) -- slightly *below* the primary
model's default-hyperparameter F1 0.839, and its bare-domain sanity check (1/12) is no better than
the primary's (2/12). This is a real search, not a few hand-picked configs, and it confirms rather
than overturns the earlier finding (§5.2, now correctly hedged as "strong evidence, not proof"):
hyperparameter tuning is not the lever that closes this gap on this dataset. Saved as a comparison
artifact only (`saved_models/experiments/`), consistent with treating this as "implemented
experiment infrastructure," not evidence of improved performance.

### 14.5 Consolidated results table (cite this one)

Every trained-model number this project has produced, in one place, all from the current pipeline
except where noted. All are URL-only; none include an email/BERT component (§8, §10.5 -- not
attempted, out of scope for this pass).

| Configuration | Split | Test accuracy | Test precision | Test recall | Test F1 | Sanity check (legit/phish) |
|---|---|---|---|---|---|---|
| **Primary (current default hyperparameters, all cleaning)** | Domain-grouped | 0.841 | 0.841 | 0.837 | **0.839** | 2/12, 4/4 |
| Tuned (RandomizedSearchCV + StratifiedGroupKFold) | Domain-grouped | 0.830 | 0.813 | 0.852 | 0.832 | 1/12, 4/4 |
| No domain-based cleaning | Domain-grouped | 0.830 | 0.815 | 0.840 | 0.827 | not re-run |
| Random per-URL split (methodology comparison, not a generalization claim) | Random | 0.879 | 0.878 | 0.880 | 0.879 | not re-run |
| `rule_based` (live default) | N/A -- live API, not this table's train/test split | — | 0.614 | **0.0169** | 0.033 | N/A |

**Bottom line**: F1 0.839 (primary, domain-grouped) is the number to cite for the trained URL
model. It does not meet the spec's F1≥0.90 target. Tuning does not close the gap. The live default
(`rule_based`) is far worse on recall specifically and is not a functioning phishing detector.
Neither backend is ready to ship; see §8 for the path forward (better/differently-sourced dataset,
or a narrower URL-only thesis scope).

---

## 15. Reproducing the API evaluation

Run from `backend` with the installed virtual environment. These commands work
in PowerShell and do not need shell-specific environment-variable syntax:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluation.managed_run --backend trained
.\.venv\Scripts\python.exe -m ml.evaluation.managed_run --backend rule_based
```

Each command creates a unique directory under `ml/evaluation/runs/`, creates a
fresh SQLite database and temporary administrator, starts a loopback-only server
with an explicit backend, evaluates it, and stops that server in a `finally` block.
Existing servers, databases, and result files are preserved. For a smoke test add
`--limit 20`; smoke results are not full-dataset research results.

The primary trained artifact must already exist at
`saved_models/url_classifier.joblib`. Rebuild it when needed:

```powershell
.\.venv\Scripts\python.exe -m ml.training.build_features
.\.venv\Scripts\python.exe -m ml.training.train_url_classifier
```

The evaluator verifies the server's loaded-model identity and SQLite database
identity through the admin-only `/api/v1/metrics/runtime` endpoint. It saves model,
test-set, source-code, and prediction-file hashes, installed package versions,
per-request predictions, validation rejections, independent classification metrics,
and HTTP round-trip latency. The capacity probe uses a seeded sample of valid
inputs and records status counts, successful throughput, and concurrent latency.
The model is loaded before timing, so these are warm-model measurements.

The standalone evaluator can also target an already running server:

```powershell
$env:EVALUATION_PASSWORD = 'your-local-evaluation-password'
.\.venv\Scripts\python.exe -m ml.evaluation.run_evaluation --base-url http://127.0.0.1:8123 --admin-email eval-admin@example.com --classifier-backend trained --db-path 'C:\absolute\path\eval.db' --output 'C:\absolute\path\new-results.json'
```

The server must use that exact database, an empty allowlist, and no existing
logs. Output files must not exist. Authentication, runtime identity, record counts,
predictions, and metrics are checked; unexpected failures are reported separately
from input-validation rejections.

The current held-out data has been inspected repeatedly during development. Treat
these runs as development evaluations, not an untouched final generalization test.
The research still needs an independent external test set and a controlled cleaning
comparison if causal claims are intended. Never treat spam labels as phishing
labels without an explicit, supported mapping.

---

## 16. Email classifier wired in; classifier_backend split into two settings (2026-09-06)

The "labeled email data plus an email model for the original hybrid URL/email thesis scope" gap
noted above is now closed — see `README_EMAIL.md` for the full dataset audit, training, and
results (TF-IDF + Random Forest, F1 0.980, meets every spec target, no known bare-domain-style
failure mode).

Wiring both classifiers into `app/pipeline/classifiers.py` under one `CLASSIFIER_BACKEND` setting
would have forced an all-or-nothing choice between the two very different backends this document
argues for keeping separate — so `TrainedURLClassifier` was generalized into `TrainedClassifier`
(handles either/both) and the single setting was split into `URL_CLASSIFIER_BACKEND` and
`EMAIL_CLASSIFIER_BACKEND`, resolved independently. **§6's decision to keep the URL side on
`rule_based` is unchanged and still applies** — nothing in this section revisits that. The local
`.env` now runs `URL_CLASSIFIER_BACKEND=rule_based` (per §6) with `EMAIL_CLASSIFIER_BACKEND=trained`
(safe, per `README_EMAIL.md`). `ml/evaluation/managed_run.py`/`run_evaluation.py` are URL-only
tools and were updated to set/read `URL_CLASSIFIER_BACKEND`/`url_classifier_backend` specifically;
their `--backend`/`--classifier-backend` CLI flags are unchanged.

Verified against the real running FastAPI server (not just direct model calls): a real phishing
email and a real legitimate email both classify correctly via the trained email model; `google.com`
correctly classifies as legitimate via the URL side's rule-based heuristic (confirming the split
actually prevents the bare-domain regression, not just in theory).
