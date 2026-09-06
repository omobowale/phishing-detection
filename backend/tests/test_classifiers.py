import pytest

from app.core.config import settings
from app.pipeline import classifiers


@pytest.fixture(autouse=True)
def _clear_classifier_cache():
    # _get_classifier is process-wide (lru_cache), so tests that flip
    # settings.url_classifier_backend/email_classifier_backend must not leak a
    # cached instance into other tests or across runs.
    classifiers._get_classifier.cache_clear()
    yield
    classifiers._get_classifier.cache_clear()


def test_get_classifier_is_cached_not_reloaded_per_call(monkeypatch):
    """Regression test: TrainedClassifier used to be constructed (and its
    joblib model file deserialized from disk) on every single call to
    get_classifier(), i.e. every /detect request."""
    monkeypatch.setattr(settings, "url_classifier_backend", "rule_based")
    monkeypatch.setattr(settings, "email_classifier_backend", "rule_based")
    first = classifiers.get_classifier()
    second = classifiers.get_classifier()
    assert first is second


def test_combine_default_threshold_matches_model_predict_semantics():
    """Regression test: TrainedURLClassifier used to share RuleBasedClassifier's
    hand-tuned 0.4 cutoff, while ml/training/train_url_classifier.py's offline
    evaluation used model.predict()'s standard 0.5 cutoff -- the same score
    could get a different verdict live than what its own reported metrics
    claimed. _combine() now defaults to 0.5; RuleBasedClassifier opts into 0.4
    explicitly for its own separately-tuned heuristic scoring."""
    score, weight_total = 4.0, 10.0  # confidence = 0.40
    assert classifiers._combine(score, weight_total, threshold=0.4)[0].value == "phishing"
    assert classifiers._combine(score, weight_total)[0].value == "legitimate"


_URL_MODEL_PATH = classifiers.SAVED_MODELS_DIR / "url_classifier.joblib"
_EMAIL_MODEL_PATH = classifiers.SAVED_MODELS_DIR / "email_classifier.joblib"


@pytest.mark.skipif(
    not _URL_MODEL_PATH.exists(),
    reason="no trained model artifact at ml/saved_models/url_classifier.joblib "
    "(run: python -m ml.training.train_url_classifier)",
)
def test_trained_classifier_predicts_url_end_to_end(monkeypatch):
    """url_classifier_backend="trained" with email left at "rule_based" (the two
    are independent settings -- see config.py -- since the URL and email trained
    models have very different maturity)."""
    from app.pipeline.feature_extraction_url import extract_url_features

    monkeypatch.setattr(settings, "url_classifier_backend", "trained")
    monkeypatch.setattr(settings, "email_classifier_backend", "rule_based")
    clf = classifiers.get_classifier()
    assert isinstance(clf, classifiers.TrainedClassifier)
    assert clf.email_model is None  # rule-based fallback for the email side

    features = extract_url_features("http://192.168.1.1/wp-admin/login.php")
    prediction, confidence = clf.predict(features, None)
    assert prediction.value in ("phishing", "legitimate")
    assert 0 <= confidence <= 1


@pytest.mark.skipif(
    not _EMAIL_MODEL_PATH.exists(),
    reason="no trained model artifact at ml/saved_models/email_classifier.joblib "
    "(run: python -m ml.training.train_email_classifier)",
)
def test_trained_classifier_predicts_email_end_to_end(monkeypatch):
    """email_classifier_backend="trained" with url left at "rule_based"."""
    from app.pipeline.feature_extraction_email import extract_email_features

    monkeypatch.setattr(settings, "url_classifier_backend", "rule_based")
    monkeypatch.setattr(settings, "email_classifier_backend", "trained")
    clf = classifiers.get_classifier()
    assert isinstance(clf, classifiers.TrainedClassifier)
    assert clf.url_model is None  # rule-based fallback for the url side

    phishing_features = extract_email_features(
        "Subject: Account Suspended\n\nDear Customer, your account has been suspended. "
        "Click here to verify your password and identity immediately or lose access."
    )
    prediction, confidence = clf.predict(None, phishing_features)
    assert prediction.value in ("phishing", "legitimate")
    assert 0 <= confidence <= 1


@pytest.mark.skipif(
    not (_URL_MODEL_PATH.exists() and _EMAIL_MODEL_PATH.exists()),
    reason="needs both trained model artifacts",
)
def test_trained_classifier_both_backends_independently_trained(monkeypatch):
    """Regression test: url_classifier_backend and email_classifier_backend
    resolve independently through get_classifier() -- setting both to "trained"
    loads both models onto one TrainedClassifier instance."""
    from app.pipeline.feature_extraction_email import extract_email_features
    from app.pipeline.feature_extraction_url import extract_url_features

    monkeypatch.setattr(settings, "url_classifier_backend", "trained")
    monkeypatch.setattr(settings, "email_classifier_backend", "trained")
    clf = classifiers.get_classifier()
    assert clf.url_model is not None
    assert clf.email_model is not None

    prediction, confidence = clf.predict(
        extract_url_features("http://192.168.1.1/wp-admin/login.php"),
        extract_email_features("Dear Customer, verify your account now or it will be closed."),
    )
    assert prediction.value in ("phishing", "legitimate")
    assert 0 <= confidence <= 1
