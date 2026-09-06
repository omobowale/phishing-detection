import pytest

from app.core.config import settings
from app.pipeline import classifiers


@pytest.fixture(autouse=True)
def _clear_classifier_cache():
    # _get_classifier_for_backend is process-wide (lru_cache), so tests that
    # flip settings.classifier_backend must not leak a cached instance into
    # other tests or across runs.
    classifiers._get_classifier_for_backend.cache_clear()
    yield
    classifiers._get_classifier_for_backend.cache_clear()


def test_get_classifier_is_cached_not_reloaded_per_call(monkeypatch):
    """Regression test: TrainedURLClassifier used to be constructed (and its
    joblib model file deserialized from disk) on every single call to
    get_classifier(), i.e. every /detect request."""
    monkeypatch.setattr(settings, "classifier_backend", "rule_based")
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


_PRIMARY_MODEL_PATH = classifiers.SAVED_MODELS_DIR / "url_classifier.joblib"


@pytest.mark.skipif(
    not _PRIMARY_MODEL_PATH.exists(),
    reason="no trained model artifact at ml/saved_models/url_classifier.joblib "
    "(run: python -m ml.training.train_url_classifier)",
)
def test_trained_classifier_predicts_end_to_end(monkeypatch):
    from app.pipeline.feature_extraction_url import extract_url_features

    monkeypatch.setattr(settings, "classifier_backend", "trained")
    clf = classifiers.get_classifier()
    assert isinstance(clf, classifiers.TrainedURLClassifier)

    features = extract_url_features("http://192.168.1.1/wp-admin/login.php")
    prediction, confidence = clf.predict(features, None)
    assert prediction.value in ("phishing", "legitimate")
    assert 0 <= confidence <= 1
