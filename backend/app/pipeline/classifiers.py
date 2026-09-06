"""Classification engine (spec section 2/8).

Defines a common `BaseClassifier` interface so Random Forest, XGBoost, and the
fine-tuned BERT model can be swapped in later without touching the API layer.
Until trained models exist under /ml/saved_models, `get_classifier()` returns
`RuleBasedClassifier` -- a heuristic placeholder that keeps /detect functional
end-to-end during backend/frontend development.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import joblib

from app.core.config import settings
from app.models.detection_log import Prediction

SAVED_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "saved_models"

# Weighted heuristic checks for email-side signals. There is no trained email/BERT
# model yet (build order phase 2 covers URL only so far), so both RuleBasedClassifier
# and TrainedURLClassifier fall back to this for the email_features half of a request.
_EMAIL_CHECKS = [
    (lambda f: f.get("reply_to_mismatch"), 2.5),
    (lambda f: f.get("urgency_keyword_count", 0) >= 2, 2.0),
    (lambda f: f.get("has_headers") and not f.get("spf_pass"), 1.5),
    (lambda f: f.get("has_headers") and not f.get("dkim_pass"), 1.5),
    (lambda f: f.get("url_count_in_body", 0) >= 3, 1.0),
    (lambda f: f.get("all_caps_word_count", 0) >= 3, 0.5),
]


def _email_score(email_features: dict) -> tuple[float, float]:
    """Returns (weighted phishing score, total weight) for the email checks."""
    score = weight_total = 0.0
    for check, weight in _EMAIL_CHECKS:
        weight_total += weight
        if check(email_features):
            score += weight
    return score, weight_total


def _combine(score: float, weight_total: float, threshold: float = 0.5) -> tuple[Prediction, float]:
    """threshold defaults to 0.5 -- the standard "more likely than not" cutoff,
    and the same one sklearn/xgboost's own .predict() uses internally, which is
    what ml/training/train_url_classifier.py evaluates against. TrainedURLClassifier
    relies on that default so its live decision matches its offline evaluation;
    RuleBasedClassifier explicitly passes its own separately-tuned 0.4 (a
    hand-picked cutoff for its weighted-heuristic score, unrelated to any
    probability calibration, so there's no reason it should match the ML
    default)."""
    if weight_total == 0:
        return Prediction.legitimate, 0.5
    confidence = score / weight_total
    prediction = Prediction.phishing if confidence >= threshold else Prediction.legitimate
    # report confidence in the predicted direction, not raw phishing-score
    reported_confidence = confidence if prediction == Prediction.phishing else 1 - confidence
    return prediction, round(reported_confidence, 4)


class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, url_features: dict | None, email_features: dict | None) -> tuple[Prediction, float]:
        """Return (prediction, confidence_score in [0, 1])."""
        raise NotImplementedError


class RuleBasedClassifier(BaseClassifier):
    """Weighted heuristic scorer. Not a substitute for the trained RF/XGBoost/BERT
    models -- it exists so the pipeline has a real, testable decision path before
    those models are trained (build order phase 2 in the spec)."""

    def predict(self, url_features: dict | None, email_features: dict | None) -> tuple[Prediction, float]:
        score = 0.0
        weight_total = 0.0

        if url_features:
            checks = [
                (url_features.get("is_ip_address"), 3.0),
                (url_features.get("has_at_symbol"), 2.0),
                (url_features.get("suspicious_tld"), 2.0),
                (url_features.get("brand_keyword_outside_domain"), 3.0),
                (not url_features.get("has_https"), 1.0),
                (url_features.get("subdomain_count", 0) >= 3, 1.5),
                (url_features.get("special_char_count", 0) >= 5, 1.0),
            ]
            for triggered, weight in checks:
                weight_total += weight
                if triggered:
                    score += weight

        if email_features:
            email_s, email_w = _email_score(email_features)
            score += email_s
            weight_total += email_w

        return _combine(score, weight_total, threshold=0.4)


class TrainedURLClassifier(BaseClassifier):
    """Uses a trained RF/XGBoost model (see ml/training/train_url_classifier.py)
    for the URL side of a request. Falls back to the rule-based email checks for
    email_features, since no trained email/BERT model exists yet."""

    # a trained model's phishing-probability vote is weighted heavily relative to
    # the email heuristic checks, since it's evaluated evidence rather than a guess.
    URL_MODEL_WEIGHT = 10.0

    def __init__(self, model, feature_columns: list[str]):
        self.model = model
        self.feature_columns = feature_columns

    def predict(self, url_features: dict | None, email_features: dict | None) -> tuple[Prediction, float]:
        score = 0.0
        weight_total = 0.0

        if url_features:
            vector = [[url_features.get(col, 0) for col in self.feature_columns]]
            phishing_proba = self.model.predict_proba(vector)[0][1]
            score += phishing_proba * self.URL_MODEL_WEIGHT
            weight_total += self.URL_MODEL_WEIGHT

        if email_features:
            email_s, email_w = _email_score(email_features)
            score += email_s
            weight_total += email_w

        return _combine(score, weight_total)


def _load_trained_url_classifier() -> TrainedURLClassifier:
    # Fixed, single filename -- comparison/experiment runs
    # (ml/training/train_url_classifier.py's ARTIFACT_SUFFIX) write under
    # saved_models/experiments/ instead, so there is no glob pattern here that
    # could ever accidentally match one of them.
    model_path = SAVED_MODELS_DIR / "url_classifier.joblib"
    if not model_path.exists():
        raise NotImplementedError(
            "classifier_backend is 'trained' but no model found at "
            f"{model_path}. Run: python -m ml.training.train_url_classifier"
        )
    model_bytes = model_path.read_bytes()
    bundle = joblib.load(BytesIO(model_bytes))
    classifier = TrainedURLClassifier(bundle["model"], bundle["feature_columns"])
    classifier.model_sha256 = sha256(model_bytes).hexdigest()
    classifier.model_name = bundle.get("model_name", type(bundle["model"]).__name__)
    return classifier


@lru_cache(maxsize=4)
def _get_classifier_for_backend(backend: str) -> BaseClassifier:
    if backend == "rule_based":
        return RuleBasedClassifier()
    if backend == "trained":
        return _load_trained_url_classifier()
    raise NotImplementedError(
        f"Classifier backend '{backend}' is not wired up yet. Train the model "
        "(see /ml/training) and add a loader here that implements BaseClassifier."
    )


def get_classifier() -> BaseClassifier:
    # Cached per backend value (not just once globally) so the expensive part
    # -- deserializing the joblib model file -- happens once per distinct
    # backend, not on every /detect request, while still letting tests switch
    # classifier_backend within a single process and get the right instance.
    return _get_classifier_for_backend(settings.classifier_backend)
