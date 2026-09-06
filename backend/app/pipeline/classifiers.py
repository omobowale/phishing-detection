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

# Weighted structural checks for url-side signals, shared by RuleBasedClassifier
# and by TrainedClassifier as its fallback when no trained URL model is loaded.
_URL_CHECKS = [
    (lambda f: f.get("is_ip_address"), 3.0),
    (lambda f: f.get("has_at_symbol"), 2.0),
    (lambda f: f.get("suspicious_tld"), 2.0),
    (lambda f: f.get("brand_keyword_outside_domain"), 3.0),
    (lambda f: not f.get("has_https"), 1.0),
    (lambda f: f.get("subdomain_count", 0) >= 3, 1.5),
    (lambda f: f.get("special_char_count", 0) >= 5, 1.0),
]

# Weighted heuristic checks for email-side signals, shared by RuleBasedClassifier
# and by TrainedClassifier as its fallback when no trained email model is loaded.
_EMAIL_CHECKS = [
    (lambda f: f.get("reply_to_mismatch"), 2.5),
    (lambda f: f.get("urgency_keyword_count", 0) >= 2, 2.0),
    (lambda f: f.get("has_headers") and not f.get("spf_pass"), 1.5),
    (lambda f: f.get("has_headers") and not f.get("dkim_pass"), 1.5),
    (lambda f: f.get("url_count_in_body", 0) >= 3, 1.0),
    (lambda f: f.get("all_caps_word_count", 0) >= 3, 0.5),
]


def _url_score(url_features: dict) -> tuple[float, float]:
    """Returns (weighted phishing score, total weight) for the url checks."""
    score = weight_total = 0.0
    for check, weight in _URL_CHECKS:
        weight_total += weight
        if check(url_features):
            score += weight
    return score, weight_total


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
    what ml/training/train_url_classifier.py evaluates against. TrainedClassifier
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
            url_s, url_w = _url_score(url_features)
            score += url_s
            weight_total += url_w

        if email_features:
            email_s, email_w = _email_score(email_features)
            score += email_s
            weight_total += email_w

        return _combine(score, weight_total, threshold=0.4)


class TrainedClassifier(BaseClassifier):
    """Uses trained models (ml/training/train_url_classifier.py and
    train_email_classifier.py) for whichever side of a request has one loaded.
    Falls back to the same rule-based heuristics RuleBasedClassifier uses for
    whichever side is configured as "rule_based" (or has no trained model on
    disk) -- url_classifier_backend and email_classifier_backend are independent
    settings, so this instance may end up trained on one side only."""

    # A trained model's phishing-probability vote is weighted heavily relative
    # to the heuristic checks, since it's evaluated evidence rather than a guess.
    URL_MODEL_WEIGHT = 10.0
    EMAIL_MODEL_WEIGHT = 10.0

    def __init__(
        self,
        url_model=None,
        url_feature_columns: list[str] | None = None,
        email_model=None,
        email_vectorizer=None,
    ):
        self.url_model = url_model
        self.url_feature_columns = url_feature_columns
        self.email_model = email_model
        self.email_vectorizer = email_vectorizer

    def predict(self, url_features: dict | None, email_features: dict | None) -> tuple[Prediction, float]:
        score = 0.0
        weight_total = 0.0

        if url_features:
            if self.url_model is not None:
                vector = [[url_features.get(col, 0) for col in self.url_feature_columns]]
                phishing_proba = self.url_model.predict_proba(vector)[0][1]
                score += phishing_proba * self.URL_MODEL_WEIGHT
                weight_total += self.URL_MODEL_WEIGHT
            else:
                url_s, url_w = _url_score(url_features)
                score += url_s
                weight_total += url_w

        if email_features:
            if self.email_model is not None:
                # tokens_text: see feature_extraction_email.py -- the same
                # strip_email_headers() -> preprocess_email_text() pipeline
                # build_email_features.py used to train this vectorizer/model.
                vector = self.email_vectorizer.transform([email_features.get("tokens_text", "")])
                phishing_proba = self.email_model.predict_proba(vector)[0][1]
                score += phishing_proba * self.EMAIL_MODEL_WEIGHT
                weight_total += self.EMAIL_MODEL_WEIGHT
            else:
                email_s, email_w = _email_score(email_features)
                score += email_s
                weight_total += email_w

        return _combine(score, weight_total)


def _load_url_model() -> tuple:
    # Fixed, single filename -- comparison/experiment runs
    # (ml/training/train_url_classifier.py's ARTIFACT_SUFFIX) write under
    # saved_models/experiments/ instead, so there is no glob pattern here that
    # could ever accidentally match one of them.
    model_path = SAVED_MODELS_DIR / "url_classifier.joblib"
    if not model_path.exists():
        return None, None, None, None
    model_bytes = model_path.read_bytes()
    bundle = joblib.load(BytesIO(model_bytes))
    model_name = bundle.get("model_name", type(bundle["model"]).__name__)
    return bundle["model"], bundle["feature_columns"], sha256(model_bytes).hexdigest(), model_name


def _load_email_model() -> tuple:
    model_path = SAVED_MODELS_DIR / "email_classifier.joblib"
    if not model_path.exists():
        return None, None, None, None
    model_bytes = model_path.read_bytes()
    bundle = joblib.load(BytesIO(model_bytes))
    model_name = bundle.get("model_name", type(bundle["model"]).__name__)
    return bundle["model"], bundle["vectorizer"], sha256(model_bytes).hexdigest(), model_name


def _resolve_url_component(backend: str) -> tuple:
    if backend == "rule_based":
        return None, None, None, "rule_based"
    if backend == "trained":
        url_model, url_feature_columns, url_sha256, url_name = _load_url_model()
        if url_model is None:
            raise NotImplementedError(
                "url_classifier_backend is 'trained' but no model found at "
                f"{SAVED_MODELS_DIR / 'url_classifier.joblib'}. Run: "
                "python -m ml.training.train_url_classifier"
            )
        return url_model, url_feature_columns, url_sha256, url_name
    raise NotImplementedError(f"url_classifier_backend '{backend}' must be 'rule_based' or 'trained'.")


def _resolve_email_component(backend: str) -> tuple:
    if backend == "rule_based":
        return None, None, None, "rule_based"
    if backend == "trained":
        email_model, email_vectorizer, email_sha256, email_name = _load_email_model()
        if email_model is None:
            raise NotImplementedError(
                "email_classifier_backend is 'trained' but no model found at "
                f"{SAVED_MODELS_DIR / 'email_classifier.joblib'}. Run: "
                "python -m ml.training.train_email_classifier"
            )
        return email_model, email_vectorizer, email_sha256, email_name
    raise NotImplementedError(f"email_classifier_backend '{backend}' must be 'rule_based' or 'trained'.")


@lru_cache(maxsize=8)
def _get_classifier(url_backend: str, email_backend: str) -> BaseClassifier:
    # URL and email are resolved independently -- see config.py's comment on why
    # these are two separate settings rather than one shared "classifier_backend":
    # the two trained models have very different maturity right now.
    if url_backend == "rule_based" and email_backend == "rule_based":
        return RuleBasedClassifier()

    url_model, url_feature_columns, url_sha256, url_name = _resolve_url_component(url_backend)
    email_model, email_vectorizer, email_sha256, email_name = _resolve_email_component(email_backend)

    classifier = TrainedClassifier(url_model, url_feature_columns, email_model, email_vectorizer)
    classifier.model_name = url_name
    classifier.model_sha256 = url_sha256
    classifier.email_model_name = email_name
    classifier.email_model_sha256 = email_sha256
    return classifier


def get_classifier() -> BaseClassifier:
    # Cached per (url_backend, email_backend) pair, not just once globally, so
    # the expensive part -- deserializing the joblib model files -- happens once
    # per distinct combination, not on every /detect request, while still
    # letting tests flip either setting within a single process and get the
    # right instance.
    return _get_classifier(settings.url_classifier_backend, settings.email_classifier_backend)
