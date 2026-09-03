"""Classification engine (spec section 2/8).

Defines a common `BaseClassifier` interface so Random Forest, XGBoost, and the
fine-tuned BERT model can be swapped in later without touching the API layer.
Until trained models exist under /ml/saved_models, `get_classifier()` returns
`RuleBasedClassifier` -- a heuristic placeholder that keeps /detect functional
end-to-end during backend/frontend development.
"""

from abc import ABC, abstractmethod

from app.core.config import settings
from app.models.detection_log import Prediction


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
            checks = [
                (email_features.get("reply_to_mismatch"), 2.5),
                (email_features.get("urgency_keyword_count", 0) >= 2, 2.0),
                (email_features.get("has_headers") and not email_features.get("spf_pass"), 1.5),
                (email_features.get("has_headers") and not email_features.get("dkim_pass"), 1.5),
                (email_features.get("url_count_in_body", 0) >= 3, 1.0),
                (email_features.get("all_caps_word_count", 0) >= 3, 0.5),
            ]
            for triggered, weight in checks:
                weight_total += weight
                if triggered:
                    score += weight

        if weight_total == 0:
            return Prediction.legitimate, 0.5

        confidence = score / weight_total
        prediction = Prediction.phishing if confidence >= 0.4 else Prediction.legitimate
        # report confidence in the predicted direction, not raw phishing-score
        reported_confidence = confidence if prediction == Prediction.phishing else 1 - confidence
        return prediction, round(reported_confidence, 4)


def get_classifier() -> BaseClassifier:
    backend = settings.classifier_backend
    if backend == "rule_based":
        return RuleBasedClassifier()
    raise NotImplementedError(
        f"Classifier backend '{backend}' is not wired up yet. Train the model "
        "(see /ml/training) and add a loader here that implements BaseClassifier."
    )
