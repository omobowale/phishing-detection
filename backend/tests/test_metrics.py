import pytest

from app.models.detection_log import DetectionLog, InputType, Prediction
from tests.conftest import register_and_login


def test_metrics_requires_auth(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 401


def test_metrics_requires_admin(client):
    user_headers = register_and_login(client, email="plain-metrics@example.com", role="end_user")
    resp = client.get("/api/v1/metrics", headers=user_headers)
    assert resp.status_code == 403


def test_metrics_computes_accuracy_precision_recall_f1(client):
    """/detect has no ground-truth field by design, so this seeds labeled log
    rows directly via the DB (matching how ml/evaluation/run_evaluation.py
    attaches ground truth after the fact) and checks /metrics' math by hand:
    2 true positives, 1 false negative, 1 false positive, 1 true negative ->
    accuracy=3/5, precision=2/3, recall=2/3.
    """
    admin_headers = register_and_login(client, email="admin-metrics@example.com", role="admin")

    db = client.session_factory()
    try:
        outcomes = [
            (Prediction.phishing, Prediction.phishing),      # true positive
            (Prediction.phishing, Prediction.phishing),      # true positive
            (Prediction.legitimate, Prediction.phishing),    # false negative
            (Prediction.phishing, Prediction.legitimate),    # false positive
            (Prediction.legitimate, Prediction.legitimate),  # true negative
        ]
        for prediction, actual in outcomes:
            db.add(DetectionLog(
                user_id=None,
                input_type=InputType.url,
                input_data="url: http://example.com",
                prediction=prediction,
                confidence_score=0.9,
                processing_time=1.0,
                actual_label=actual,
            ))
        db.commit()
    finally:
        db.close()

    resp = client.get("/api/v1/metrics", headers=admin_headers)
    assert resp.status_code == 200
    body = resp.json()

    assert body["labeled_sample_size"] == 5
    assert body["total_requests"] == 5
    assert body["accuracy"] == pytest.approx(3 / 5)
    assert body["precision"] == pytest.approx(2 / 3)
    assert body["recall"] == pytest.approx(2 / 3)
    assert body["f1_score"] == pytest.approx(2 * (2 / 3 * 2 / 3) / (2 / 3 + 2 / 3))


def test_metrics_with_no_labeled_logs_returns_null_quality_metrics(client):
    admin_headers = register_and_login(client, email="admin-metrics-empty@example.com", role="admin")

    resp = client.get("/api/v1/metrics", headers=admin_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["labeled_sample_size"] == 0
    assert body["accuracy"] is None
    assert body["precision"] is None
    assert body["recall"] is None
    assert body["f1_score"] is None
