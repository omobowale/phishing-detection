import sqlite3

import httpx
import pandas as pd
import pytest

from ml.evaluation import run_evaluation as evaluation
from tests.conftest import register_and_login


def install_transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(evaluation.httpx, "Client", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(handler)))


def test_capacity_separates_http_rejections_server_errors_and_transport_errors(monkeypatch):
    def handler(request):
        if b"offline" in request.content:
            raise httpx.ConnectError("offline", request=request)
        status = 400 if b"invalid" in request.content else 500 if b"broken" in request.content else 200
        return httpx.Response(status, json={})
    install_transport(monkeypatch, handler)
    result = evaluation._throughput_probe("http://test", ["valid", "invalid", "broken", "offline"], 2)
    assert result["status_counts"] == {"200": 1, "400": 1, "500": 1, "transport_error": 1}
    assert result["successful_requests"] == 1
    assert result["error_rate"] == .75
    assert result["attempted_rps"] == pytest.approx(4 * result["throughput_rps"])


def test_sequential_pass_preserves_rejections_and_successful_predictions(monkeypatch):
    def handler(request):
        if b"invalid" in request.content:
            return httpx.Response(400, json={"detail": "invalid URL"})
        return httpx.Response(200, json={"classification": "phishing", "confidence_score": .9})
    install_transport(monkeypatch, handler)
    accepted, rejected = evaluation._run_detection_pass("http://test", pd.DataFrame(
        {"url": ["valid", "invalid"], "label": ["phishing", "legitimate"]}))
    assert len(accepted) == len(rejected) == 1
    assert rejected[0]["true_label"] == "legitimate"
    assert rejected[0]["status_code"] == 400
    assert evaluation.classification_metrics(accepted)["recall"] == 1


def test_email_evaluation_uses_email_payload_and_multiline_log_keys(monkeypatch, tmp_path):
    import json
    content = "Subject: Notice\n\nPlease verify your account."
    def handler(request):
        assert json.loads(request.content) == {"email_text": content}
        return httpx.Response(200, json={"classification": "phishing", "confidence_score": .9})
    install_transport(monkeypatch, handler)
    accepted, rejected = evaluation._run_detection_pass("http://test", pd.DataFrame(
        {"text": [content], "label": ["phishing"]}), "email_text")
    assert not rejected
    path = tmp_path / "email.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE detection_logs (id INTEGER,input_data TEXT,prediction TEXT,actual_label TEXT)")
        db.execute("INSERT INTO detection_logs VALUES (1,?,'phishing',NULL)", (f"email_text: {content}",))
    evaluation._label_logs(path, accepted)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT actual_label FROM detection_logs").fetchone()[0] == "phishing"


def test_labeling_rejects_mismatched_predictions_without_partial_updates(tmp_path):
    path = tmp_path / "eval.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE detection_logs (id INTEGER,input_data TEXT,prediction TEXT,actual_label TEXT)")
        db.executemany("INSERT INTO detection_logs VALUES (?,?,?,NULL)",
                       [(1, "url: a.com", "legitimate"), (2, "url: b.com", "phishing")])
    results = [{"url": url, "true_label": "legitimate", "classification": "legitimate"}
               for url in ["a.com", "b.com"]]
    with pytest.raises(RuntimeError, match="do not match"):
        evaluation._label_logs(path, results)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(actual_label) FROM detection_logs").fetchone()[0] == 0
    results[1]["classification"] = "phishing"
    evaluation._label_logs(path, results)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(actual_label) FROM detection_logs").fetchone()[0] == 2


def test_runtime_identity_requires_admin(client):
    assert client.get("/api/v1/metrics/runtime").status_code == 401
    user = register_and_login(client)
    assert client.get("/api/v1/metrics/runtime", headers=user).status_code == 403
    admin = register_and_login(client, email="admin-runtime@example.com", role="admin")
    response = client.get("/api/v1/metrics/runtime", headers=admin)
    assert response.status_code == 200
    assert response.json()["url_classifier_backend"] == "rule_based"
    assert response.json()["email_classifier_backend"] == "rule_based"
    assert response.json()["database_identity"] is None  # Test fixture uses an in-memory DB.


def test_detection_timing_update_does_not_refresh_expired_row(client):
    """A post-commit SELECT followed by UPDATE caused WAL locking under load."""
    from sqlalchemy import event
    engine = client.session_factory.kw["bind"]
    statements = []
    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement.lstrip().upper())
    event.listen(engine, "before_cursor_execute", capture)
    try:
        response = client.post("/api/v1/detect", json={"url": "https://example.com"})
        assert response.status_code == 200
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    insert_at = next(i for i, sql in enumerate(statements) if sql.startswith("INSERT INTO DETECTION_LOGS"))
    update_at = next(i for i, sql in enumerate(statements) if sql.startswith("UPDATE DETECTION_LOGS"))
    assert not any(sql.startswith("SELECT") for sql in statements[insert_at + 1:update_at])


def test_enabling_email_model_does_not_change_url_heuristic_threshold():
    from app.pipeline.classifiers import RuleBasedClassifier, TrainedClassifier
    from app.pipeline.feature_extraction_url import extract_url_features
    features = extract_url_features("http://192.168.1.1/login@account")
    expected = RuleBasedClassifier().predict(features, None)
    assert expected[0].value == "phishing"
    assert TrainedClassifier(email_model=object()).predict(features, None) == expected


def test_enabling_url_model_does_not_change_email_heuristic_threshold():
    from app.pipeline.classifiers import RuleBasedClassifier, TrainedClassifier
    features = {"reply_to_mismatch": True, "url_count_in_body": 3, "all_caps_word_count": 3}
    expected = RuleBasedClassifier().predict(None, features)
    assert expected[0].value == "phishing"
    assert TrainedClassifier(url_model=object()).predict(None, features) == expected
