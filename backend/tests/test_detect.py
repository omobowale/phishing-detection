def test_detect_requires_at_least_one_field(client):
    resp = client.post("/api/v1/detect", json={})
    assert resp.status_code == 400


def test_detect_suspicious_url_flagged_phishing(client):
    resp = client.post("/api/v1/detect", json={"url": "http://192.168.1.1/paypal-login@secure"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["classification"] == "phishing"
    assert 0 <= body["confidence_score"] <= 1
    assert body["whitelisted"] is False


def test_detect_plain_https_url_legitimate(client):
    resp = client.post("/api/v1/detect", json={"url": "https://example.com/"})
    assert resp.status_code == 200
    assert resp.json()["classification"] == "legitimate"


def test_detect_email_text(client):
    resp = client.post("/api/v1/detect", json={
        "email_text": "Hi team, the meeting notes are attached. Thanks!",
    })
    assert resp.status_code == 200
    assert resp.json()["classification"] in ("phishing", "legitimate")


def test_whitelisted_domain_bypasses_pipeline(client):
    from tests.conftest import register_and_login

    admin_headers = register_and_login(client, email="admin@example.com", role="admin")
    client.post("/api/v1/whitelist", json={"domain": "example.com"}, headers=admin_headers)

    resp = client.post("/api/v1/detect", json={"url": "https://example.com/anything"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["whitelisted"] is True
    assert body["classification"] == "legitimate"
    assert body["confidence_score"] == 1.0
