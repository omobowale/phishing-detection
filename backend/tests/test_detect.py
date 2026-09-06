from tests.conftest import register_and_login

# Deterministic phishing-shaped email: real header lines (From/Reply-To
# mismatch) plus several urgency keywords, which reliably crosses
# RuleBasedClassifier's threshold (see app/pipeline/classifiers.py). Plain body
# text with no headers can't exercise the header-based checks at all.
_PHISHING_EMAIL = (
    "From: security@bank-support.com\n"
    "Reply-To: totally-different@scam.example\n"
    "Subject: Urgent account verification required\n"
    "\n"
    "Please click here to verify your account immediately, your access will be suspended."
)


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


def test_detect_benign_email_legitimate(client):
    resp = client.post("/api/v1/detect", json={
        "email_text": "Hi team, the meeting notes are attached. Thanks!",
    })
    assert resp.status_code == 200
    assert resp.json()["classification"] == "legitimate"


def test_detect_phishing_shaped_email_flagged_phishing(client):
    # unlike the old version of this test, this actually exercises detection
    # quality instead of accepting either outcome.
    resp = client.post("/api/v1/detect", json={"email_text": _PHISHING_EMAIL})
    assert resp.status_code == 200
    assert resp.json()["classification"] == "phishing"


def test_whitelisted_domain_bypasses_pipeline(client):
    admin_headers = register_and_login(client, email="admin@example.com", role="admin")
    client.post("/api/v1/whitelist", json={"domain": "example.com"}, headers=admin_headers)

    resp = client.post("/api/v1/detect", json={"url": "https://example.com/anything"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["whitelisted"] is True
    assert body["classification"] == "legitimate"
    assert body["confidence_score"] == 1.0


def test_whitelist_userinfo_bypass_is_blocked(client):
    """Regression test for a confirmed, live vulnerability: get_domain() used
    to do `netloc.split(":")[0]`, which takes the userinfo *username* rather
    than the real host for a "user:pass@host" netloc. An attacker could embed
    a trusted domain name as fake userinfo in front of their own host and get
    it treated as whitelisted. Fixed by using urlparse(...).hostname."""
    admin_headers = register_and_login(client, email="admin-bypass@example.com", role="admin")
    client.post("/api/v1/whitelist", json={"domain": "trusted.com"}, headers=admin_headers)

    resp = client.post(
        "/api/v1/detect",
        json={"url": "https://trusted.com:password@evil.example/"},
    )
    assert resp.status_code == 200
    assert resp.json()["whitelisted"] is False


def test_mixed_submission_still_analyzes_email_when_url_whitelisted(client):
    """Regression test: a whitelisted URL used to short-circuit the entire
    request, skipping analysis of an accompanying email_text entirely. Now
    only a URL-only submission gets the whitelist fast path; a mixed
    submission still runs the email through the classifier."""
    admin_headers = register_and_login(client, email="admin-mixed@example.com", role="admin")
    client.post("/api/v1/whitelist", json={"domain": "trusted.com"}, headers=admin_headers)

    resp = client.post(
        "/api/v1/detect",
        json={"url": "https://trusted.com/", "email_text": _PHISHING_EMAIL},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["whitelisted"] is True  # the URL itself is still reported as trusted
    assert body["classification"] == "phishing"  # but the email must still be analyzed
    assert body["confidence_score"] != 1.0  # not the whitelist-only fast path's fixed 1.0


def test_detect_rejects_whitespace_only_url(client):
    resp = client.post("/api/v1/detect", json={"url": "   "})
    assert resp.status_code == 400


def test_detect_rejects_whitespace_only_email(client):
    resp = client.post("/api/v1/detect", json={"email_text": "   "})
    assert resp.status_code == 400


def test_detect_rejects_url_shaped_garbage(client):
    resp = client.post("/api/v1/detect", json={"url": "not a url"})
    assert resp.status_code == 400


def test_detect_rejects_malformed_url_without_500(client):
    """Regression test: "http://[" used to raise an unhandled exception deep in
    urlparse (invalid IPv6 URL), returning a 500. Malformed input is a client
    error and must be a 400, per spec section 6."""
    resp = client.post("/api/v1/detect", json={"url": "http://["})
    assert resp.status_code == 400


def test_detect_rejects_oversized_url(client):
    resp = client.post("/api/v1/detect", json={"url": "http://example.com/" + "a" * 3000})
    assert resp.status_code == 400


def test_pipeline_failure_returns_500_not_a_crash(client, monkeypatch):
    """spec section 6: model/pipeline failures -> 500 with a logged error, not
    an unhandled exception reaching the client as a raw traceback."""
    from app.api import detect as detect_module

    class _BrokenClassifier:
        def predict(self, url_features, email_features):
            raise RuntimeError("boom")

    monkeypatch.setattr(detect_module, "get_classifier", lambda: _BrokenClassifier())
    resp = client.post("/api/v1/detect", json={"url": "http://example.com"})
    assert resp.status_code == 500
    assert "detail" in resp.json()
