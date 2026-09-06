from tests.conftest import register_and_login


def test_anonymous_cannot_list_logs(client):
    resp = client.get("/api/v1/logs")
    assert resp.status_code == 401


def test_own_detection_shows_up_in_own_logs(client):
    user_headers = register_and_login(client, email="log-owner@example.com", role="end_user")

    resp = client.post("/api/v1/detect", json={"url": "http://example.com"}, headers=user_headers)
    assert resp.status_code == 200

    resp = client.get("/api/v1/logs", headers=user_headers)
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1


def test_non_admin_cannot_see_other_users_logs(client):
    owner_headers = register_and_login(client, email="log-owner-2@example.com", role="end_user")
    other_headers = register_and_login(client, email="log-other@example.com", role="end_user")

    client.post("/api/v1/detect", json={"url": "http://example.com"}, headers=owner_headers)

    resp = client.get("/api/v1/logs", headers=other_headers)
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_admin_sees_all_logs_including_anonymous(client):
    admin_headers = register_and_login(client, email="admin-logs@example.com", role="admin")

    # anonymous request -- no Authorization header at all
    resp = client.post("/api/v1/detect", json={"url": "http://example.com"})
    assert resp.status_code == 200

    resp = client.get("/api/v1/logs", headers=admin_headers)
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any(item["user_id"] is None for item in items)


def test_non_owner_cannot_fetch_specific_log_by_id(client):
    owner_headers = register_and_login(client, email="log-owner-3@example.com", role="end_user")
    other_headers = register_and_login(client, email="log-other-2@example.com", role="end_user")

    client.post("/api/v1/detect", json={"url": "http://example.com"}, headers=owner_headers)
    log_id = client.get("/api/v1/logs", headers=owner_headers).json()["items"][0]["id"]

    resp = client.get(f"/api/v1/logs/{log_id}", headers=other_headers)
    assert resp.status_code == 403

    resp = client.get(f"/api/v1/logs/{log_id}", headers=owner_headers)
    assert resp.status_code == 200


def test_get_log_not_found(client):
    user_headers = register_and_login(client, email="log-notfound@example.com", role="end_user")
    resp = client.get("/api/v1/logs/999999", headers=user_headers)
    assert resp.status_code == 404
