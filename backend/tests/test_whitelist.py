from tests.conftest import register_and_login


def test_non_admin_cannot_manage_whitelist(client):
    user_headers = register_and_login(client, email="plain@example.com", role="end_user")
    resp = client.post("/api/v1/whitelist", json={"domain": "example.com"}, headers=user_headers)
    assert resp.status_code == 403


def test_anonymous_cannot_manage_whitelist(client):
    resp = client.get("/api/v1/whitelist")
    assert resp.status_code == 401


def test_admin_whitelist_crud(client):
    admin_headers = register_and_login(client, email="admin2@example.com", role="admin")

    resp = client.post("/api/v1/whitelist", json={"domain": "trusted.com"}, headers=admin_headers)
    assert resp.status_code == 201
    entry_id = resp.json()["id"]

    resp = client.get("/api/v1/whitelist", headers=admin_headers)
    assert resp.status_code == 200
    assert any(e["domain"] == "trusted.com" for e in resp.json())

    resp = client.delete(f"/api/v1/whitelist/{entry_id}", headers=admin_headers)
    assert resp.status_code == 204

    resp = client.get("/api/v1/whitelist", headers=admin_headers)
    assert all(e["domain"] != "trusted.com" for e in resp.json())
