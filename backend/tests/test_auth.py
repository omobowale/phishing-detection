def test_register_and_login(client):
    resp = client.post("/api/v1/auth/register", json={
        "name": "Alice", "email": "alice@example.com", "password": "secret123",
    })
    assert resp.status_code == 201

    resp = client.post("/api/v1/auth/login", data={"username": "alice@example.com", "password": "secret123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password(client):
    client.post("/api/v1/auth/register", json={
        "name": "Bob", "email": "bob@example.com", "password": "secret123",
    })
    resp = client.post("/api/v1/auth/login", data={"username": "bob@example.com", "password": "wrong"})
    assert resp.status_code == 401


def test_duplicate_registration_rejected(client):
    payload = {"name": "Carl", "email": "carl@example.com", "password": "secret123"}
    assert client.post("/api/v1/auth/register", json=payload).status_code == 201
    assert client.post("/api/v1/auth/register", json=payload).status_code == 400


def test_self_registration_cannot_grant_admin_role(client):
    resp = client.post("/api/v1/auth/register", json={
        "name": "Dana", "email": "dana@example.com", "password": "secret123", "role": "admin",
    })
    assert resp.status_code == 201
    assert resp.json()["role"] == "end_user"


def test_me_returns_current_user(client):
    from tests.conftest import register_and_login

    headers = register_and_login(client, email="erin@example.com")
    resp = client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["email"] == "erin@example.com"


def test_me_requires_auth(client):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401
