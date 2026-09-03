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
