import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 - ensures all models are registered on Base
from app.db.base import Base, get_db
from app.main import app
from app.models.user import User, UserRole


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        test_client.session_factory = TestingSessionLocal
        yield test_client
    app.dependency_overrides.clear()


def register_and_login(client, email="user@example.com", role="end_user", password="password123"):
    """Registers (always as end_user, per the API) and logs in. When role="admin"
    is requested, promotes the user directly via the DB -- there's no self-service
    admin signup endpoint, by design (see schemas/auth.py)."""
    client.post("/api/v1/auth/register", json={
        "name": "Test User", "email": email, "password": password,
    })

    if role == "admin":
        db = client.session_factory()
        try:
            user = db.query(User).filter(User.email == email).first()
            user.role = UserRole.admin
            db.commit()
        finally:
            db.close()

    resp = client.post("/api/v1/auth/login", data={"username": email, "password": password})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
