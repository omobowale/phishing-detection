"""Create or promote an admin user.

POST /auth/register always creates end_user accounts (see schemas/auth.py) --
self-service admin signup would make the whitelist's admin-only gate meaningless.
This script is the only way to create an admin account, run directly against the
DB, for local dev / initial bootstrap.

Usage:
    python -m scripts.create_admin admin@example.com "Admin Name" supersecret
"""

import sys

from app.core.security import hash_password
from app.db.base import Base, SessionLocal, engine
from app.models.user import User, UserRole


def create_admin(email: str, name: str, password: str) -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.role = UserRole.admin
            print(f"Promoted existing user {email} to admin.")
        else:
            user = User(name=name, email=email, hashed_password=hash_password(password), role=UserRole.admin)
            db.add(user)
            print(f"Created admin user {email}.")
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    create_admin(email=sys.argv[1], name=sys.argv[2], password=sys.argv[3])
