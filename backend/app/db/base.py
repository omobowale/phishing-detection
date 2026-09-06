from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

_is_sqlite = settings.database_url.startswith("sqlite")
connect_args = {"check_same_thread": False} if _is_sqlite else {}
engine = create_engine(settings.database_url, connect_args=connect_args)

if _is_sqlite:
    # default SQLite journal mode does a rollback-journal create/delete on every
    # commit, which measured ~4-5x slower than WAL mode under this app's per-request
    # commit pattern (see ml/README.md phase 5 notes) -- WAL uses a single
    # append-only log instead, avoiding that per-commit file churn.
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
