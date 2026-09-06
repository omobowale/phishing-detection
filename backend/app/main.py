import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, detect, logs, metrics, whitelist
from app.core.config import DEFAULT_SECRET_KEY, settings
from app.db.base import Base, engine
from app.pipeline.classifiers import get_classifier
from app.pipeline.feature_extraction_email import extract_email_features
from app.pipeline.feature_extraction_url import extract_url_features

logger = logging.getLogger("phishing_detection.startup")


def _check_secret_key() -> None:
    if settings.secret_key != DEFAULT_SECRET_KEY:
        return
    message = (
        "SECRET_KEY is still the insecure default ('dev-secret-change-me'). "
        "Every JWT this process issues can be forged by anyone who reads this "
        "source code. Set a real random SECRET_KEY via the environment/.env "
        "before deploying."
    )
    if settings.environment == "production":
        raise RuntimeError(message)
    logger.warning(message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_secret_key()
    # dev convenience; use Alembic migrations instead once this needs to run
    # against a shared/production database.
    Base.metadata.create_all(bind=engine)
    # Complete lazy corpus/model initialization before accepting traffic. This
    # also makes a missing configured artifact a startup error, not a user 500.
    started = time.perf_counter()
    classifier = get_classifier()
    classifier.predict(extract_url_features("https://example.com/"), None)
    classifier.predict(None, extract_email_features("The meeting notes are ready."))
    logger.info("Classification pipeline ready after %.2f seconds", time.perf_counter() - started)
    yield


app = FastAPI(title="Phishing Detection API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # spec section 6: malformed/empty input must be a 400 with a clear message,
    # not FastAPI's default 422. exc.errors() can include a "ctx" dict holding the
    # raw exception object (e.g. from a model_validator), which isn't JSON
    # serializable, so only pass through the serializable fields.
    errors = [
        {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=400, content={"detail": errors})


API_PREFIX = "/api/v1"
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(detect.router, prefix=API_PREFIX)
app.include_router(whitelist.router, prefix=API_PREFIX)
app.include_router(logs.router, prefix=API_PREFIX)
app.include_router(metrics.router, prefix=API_PREFIX)


@app.get("/health")
def health():
    return {"status": "ok"}
