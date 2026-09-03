from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, detect, logs, metrics, whitelist
from app.core.config import settings
from app.db.base import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # dev convenience; use Alembic migrations instead once this needs to run
    # against a shared/production database.
    Base.metadata.create_all(bind=engine)
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
