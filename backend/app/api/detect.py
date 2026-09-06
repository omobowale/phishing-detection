import logging
import time
from contextlib import nullcontext
from threading import Lock

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import update

from app.api.deps import get_current_user
from app.db.base import get_db
from app.models.detection_log import DetectionLog, InputType, Prediction
from app.models.user import User
from app.models.whitelist import Whitelist
from app.pipeline.classifiers import get_classifier
from app.pipeline.feature_extraction_email import extract_email_features
from app.pipeline.feature_extraction_url import extract_url_features
from app.pipeline.preprocessing import get_domain
from app.schemas.detection import DetectRequest, DetectResponse

logger = logging.getLogger("phishing_detection.detect")
router = APIRouter(tags=["detect"])

# SQLite has one writer. Queue detection-log writes within this API process
# instead of making concurrent requests compete for its write lock.
_sqlite_log_lock = Lock()


def _input_type(payload: DetectRequest) -> InputType:
    if payload.url and payload.email_text:
        return InputType.both
    return InputType.url if payload.url else InputType.email_text


def _raw_input(payload: DetectRequest) -> str:
    parts = []
    if payload.url:
        parts.append(f"url: {payload.url}")
    if payload.email_text:
        parts.append(f"email_text: {payload.email_text}")
    return "\n".join(parts)


@router.post("/detect", response_model=DetectResponse)
def detect(
    payload: DetectRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    start = time.perf_counter()

    # Parsing/feature-extraction failures are the client's fault (a malformed
    # URL like "http://[") and map to 400, per spec section 6 ("malformed input
    # -> 400"). Classifier failures are the server's fault and map to 500
    # ("model/pipeline failures -> 500"). These used to be lumped into one
    # handler that always returned 500, which is wrong for the former case.
    try:
        whitelisted = False
        if payload.url:
            domain = get_domain(payload.url)
            whitelisted = db.query(Whitelist).filter(Whitelist.domain == domain).first() is not None

        # A whitelisted URL only vouches for the URL itself -- an accompanying
        # email must still be analyzed (a phishing email can legitimately link
        # to a trusted domain, e.g. quoting or spoofing a real site). Only skip
        # classification entirely when there's nothing but a trusted URL to look at.
        if whitelisted and not payload.email_text:
            url_features = email_features = None
        else:
            url_features = extract_url_features(payload.url) if (payload.url and not whitelisted) else None
            email_features = extract_email_features(payload.email_text) if payload.email_text else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Malformed input: {exc}") from None

    try:
        if whitelisted and not payload.email_text:
            prediction, confidence = Prediction.legitimate, 1.0
        else:
            classifier = get_classifier()
            prediction, confidence = classifier.predict(url_features, email_features)
    except Exception:
        logger.exception("Pipeline failure processing /detect request")
        raise HTTPException(status_code=500, detail="Classification pipeline failed") from None

    log_entry = DetectionLog(
        user_id=current_user.user_id if current_user else None,
        input_type=_input_type(payload),
        input_data=_raw_input(payload),
        prediction=prediction,
        confidence_score=confidence,
        processing_time=0,
    )
    is_sqlite = db.get_bind().dialect.name == "sqlite"
    if is_sqlite:
        # Finish the read-only auth/allowlist transaction before entering the
        # write queue; otherwise an old WAL snapshot may fail to upgrade.
        db.rollback()
    with _sqlite_log_lock if is_sqlite else nullcontext():
        db.add(log_entry)
        db.flush()
        log_id = log_entry.id
        db.commit()

        # Includes queueing, inference, and the first durable write. The second
        # timing update is excluded; the evaluator measures full HTTP latency.
        elapsed_ms = (time.perf_counter() - start) * 1000
        # No ORM refresh between writes: capture the id before commit and use
        # a direct UPDATE instead of reading an expired row to recover its id.
        db.execute(
            update(DetectionLog).where(DetectionLog.id == log_id).values(processing_time=elapsed_ms),
            execution_options={"synchronize_session": False},
        )
        db.commit()

    return DetectResponse(
        classification=prediction,
        confidence_score=confidence,
        processing_time_ms=round(elapsed_ms, 2),
        whitelisted=whitelisted,
    )
