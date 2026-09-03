import logging
import time

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import Session

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

    if payload.url:
        domain = get_domain(payload.url)
        whitelisted = db.query(Whitelist).filter(Whitelist.domain == domain).first() is not None
        if whitelisted:
            elapsed_ms = (time.perf_counter() - start) * 1000
            db.add(DetectionLog(
                user_id=current_user.user_id if current_user else None,
                input_type=_input_type(payload),
                input_data=_raw_input(payload),
                prediction=Prediction.legitimate,
                confidence_score=1.0,
                processing_time=elapsed_ms,
            ))
            db.commit()
            return DetectResponse(
                classification=Prediction.legitimate,
                confidence_score=1.0,
                processing_time_ms=round(elapsed_ms, 2),
                whitelisted=True,
            )

    try:
        url_features = extract_url_features(payload.url) if payload.url else None
        email_features = extract_email_features(payload.email_text) if payload.email_text else None
        classifier = get_classifier()
        prediction, confidence = classifier.predict(url_features, email_features)
    except Exception:
        logger.exception("Pipeline failure processing /detect request")
        raise HTTPException(status_code=500, detail="Classification pipeline failed") from None

    elapsed_ms = (time.perf_counter() - start) * 1000

    db.add(DetectionLog(
        user_id=current_user.user_id if current_user else None,
        input_type=_input_type(payload),
        input_data=_raw_input(payload),
        prediction=prediction,
        confidence_score=confidence,
        processing_time=elapsed_ms,
    ))
    db.commit()

    return DetectResponse(
        classification=prediction,
        confidence_score=confidence,
        processing_time_ms=round(elapsed_ms, 2),
        whitelisted=False,
    )
