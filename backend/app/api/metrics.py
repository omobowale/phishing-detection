from fastapi import APIRouter, Depends
from hashlib import sha256
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.db.base import get_db
from app.models.detection_log import DetectionLog, Prediction
from app.models.user import User
from app.schemas.metrics import Metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics/runtime")
def runtime_info(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    """Admin-only identity of the loaded model and database, for evaluation."""
    from app.core.config import settings
    from app.pipeline.classifiers import get_classifier

    classifier = get_classifier()
    url = db.get_bind().url
    database_identity = None
    if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
        database_identity = sha256(str(Path(url.database).resolve()).encode()).hexdigest()
    return {
        "classifier_backend": settings.classifier_backend,
        "model_name": getattr(classifier, "model_name", type(classifier).__name__),
        "model_sha256": getattr(classifier, "model_sha256", None),
        "database_identity": database_identity,
    }


@router.get("/metrics", response_model=Metrics)
def get_metrics(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    logs = db.query(DetectionLog).all()

    total_requests = len(logs)
    phishing_count = sum(1 for l in logs if l.prediction == Prediction.phishing)
    legitimate_count = total_requests - phishing_count

    average_latency_ms = (
        sum(l.processing_time for l in logs) / total_requests if total_requests else None
    )

    throughput_rps = None
    if total_requests >= 2:
        timestamps = sorted(l.timestamp for l in logs)
        span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        if span_seconds > 0:
            throughput_rps = total_requests / span_seconds

    # accuracy/precision/recall/F1 can only be computed against logs with a known
    # ground-truth label (see actual_label on DetectionLog) -- i.e. a labeled
    # evaluation run (build order phase 5), not organic production traffic.
    labeled = [l for l in logs if l.actual_label is not None]
    accuracy = precision = recall = f1 = None
    if labeled:
        y_true = [1 if l.actual_label == Prediction.phishing else 0 for l in labeled]
        y_pred = [1 if l.prediction == Prediction.phishing else 0 for l in labeled]
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(labeled)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    return Metrics(
        total_requests=total_requests,
        phishing_count=phishing_count,
        legitimate_count=legitimate_count,
        average_latency_ms=average_latency_ms,
        throughput_rps=throughput_rps,
        labeled_sample_size=len(labeled),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )
