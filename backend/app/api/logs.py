from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import require_user
from app.db.base import get_db
from app.models.detection_log import DetectionLog, Prediction
from app.models.user import User, UserRole
from app.schemas.log import DetectionLogOut, PaginatedLogs

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("", response_model=PaginatedLogs)
def list_logs(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_user),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    classification: Prediction | None = None,
    user_id: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
):
    query = db.query(DetectionLog)

    # non-admins can only ever see their own logs, regardless of the user_id filter
    if current_user.role != UserRole.admin:
        query = query.filter(DetectionLog.user_id == current_user.user_id)
    elif user_id is not None:
        query = query.filter(DetectionLog.user_id == user_id)

    if classification is not None:
        query = query.filter(DetectionLog.prediction == classification)
    if start_date is not None:
        query = query.filter(DetectionLog.timestamp >= start_date)
    if end_date is not None:
        query = query.filter(DetectionLog.timestamp <= end_date)

    total = query.count()
    items = (
        query.order_by(DetectionLog.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return PaginatedLogs(total=total, page=page, page_size=page_size, items=items)


@router.get("/{log_id}", response_model=DetectionLogOut)
def get_log(log_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_user)):
    log = db.query(DetectionLog).filter(DetectionLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log entry not found")
    if current_user.role != UserRole.admin and log.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this log entry")
    return log
