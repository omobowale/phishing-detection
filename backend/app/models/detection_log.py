import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class InputType(str, enum.Enum):
    url = "url"
    email_text = "email_text"
    both = "both"


class Prediction(str, enum.Enum):
    phishing = "phishing"
    legitimate = "legitimate"


class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    input_type: Mapped[InputType] = mapped_column(Enum(InputType))
    input_data: Mapped[str] = mapped_column(Text)
    prediction: Mapped[Prediction] = mapped_column(Enum(Prediction))
    confidence_score: Mapped[float] = mapped_column(Float)
    processing_time: Mapped[float] = mapped_column(Float)  # ms
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    # extension beyond the ERD: ground-truth label for a request, set when a log
    # entry belongs to a labeled evaluation run. GET /metrics needs this to compute
    # accuracy/precision/recall/F1 (section 5/9 of the spec) -- without it there's
    # no way to know if a prediction was correct.
    actual_label: Mapped[Prediction | None] = mapped_column(Enum(Prediction), nullable=True)

    user = relationship("User", back_populates="detection_logs")
