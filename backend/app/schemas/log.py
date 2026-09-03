from datetime import datetime

from pydantic import BaseModel

from app.models.detection_log import InputType, Prediction


class DetectionLogOut(BaseModel):
    id: int
    user_id: int | None
    input_type: InputType
    input_data: str
    prediction: Prediction
    confidence_score: float
    processing_time: float
    timestamp: datetime
    actual_label: Prediction | None

    model_config = {"from_attributes": True}


class PaginatedLogs(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[DetectionLogOut]
