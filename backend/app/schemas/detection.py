from pydantic import BaseModel, model_validator

from app.models.detection_log import Prediction


class DetectRequest(BaseModel):
    url: str | None = None
    email_text: str | None = None

    @model_validator(mode="after")
    def at_least_one_field(self):
        if not self.url and not self.email_text:
            raise ValueError("at least one of 'url' or 'email_text' is required")
        return self


class DetectResponse(BaseModel):
    classification: Prediction
    confidence_score: float
    processing_time_ms: float
    whitelisted: bool
