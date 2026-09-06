import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.detection_log import Prediction
from app.pipeline.preprocessing import normalize_url

# practical caps, not protocol limits: bound payload size so a client can't
# send an arbitrarily large body into the feature-extraction/NLTK pipeline.
MAX_URL_LENGTH = 2048
MAX_EMAIL_LENGTH = 50_000

# Deliberately permissive -- this only needs to reject obviously-not-a-url
# input ("not a url", stray whitespace/control characters), not fully validate
# RFC 1035 hostnames. Ordinary domains and IPv4 addresses both match.
#
# Underscores are allowed in a label even though RFC 1035 forbids them: real
# hostnames using them are common enough in practice (some DNS servers/
# browsers accept them, and this app's own held-out phishing/URL dataset
# contains legitimate-labeled examples like "good_guild.w.interia.pl") that
# rejecting them would refuse ordinary input a real user might submit, which
# is worse than being slightly more permissive than the RFC.
_HOSTNAME_RE = re.compile(
    r"^[a-zA-Z0-9]([a-zA-Z0-9_-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9_-]{0,61}[a-zA-Z0-9])?)*$"
)


class DetectRequest(BaseModel):
    url: str | None = Field(default=None, max_length=MAX_URL_LENGTH)
    email_text: str | None = Field(default=None, max_length=MAX_EMAIL_LENGTH)

    @field_validator("url", "email_text", mode="before")
    @classmethod
    def _trim(cls, v):
        # a whitespace-only value must behave as "not provided", not as a
        # present-but-blank value that sails through the rest of the pipeline.
        return v.strip() if isinstance(v, str) else v

    @field_validator("url")
    @classmethod
    def _validate_url_shape(cls, v: str | None) -> str | None:
        if not v:
            return v
        # normalize_url()/urlparse() can raise ValueError on structurally
        # invalid input (e.g. "http://[") -- letting that propagate here is
        # intentional: pydantic turns a validator's ValueError into a normal
        # 400 validation error instead of it reaching the handler as an
        # unhandled exception.
        host = urlparse(normalize_url(v)).hostname
        if not host or not _HOSTNAME_RE.match(host):
            raise ValueError(f"'{v}' does not look like a valid URL/hostname")
        return v

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
