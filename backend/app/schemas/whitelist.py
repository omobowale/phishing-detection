from datetime import datetime

from pydantic import BaseModel


class WhitelistCreate(BaseModel):
    domain: str


class WhitelistOut(BaseModel):
    id: int
    domain: str
    added_by: int
    date_added: datetime

    model_config = {"from_attributes": True}
