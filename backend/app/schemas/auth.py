from pydantic import BaseModel, EmailStr

from app.models.user import UserRole


class UserRegister(BaseModel):
    """Public self-registration. No `role` field on purpose - the API always
    creates end_user accounts here; promoting to admin is a separate,
    non-self-service action (see backend/scripts/create_admin.py)."""

    name: str
    email: EmailStr
    password: str


class UserOut(BaseModel):
    user_id: int
    name: str
    email: EmailStr
    role: UserRole

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
