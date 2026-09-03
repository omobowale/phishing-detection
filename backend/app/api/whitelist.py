from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.db.base import get_db
from app.models.user import User
from app.models.whitelist import Whitelist
from app.schemas.whitelist import WhitelistCreate, WhitelistOut

router = APIRouter(prefix="/whitelist", tags=["whitelist"])


@router.get("", response_model=list[WhitelistOut])
def list_whitelist(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    return db.query(Whitelist).order_by(Whitelist.date_added.desc()).all()


@router.post("", response_model=WhitelistOut, status_code=201)
def add_whitelist_entry(
    payload: WhitelistCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    domain = payload.domain.strip().lower()
    if not domain:
        raise HTTPException(status_code=400, detail="domain must not be empty")
    if db.query(Whitelist).filter(Whitelist.domain == domain).first():
        raise HTTPException(status_code=400, detail="Domain already whitelisted")

    entry = Whitelist(domain=domain, added_by=current_user.user_id)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.delete("/{entry_id}", status_code=204)
def delete_whitelist_entry(entry_id: int, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    entry = db.query(Whitelist).filter(Whitelist.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Whitelist entry not found")
    db.delete(entry)
    db.commit()
