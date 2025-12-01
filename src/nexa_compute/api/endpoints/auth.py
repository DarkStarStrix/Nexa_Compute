from datetime import datetime
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from nexa_compute.api.auth import generate_api_key, rotate_api_key
from nexa_compute.api.database import ApiKeyDB, UserDB, get_db
from nexa_compute.api.middleware import get_rate_limited_user

router = APIRouter()

class ApiKeyResponse(BaseModel):
    key_id: str
    name: str
    prefix: str
    created_at: datetime
    # Raw key is ONLY returned on creation
    raw_key: str | None = None

class CreateApiKeyRequest(BaseModel):
    name: str

@router.post("/api-keys", response_model=ApiKeyResponse)
def create_api_key(
    request: CreateApiKeyRequest,
    user: Annotated[UserDB, Depends(get_rate_limited_user)],
    db: Annotated[Session, Depends(get_db)],
):
    key_id, raw_key, key_hash = generate_api_key()
    prefix = f"{raw_key[:10]}..."

    new_key = ApiKeyDB(
        key_id=key_id,
        key_hash=key_hash,
        key_prefix=prefix,
        name=request.name,
        user_id=user.user_id,
        created_at=datetime.utcnow(),
        is_active=True
    )
    
    db.add(new_key)
    db.commit()
    db.refresh(new_key)
    
    return ApiKeyResponse(
        key_id=new_key.key_id,
        name=new_key.name,
        prefix=new_key.key_prefix,
        created_at=new_key.created_at,
        raw_key=raw_key # Return raw key only this once
    )

@router.get("/api-keys", response_model=List[ApiKeyResponse])
def list_api_keys(
    user: Annotated[UserDB, Depends(get_rate_limited_user)],
    db: Annotated[Session, Depends(get_db)],
):
    keys = (
        db.query(ApiKeyDB)
        .filter(
            ApiKeyDB.user_id == user.user_id,
            ApiKeyDB.is_active.is_(True),
        )
        .all()
    )
    
    return [
        ApiKeyResponse(
            key_id=k.key_id,
            name=k.name,
            prefix=k.key_prefix,
            created_at=k.created_at,
            raw_key=None # Never return raw key on list
        ) for k in keys
    ]

@router.delete("/api-keys/{key_id}")
def revoke_api_key(
    key_id: str,
    user: Annotated[UserDB, Depends(get_rate_limited_user)],
    db: Annotated[Session, Depends(get_db)],
):
    key = db.query(ApiKeyDB).filter(
        ApiKeyDB.key_id == key_id,
        ApiKeyDB.user_id == user.user_id
    ).first()
    
    if not key:
        raise HTTPException(status_code=404, detail="API Key not found")
        
    key.is_active = False
    db.commit()
    
    return {"status": "revoked"}


class RotateApiKeyRequest(BaseModel):
    name: str | None = None


@router.post("/api-keys/rotate", response_model=ApiKeyResponse)
def rotate_api_key_endpoint(
    request: RotateApiKeyRequest,
    user: Annotated[UserDB, Depends(get_rate_limited_user)],
    db: Annotated[Session, Depends(get_db)],
):
    raw_key, record = rotate_api_key(user, db, name=request.name)
    return ApiKeyResponse(
        key_id=record.key_id,
        name=record.name,
        prefix=record.key_prefix,
        created_at=record.created_at,
        raw_key=raw_key,
    )
