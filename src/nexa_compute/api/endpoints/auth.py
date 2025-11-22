from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from datetime import datetime
from nexa_compute.api.database import get_db, ApiKeyDB, UserDB
from nexa_compute.api.auth import get_api_key, generate_api_key

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
    user: UserDB = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    raw_key, key_hash = generate_api_key()
    prefix = raw_key[:10] + "..."
    
    import uuid
    key_id = f"key_{uuid.uuid4().hex[:12]}"
    
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
    user: UserDB = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    keys = db.query(ApiKeyDB).filter(
        ApiKeyDB.user_id == user.user_id,
        ApiKeyDB.is_active == True
    ).all()
    
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
    user: UserDB = Depends(get_api_key),
    db: Session = Depends(get_db)
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
