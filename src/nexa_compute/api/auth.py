import secrets
import hashlib
from fastapi import Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import Session
from nexa_compute.api.database import get_db, ApiKeyDB, UserDB

API_KEY_NAME = "X-Nexa-Api-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(
    api_key_header: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> UserDB:
    if not api_key_header:
        # For now, allow unauthenticated access for dashboard/testing if configured
        # In production, this should raise HTTPException
        # raise HTTPException(status_code=403, detail="Could not validate credentials")
        return UserDB(user_id="default_user", email="demo@nexa.ai") 

    # Hash the key to look it up
    # In a real implementation, we'd use a secure hash (argon2/bcrypt) but for API keys SHA256 is often used for speed if high entropy
    key_hash = hashlib.sha256(api_key_header.encode()).hexdigest()
    
    api_key_record = db.query(ApiKeyDB).filter(
        ApiKeyDB.key_hash == key_hash,
        ApiKeyDB.is_active == True
    ).first()
    
    if not api_key_record:
        raise HTTPException(status_code=403, detail="Invalid API Key")
        
    # Update last used
    # api_key_record.last_used_at = datetime.utcnow()
    # db.commit()
    
    return api_key_record.user

def generate_api_key() -> tuple[str, str]:
    """Generate a new API key. Returns (raw_key, key_hash)."""
    raw_key = f"nexa_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    return raw_key, key_hash
