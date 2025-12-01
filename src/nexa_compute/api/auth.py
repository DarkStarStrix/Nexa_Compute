import hashlib
import os
import secrets
from datetime import datetime
from typing import Annotated, Optional, Tuple

from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import Session

from nexa_compute.api.database import ApiKeyDB, UserDB, get_db

API_KEY_NAME = "X-Nexa-Api-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

_AUTH_MODE = os.getenv("NEXA_API_AUTH_MODE", "production").lower()
_ALLOW_ANON = os.getenv("NEXA_API_ALLOW_ANON", "false").lower() in {"1", "true", "yes"}


def _allow_unauthenticated() -> bool:
    return _ALLOW_ANON or _AUTH_MODE in {"development", "dev", "test", "testing"}


def get_api_key(
    api_key_header: Annotated[Optional[str], Security(api_key_header)],
    db: Annotated[Session, Depends(get_db)],
) -> UserDB:
    """Validate API key header, optionally allowing dev-mode bypass."""
    if not api_key_header:
        if _allow_unauthenticated():
            dev_user = UserDB(user_id="dev-user", email="dev@nexa.run", is_active=True)
            dev_user._current_api_key_hash = None  # type: ignore[attr-defined]
            dev_user._current_api_key_id = None  # type: ignore[attr-defined]
            return dev_user
        raise HTTPException(status_code=403, detail="Missing API key")

    key_hash = hash_api_key(api_key_header)

    api_key_record = (
        db.query(ApiKeyDB)
        .filter(ApiKeyDB.key_hash == key_hash, ApiKeyDB.is_active.is_(True))
        .first()
    )

    if not api_key_record or not api_key_record.user or not api_key_record.user.is_active:
        raise HTTPException(status_code=403, detail="Invalid API key")

    api_key_record.last_used_at = datetime.utcnow()
    db.add(api_key_record)
    db.commit()

    user = api_key_record.user
    user._current_api_key_hash = key_hash  # type: ignore[attr-defined]
    user._current_api_key_id = api_key_record.key_id  # type: ignore[attr-defined]
    return user


def hash_api_key(raw_key: str) -> str:
    """Return SHA256 hash of an API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> Tuple[str, str, str]:
    """Generate a new API key. Returns (key_id, raw_key, key_hash)."""
    raw_key = f"nexa_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(raw_key)
    key_id = secrets.token_hex(8)
    return key_id, raw_key, key_hash


def rotate_api_key(
    user: UserDB,
    db: Session,
    *,
    name: Optional[str] = None,
) -> tuple[str, ApiKeyDB]:
    """Rotate an API key for the given user, deactivating previous active keys."""
    key_id, raw_key, key_hash = generate_api_key()

    # Mark existing keys inactive
    (
        db.query(ApiKeyDB)
        .filter(ApiKeyDB.user_id == user.user_id, ApiKeyDB.is_active.is_(True))
        .update({"is_active": False})
    )

    new_key = ApiKeyDB(
        key_id=key_id,
        key_hash=key_hash,
        key_prefix=raw_key[:8],
        name=name or "rotated-key",
        user_id=user.user_id,
        is_active=True,
    )
    db.add(new_key)
    db.commit()
    db.refresh(new_key)
    return raw_key, new_key
