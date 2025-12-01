"""API dependencies and middleware utilities (rate limiting, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Response
from sqlalchemy.orm import Session

from nexa_compute.api.auth import get_api_key
from nexa_compute.api.config import get_settings
from nexa_compute.api.database import ApiKeyRateLimitDB, UserDB, get_db


@dataclass(frozen=True)
class RateLimitInfo:
    limit: int
    remaining: int
    reset_epoch: int


class RateLimiter:
    """Simple fixed-window rate limiter backed by the database."""

    def __init__(self, window_seconds: int, max_requests: int) -> None:
        self.window_seconds = max(1, window_seconds)
        self.max_requests = max(1, max_requests)

    def hit(self, key_hash: str, db: Session) -> RateLimitInfo:
        if not key_hash:
            # Missing key hash (development mode) -> skip enforcement
            return RateLimitInfo(limit=self.max_requests, remaining=self.max_requests, reset_epoch=0)

        now = datetime.utcnow()
        window_start = self._window_start(now)
        window_end = window_start + timedelta(seconds=self.window_seconds)

        record = (
            db.query(ApiKeyRateLimitDB)
            .filter(
                ApiKeyRateLimitDB.key_hash == key_hash,
                ApiKeyRateLimitDB.window_start == window_start,
            )
            .first()
        )

        if record is None:
            record = ApiKeyRateLimitDB(
                key_hash=key_hash,
                window_start=window_start,
                count=0,
            )
            db.add(record)

        record.count += 1

        if record.count > self.max_requests:
            db.rollback()
            retry_after = int((window_end - now).total_seconds()) or 1
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        self._cleanup(db, window_start)
        db.commit()

        remaining = max(self.max_requests - record.count, 0)
        reset_epoch = int(window_end.replace(tzinfo=timezone.utc).timestamp())
        return RateLimitInfo(limit=self.max_requests, remaining=remaining, reset_epoch=reset_epoch)

    def _window_start(self, now: datetime) -> datetime:
        epoch = int(now.timestamp())
        window_start_epoch = epoch - (epoch % self.window_seconds)
        return datetime.fromtimestamp(window_start_epoch, tz=timezone.utc).replace(tzinfo=None)

    def _cleanup(self, db: Session, current_window: datetime) -> None:
        cutoff = current_window - timedelta(seconds=self.window_seconds * 2)
        (
            db.query(ApiKeyRateLimitDB)
            .filter(ApiKeyRateLimitDB.window_start < cutoff)
            .delete(synchronize_session=False)
        )


def get_rate_limited_user(
    response: Response,
    user: Annotated[UserDB, Depends(get_api_key)],
    db: Annotated[Session, Depends(get_db)],
) -> UserDB:
    """Dependency enforcing per-key rate limiting and returning the authenticated user."""
    settings = get_settings()
    limiter = RateLimiter(
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        max_requests=settings.RATE_LIMIT_REQUESTS_PER_WINDOW,
    )
    key_hash = getattr(user, "_current_api_key_hash", None)
    info = limiter.hit(key_hash, db)

    if info.reset_epoch:
        response.headers["X-RateLimit-Limit"] = str(info.limit)
        response.headers["X-RateLimit-Remaining"] = str(info.remaining)
        response.headers["X-RateLimit-Reset"] = str(info.reset_epoch)

    return user

