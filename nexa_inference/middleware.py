"""API middleware for authentication and rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class LicenseTier:
    """License tier definitions."""

    FREE = "free"
    PRO = "pro"
    ACADEMIC = "academic"
    COMMERCIAL = "commercial"


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self):
        self.requests: dict[str, list[float]] = defaultdict(list)

    def check_rate_limit(self, api_key: str, tier: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is within rate limit.

        Args:
            api_key: API key identifier
            tier: License tier
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            True if within limit, False otherwise
        """
        if tier in [LicenseTier.ACADEMIC, LicenseTier.COMMERCIAL]:
            return True

        now = time.time()
        key = f"{api_key}:{tier}"
        self.requests[key] = [req_time for req_time in self.requests[key] if now - req_time < window_seconds]

        if len(self.requests[key]) >= max_requests:
            return False

        self.requests[key].append(now)
        return True


class APIKeyValidator:
    """Validates API keys and returns license tier."""

    def __init__(self, api_keys: Optional[dict[str, str]] = None):
        """Initialize validator with API key mapping.

        Args:
            api_keys: Dictionary mapping API keys to license tiers
        """
        self.api_keys = api_keys or {}

    def validate(self, api_key: Optional[str]) -> tuple[bool, str]:
        """Validate API key and return tier.

        Args:
            api_key: API key to validate

        Returns:
            Tuple of (is_valid, tier)
        """
        if not api_key:
            return False, LicenseTier.FREE

        tier = self.api_keys.get(api_key, LicenseTier.FREE)
        return True, tier


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting based on license tier."""

    def __init__(self, app, rate_limiter: RateLimiter, api_validator: APIKeyValidator):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.api_validator = api_validator
        self.tier_limits = {
            LicenseTier.FREE: {"max_requests": 100, "window_seconds": 3600},
            LicenseTier.PRO: {"max_requests": 1000, "window_seconds": 3600},
            LicenseTier.ACADEMIC: {"max_requests": float("inf"), "window_seconds": 3600},
            LicenseTier.COMMERCIAL: {"max_requests": float("inf"), "window_seconds": 3600},
        }

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        api_key = request.headers.get("X-API-Key")
        is_valid, tier = self.api_validator.validate(api_key)

        if not is_valid and api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

        limits = self.tier_limits.get(tier, self.tier_limits[LicenseTier.FREE])
        max_requests = limits["max_requests"]
        window_seconds = limits["window_seconds"]

        if max_requests != float("inf"):
            key_id = api_key or "anonymous"
            if not self.rate_limiter.check_rate_limit(key_id, tier, int(max_requests), window_seconds):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded for tier {tier}",
                )

        request.state.api_key = api_key
        request.state.tier = tier

        response = await call_next(request)
        return response

