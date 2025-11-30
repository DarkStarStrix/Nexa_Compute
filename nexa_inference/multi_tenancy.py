"""Multi-tenancy and organization scoping utilities."""

from __future__ import annotations

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware


class OrganizationMiddleware(BaseHTTPMiddleware):
    """Ensure organization ID is supplied and stored on request state."""

    def __init__(self, app, *, require_org: bool = True) -> None:
        super().__init__(app)
        self.require_org = require_org

    async def dispatch(self, request: Request, call_next):
        org_id = request.headers.get("X-Org-ID")
        if self.require_org and not org_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing X-Org-ID header")
        request.state.org_id = org_id
        response = await call_next(request)
        if org_id:
            response.headers["X-Org-ID"] = org_id
        return response

