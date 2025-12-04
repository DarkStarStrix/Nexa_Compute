from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from nexa_compute.api.database import get_db
from nexa_compute.core.storage import get_storage

router = APIRouter()


@router.get("/health")
def liveness() -> dict[str, str]:
    """Basic liveness probe."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/ready")
def readiness(db: Annotated[Session, Depends(get_db)]) -> dict[str, object]:
    """Readiness probe verifying DB and storage reachability."""
    status = {"database": False, "storage": False}
    try:
        db.execute(text("SELECT 1"))
        status["database"] = True
    except Exception as exc:  # pragma: no cover - defensive
        status["database_error"] = repr(exc)

    try:
        storage = get_storage()
        # Touch sentinel directories to ensure they are accessible
        (storage.durable_root / ".health").touch(exist_ok=True)
        status["storage"] = True
    except Exception as exc:  # pragma: no cover - defensive
        status["storage_error"] = repr(exc)

    if not status["database"] or not status["storage"]:
        raise HTTPException(status_code=503, detail=status)

    return {"status": "ready", **status}


@router.get("/health/metrics")
def health_metrics(db: Annotated[Session, Depends(get_db)]) -> dict[str, object]:
    """Operational metrics surface for monitoring."""
    jobs_total = db.execute(text("SELECT COUNT(*) FROM jobs")).scalar() or 0
    workers_total = db.execute(text("SELECT COUNT(*) FROM workers")).scalar() or 0
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "jobs_total": jobs_total,
        "workers_total": workers_total,
    }

