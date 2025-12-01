"""Prometheus metrics endpoint."""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

