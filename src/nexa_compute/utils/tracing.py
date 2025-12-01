"""Distributed tracing utilities using OpenTelemetry.

This module provides a centralized way to configure and use distributed tracing
across the NexaCompute platform. It supports:
1. Automatic instrumentation of FastAPI, requests, and other libraries
2. Manual instrumentation via decorators and context managers
3. Context propagation across async boundaries and thread pools
4. Export to standard OTLP collectors (e.g. Jaeger, Tempo)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

LOGGER = logging.getLogger(__name__)
_TRACER_NAME = "nexa.compute"

T = TypeVar("T")


def configure_tracing(
    service_name: str = "nexa-compute",
    endpoint: Optional[str] = None,
    instrument_fastapi: bool = False,
    instrument_requests: bool = True,
) -> None:
    """Configure the global tracer provider and instrumentation.

    Args:
        service_name: logical name of the service for traces
        endpoint: OTLP gRPC endpoint (e.g. "localhost:4317"). If None,
                 reads from OTEL_EXPORTER_OTLP_ENDPOINT env var.
        instrument_fastapi: whether to auto-instrument FastAPI apps (requires app instance later)
        instrument_requests: whether to auto-instrument the requests library
    """
    # Check if tracing is disabled
    if os.getenv("NEXA_TRACING_ENABLED", "true").lower() == "false":
        LOGGER.info("tracing_disabled")
        return

    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    # Configure exporter
    otlp_endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    try:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        LOGGER.info("tracing_configured", extra={"endpoint": otlp_endpoint, "service": service_name})
    except Exception as exc:
        LOGGER.warning("tracing_configuration_failed", extra={"error": str(exc)})
        return

    # Auto-instrumentation
    if instrument_requests:
        RequestsInstrumentor().instrument()
        LOGGER.debug("requests_instrumented")

    # Note: FastAPI instrumentation usually requires the app instance,
    # so it's often done in the main.py or where the app is created.
    # We expose instrument_app() for that.


def instrument_app(app: Any) -> None:
    """Instrument a FastAPI application."""
    if os.getenv("NEXA_TRACING_ENABLED", "true").lower() == "false":
        return
    FastAPIInstrumentor.instrument_app(app)
    LOGGER.info("fastapi_app_instrumented")


def get_tracer(name: str = _TRACER_NAME) -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)


def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to wrap a function execution in a span.

    Args:
        name: name of the span
        attributes: initial attributes to set on the span
        kind: kind of span (client, server, internal, producer, consumer)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            tracer = get_tracer()
            with tracer.start_as_current_span(name, kind=kind) as span:
                if attributes:
                    span.set_attributes(attributes)
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            tracer = get_tracer()
            with tracer.start_as_current_span(name, kind=kind) as span:
                if attributes:
                    span.set_attributes(attributes)
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper

    return decorator


@contextmanager
def span_context(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> Generator[Span, None, None]:
    """Context manager for manual spans."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            span.set_attributes(attributes)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
