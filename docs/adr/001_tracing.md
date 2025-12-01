# ADR-001: Distributed Tracing with OpenTelemetry

## Status
Accepted

## Context
As the NexaCompute platform scales, debugging distributed systems (API -> Worker -> Training Job) becomes increasingly difficult. We need a way to trace requests across service boundaries to identify bottlenecks and failures.

## Decision
We will adopt **OpenTelemetry (OTEL)** as the standard for distributed tracing.

### Rationale
1. **Vendor Neutral**: OTEL is the industry standard and supports exporting to Jaeger, Zipkin, Prometheus, and commercial vendors (Datadog, Honeycomb).
2. **Ecosystem Support**: Strong support for Python, FastAPI, and PyTorch.
3. **Future Proof**: Allows switching backends without code changes.

## Implementation
- Use `opentelemetry-instrumentation-fastapi` for API automatic instrumentation.
- Use manual `Tracer` for internal components (Scheduler, Trainer).
- Propagate trace context via HTTP headers (`traceparent`).
- Export traces via OTLP/gRPC to a collector.

## Consequences
- **Positive**: End-to-end visibility, standardized logging context.
- **Negative**: Slight performance overhead (<1% latency), operational complexity of managing a collector.

