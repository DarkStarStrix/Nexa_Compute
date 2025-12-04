# NexaCompute Test Suite

## Quick Start

Run tests using the Rich CLI:

```bash
# Interactive mode - pick what to test
python tests/cli.py run --interactive

# Run specific modules
python tests/cli.py run --modules rust,v4

# Run entire test suite
python tests/cli.py run --modules all

# Quick test (Rust + V4 only)
python tests/cli.py quick

# List all available test modules
python tests/cli.py list
```

## Available Test Modules

- `rust` - Rust module Python bindings
- `v4` - V4 specification compliance
- `api` - API endpoint tests
- `integration` - End-to-end integration tests
- `sdk` - Python SDK client tests
- `deployment` - Deployment readiness tests
- `infra` - Infrastructure tests
- `config` - Configuration system tests
- `registry` - Registry and versioning tests
- `resilience` - Error handling tests
- `inference` - Inference feature tests
- `monitoring` - Observability tests
- `load` - Performance and stress tests

## Direct pytest Usage

You can also run pytest directly:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_rust_modules.py -v

# With coverage (requires pytest-cov)
pytest tests/ --cov=src/nexa_compute --cov-report=term-missing
```
