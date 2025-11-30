# Nexa Forge Test Suite

This directory contains comprehensive tests for Nexa Forge API, SDK, and deployment readiness.

## Test Structure

- **test_api.py** - API endpoint tests (jobs, workers, billing, artifacts)
- **test_sdk.py** - SDK client tests (initialization, job methods, convenience methods)
- **test_integration.py** - Integration tests (component wiring, end-to-end workflows)
- **test_deployment.py** - Deployment readiness tests (configuration, wiring, environment)
- **test_infra.py** - Infrastructure tests (directory structure, CLI, modules)
- **test_config.py** - Configuration tests (YAML validation, compute plans)
- **test_registry.py** - Registry tests (model registry functionality)

## Running Tests

### Install Dependencies
```bash
/opt/homebrew/bin/uv pip sync requirements/requirements-dev.lock
```

### Run All Tests
```bash
pytest tests/ -v
# or use the helper entrypoint
python tests/main.py -v
```

### Run Specific Test Suites
```bash
# API tests
pytest tests/test_api.py -v

# SDK tests
pytest tests/test_sdk.py -v

# Integration tests
pytest tests/test_integration.py -v

# Deployment readiness tests
pytest tests/test_deployment.py -v
```

### Run with Markers
```bash
pytest -m api -v          # API tests only
pytest -m sdk -v          # SDK tests only
pytest -m integration -v   # Integration tests only
pytest -m deployment -v    # Deployment tests only
pytest -m infra -v         # Infrastructure tests only
```

## Test Coverage

### API Tests (test_api.py)
- Health check endpoint
- Job creation (generate, train, audit, distill, evaluate, deploy)
- Job listing and retrieval
- Job status updates
- Worker registration and management
- Worker heartbeat
- Job assignment to workers
- Billing summary retrieval
- Artifact listing and retrieval

### SDK Tests (test_sdk.py)
- Client initialization (default, custom, env vars)
- Job submission methods
- Job retrieval methods
- Convenience methods (generate, audit, distill, train, evaluate, deploy)
- Error handling

### Integration Tests (test_integration.py)
- Component import verification
- Storage registry integration
- Worker processor integration
- Server config integration
- Deployment readiness checks
- End-to-end job workflows
- Storage backend integration
- Worker agent integration

### Deployment Tests (test_deployment.py)
- API endpoint registration
- Database initialization
- Storage backend configuration
- Service initialization (billing, job manager, worker registry)
- Component wiring verification
- Environment configuration
- SDK deployment readiness

## Test Fixtures

- `test_db` - Temporary in-memory database
- `client` - FastAPI TestClient instance
- `artifacts_dir` - Temporary artifacts directory
- `test_artifacts_dir` - Session-scoped artifacts directory
- `setup_test_env` - Automatic environment variable setup

## Notes

- Tests use in-memory SQLite database by default
- Tests create temporary directories for artifacts
- Environment variables are automatically set for testing
- Some tests may skip if optional dependencies are missing

