"""Deployment readiness checklist and verification.

This module verifies that all components are properly wired for deployment.
"""

import os
from pathlib import Path
from typing import List, Tuple

def check_storage_setup() -> List[Tuple[str, bool, str]]:
    """Check storage configuration."""
    issues = []
    
    # Check storage backend config
    backend = os.getenv("STORAGE_BACKEND", "local")
    if backend == "local":
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        if not artifacts_dir.exists():
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            issues.append(("Storage", True, f"Created artifacts directory: {artifacts_dir}"))
        else:
            issues.append(("Storage", True, f"Artifacts directory exists: {artifacts_dir}"))
    elif backend in ["s3", "do_spaces"]:
        required_vars = ["STORAGE_BUCKET", "STORAGE_ACCESS_KEY", "STORAGE_SECRET_KEY"]
        missing = [v for v in required_vars if not os.getenv(v)]
        if missing:
            issues.append(("Storage", False, f"Missing env vars: {', '.join(missing)}"))
        else:
            issues.append(("Storage", True, "S3/DO Spaces configured"))
    
    return issues

def check_database_setup() -> List[Tuple[str, bool, str]]:
    """Check database configuration."""
    issues = []
    
    db_url = os.getenv("DATABASE_URL", "sqlite:///./var/nexa_forge.db")
    if db_url.startswith("sqlite"):
        db_path = db_url.replace("sqlite:///", "")
        db_file = Path(db_path)
        if db_file.exists():
            issues.append(("Database", True, f"Database file exists: {db_path}"))
        else:
            issues.append(("Database", True, f"Database will be created: {db_path}"))
    else:
        issues.append(("Database", True, f"Using remote database: {db_url.split('@')[1] if '@' in db_url else db_url}"))
    
    return issues

def check_api_endpoints() -> List[Tuple[str, bool, str]]:
    """Check API endpoints are registered."""
    issues = []
    
    try:
        from nexa_compute.api.main import app
        routes = [r.path for r in app.routes]
        required_routes = [
            "/api/jobs",
            "/api/workers",
            "/api/billing",
            "/api/auth",
            "/api/artifacts",
            "/health"
        ]
        
        for route in required_routes:
            if any(route in r for r in routes):
                issues.append(("API", True, f"Route registered: {route}"))
            else:
                issues.append(("API", False, f"Route missing: {route}"))
    except Exception as e:
        issues.append(("API", False, f"Failed to check routes: {e}"))
    
    return issues

def check_worker_agent() -> List[Tuple[str, bool, str]]:
    """Check worker agent setup."""
    issues = []
    
    try:
        from nexa_compute.api.worker_agent import WorkerAgent
        issues.append(("Worker Agent", True, "Worker agent module loads"))
    except Exception as e:
        issues.append(("Worker Agent", False, f"Failed to load: {e}"))
    
    return issues

def verify_deployment_readiness() -> dict:
    """Run all checks and return summary."""
    all_issues = []
    all_issues.extend(check_storage_setup())
    all_issues.extend(check_database_setup())
    all_issues.extend(check_api_endpoints())
    all_issues.extend(check_worker_agent())
    
    passed = sum(1 for _, status, _ in all_issues if status)
    total = len(all_issues)
    
    return {
        "ready": passed == total,
        "checks_passed": passed,
        "checks_total": total,
        "details": all_issues
    }

if __name__ == "__main__":
    result = verify_deployment_readiness()
    print(f"\n{'='*60}")
    print(f"Deployment Readiness Check")
    print(f"{'='*60}\n")
    
    for category, status, message in result["details"]:
        status_icon = "✓" if status else "✗"
        print(f"{status_icon} [{category}] {message}")
    
    print(f"\n{'='*60}")
    print(f"Status: {result['checks_passed']}/{result['checks_total']} checks passed")
    print(f"Ready: {'YES' if result['ready'] else 'NO'}")
    print(f"{'='*60}\n")

