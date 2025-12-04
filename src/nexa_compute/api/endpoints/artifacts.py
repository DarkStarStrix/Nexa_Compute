"""Artifacts API endpoint for managing datasets, checkpoints, and reports."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from nexa_compute.api.database import get_db
from nexa_compute.api.models import JobResponse
from nexa_compute.api.services.job_manager import JobManager
from nexa_compute.api.config import get_settings
import os
import sys

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from infra.storage.backends import get_storage_backend, StorageBackend
except ImportError:
    # Fallback if running from different context
    try:
        from src.infra.storage.backends import get_storage_backend, StorageBackend
    except ImportError:
        # Last resort - define minimal interface
        class StorageBackend:
            def upload(self, local_path, remote_key): pass
            def download(self, remote_key, local_path): pass
            def exists(self, remote_key): return False
        
        def get_storage_backend(*args, **kwargs):
            return StorageBackend()

router = APIRouter()

def get_storage() -> StorageBackend:
    """Get storage backend from config."""
    settings = get_settings()
    backend_type = os.getenv("STORAGE_BACKEND", settings.STORAGE_BACKEND)
    
    if backend_type == "local":
        base_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        return get_storage_backend("local", base_dir=base_dir)
    elif backend_type in ["s3", "do_spaces"]:
        return get_storage_backend(
            backend_type,
            bucket=os.getenv("STORAGE_BUCKET", settings.S3_BUCKET or ""),
            access_key=os.getenv("STORAGE_ACCESS_KEY", ""),
            secret_key=os.getenv("STORAGE_SECRET_KEY", ""),
            region=os.getenv("STORAGE_REGION", "us-east-1"),
            endpoint_url=os.getenv("STORAGE_ENDPOINT_URL")
        )
    else:
        return get_storage_backend("local")

@router.get("/")
def list_artifacts(
    artifact_type: Optional[str] = Query(None, description="Filter by type: dataset, checkpoint, report"),
    limit: int = Query(100, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    storage: StorageBackend = Depends(get_storage),
    db: Session = Depends(get_db)
):
    """List artifacts."""
    artifacts = []
    
    # Use storage registry if available
    try:
        from storage.registry import ARTIFACTS_DIR
        artifacts_dir = Path(ARTIFACTS_DIR)
    except ImportError:
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    
    if not artifacts_dir.exists():
        return []
    
    # Scan for artifacts
    for artifact_type_dir in ["datasets", "checkpoints", "evals"]:
        type_path = artifacts_dir / artifact_type_dir
        if not type_path.exists():
            continue
            
        if artifact_type and artifact_type_dir != artifact_type + "s":
            continue
            
        for item in type_path.iterdir():
            if item.is_file() and item.suffix in [".parquet", ".pt", ".json", ".pth"]:
                size = item.stat().st_size
                artifacts.append({
                    "id": f"{artifact_type_dir}_{item.stem}",
                    "name": item.name,
                    "type": artifact_type_dir.rstrip("s"),
                    "size": size,
                    "size_human": _format_size(size),
                    "uri": f"file://{item.absolute()}",
                    "created": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
            elif item.is_dir():
                # Check for COMPLETE marker (artifact protocol)
                complete_marker = item / "COMPLETE"
                if complete_marker.exists():
                    meta_file = item / "meta.json"
                    if meta_file.exists():
                        import json
                        try:
                            meta = json.loads(meta_file.read_text())
                            artifacts.append({
                                "id": f"{artifact_type_dir}_{item.name}",
                                "name": item.name,
                                "type": artifact_type_dir.rstrip("s"),
                                "size": meta.get("bytes", 0),
                                "size_human": _format_size(meta.get("bytes", 0)),
                                "uri": meta.get("uri", f"file://{item.absolute()}"),
                                "created": meta.get("created_at", datetime.fromtimestamp(item.stat().st_mtime).isoformat())
                            })
                        except Exception as exc:
                            from nexa_compute.core.logging import get_logger
                            logger = get_logger(__name__)
                            logger.warning(
                                "artifact_metadata_read_failed",
                                extra={"path": str(item), "error": str(exc)},
                            )
                            continue
    
    # Sort by created date (newest first)
    artifacts.sort(key=lambda x: x["created"], reverse=True)
    
    return artifacts[skip:skip+limit]

@router.get("/{artifact_id}")
def get_artifact(
    artifact_id: str,
    storage: StorageBackend = Depends(get_storage),
    db: Session = Depends(get_db)
):
    """Get artifact details."""
    # Try to use storage registry
    try:
        from storage.registry import (
            get_dataset_uri,
            get_checkpoint_uri,
            get_eval_uri,
            get_deployment_info
        )
        
        parts = artifact_id.split("_", 1)
        if len(parts) == 2:
            artifact_type, name = parts
            try:
                if artifact_type == "datasets":
                    uri = get_dataset_uri(name)
                elif artifact_type == "checkpoints":
                    uri = get_checkpoint_uri(name)
                elif artifact_type == "evals":
                    uri = get_eval_uri(name)
                else:
                    raise HTTPException(status_code=404, detail="Invalid artifact type")
                
                return {"id": artifact_id, "uri": uri, "type": artifact_type.rstrip("s")}
            except FileNotFoundError:
                # Artifact not found in registry
                pass
    except ImportError:
        pass
    
    # Fallback: return basic info
    return {"id": artifact_id, "message": "Artifact details endpoint"}

@router.get("/{artifact_id}/download")
def download_artifact(
    artifact_id: str,
    storage: StorageBackend = Depends(get_storage),
    db: Session = Depends(get_db)
):
    """Get download URL for artifact."""
    # Extract artifact info from ID
    parts = artifact_id.split("_", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=404, detail="Invalid artifact ID")
    
    artifact_type, name = parts
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    artifact_path = artifacts_dir / f"{artifact_type}s" / name
    
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    # Return URI (in production, generate signed URL for S3)
    return {"uri": f"file://{artifact_path.absolute()}", "download_url": f"/api/artifacts/{artifact_id}/file"}

def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
