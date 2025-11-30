"""FastAPI dashboard for pipeline monitoring."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="NexaData Pipeline Monitor", version="1.0.0")


def find_manifest_files(root_dir: Path) -> List[Path]:
    """Find all manifest files in directory tree.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of manifest file paths
    """
    manifests = []
    if not root_dir.exists():
        return manifests
    
    for manifest_path in root_dir.rglob("dataset_manifest.json"):
        manifests.append(manifest_path)
    
    return manifests


def load_manifest(manifest_path: Path) -> Optional[Dict]:
    """Load manifest from file.
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        Manifest dictionary or None if error
    """
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load manifest {manifest_path}: {e}")
        return None


@app.get("/runs")
def list_runs(root_dir: str = "data/shards") -> List[Dict]:
    """List all pipeline runs.
    
    Args:
        root_dir: Root directory to search for manifests
        
    Returns:
        List of run summaries
    """
    root = Path(root_dir)
    manifests = find_manifest_files(root)
    
    runs = []
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        if not manifest:
            continue
        
        runs.append({
            "run_id": manifest.get("run_id"),
            "dataset": manifest.get("dataset"),
            "status": manifest.get("status", "unknown"),
            "created_at": manifest.get("created_at"),
            "num_shards": manifest.get("num_shards", 0),
            "dataset_hash": manifest.get("dataset_hash"),
            "manifest_path": str(manifest_path),
        })
    
    return runs


@app.get("/runs/{run_id}")
def get_run_details(run_id: str, root_dir: str = "data/shards") -> Dict:
    """Get detailed information about a specific run.
    
    Args:
        run_id: Run ID to retrieve
        root_dir: Root directory to search for manifests
        
    Returns:
        Run details dictionary
    """
    root = Path(root_dir)
    manifests = find_manifest_files(root)
    
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        if not manifest:
            continue
        
        if manifest.get("run_id") == run_id:
            return manifest
    
    raise HTTPException(status_code=404, detail=f"Run {run_id} not found")


@app.get("/runs/{run_id}/metrics")
def get_run_metrics(run_id: str, root_dir: str = "data/shards") -> Dict:
    """Get metrics for a specific run.
    
    Args:
        run_id: Run ID
        root_dir: Root directory to search
        
    Returns:
        Metrics dictionary
    """
    root = Path(root_dir)
    manifests = find_manifest_files(root)
    
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        if not manifest:
            continue
        
        if manifest.get("run_id") == run_id:
            metrics_path = manifest_path.parent / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    return json.load(f)
            else:
                return manifest.get("metrics", {})
    
    raise HTTPException(status_code=404, detail=f"Run {run_id} not found")


@app.get("/runs/{run_id}/shards")
def get_run_shards(run_id: str, root_dir: str = "data/shards") -> Dict:
    """Get shard list for a specific run.
    
    Args:
        run_id: Run ID
        root_dir: Root directory to search
        
    Returns:
        Shard information dictionary
    """
    root = Path(root_dir)
    manifests = find_manifest_files(root)
    
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        if not manifest:
            continue
        
        if manifest.get("run_id") == run_id:
            shards = manifest.get("shards", [])
            return {
                "run_id": run_id,
                "num_shards": len(shards),
                "shards": shards,
            }
    
    raise HTTPException(status_code=404, detail=f"Run {run_id} not found")


@app.get("/runs/{run_id}/timeseries")
def get_run_timeseries(run_id: str, root_dir: str = "data/shards") -> List[Dict]:
    """Get time series data for a specific run.
    
    Args:
        run_id: Run ID
        root_dir: Root directory to search
        
    Returns:
        Time series data list
    """
    root = Path(root_dir)
    manifests = find_manifest_files(root)
    
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        if not manifest:
            continue
        
        if manifest.get("run_id") == run_id:
            timeseries_path = manifest_path.parent / "resource_timeseries.json"
            if timeseries_path.exists():
                with open(timeseries_path) as f:
                    return json.load(f)
            else:
                return []
    
    raise HTTPException(status_code=404, detail=f"Run {run_id} not found")


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Serve the dashboard.
    
    Args:
        host: Host to bind
        port: Port to bind
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()

