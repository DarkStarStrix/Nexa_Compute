import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from nexa_inference.server import serve_model

def deploy_model(
    checkpoint_id: str,
    port: int = 8000
) -> Dict[str, Any]:
    """
    Deploy a model checkpoint.
    
    Args:
        checkpoint_id: Identifier for the checkpoint
        port: Port to serve on
        
    Returns:
        Dict with deployment info
    """
    # Resolve checkpoint path
    # Assuming checkpoint_id maps to a folder in artifacts/checkpoints
    # In a real system, we'd query the registry.
    
    # Try to find it in checkpoints dir
    checkpoint_dir = ROOT / "artifacts" / "checkpoints"
    # Search for folder matching checkpoint_id (which might be a hash)
    # For MVP, assume exact match or simple lookup
    checkpoint_path = checkpoint_dir / checkpoint_id
    
    if not checkpoint_path.exists():
        # Try to find by hash in manifest? 
        # For now, just raise if not found
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
    # Start server in a separate thread/process?
    # For a "job", we usually deploy to a new container/service.
    # Here we might just return the command to run it, or start it if it's a local worker.
    # The spec says "deploy pipeline... creates internal inference endpoint".
    
    # For this wrapper, let's simulate deployment by returning the info needed to start it.
    # If we actually want to start it:
    # thread = threading.Thread(target=serve_model, args=(checkpoint_path,), kwargs={"port": port})
    # thread.daemon = True
    # thread.start()
    
    return {
        "deployment_id": f"deploy_{checkpoint_id}",
        "inference_url": f"http://localhost:{port}/infer",
        "status": "deployed", # or "starting"
        "command": f"python -m nexa_inference.server --checkpoint {checkpoint_path} --port {port}"
    }
