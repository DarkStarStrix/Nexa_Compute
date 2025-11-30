"""Integration utilities for src/storage, src/server, src/utils, and src/workers.

This ensures all legacy components work with the new API system.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Callable
import os
import sys

# Add src to path for imports - ensure we're using absolute paths
SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def get_storage_registry():
    """Get storage registry functions from src/storage."""
    try:
        from storage.registry import (
            get_dataset_uri,
            get_checkpoint_uri,
            get_eval_uri,
            get_deployment_info
        )
        return {
            "get_dataset_uri": get_dataset_uri,
            "get_checkpoint_uri": get_checkpoint_uri,
            "get_eval_uri": get_eval_uri,
            "get_deployment_info": get_deployment_info
        }
    except ImportError as e:
        print(f"Warning: Could not import storage registry: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error importing storage registry: {e}")
        return None

def get_worker_processor() -> Optional[Callable]:
    """Get worker processor from src/workers."""
    try:
        from workers.worker import process_job
        return process_job
    except ImportError as e:
        print(f"Warning: Could not import worker processor: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error importing worker processor: {e}")
        return None

def get_server_config():
    """Get server config from src/server."""
    try:
        from server.config import Config
        return Config
    except ImportError as e:
        print(f"Warning: Could not import server config: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error importing server config: {e}")
        return None

def get_server_models():
    """Get server models from src/server."""
    try:
        from server.models import BaseJob
        return {"BaseJob": BaseJob}
    except ImportError as e:
        print(f"Warning: Could not import server models: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error importing server models: {e}")
        return None

def check_integration() -> Dict[str, Any]:
    """Check integration status of all components."""
    results = {
        "storage_registry": get_storage_registry() is not None,
        "worker_processor": get_worker_processor() is not None,
        "server_config": get_server_config() is not None,
        "server_models": get_server_models() is not None,
    }
    
    results["all_available"] = all(results.values())
    return results

if __name__ == "__main__":
    status = check_integration()
    print("Integration Status:")
    for component, available in status.items():
        if component != "all_available":
            status_icon = "✓" if available else "✗"
            print(f"  {status_icon} {component}: {'Available' if available else 'Not Available'}")
    print(f"\nAll components available: {'✓ YES' if status['all_available'] else '✗ NO'}")
