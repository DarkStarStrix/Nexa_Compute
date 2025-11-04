"""Inference server for serving trained models via FastAPI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(f"Required dependencies missing: {e}. Install with: pip install torch fastapi uvicorn") from e

try:
    from nexa_train.models import DEFAULT_MODEL_REGISTRY
except ImportError:
    # Fallback for when nexa_train not available
    DEFAULT_MODEL_REGISTRY = None


class InferenceRequest(BaseModel):
    """Request schema for inference."""
    
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    parameters: Optional[Dict] = None


class InferenceResponse(BaseModel):
    """Response schema for inference."""
    
    text: str
    tokens: int
    latency_ms: float
    model_id: str


class InferenceServer:
    """FastAPI server for model inference."""
    
    def __init__(self, checkpoint_path: Path, config_path: Optional[Path] = None):
        """Initialize inference server with a trained model."""
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path, config_path)
        self.model.eval()
        self.model_id = checkpoint_path.parent.name
        
        self.app = FastAPI(title="NexaCompute Inference Server")
        self._setup_routes()
    
    def _load_model(self, checkpoint_path: Path, config_path: Optional[Path]) -> torch.nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Try to extract model config from checkpoint
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        elif config_path and config_path.exists():
            import yaml
            config = yaml.safe_load(config_path.read_text())
            model_config = config.get("model", {})
        else:
            raise ValueError("Cannot determine model architecture. Provide config_path or ensure checkpoint contains model_config.")
        
        # Build model from registry
        model_name = model_config.get("name", "mlp_classifier")
        if DEFAULT_MODEL_REGISTRY:
            model = DEFAULT_MODEL_REGISTRY.build(model_name, model_config.get("parameters", {}))
        else:
            # Fallback: load model directly from checkpoint
            raise ValueError("Model registry not available. Ensure nexa_train is installed.")
        
        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        def health():
            return {"status": "healthy", "model": self.model_id, "device": str(self.device)}
        
        @self.app.post("/infer", response_model=InferenceResponse)
        def infer(request: InferenceRequest):
            """Run inference on a prompt."""
            import time
            
            start_time = time.perf_counter()
            
            # Simple tokenization (extend for production with proper tokenizer)
            # For now, assume text input
            try:
                # This is a placeholder - implement proper tokenization based on model type
                # For text generation models, use the model's tokenizer
                with torch.no_grad():
                    # Placeholder inference logic
                    # In production, use proper tokenization and generation
                    output = f"Model inference for: {request.prompt[:100]}..."
                    tokens = len(output.split())
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                return InferenceResponse(
                    text=output,
                    tokens=tokens,
                    latency_ms=round(latency_ms, 2),
                    model_id=self.model_id
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/model/info")
        def model_info():
            """Get model information."""
            return {
                "model_id": self.model_id,
                "device": str(self.device),
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "checkpoint_path": str(self.checkpoint_path),
            }


def serve_model(
    checkpoint_path: Path,
    config_path: Optional[Path] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Serve a trained model via FastAPI."""
    import uvicorn
    
    server = InferenceServer(checkpoint_path, config_path)
    uvicorn.run(server.app, host=host, port=port)

