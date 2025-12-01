"""
Core Inference Server implementation for NexaCompute.

This server handles:
1. Model loading and initialization
2. Request processing and routing
3. Metrics and monitoring integration
4. Health checks and readiness probes
"""

import logging
import time
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from nexa_compute.api.config import get_settings
from nexa_compute.experiments import get_ab_manager
from nexa_compute.monitoring.model_monitor import ModelMonitor
from nexa_compute.utils.metrics import MetricsRegistry
from nexa_compute.utils.tracing import configure_tracing, instrument_app, trace_span

LOGGER = logging.getLogger(__name__)
settings = get_settings()

class InferenceServer:
    def __init__(self, model_name: str, model_version: str = "latest") -> None:
        self.model_name = model_name
        self.version = model_version
        self.monitor = ModelMonitor(model_name, model_version)
        
        self.app = FastAPI(title=f"Nexa Inference: {model_name}")
        self._setup_middleware()
        self._setup_routes()
        self._setup_tracing()
        
    def _setup_tracing(self) -> None:
        configure_tracing(service_name=f"nexa-inference-{self.model_name}", instrument_fastapi=True)
        instrument_app(self.app)

    def _setup_middleware(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            duration = time.time() - start_time
            
            MetricsRegistry.record_api_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                latency=duration,
            )
            return response

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "ok", "model": self.model_name, "version": self.version}

        @self.app.post("/v1/models/{model_name}:predict")
        @trace_span("predict")
        async def predict(model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            if model_name != self.model_name:
                raise HTTPException(status_code=404, detail="Model not found")
                
            start_time = time.time()
            
            # A/B Testing: Determine variant
            user_id = payload.get("user_id", "anonymous")
            variant = get_ab_manager().get_variant(f"ab_{self.model_name}", user_id)
            
            # TODO: Route to variant implementation (currently just using main model)
            # In a real implementation, this would load/call the specific variant weights
            
            try:
                # Simulated inference
                result = self._run_inference(payload["inputs"])
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Log for monitoring
                self.monitor.log_inference(
                    inputs=payload["inputs"],
                    outputs=result,
                    latency_ms=latency_ms,
                )
                
                # Check for drift periodically (async in background ideally)
                self.monitor.check_drift()
                
                # Log A/B test metrics
                get_ab_manager().log_metric(f"ab_{self.model_name}", variant, "latency", latency_ms)
                
                return {"outputs": result, "variant": variant}
                
            except Exception as exc:
                LOGGER.error("inference_failed", extra={"error": str(exc)})
                raise HTTPException(status_code=500, detail="Inference failed")

    def _run_inference(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for actual model execution
        # This would use vLLM, Torch, ONNX, etc.
        time.sleep(0.05) # Simulate compute
        return {"score": 0.95, "label": "positive"}

    def run(self, port: int = 8000) -> None:
        MetricsRegistry.start_server(port + 1) # Metrics on separate port
        uvicorn.run(self.app, host="0.0.0.0", port=port)


def serve_model(model_name: str, port: int = 8000) -> None:
    """Serve a model."""
    server = InferenceServer(model_name)
    server.run(port=port)


if __name__ == "__main__":
    server = InferenceServer("test-model")
    server.run()
