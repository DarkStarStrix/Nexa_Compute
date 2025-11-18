"""FastAPI inference server for Hugging Face causal models."""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class InferenceRequest(BaseModel):
    prompt: str
    temperature: float = 0.2
    max_tokens: int = 512


class InferenceResponse(BaseModel):
    text: str
    tokens: int
    latency_ms: float
    model_id: str


@lru_cache(maxsize=1)
def _load_model(model_id: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model


def build_app(model_id: str) -> FastAPI:
    tokenizer, model = _load_model(model_id)
    app = FastAPI(title="Nexa Scientific Inference", version="1.0.0")

    @app.post("/infer", response_model=InferenceResponse)
    def infer(request: InferenceRequest) -> InferenceResponse:
        input_ids = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        completion = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_tokens = output_ids.shape[-1] - input_ids["input_ids"].shape[-1]
        return InferenceResponse(
            text=completion,
            tokens=int(generated_tokens),
            latency_ms=float(latency_ms),
            model_id=model_id,
        )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "model_id": model_id}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a Hugging Face model via FastAPI.")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args.model_id)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

