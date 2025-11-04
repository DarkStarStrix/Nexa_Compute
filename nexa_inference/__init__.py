"""Inference module for serving trained models."""

from .server import InferenceServer, serve_model

__all__ = ["InferenceServer", "serve_model"]

