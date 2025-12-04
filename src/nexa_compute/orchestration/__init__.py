"""High-level orchestration entry points.

Responsibility: Wires together config, data, models, training, and evaluation into complete 
end-to-end training pipelines with manifest tracking and distributed execution support.
"""

from .pipeline import TrainingPipeline

__all__ = ["TrainingPipeline"]
