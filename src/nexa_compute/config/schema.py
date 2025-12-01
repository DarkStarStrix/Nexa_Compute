"""Pydantic schemas defining configuration contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from nexa_compute.core.project_registry import (
    DEFAULT_PROJECT_REGISTRY,
    ProjectRegistryError,
    validate_project_slug,
)


class ProjectConfig(BaseModel):
    slug: str = "scientific_assistant"
    name: str = "Scientific Assistant"
    owner: Optional[str] = None

    @model_validator(mode="after")
    def _validate_slug(cls, value: "ProjectConfig") -> "ProjectConfig":
        validate_project_slug(value.slug)
        try:
            DEFAULT_PROJECT_REGISTRY.ensure_registered(value.slug)
        except ProjectRegistryError:
            # Allow unregistered projects during bootstrap; registry can be refreshed later.
            pass
        return value


class DataSplitConfig(BaseModel):
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    @model_validator(mode="after")
    def check_sum(cls, values: "DataSplitConfig") -> "DataSplitConfig":
        total = values.train + values.validation + values.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Train/validation/test splits must sum to 1.0")
        return values


class DataConfig(BaseModel):
    dataset_name: str = "synthetic"
    dataset_version: str | None = None
    source_uri: str | None = None
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    augmentations: List[str] = Field(default_factory=list)
    cache_dir: str = "data/cache"


class ModelConfig(BaseModel):
    name: str = "mlp_classifier"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    pretrained: bool = False
    checkpoint_path: Optional[str] = None


class OptimizerConfig(BaseModel):
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    eps: float = 1e-8
    betas: Optional[List[float]] = None
    args: Dict[str, Any] = Field(default_factory=dict)


class SchedulerConfig(BaseModel):
    name: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)


class DistributedConfig(BaseModel):
    backend: str = "nccl"
    world_size: int = 1
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    seed: int = 42


class CheckpointConfig(BaseModel):
    dir: str = "artifacts/checkpoints"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 3
    save_every_n_epochs: int = 1
    resume_from: Optional[str] = None


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"
    tensorboard: bool = True
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: Optional[str] = None
    flush_steps: int = 50


class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "f1"])
    batch_size: int = 128
    save_predictions: bool = True
    output_dir: str = "artifacts/evaluation"
    generate_confusion_matrix: bool = True
    generate_calibration_plot: bool = False
    calibration_bins: int = 10


class ExperimentConfig(BaseModel):
    name: str = "debug"
    output_dir: str = "artifacts/runs"
    tags: Dict[str, str] = Field(default_factory=dict)
    notes: Optional[str] = None


class TrainingLoopConfig(BaseModel):
    epochs: int = 10
    max_steps: Optional[int] = None
    log_every_n_steps: int = 10
    val_every_n_steps: Optional[int] = None
    gradient_clip_norm: Optional[float] = None
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    timeout_seconds: Optional[int] = None


class TrainingConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingLoopConfig = Field(default_factory=TrainingLoopConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def output_directory(self) -> Path:
        return Path(self.experiment.output_dir) / self.experiment.name

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
    }

    @model_validator(mode="after")
    def _ensure_project_scoping(cls, value: "TrainingConfig") -> "TrainingConfig":
        slug = value.project.slug

        def _ensure_path(current: str, expected_prefix: str) -> str:
            if current.startswith(expected_prefix):
                return current
            return f"{expected_prefix.rstrip('/')}/{Path(current).name}"

        if value.experiment.output_dir == "artifacts/runs":
            value.experiment.output_dir = f"artifacts/{slug}/runs"
        elif not value.experiment.output_dir.startswith(f"artifacts/{slug}"):
            value.experiment.output_dir = _ensure_path(value.experiment.output_dir, f"artifacts/{slug}")

        if value.evaluation.output_dir == "artifacts/evaluation":
            value.evaluation.output_dir = f"artifacts/{slug}/eval"
        elif not value.evaluation.output_dir.startswith(f"artifacts/{slug}"):
            value.evaluation.output_dir = _ensure_path(value.evaluation.output_dir, f"artifacts/{slug}")

        checkpoint_dir = value.training.checkpoint.dir
        if checkpoint_dir == "artifacts/checkpoints":
            value.training.checkpoint.dir = f"artifacts/{slug}/checkpoints"
        elif not checkpoint_dir.startswith(f"artifacts/{slug}"):
            value.training.checkpoint.dir = _ensure_path(checkpoint_dir, f"artifacts/{slug}/checkpoints")

        log_dir = value.training.logging.log_dir
        if log_dir == "logs":
            value.training.logging.log_dir = f"logs/{slug}"
        elif not log_dir.startswith(f"logs/{slug}"):
            value.training.logging.log_dir = _ensure_path(log_dir, f"logs/{slug}")

        return value
