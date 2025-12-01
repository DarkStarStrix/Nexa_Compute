"""Reusable workflow templates for common ML tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional

from nexa_compute.orchestration.workflow import WorkflowBuilder, WorkflowDefinition


def training_workflow(
    model_name: str,
    dataset_name: str,
    dataset_version: str = "latest",
    epochs: int = 3,
    batch_size: int = 32,
) -> WorkflowDefinition:
    """Standard training pipeline: Data Prep -> Train -> Evaluate."""
    
    builder = WorkflowBuilder(f"train_{model_name}")
    
    # Step 1: Data Preparation
    builder.add_step(
        step_id="data_prep",
        uses="nexa_compute.steps.data.prepare",
        params={
            "dataset": dataset_name,
            "version": dataset_version,
        },
        outputs={"train_split": "train_data", "eval_split": "eval_data"},
    )
    
    # Step 2: Training
    builder.add_step(
        step_id="train",
        uses="nexa_compute.steps.train.train_model",
        inputs={"train_data": "train_data"},
        params={
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        outputs={"checkpoint": "model_checkpoint"},
        depends_on=["data_prep"],
        backend="slurm", # Offload heavy compute
    )
    
    # Step 3: Evaluation
    builder.add_step(
        step_id="evaluate",
        uses="nexa_compute.steps.eval.evaluate_model",
        inputs={
            "checkpoint": "model_checkpoint",
            "eval_data": "eval_data",
        },
        depends_on=["train"],
    )
    
    return builder.build()


def batch_inference_workflow(
    model_name: str,
    input_path: str,
    output_path: str,
) -> WorkflowDefinition:
    """Batch inference pipeline."""
    
    builder = WorkflowBuilder(f"infer_{model_name}")
    
    builder.add_step(
        step_id="batch_predict",
        uses="nexa_compute.steps.infer.batch_predict",
        params={
            "model": model_name,
            "input_path": input_path,
            "output_path": output_path,
        },
        backend="slurm",
    )
    
    return builder.build()

