# Workflow Templates

NexaCompute provides reusable templates for common ML workflows.

## Training Workflow

The `training_workflow` template creates a standard pipeline:
1. **Data Preparation**: Materializes dataset splits.
2. **Training**: Runs model training (optionally on Slurm).
3. **Evaluation**: Computes metrics on the validation set.

### Usage

```python
from nexa_compute.orchestration.templates import training_workflow
from nexa_compute.orchestration.scheduler import get_scheduler

# Create workflow definition
workflow = training_workflow(
    model_name="bert-base-uncased",
    dataset_name="glue/mrpc",
    epochs=5
)

# Register and trigger
get_scheduler().register_workflow(workflow)
run_id = get_scheduler().trigger_workflow(workflow.name)
print(f"Started run: {run_id}")
```

## Batch Inference Workflow

The `batch_inference_workflow` template runs offline inference on large datasets.

### Usage

```python
from nexa_compute.orchestration.templates import batch_inference_workflow

workflow = batch_inference_workflow(
    model_name="my-classifier:v1",
    input_path="s3://bucket/inputs.jsonl",
    output_path="s3://bucket/predictions.jsonl"
)
```

## Custom Workflows

You can build custom workflows using the `WorkflowBuilder`:

```python
from nexa_compute.orchestration.workflow import WorkflowBuilder

builder = WorkflowBuilder("custom_pipeline")
builder.add_step(
    step_id="step1",
    uses="my_module.step_function",
)
builder.add_step(
    step_id="step2",
    uses="my_module.next_step",
    depends_on=["step1"]
)
workflow = builder.build()
```

