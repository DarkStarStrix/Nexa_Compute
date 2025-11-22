# Nexa Forge Python SDK

Official Python SDK for [Nexa Forge](https://nexa.ai) - the AI foundry for data generation, distillation, training, and evaluation.

## Installation

```bash
pip install nexa-forge
```

Or install from source:

```bash
cd sdk/
pip install -e .
```

## Quick Start

### 1. Get Your API Key

Visit your [Nexa Forge Dashboard](http://localhost:3000/dashboard/settings) to generate an API key.

### 2. Initialize the Client

```python
from nexa_forge import NexaForgeClient

client = NexaForgeClient(api_key="your_api_key_here")
```

You can also set the API key as an environment variable:

```bash
export NEXA_API_KEY="your_api_key_here"
```

Then initialize without passing the key:

```python
client = NexaForgeClient()
```

## Usage Examples

### Data Generation

Generate synthetic data for a specific domain:

```python
job = client.generate(
    domain="medical_imaging",
    num_samples=1000,
    params={"resolution": "1024x1024"}
)

print(f"Job ID: {job['job_id']}")
print(f"Status: {job['status']}")
```

### Data Audit

Audit a dataset for quality issues:

```python
job = client.audit(
    dataset_uri="s3://my-bucket/dataset.parquet"
)
```

### Model Distillation

Distill a large teacher model into a smaller student:

```python
job = client.distill(
    teacher_model="gpt-4",
    student_model="llama-3-8b",
    dataset_uri="s3://my-bucket/distillation-data.parquet"
)
```

### Training

Fine-tune a model on your dataset:

```python
job = client.train(
    model_id="llama-3-8b",
    dataset_uri="s3://my-bucket/training-data.parquet",
    epochs=3,
    learning_rate=1e-5
)
```

### Evaluation

Run benchmarks on a trained model:

```python
job = client.evaluate(
    model_id="my-finetuned-model-v1",
    benchmark="mmlu"
)
```

### Model Deployment

Deploy a model to an inference endpoint:

```python
job = client.deploy(
    model_id="my-finetuned-model-v1",
    region="us-east-1"
)
```

## Monitoring Jobs

### Get Job Status

```python
status = client.get_job(job_id="job_abc123")
print(f"Status: {status['status']}")
print(f"Result: {status.get('result')}")
```

### List Jobs

```python
# List all jobs
jobs = client.list_jobs(limit=100)

# Filter by status
completed_jobs = client.list_jobs(status="completed")
```

## API Reference

### `NexaForgeClient`

#### Constructor

- `api_key` (str, optional): Your Nexa Forge API key. Defaults to `NEXA_API_KEY` env var.
- `api_url` (str, optional): API endpoint URL. Defaults to `http://localhost:8000/api`.

#### Methods

**Job Submission:**

- `generate(domain: str, num_samples: int, **kwargs)` → dict
- `audit(dataset_uri: str, **kwargs)` → dict
- `distill(teacher_model: str, student_model: str, dataset_uri: str, **kwargs)` → dict
- `train(model_id: str, dataset_uri: str, epochs: int, **kwargs)` → dict
- `evaluate(model_id: str, benchmark: str, **kwargs)` → dict
- `deploy(model_id: str, region: str, **kwargs)` → dict

**Job Management:**

- `get_job(job_id: str)` → dict
- `list_jobs(limit: int, status: str)` → List[dict]

## Error Handling

```python
try:
    job = client.generate(domain="biology", num_samples=100)
except requests.HTTPError as e:
    print(f"API Error: {e}")
```

## License

MIT
