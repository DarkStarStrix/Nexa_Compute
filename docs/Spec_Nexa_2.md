
# Nexa API – Final SPEC.md

A remote, Tinker-style API built on top of NexaCompute’s existing engine room.

---

## 0. Overview

**Nexa API** is a remote service that lets users run an end-to-end model pipeline via simple API calls:

- **`audit`** – LLM-as-a-judge data quality audit  
- **`distill`** – teacher-model distillation to generate SFT-ready data  
- **`train`** – fine-tune/train a model  
- **`evaluate`** – evaluate a checkpoint with LLM-based or task-based metrics  
- **`deploy`** – deploy a trained checkpoint for inference  

Under the hood, all heavy work is done by **NexaCompute** (your existing system), running on GPU workers. Users **never** run training loops or touch infra.

---

## 1. Golden Rules

These are the constraints that keep the system sane and maintainable.

### Rule 1 — Everything lives in `src/nexacompute/`

All platform logic stays under:

```

src/nexacompute/
sdk/
server/
workers/
core/
storage/
billing/
utils/

````

No orchestration or platform logic is defined outside this tree.

---

### Rule 2 — Users call APIs; you run the loop

Nexa is a **remote API**, not a local library.

User mental model:

```python
from nexa import NexaClient

client = NexaClient(api_key="...")

audit_job   = client.audit("s3://bucket/my_data.parquet")
distill_job = client.distill("ds_123")
train_job   = client.train("ds_123d", model="Mistral-7B", epochs=3)
eval_job    = client.evaluate("ckpt_456")
deploy_job  = client.deploy("ckpt_456")

status = client.status(train_job["job_id"])
````

You manage:

* GPUs
* pipelines
* retries
* artifacts
* evaluation
* deployment

They just call functions.

---

### Rule 3 — DO = Control Plane, GPU Provider = Workers

Infra roles:

* **DigitalOcean VPS** (control plane):

  * FastAPI HTTP API
  * job queue
  * job state tracking
  * log streaming endpoint (SSE/WebSocket)
  * Stripe integration / metering
  * worker registry

* **Prime Intellect / other GPU boxes** (workers):

  * run NexaCompute pipelines
  * poll for jobs from control plane
  * stream logs + results back

NexaCompute code is deployed on both, but heavy jobs run on workers.

---

## 2. High-Level Architecture

```
User Code
   │
   ▼
 Nexa SDK (client library)
   │   (HTTP only)
   ▼
 DigitalOcean VPS (Control Plane)
 ┌─────────────────────────────────────┐
 │ FastAPI routes: /audit, /distill…  │
 │ Redis/DB job queue                 │
 │ Job table/state                    │
 │ SSE/WS log streaming               │
 │ Stripe billing hooks               │
 │ Worker registry + scheduler        │
 └─────────────────────────────────────┘
   │
   ▼
 GPU Workers (Prime Intellect, etc.)
 ┌─────────────────────────────────────┐
 │ Poll for jobs                      │
 │ Run NexaCompute pipelines          │
 │ Stream logs + results              │
 └─────────────────────────────────────┘
   │
   ▼
Artifacts Storage (DO Spaces / S3 / local FS)
```

---

## 3. External API (HTTP)

All operations are:

* asynchronous
* created via `POST`
* return a `job_id`
* monitored via `GET /status/{job_id}`

### 3.1 Endpoints

---

#### POST `/audit`

Request:

```json
{
  "dataset_uri": "s3://... or hf://... or internal://ds_123"
}
```

Response:

```json
{
  "job_id": "job_audit_001"
}
```

---

#### POST `/distill`

```json
{
  "dataset_id": "ds_123",
  "teacher": "openai/gpt-4o-mini"
}
```

Response:

```json
{
  "job_id": "job_distill_001"
}
```

---

#### POST `/train`

```json
{
  "dataset_id": "ds_123d",
  "model": "Mistral-7B",
  "epochs": 3
}
```

Response:

```json
{
  "job_id": "job_train_001"
}
```

---

#### POST `/evaluate`

```json
{
  "checkpoint_id": "ckpt_456"
}
```

Response:

```json
{
  "job_id": "job_eval_001"
}
```

---

#### POST `/deploy`

```json
{
  "checkpoint_id": "ckpt_456"
}
```

Response:

```json
{
  "job_id": "job_deploy_001"
}
```

---

#### GET `/status/{job_id}`

```json
{
  "job_id": "job_train_001",
  "job_type": "train",
  "status": "running",
  "created_at": "2025-11-20T19:00:00Z",
  "updated_at": "2025-11-20T19:05:30Z",
  "result": null,
  "logs_uri": "s3://nexa-logs/job_train_001.log",
  "artifacts_uri": "s3://nexa-artifacts/jobs/job_train_001/",
  "error": null
}
```

---

## 4. SDK Surface

The SDK is a thin wrapper around these HTTP endpoints.

### 4.1 Layout

```
src/nexacompute/sdk/
    __init__.py
    client.py
    models.py
```

### 4.2 NexaClient

```python
class NexaClient:
    def audit(self, dataset_uri) -> dict
    def distill(self, dataset_id, teacher="openai/gpt-4o-mini") -> dict
    def train(self, dataset_id, model="Mistral-7B", epochs=3) -> dict
    def evaluate(self, checkpoint_id) -> dict
    def deploy(self, checkpoint_id) -> dict
    def status(self, job_id) -> dict
```

---

## 5. Server Layout

```
src/nexacompute/server/
    api.py
    endpoints/
        audit.py
        distill.py
        train.py
        evaluate.py
        deploy.py
    models.py
    jobs.py
    queue.py
    scheduler.py
    streams.py
    auth.py
```

The API server:

* creates jobs
* enqueues jobs
* streams logs
* returns results

---

## 6. Job Model & State Machine

### 6.1 BaseJob Schema

```python
class BaseJob(BaseModel):
    job_id: str
    job_type: str
    user_id: str
    payload: Dict[str, Any]
    status: str
    result: Optional[Dict[str, Any]] = None
    logs_uri: Optional[str] = None
    artifacts_uri: Optional[str] = None
    error: Optional[str] = None
```

### 6.2 Request Schemas

```python
class AuditRequest(BaseModel):
    dataset_uri: str

class DistillRequest(BaseModel):
    dataset_id: str
    teacher: str

class TrainRequest(BaseModel):
    dataset_id: str
    model: str
    epochs: int

class EvaluateRequest(BaseModel):
    checkpoint_id: str

class DeployRequest(BaseModel):
    checkpoint_id: str
```

### 6.3 Job States

* `pending`
* `running`
* `completed`
* `failed`

---

## 7. Artifact Registry

Canonical layout:

```
artifacts/
    datasets/
    distill/
    checkpoints/
    evals/
    deployments/
```

Each artifact has a `manifest.json` containing:

* creation time
* job_id
* input sources
* config
* metrics
* hash

Registry provides helpers:

```python
get_dataset_uri(id)
get_checkpoint_uri(id)
get_eval_uri(id)
get_deployment_info(id)
```

---

## 8. Core Pipelines (Engine Room)

All heavy logic in NexaCompute stays in:

```
src/nexacompute/core/
```

### 8.1 Audit Pipeline

Input:

```json
{ "dataset_uri": "..." }
```

Output:

```json
{
  "dataset_id": "ds_123",
  "overall_quality_score": 4.2,
  "quality_tier": "A-",
  "metrics": {...},
  "scored_dataset_uri": "s3://.../datasets/ds_123/scored.parquet"
}
```

### 8.2 Distill Pipeline

```json
{
  "distilled_dataset_id": "ds_123d",
  "sft_dataset_uri": "s3://.../distill/ds_123d/sft.parquet",
  "num_samples": 10000,
  "estimated_tokens": 2.5e6
}
```

### 8.3 Train Pipeline

```json
{
  "checkpoint_id": "ckpt_456",
  "checkpoint_uri": "s3://.../checkpoints/ckpt_456/",
  "train_metrics_uri": "...",
  "gpu_hours": 3.2
}
```

### 8.4 Evaluate Pipeline

```json
{
  "eval_id": "ev_789",
  "scores": {...},
  "report_uri": "..."
}
```

### 8.5 Deploy Pipeline

```json
{
  "deployment_id": "dp_012",
  "inference_url": "https://models.nexa.run/dp_012",
  "hf_repo": "allan/nexa-ckpt-456"
}
```

---

## 9. Pricing Model

A simple metered system:

### Audit

```
$0.20 per 1,000 rows
```

### Distill

```
$0.30 per 1M tokens generated
```

### Train (GPU Hours)

```
A100: $1.25/hr
```

### Evaluate

```
$0.10 per 100 samples
```

### Deploy

```
$2/month
```

`billing/usage.py` and `billing/stripe.py` compute & record charges.

---

## 10. Worker Runtime (GPU Nodes)

Workers run the pipelines remotely.

Pseudo:

```python
while True:
    job = poll_job(worker_id)
    if not job:
        sleep(1)
        continue

    update(job, status="running")

    try:
        result = run_engine(job)
        save_artifacts(result)
        update(job, result=result, status="completed")
    except Exception as e:
        update(job, error=str(e), status="failed")
```

Workers only need:

* job schema
* core engines
* registry
* config

Everything else stays in the control plane.

---

## 11. Control Plane Logic

DigitalOcean server handles:

* API → job creation
* job queueing
* worker assignment
* log streaming
* Stripe billing

Minimal viable scheduler:

```python
def schedule():
    job = queue.pop_next()
    worker = pick_available_worker()
    assign_job(worker, job)
```

---

## 12. End-to-End Lifecycle

1. User uploads dataset
2. User calls `client.audit()`
3. Control plane creates `AuditJob`
4. Worker runs data audit pipeline
5. User receives dataset quality + scored parquet
6. User calls `client.distill()`
7. Worker generates SFT dataset
8. User calls `client.train()`
9. Worker trains model
10. User calls `client.evaluate()`
11. Worker runs evaluation
12. User calls `client.deploy()`
13. Worker deploys checkpoint

You orchestrate *nothing* manually — Nexa API handles it.

---

## 13. MVP Build Order

1. Define job schemas
2. Implement FastAPI endpoints
3. Implement job queue
4. Implement local worker
5. Refactor audit script → `audit_engine.py`
6. Attach minimal registry
7. Build SDK
8. Deploy control plane to DO
9. Attach PI GPU worker
10. Run end-to-end test

```
