# Nexa Forge Platform Overview

## What is Nexa Forge?

Nexa Forge is an **API-first AI foundry platform** designed for orchestrating data generation, model distillation, training, and evaluation workflows on ephemeral GPU compute. Users interact programmatically via the Python SDK, while the dashboard provides management and observability.

---

## Architecture

### Core Components

1. **Backend API** (`src/nexa_compute/api/`)
   - FastAPI-based REST API
   - Job orchestration and worker management
   - API key authentication
   - Metered billing tracking

2. **Python SDK** (`sdk/nexa_forge/`)
   - Official client library
   - Simple interface for all job types
   - Environment variable support

3. **Dashboard** (`frontend/`)
   - Next.js web interface
   - API key management
   - Job monitoring and billing
   - Usage analytics

4. **Worker Agents**
   - Pull-based job execution
   - GPU worker registration
   - Heartbeat system

---

## User Workflow

### 1. User Onboarding

1. User accesses the dashboard at `http://localhost:3000`
2. Navigates to **Settings** → **API Keys**
3. Clicks **Generate New Key**
4. Modal appears with the full key (shown **only once**)
5. User copies key and stores it securely

### 2. SDK Installation

```bash
pip install nexa-forge
```

### 3. Programmatic Usage

```python
from nexa_forge import NexaForgeClient

# Initialize with API key
client = NexaForgeClient(api_key="nexa_abc123...")

# Submit jobs
job = client.generate(domain="biology", num_samples=100)
print(f"Job ID: {job['job_id']}")

# Monitor status
status = client.get_job(job['job_id'])
print(f"Status: {status['status']}")
```

### 4. Dashboard Monitoring

Users can:

- View job execution in **Jobs** tab
- Monitor worker fleet in **Workers** tab
- Track costs in **Billing** tab
- Browse artifacts (datasets, checkpoints) in **Artifacts** tab

---

## API Endpoints

### Authentication

- `POST /api/auth/api-keys` - Generate new API key
- `GET /api/auth/api-keys` - List user's API keys
- `DELETE /api/auth/api-keys/{key_id}` - Revoke a key

### Jobs

- `POST /api/jobs/{job_type}` - Submit a job (generate, audit, distill, train, evaluate, deploy)
- `GET /api/jobs/{job_id}` - Get job status
- `GET /api/jobs/` - List jobs (with filtering)

### Workers

- `POST /api/workers/register` - Register a worker
- `POST /api/workers/heartbeat` - Send heartbeat
- `POST /api/workers/next_job` - Poll for next job
- `GET /api/workers/` - List all workers

### Billing

- `GET /api/billing/summary` - Get usage and cost summary

---

## SDK Methods

### Data Operations

```python
# Generate synthetic data
client.generate(domain="medical", num_samples=1000)

# Audit dataset quality
client.audit(dataset_uri="s3://bucket/data.parquet")
```

### Model Operations

```python
# Distill a large model
client.distill(
    teacher_model="gpt-4",
    student_model="llama-3-8b",
    dataset_uri="s3://bucket/dataset.parquet"
)

# Fine-tune a model
client.train(
    model_id="llama-3-8b",
    dataset_uri="s3://bucket/train.parquet",
    epochs=3
)

# Evaluate model performance
client.evaluate(model_id="my-model-v1", benchmark="mmlu")

# Deploy to inference endpoint
client.deploy(model_id="my-model-v1", region="us-east-1")
```

---

## Security & Best Practices

### API Key Management

1. **Generation**: Keys are generated with high entropy using `secrets.token_urlsafe(32)`
2. **Storage**: Only the SHA256 hash is stored in the database
3. **Display**: Raw key is shown **only once** during creation
4. **Revocation**: Users can revoke keys at any time from the dashboard

### Authentication Flow

```
User Request → API → get_api_key() → Validate Hash → Return User
```

If no key or invalid key → 403 Forbidden

---

## Billing & Metering

### Tracked Resources

| Resource Type | Unit | Rate |
|--------------|------|------|
| GPU Hours | per hour | $2.50 |
| Input Tokens | per 1M | $10.00 |
| Output Tokens | per 1M | $30.00 |
| Storage | per GB/month | $0.02 |

### Usage Tracking

Every job execution automatically records:

- GPU time consumed
- Tokens processed (input/output)
- Storage used

Users can view:

- Real-time cost breakdown
- Usage trends over time
- Cost per job type

---

## Development & Testing

### Running Locally

1. **Start Backend**:

   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   uvicorn nexa_compute.api.main:app --port 8000
   ```

2. **Start Frontend**:

   ```bash
   cd frontend
   npm run dev
   ```

3. **Access**:
   - Dashboard: <http://localhost:3000>
   - API Docs: <http://localhost:8000/docs>

### Test Data Population

```bash
python scripts/populate_test_data.py
```

This creates mock workers and jobs for testing.

### SDK Demo

```bash
python sdk/demo.py
```

---

## Deployment

### Docker Compose (Recommended)

```bash
./scripts/start_forge.sh
```

This starts:

- Backend API (port 8000)
- Frontend Dashboard (port 3000)
- Worker agent (background)

### Production Considerations

1. **Database**: Migrate from SQLite to PostgreSQL
2. **Authentication**: Add proper user registration/login
3. **API Keys**: Consider rate limiting per key
4. **Workers**: Deploy on GPU instances (RunPod, Lambda Labs, etc.)
5. **Storage**: Integrate with S3 for artifact storage
6. **Monitoring**: Add observability (Prometheus, Grafana)

---

## Next Steps

### For Platform Development

- [ ] Add user registration/login
- [ ] Implement artifact storage (S3 integration)
- [ ] Add worker health checks and auto-scaling
- [ ] Integrate Stripe for payment processing
- [ ] Add comprehensive error handling

### For Users

1. Generate your API key from the dashboard
2. Install the SDK: `pip install nexa-forge`
3. Start submitting jobs!

---

## Support

- **Documentation**: <http://localhost:3000/docs>
- **API Reference**: <http://localhost:8000/docs>
- **GitHub**: [github.com/nexa-ai/nexa-forge](https://github.com)
