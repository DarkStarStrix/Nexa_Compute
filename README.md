# Nexa Compute & Nexa Forge

> **AI Infrastructure Platform with Managed API Service**

A complete AI foundry platform for orchestrating data generation, model distillation, training, and evaluation on ephemeral GPU compute.

## Quick Start

### Local Development

```bash
# 1. Start Backend API
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
uvicorn nexa_compute.api.main:app --port 8000

# 2. Start Frontend Dashboard (new terminal)
cd frontend && npm run dev

# 3. Start Mintlify Docs (new terminal)
cd docs/mintlify && npx mintlify dev

# 4. Populate Demo Data
python scripts/create_dashboard_demo.py
```

## Project Structure

```
Nexa_compute/
├── src/nexa_compute/api/          # FastAPI backend
│   ├── main.py                    # Main application
│   ├── auth.py                    # API key authentication
│   ├── database.py                # SQLAlchemy models
│   ├── endpoints/                 # API routes
│   └── services/                  # Business logic
├── frontend/                      # Next.js dashboard
│   ├── app/                       # App Router pages
│   ├── components/                # React components
│   └── lib/                       # API client
├── sdk/                           # Python SDK
│   ├── nexa_forge/               # Client library
│   ├── setup.py                  # Package config
│   └── demo.py                   # Demo script
├── docs/mintlify/                # Documentation
│   ├── mint.json                 # Mintlify config
│   ├── *.mdx                     # Doc pages
│   └── logo/                     # Branding assets
└── scripts/                      # Utility scripts
```

## Core Features

### Backend (FastAPI)

- ✅ 6 job types (generate, audit, distill, train, evaluate, deploy)
- ✅ Worker management & orchestration
- ✅ API key authentication (SHA256 hashed)
- ✅ Metered billing tracking
- ✅ Real-time job status & logs

### Frontend (Next.js)

- ✅ Real-time dashboard with metrics
- ✅ Expandable job logs
- ✅ Worker fleet monitoring
- ✅ Billing analytics with charts
- ✅ Secure API key management
- ✅ Dark theme UI

### Python SDK

- ✅ Simple client API
- ✅ All 6 job types supported
- ✅ Environment variable config
- ✅ Comprehensive documentation

### Documentation (Mintlify)

- ✅ Getting started guides
- ✅ API reference
- ✅ Architecture diagrams
- ✅ Pricing information
- ✅ SDK examples

## 💰 Freemium Model

| Feature | Free Tier | Pro Plan |
|---------|-----------|----------|
| **GPU Hours/Month** | 10 | Unlimited |
| **Concurrent Jobs** | 2 | 50 |
| **Job Retention** | 7 days | 90 days |
| **Support** | Community | Priority |
| **SLA** | None | 99.9% |
| **Price** | $0 | $99/mo + usage |

## Python SDK Usage

```python
from nexa_forge import NexaForgeClient

# Initialize client
client = NexaForgeClient(api_key="nexa_...")

# Generate data
job1 = client.generate(domain="biology", num_samples=100)

# Train model
job2 = client.train(
    model_id="llama-3-8b",
    dataset_uri="s3://bucket/data.parquet",
    epochs=3
)

# Monitor jobs
status = client.get_job(job1['job_id'])
all_jobs = client.list_jobs(limit=10)
```

## Security

- API keys hashed with SHA256
- One-time key display on creation
- Secure modal with warnings
- Revocation support
- Ready for rate limiting

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Next.js 16, Tailwind CSS, Recharts
- **SDK**: Python 3.11+
- **Docs**: Mintlify
- **Deployment**: Docker Compose ready

## Deployment

### Docker Compose

```bash
./scripts/start_forge.sh
```

### Production

- Frontend → Vercel
- Backend → Railway/Render
- Docs → Mintlify
- Database → PostgreSQL

## Documentation

Full documentation available at <http://localhost:3001>

Key sections:

- **Getting Started**: Quick setup guide
- **Architecture**: Platform overview
- **API Reference**: Complete endpoint docs
- **SDK Guide**: Python client usage
- **Pricing**: Freemium model details

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

**Built with ❤️ using FastAPI, Next.js, and Mintlify**
