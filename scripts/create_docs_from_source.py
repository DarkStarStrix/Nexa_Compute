#!/usr/bin/env python3
"""
Create comprehensive documentation from source materials.
Focus on WHAT is built and HOW it works, not implementation details.
"""

import re
from pathlib import Path
import json

docs_dir = Path("/Users/allanmurimiwandia/Nexa_compute/docs")
mintlify_dir = docs_dir / "mintlify"
src_dir = Path("/Users/allanmurimiwandia/Nexa_compute/src")
sdk_dir = Path("/Users/allanmurimiwandia/Nexa_compute/sdk")

# Documentation sources
NEXA_COMPUTE_SOURCES = {
    "overview": {
        "sources": ["Overview_of_Project/README.md", "MAIN_README.md"],
        "dest": "compute/overview.mdx",
        "title": "Nexa Compute Overview",
        "description": "Open-source ML toolbox for distributed training and evaluation"
    },
    "quickstart": {
        "sources": ["QUICK_START.md"],
        "dest": "compute/quickstart.mdx",
        "title": "Quick Start Guide",
        "description": "Get started with Nexa Compute in minutes"
    },
    "operations": {
        "sources": ["Overview_of_Project/RUNBOOK.md"],
        "dest": "compute/operations.mdx",
        "title": "Operations Guide",
        "description": "How to operate and maintain Nexa Compute"
    },
    "policies": {
        "sources": ["Overview_of_Project/POLICY.md"],
        "dest": "compute/policies.mdx",
        "title": "Policies & Best Practices",
        "description": "Development policies and conventions"
    },
    "pipelines": {
        "sources": ["Pipeline/PIPELINE_GUIDE.md"],
        "dest": "compute/pipelines.mdx",
        "title": "ML Pipelines",
        "description": "How pipelines work in Nexa Compute"
    },
}

NEXA_FORGE_SOURCES = {
    "overview": {
        "sources": ["Nexa_Forge.md", "PLATFORM_OVERVIEW.md"],
        "dest": "forge/overview.mdx",
        "title": "Nexa Forge Overview",
        "description": "Managed API service for AI workflows"
    },
}

def extract_relevant_content(content, title):
    """Extract content, removing build instructions and specs."""
    
    # Remove sections about implementation
    content = re.sub(r'##\s+(Implementation|Build|Setup)\s+.*?(?=\n##|\Z)', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove TODOs and implementation notes
    content = re.sub(r'(?:TODO|FIXME|NOTE TO SELF):.*?(?=\n\n|\Z)', '', content, flags=re.DOTALL)
    
    # Remove execution summaries
    content = re.sub(r'##\s+Execution\s+Summary.*?(?=\n##|\Z)', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    return content

def create_clean_mdx(sources, dest, title, description):
    """Combine multiple source files into one clean MDX."""
    combined_content = []
    
    for source in sources:
        source_path = docs_dir / source
        if source_path.exists():
            content = source_path.read_text(encoding='utf-8')
            content = extract_relevant_content(content, title)
            
            # Remove existing frontmatter
            content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
            
            # Remove triple heading hashes (we'll use hierarchy)
            content = re.sub(r'^###\s+', '## ', content, flags=re.MULTILINE)
            
            combined_content.append(content)
    
    if not combined_content:
        return False
    
    # Create frontmatter
    frontmatter = f'''---
title: "{title}"
description: "{description}"
---

'''
    
    # Combine all content
    final_content = frontmatter + '\n\n'.join(combined_content)
    
    # Clean up
    final_content = re.sub(r'\n{3,}', '\n\n', final_content)  # Max 2 newlines
    final_content = re.sub(r'<[^>]+>', '', final_content)  # Remove HTML/JSX
    
    # Write
    dest_path = mintlify_dir / dest
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(final_content, encoding='utf-8')
    
    print(f"✓ Created {dest}")
    return True

def scan_sdk_for_docs():
    """Extract SDK documentation."""
    sdk_readme = sdk_dir / "README.md"
    if sdk_readme.exists():
        content = sdk_readme.read_text(encoding='utf-8')
        
        frontmatter = '''---
title: "Python SDK"
description: "Official Python SDK for Nexa Forge"
---

'''
        content = frontmatter + content
        content = re.sub(r'<[^>]+>', '', content)
        
        dest = mintlify_dir / "forge/sdk.mdx"
        dest.write_text(content, encoding='utf-8')
        print("✓ Created forge/sdk.mdx")

def create_api_reference():
    """Create API reference from actual endpoints."""
    
    api_content = '''---
title: "API Reference"
description: "Complete REST API reference for Nexa Forge"
---

# API Endpoints

## Authentication

All API requests require an API key in the `X-Nexa-Api-Key` header.

```bash
curl -H "X-Nexa-Api-Key: YOUR_KEY" https://api.nexa.ai/api/jobs
```

## Jobs API

### Submit a Job

**Endpoint:** `POST /api/jobs/{job_type}`

**Job Types:**
- `generate` - Generate synthetic data
- `audit` - Audit dataset quality
- `distill` - Distill models
- `train` - Train/fine-tune models
- `evaluate` - Run evaluations
- `deploy` - Deploy models

**Example:**

```python
from nexa_forge import NexaForgeClient

client = NexaForgeClient(api_key="YOUR_KEY")
job = client.generate(domain="medical", num_samples=100)
```

### Get Job Status

**Endpoint:** `GET /api/jobs/{job_id}`

Returns current job status, logs, and results.

### List Jobs

**Endpoint:** `GET /api/jobs/`

**Query Parameters:**
- `skip` - Pagination offset
- `limit` - Number of results
- `status` - Filter by status

## Workers API

### Register Worker

**Endpoint:** `POST /api/workers/register`

Register a new GPU worker to the pool.

### Worker Heartbeat

**Endpoint:** `POST /api/workers/heartbeat`

Send periodic heartbeat to maintain worker registration.

## Billing API

### Get Billing Summary

**Endpoint:** `GET /api/billing/summary`

Returns usage and cost breakdown.

## Rate Limits

- **Free Tier:** 100 requests/minute
- **Pro Tier:** 1000 requests/minute
'''
    
    dest = mintlify_dir / "forge/api-reference.mdx"
    dest.write_text(api_content, encoding='utf-8')
    print("✓ Created forge/api-reference.mdx")

def create_dashboard_docs():
    """Create dashboard documentation."""
    
    dash_content = '''---
title: "Dashboard"
description: "Web dashboard for monitoring and managing AI workflows"
---

# Nexa Forge Dashboard

The Nexa Forge dashboard provides a comprehensive view of your AI operations.

## Features

### Job Management
- View all submitted jobs
- Real-time status updates
- Expandable logs for debugging
- Filter by status (running, completed, failed)

### Worker Fleet
- Monitor GPU worker status
- View worker specifications
- Track current job assignments

### Billing & Usage
- Real-time cost tracking
- Usage breakdown by resource type
- Cost visualization with charts
- Invoice history

### API Keys
- Generate new API keys
- View active keys (prefix only)
- Revoke compromised keys
- Security best practices

## Access

The dashboard is available at `http://localhost:3000` for local deployments.

### Navigation

- **Overview** - Metrics and recent activity
- **Jobs** - Job management and logs
- **Workers** - GPU fleet status
- **Billing** - Usage and costs
- **Settings** - API key management
'''
    
    dest = mintlify_dir / "forge/dashboard.mdx"
    dest.write_text(dash_content, encoding='utf-8')
    print("✓ Created forge/dashboard.mdx")

# Main execution
print("=" * 60)
print("Creating Documentation from Source Materials")
print("=" * 60)
print()

print("Nexa Compute Documentation...")
for key, config in NEXA_COMPUTE_SOURCES.items():
    create_clean_mdx(config["sources"], config["dest"], config["title"], config["description"])

print("\nNexa Forge Documentation...")
for key, config in NEXA_FORGE_SOURCES.items():
    create_clean_mdx(config["sources"], config["dest"], config["title"], config["description"])

print("\nSDK Documentation...")
scan_sdk_for_docs()

print("\nAPI Reference...")
create_api_reference()

print("\nDashboard Documentation...")
create_dashboard_docs()

print()
print("=" * 60)
print("✓ Documentation Created!")
print("=" * 60)
