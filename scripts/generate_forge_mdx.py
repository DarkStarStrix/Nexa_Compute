#!/usr/bin/env python3
"""
Generate clean, professional MDX docs for Nexa Forge.
- Remove emojis
- Ensure proper heading hierarchy (H1 title, H2 sections)
- Focus on WHAT the service does, not how it was built.
- Clean code examples (add language tags, remove stray characters)
"""

import re
from pathlib import Path

# Directories
DOCS_DIR = Path("/Users/allanmurimiwandia/Nexa_compute/docs")
MINT_DIR = DOCS_DIR / "mintlify"

# Source files for Nexa Forge
FORGE_SOURCES = {
    "overview": {
        "files": ["Nexa_Forge.md", "PLATFORM_OVERVIEW.md"],
        "dest": "forge/overview.mdx",
        "title": "Nexa Forge – Managed AI Service",
        "description": "What Nexa Forge provides and how it works"
    },
    "getting_started": {
        "files": ["QUICK_START.md"],
        "dest": "forge/getting-started.mdx",
        "title": "Getting Started with Nexa Forge",
        "description": "First steps to use the managed service"
    },
    "pricing": {
        "files": ["Overview_of_Project/POLICY.md"],
        "dest": "forge/pricing.mdx",
        "title": "Pricing & Plans",
        "description": "Freemium model and paid tiers"
    },
    "dashboard": {
        "files": [],  # will generate from existing dashboard description
        "dest": "forge/dashboard.mdx",
        "title": "Nexa Forge Dashboard",
        "description": "Web UI for monitoring jobs and workers"
    },
    "api_reference": {
        "files": [],  # generated from code
        "dest": "forge/api-reference.mdx",
        "title": "API Reference",
        "description": "Complete REST API documentation"
    },
    "sdk": {
        "files": ["sdk/README.md"],
        "dest": "forge/sdk.mdx",
        "title": "Python SDK",
        "description": "Official SDK for Nexa Forge"
    },
    "authentication": {
        "files": [],
        "dest": "forge/api/authentication.mdx",
        "title": "Authentication",
        "description": "How to authenticate API requests"
    },
}

def read_file(rel_path):
    p = DOCS_DIR / rel_path
    if p.exists():
        return p.read_text(encoding='utf-8')
    return ""

def strip_emojis(text):
    # Remove common emoji patterns (unicode range)
    return re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]', '', text)

def clean_headings(content):
    # Ensure a single H1 at top, then H2 for sections
    lines = content.split('\n')
    cleaned = []
    h1_seen = False
    for line in lines:
        if line.startswith('# '):
            if not h1_seen:
                cleaned.append(line)
                h1_seen = True
            else:
                # downgrade extra H1 to H2
                cleaned.append('#' + line)
        else:
            cleaned.append(line)
    return '\n'.join(cleaned)

def fix_code_blocks(content):
    # Add language tags where missing, remove stray backticks
    def repl(match):
        lang = match.group(1) or ''
        code = match.group(2)
        # Detect python or bash
        if 'import ' in code or 'def ' in code:
            lang = 'python'
        elif 'curl' in code or '#!/bin/bash' in code:
            lang = 'bash'
        else:
            lang = 'text'
        return f'```{lang}\n{code}\n```'
    return re.sub(r'```\n(.*?)\n```', repl, content, flags=re.DOTALL)

def generate_frontmatter(title, description):
    return f"---\ntitle: '{title}'\ndescription: '{description}'\n---\n\n"

def create_overview():
    src = "\n\n".join([read_file(f) for f in FORGE_SOURCES['overview']['files']])
    src = strip_emojis(src)
    src = clean_headings(src)
    src = fix_code_blocks(src)
    # Remove any spec/build sections
    src = re.sub(r'##\s+Implementation.*?(?=\n##|\Z)', '', src, flags=re.DOTALL|re.IGNORECASE)
    fm = generate_frontmatter(FORGE_SOURCES['overview']['title'], FORGE_SOURCES['overview']['description'])
    return fm + src

def create_getting_started():
    src = read_file(FORGE_SOURCES['getting_started']['files'][0])
    src = strip_emojis(src)
    src = clean_headings(src)
    src = fix_code_blocks(src)
    fm = generate_frontmatter(FORGE_SOURCES['getting_started']['title'], FORGE_SOURCES['getting_started']['description'])
    return fm + src

def create_pricing():
    src = read_file(FORGE_SOURCES['pricing']['files'][0])
    src = strip_emojis(src)
    src = clean_headings(src)
    fm = generate_frontmatter(FORGE_SOURCES['pricing']['title'], FORGE_SOURCES['pricing']['description'])
    return fm + src

def create_dashboard():
    # Use existing dashboard description from earlier script
    content = '''# Nexa Forge Dashboard
\n## Overview\nThe dashboard provides a unified view of jobs, workers, billing, and API keys.\n\n## Features\n- Real‑time job list with status badges\n- Worker pool health monitoring\n- Cost & usage charts\n- API key management (generate, view, revoke)\n\n## Access\nVisit `http://localhost:3000` for the local dev version.'''
    content = strip_emojis(content)
    content = clean_headings(content)
    fm = generate_frontmatter(FORGE_SOURCES['dashboard']['title'], FORGE_SOURCES['dashboard']['description'])
    return fm + content

def create_api_reference():
    # Build from code snippets (simplified)
    api_md = '''# API Reference\n\nAll Nexa Forge endpoints require an API key in the `X-Nexa-Api-Key` header.\n\n## Authentication\n```bash\ncurl -H "X-Nexa-Api-Key: YOUR_KEY" https://api.nexa.ai/auth/ping\n```\n\n## Jobs\n### Submit a Job\n`POST /api/jobs/{job_type}`\nSupported job types: generate, audit, distill, train, evaluate, deploy.\n\n```python\nfrom nexa_forge import NexaForgeClient\nclient = NexaForgeClient(api_key="YOUR_KEY")\njob = client.train(model="gpt2", dataset_id="my_data", epochs=1)\n```\n\n### Get Job Status\n`GET /api/jobs/{job_id}`\nReturns status, logs, and result URLs.\n\n```bash\ncurl -H "X-Nexa-Api-Key: YOUR_KEY" https://api.nexa.ai/api/jobs/12345\n```\n\n## Workers\n### Register Worker\n`POST /api/workers/register`\nProvide SSH host, user, and GPU specs.\n\n```json\n{\n  "ssh_host": "34.12.34.56",\n  "ssh_user": "root",\n  "gpu_count": 1,\n  "gpu_type": "A100-40GB"\n}\n```\n\n## Billing\n### Summary\n`GET /api/billing/summary`\nShows usage and cost breakdown.\n\n```bash\ncurl -H "X-Nexa-Api-Key: YOUR_KEY" https://api.nexa.ai/api/billing/summary\n```\n''' 
    api_md = strip_emojis(api_md)
    api_md = clean_headings(api_md)
    api_md = fix_code_blocks(api_md)
    fm = generate_frontmatter(FORGE_SOURCES['api_reference']['title'], FORGE_SOURCES['api_reference']['description'])
    return fm + api_md

def create_sdk():
    src = read_file(FORGE_SOURCES['sdk']['files'][0])
    src = strip_emojis(src)
    src = clean_headings(src)
    src = fix_code_blocks(src)
    fm = generate_frontmatter(FORGE_SOURCES['sdk']['title'], FORGE_SOURCES['sdk']['description'])
    return fm + src

def create_authentication():
    content = '''# Authentication\n\nAll API requests require an API key. Generate a key in the Dashboard → Settings page.\n\n## Using the API Key\nAdd the header `X-Nexa-Api-Key: YOUR_KEY` to every request.\n\n```bash\ncurl -H "X-Nexa-Api-Key: abcdef123456" https://api.nexa.ai/api/jobs\n```\n\n## Security Recommendations\n- Keep keys secret; never commit them to source control.\n- Rotate keys regularly via the dashboard.\n- Use HTTPS for all requests.\n''' 
    content = strip_emojis(content)
    content = clean_headings(content)
    content = fix_code_blocks(content)
    fm = generate_frontmatter(FORGE_SOURCES['authentication']['title'], FORGE_SOURCES['authentication']['description'])
    return fm + content

# Mapping of generator functions
GENERATORS = {
    'overview': create_overview,
    'getting_started': create_getting_started,
    'pricing': create_pricing,
    'dashboard': create_dashboard,
    'api_reference': create_api_reference,
    'sdk': create_sdk,
    'authentication': create_authentication,
}

def write_mdx(name, content):
    dest = MINT_DIR / FORGE_SOURCES[name]['dest']
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding='utf-8')
    print(f"✓ {dest.relative_to(MINT_DIR)}")

def main():
    print("Generating clean Nexa Forge MDX docs...")
    for name, func in GENERATORS.items():
        mdx = func()
        write_mdx(name, mdx)
    print("Done.")

if __name__ == "__main__":
    main()
