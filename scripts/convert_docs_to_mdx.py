#!/usr/bin/env python3
"""
Professional MD to MDX Converter for Mintlify
Converts markdown files to properly formatted MDX with frontmatter and components.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Docs to convert - focus on architecture, how-it-works
DOCS_TO_CONVERT = {
    # Architecture & Core Concepts
    "architecture.md": {
        "dest": "core/architecture.mdx",
        "title": "Platform Architecture",
        "description": "Understand the distributed architecture of Nexa Compute",
        "group": "core"
    },
    "worker_registration.md": {
        "dest": "core/workers.mdx",
        "title": "Worker Registration & Management",
        "description": "How GPU workers are registered and orchestrated",
        "group": "core"
    },
    
    # Getting Started & Guides
    "QUICK_START.md": {
        "dest": "guides/quickstart.mdx",
        "title": "Quick Start Guide",
        "description": "Get up and running with Nexa Compute in 5 minutes",
        "group": "guides"
    },
    "SETUP.md": {
        "dest": "guides/setup.mdx",
        "title": "Installation & Setup",
        "description": "Complete setup guide for local and production environments",
        "group": "guides"
    },
    "START_HERE.md": {
        "dest": "guides/getting-started.mdx",
        "title": "Getting Started",
        "description": "Your first steps with Nexa Compute",
        "group": "guides"
    },
    "READY_TO_RUN.md": {
        "dest": "guides/deployment.mdx",
        "title": "Deployment Guide",
        "description": "Deploy Nexa Compute to production",
        "group": "guides"
    },
    
    # Forge API Docs
    "Nexa_Forge.md": {
        "dest": "forge/complete-reference.mdx",
        "title": "Complete API Reference",
        "description": "Full reference for all Nexa Forge API endpoints",
        "group": "forge"
    },
    
    # Evaluations
    "Evals.md": {
        "dest": "modules/evaluation.mdx",
        "title": "Evaluation System",
        "description": "How model evaluation works in Nexa Compute",
        "group": "modules"
    },
    
    # Overview docs
    "Overview_of_Project/ARCHITECTURE.md": {
        "dest": "core/system-design.mdx",
        "title": "System Design",
        "description": "Deep dive into Nexa Compute's system architecture",
        "group": "core"
    },
    "Overview_of_Project/DISTILLATION.md": {
        "dest": "modules/distillation.mdx",
        "title": "Model Distillation",
        "description": "Knowledge distillation from teacher to student models",
        "group": "modules"
    },
    "Overview_of_Project/PIPELINE_V2.md": {
        "dest": "modules/pipelines.mdx",
        "title": "Pipeline Architecture",
        "description": "How data flows through Nexa Compute pipelines",
        "group": "modules"
    },
    "Overview_of_Project/QUICK_START.md": {
        "dest": "guides/overview-quickstart.mdx",
        "title": "Project Quickstart",
        "description": "Quick overview of the Nexa Compute project",
        "group": "guides"
    },
    "Overview_of_Project/RUNBOOK.md": {
        "dest": "guides/runbook.mdx",
        "title": "Operations Runbook",
        "description": "Day-to-day operations and maintenance guide",
        "group": "guides"
    },
    
    # Pipeline docs
    "Pipeline/PIPELINE_GUIDE.md": {
        "dest": "modules/pipeline-guide.mdx",
        "title": "Pipeline Implementation Guide",
        "description": "Step-by-step guide to implementing ML pipelines",
        "group": "modules"
    },
}

# Skip these (specs, summaries, etc.)
SKIP_PATTERNS = [
    "Spec.md", "Spec_", "PRODUCTION_RUN", "TODAY_", "FALCON3",
    "DISTILLATION_COST", "EVAL_ISSUES", "scan_report", "SECRET_CLEANUP"
]


def should_skip(filename: str) -> bool:
    """Check if file should be skipped."""
    return any(pattern in filename for pattern in SKIP_PATTERNS)


def extract_title(content: str, fallback: str) -> str:
    """Extract title from first H1 or use fallback."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def extract_description(content: str) -> str:
    """Extract first meaningful paragraph as description."""
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Skip frontmatter, headers, and empty lines
        if line.strip() and not line.startswith('#') and not line.startswith('---'):
            # Clean up and truncate
            desc = line.strip()
            desc = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', desc)  # Remove markdown links
            return desc[:200] + ('...' if len(desc) > 200 else '')
    return "Documentation for Nexa Compute"


def format_code_blocks(content: str) -> str:
    """Ensure code blocks have proper language tags."""
    # Find code blocks without language
    def add_language(match):
        code = match.group(1)
        # Try to detect language
        if 'import ' in code or 'def ' in code or 'class ' in code:
            return f"```python\n{code}\n```"
        elif 'curl ' in code or '#!/bin/bash' in code:
            return f"```bash\n{code}\n```"
        elif '{' in code and '"' in code and ':' in code:
            return f"```json\n{code}\n```"
        else:
            return f"```text\n{code}\n```"
    
    # Replace unmarked code blocks
    content = re.sub(r'```\n(.*?)\n```', add_language, content, flags=re.DOTALL)
    return content


def format_diagrams(content: str) -> str:
    """Format ASCII diagrams properly."""
    # Wrap text-based diagrams in code blocks if not already
    lines = content.split('\n')
    result = []
    in_diagram = False
    diagram_lines = []
    
    for line in lines:
        # Detect diagram patterns
        if ('┌' in line or '│' in line or '└' in line or '─' in line or 
            '→' in line or '▼' in line or '┐' in line or '┘' in line):
            if not in_diagram:
                in_diagram = True
                diagram_lines = []
            diagram_lines.append(line)
        else:
            if in_diagram and diagram_lines:
                # Close diagram
                result.append('```text')
                result.extend(diagram_lines)
                result.append('```')
                diagram_lines = []
                in_diagram = False
            result.append(line)
    
    return '\n'.join(result)


def enhance_formatting(content: str) -> str:
    """Add MDX components for better presentation."""
    # Convert important notes to callouts
    content = re.sub(
        r'\*\*Note:\*\*\s*(.+?)(?=\n\n|\n#|\Z)',
        r'<Note>\n\1\n</Note>',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'\*\*Warning:\*\*\s*(.+?)(?=\n\n|\n#|\Z)',
        r'<Warning>\n\1\n</Warning>',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'\*\*Tip:\*\*\s*(.+?)(?=\n\n|\n#|\Z)',
        r'<Tip>\n\1\n</Tip>',
        content,
        flags=re.DOTALL
    )
    
    # Convert numbered lists to Steps where appropriate
    # Look for sections with "Steps" or "Flow" in header
    def convert_steps(match):
        header = match.group(1)
        content = match.group(2)
        
        if 'step' in header.lower() or 'flow' in header.lower():
            # Try to convert to Steps component
            lines = content.strip().split('\n')
            steps = []
            for line in lines:
                if re.match(r'^\d+\.\s+(.+)', line):
                    steps.append(re.sub(r'^\d+\.\s+(.+)', r'  <Step title="\1">\n  </Step>', line))
            
            if steps:
                return f"{header}\n\n<Steps>\n" + '\n'.join(steps) + "\n</Steps>\n"
        
        return match.group(0)
    
    content = re.sub(
        r'(###?\s+.+?(?:Steps|Flow).+?\n)((?:\d+\..+?\n)+)',
        convert_steps,
        content,
        flags=re.IGNORECASE
    )
    
    return content


def create_frontmatter(title: str, description: str, group: str = "") -> str:
    """Create MDX frontmatter."""
    frontmatter = f"""---
title: '{title}'
description: '{description}'"""
    
    if group:
        frontmatter += f"\nicon: '{get_icon_for_group(group)}'"
    
    frontmatter += "\n---\n\n"
    return frontmatter


def get_icon_for_group(group: str) -> str:
    """Get appropriate icon for content group."""
    icons = {
        'core': 'sitemap',
        'guides': 'rocket',
        'forge': 'code',
        'modules': 'puzzle-piece',
    }
    return icons.get(group, 'book')


def convert_md_to_mdx(source_path: Path, dest_path: Path, metadata: Dict) -> bool:
    """Convert a single MD file to MDX with formatting."""
    try:
        # Read source
        content = source_path.read_text(encoding='utf-8')
        
        # Remove existing frontmatter if present
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        
        # Extract or use metadata
        title = metadata.get('title', extract_title(content, source_path.stem))
        description = metadata.get('description', extract_description(content))
        group = metadata.get('group', '')
        
        # Format content
        content = format_code_blocks(content)
        content = format_diagrams(content)
        content = enhance_formatting(content)
        
        # Create frontmatter
        frontmatter = create_frontmatter(title, description, group)
        
        # Combine
        mdx_content = frontmatter + content
        
        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write MDX
        dest_path.write_text(mdx_content, encoding='utf-8')
        
        print(f"✓ {source_path.name} → {dest_path.relative_to(dest_path.parents[1])}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert {source_path.name}: {e}")
        return False


def main():
    """Main conversion process."""
    docs_dir = Path("/Users/allanmurimiwandia/Nexa_compute/docs")
    mintlify_dir = docs_dir / "mintlify"
    
    print("=" * 60)
    print("Converting MD → MDX with Professional Formatting")
    print("=" * 60)
    print()
    
    converted = 0
    failed = 0
    
    for source_file, metadata in DOCS_TO_CONVERT.items():
        source_path = docs_dir / source_file
        dest_path = mintlify_dir / metadata['dest']
        
        if not source_path.exists():
            print(f"⚠ Skipping {source_file} (not found)")
            continue
        
        if should_skip(source_file):
            print(f"⊘ Skipping {source_file} (excluded)")
            continue
        
        if convert_md_to_mdx(source_path, dest_path, metadata):
            converted += 1
        else:
            failed += 1
    
    print()
    print("=" * 60)
    print(f"✓ Converted: {converted}")
    print(f"✗ Failed: {failed}")
    print("=" * 60)
    print()
    print("Next: Update mint.json with new navigation structure")


if __name__ == "__main__":
    main()
