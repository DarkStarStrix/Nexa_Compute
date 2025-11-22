#!/usr/bin/env python3
"""
Fix all MDX parsing errors and clean up docs.
Remove broken components, fix escaping, organize content.
"""

import re
from pathlib import Path
import shutil

mintlify_dir = Path("/Users/allanmurimiwandia/Nexa_compute/docs/mintlify")

# Files to keep - only platform/architecture docs
KEEP_FILES = {
    'introduction.mdx',
    'quickstart.mdx',
    
    # Nexa Compute (Toolbox)
    'core/architecture.mdx',
    'core/system-design.mdx',
    'core/workers.mdx',
    'modules/pipelines.mdx',
    'modules/distillation.mdx',
    'modules/evaluation.mdx',
    'platform/core-concepts.mdx',
    'guides/setup.mdx',
    'guides/deployment.mdx',
    
    # Nexa Forge (Managed Service)
    'forge/overview.mdx',
    'forge/pricing.mdx',
    'forge/getting-started.mdx',
    'forge/complete-reference.mdx',
    'forge/api/authentication.mdx',
}

def clean_frontmatter(content):
    """Fix frontmatter syntax."""
    lines = content.split('\n')
    
    if not lines or lines[0].strip() != '---':
        return content
    
    end_idx = -1
    for i in range(1, min(10, len(lines))):
        if lines[i].strip() == '---':
            end_idx = i
            break
    
    if end_idx == -1:
        return content
    
    # Rebuild frontmatter with clean syntax
    new_fm = ['---']
    for line in lines[1:end_idx]:
        if ':' in line:
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            new_fm.append(f'{key}: "{val}"')
        else:
            new_fm.append(line)
    new_fm.append('---')
    
    return '\n'.join(new_fm + lines[end_idx+1:])

def remove_all_components(content):
    """Remove ALL MDX components to avoid parsing errors."""
    # Remove all JSX-style tags
    content = re.sub(r'<[A-Z][^>]*?/>', '', content)  # Self-closing
    content = re.sub(r'<[A-Z][^>]*?>.*?</[A-Z][^>]*?>', '', content, flags=re.DOTALL)  # Paired tags
    
    # Remove specific broken patterns
    content = re.sub(r'</?CardGroup[^>]*?>', '', content)
    content = re.sub(r'</?Card[^>]*?>', '', content)
    content = re.sub(r'</?CodeGroup[^>]*?>', '', content)
    content = re.sub(r'</?Steps[^>]*?>', '', content)
    content = re.sub(r'</?Step[^>]*?>', '', content)
    content = re.sub(r'</?AccordionGroup[^>]*?>', '', content)
    content = re.sub(r'</?Accordion[^>]*?>', '', content)
    content = re.sub(r'</?Note[^>]*?>', '', content)
    content = re.sub(r'</?Warning[^>]*?>', '', content)
    content = re.sub(r'</?Tip[^>]*?>', '', content)
    content = re.sub(r'</?Info[^>]*?>', '', content)
    
    # Remove img tags
    content = re.sub(r'<img[^>]*?/?>', '', content)
    
    return content

def fix_code_blocks(content):
    """Fix code block JSON escaping."""
    def fix_json_in_code(match):
        lang = match.group(1) or 'text'
        code = match.group(2)
        
        # Don't escape in text blocks - just leave as is
        if lang == 'text':
            return f'```{lang}\n{code}\n```'
        
        # For other langs, remove problematic chars
        code = code.replace('{\"', '{ "')
        code = code.replace('"}', '" }')
        
        return f'```{lang}\n{code}\n```'
    
    content = re.sub(r'```(\w+)?\n(.*?)\n```', fix_json_in_code, content, flags=re.DOTALL)
    return content

def clean_empty_lines(content):
    """Remove excessive empty lines."""
    # Remove 3+ consecutive empty lines
    content = re.sub(r'\n\n\n+', '\n\n', content)
    return content

def process_file(filepath):
    """Clean a single MDX file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        
        # Apply fixes in order
        content = clean_frontmatter(content)
        content = remove_all_components(content)
        content = fix_code_blocks(content)
        content = clean_empty_lines(content)
        
        filepath.write_text(content, encoding='utf-8')
        print(f"✓ {filepath.relative_to(mintlify_dir)}")
        
    except Exception as e:
        print(f"✗ {filepath.name}: {e}")

def cleanup_docs():
    """Remove unwanted doc files."""
    for mdx_file in mintlify_dir.rglob("*.mdx"):
        rel_path = str(mdx_file.relative_to(mintlify_dir))
        
        if rel_path not in KEEP_FILES:
            mdx_file.unlink()
            print(f"🗑️  Removed: {rel_path}")

# Main execution
print("=" * 60)
print("Fixing MDX Parsing Errors & Cleaning Docs")
print("=" * 60)
print()

print("Step 1: Removing project-specific docs...")
cleanup_docs()
print()

print("Step 2: Fixing remaining docs...")
for mdx_file in mintlify_dir.rglob("*.mdx"):
    process_file(mdx_file)

print()
print("=" * 60)
print("✓ Complete!")
print("=" * 60)
