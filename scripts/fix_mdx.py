#!/usr/bin/env python3
"""
Fix MDX formatting issues
"""

import re
from pathlib import Path

mintlify_dir = Path("/Users/allanmurimiwandia/Nexa_compute/docs/mintlify")

def fix_frontmatter(content: str) -> str:
    """Fix frontmatter escaping issues."""
    # Extract frontmatter
    match = re.search(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if not match:
        return content
    
    frontmatter = match.group(1)
    rest = content[match.end():]
    
    # Fix quotes in frontmatter values
    def fix_value(m):
        key = m.group(1)
        value = m.group(2)
        # Escape single quotes in value
        value = value.replace("'", "\\'")
        # Remove existing quotes and add clean ones
        value = value.strip('"').strip("'")
        return f"{key}: '{value}'"
    
    frontmatter = re.sub(r"(\w+):\s*['\"]?(.*?)['\"]?$", fix_value, frontmatter, flags=re.MULTILINE)
    
    return f"---\n{frontmatter}\n---\n{rest}"

def fix_code_blocks(content: str) -> str:
    """Fix code block issues."""
    # Remove JavaScript expressions in code blocks that cause parsing errors
    # Replace {variable} with {{variable}} in code blocks
    def fix_block(match):
        lang = match.group(1) or 'text'
        code = match.group(2)
        
        # Don't escape in JSX/React code
        if lang in ['jsx', 'tsx', 'react']:
            return match.group(0)
        
        # Escape single braces
       # code = re.sub(r'\{(?!\{)', '{{', code)
        # code = re.sub(r'(?<!\})\}', '}}', code)
        
        return f"```{lang}\n{code}\n```"
    
    content = re.sub(r'```(\w+)?\n(.*?)\n```', fix_block, content, flags=re.DOTALL)
    return content

def fix_mdx_components(content: str) -> str:
    """Fix MDX component syntax."""
    # Convert <Note> to proper MDX
    content = re.sub(r'<Note>\s*\n?(.*?)\n?</Note>', r'<Note>\n\1\n</Note>', content, flags=re.DOTALL)
    content = re.sub(r'<Warning>\s*\n?(.*?)\n?</Warning>', r'<Warning>\n\1\n</Warning>', content, flags=re.DOTALL)
    content = re.sub(r'<Tip>\s*\n?(.*?)\n?</Tip>', r'<Tip>\n\1\n</Tip>', content, flags=re.DOTALL)
    
    return content

def process_file(filepath: Path):
    """Process a single MDX file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        
        # Apply fixes
        content = fix_frontmatter(content)
        content = fix_code_blocks(content)
        content = fix_mdx_components(content)
        
        # Write back
        filepath.write_text(content, encoding='utf-8')
        print(f"✓ Fixed {filepath.relative_to(mintlify_dir)}")
        
    except Exception as e:
        print(f"✗ Error processing {filepath.name}: {e}")

def main():
    """Fix all MDX files."""
    print("Fixing MDX formatting issues...")
    print()
    
    for mdx_file in mintlify_dir.rglob("*.mdx"):
        process_file(mdx_file)
    
    print()
    print("Done!")

if __name__ == "__main__":
    main()
