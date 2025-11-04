#!/usr/bin/env python3
"""Launch Streamlit dashboard with command line arguments."""

import sys
import subprocess
from pathlib import Path

# Project root
project_root = Path(__file__).parent.parent

# Streamlit script
streamlit_script = project_root / "nexa_ui" / "inspect_distillation.py"

# Default args
host = "0.0.0.0"  # Allow local network access
port = 8501

# Parse args if provided
if len(sys.argv) > 1:
    host = sys.argv[1]
if len(sys.argv) > 2:
    port = int(sys.argv[2])

print("=" * 70)
print("STREAMLIT DISTILLATION INSPECTOR")
print("=" * 70)
print(f"Host: {host}")
print(f"Port: {port}")
print(f"URL: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
print("=" * 70)
print()
print("Starting Streamlit...")
print("Press Ctrl+C to stop")
print()

# Run streamlit
cmd = [
    sys.executable, "-m", "streamlit", "run",
    str(streamlit_script),
    "--server.port", str(port),
    "--server.address", host,
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false",
]

subprocess.run(cmd)

