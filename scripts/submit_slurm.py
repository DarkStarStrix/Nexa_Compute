#!/usr/bin/env python3
"""Generate (and optionally submit) Slurm sweeps from YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from nexa_infra.slurm import prepare_slurm_batch


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for building a Slurm sweep."""

    parser = argparse.ArgumentParser(description="Build Slurm job array for NexaCompute sweeps")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML sweep configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for generated artifacts (defaults to runs/manifests/slurm)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Call sbatch on the generated script",
    )
    return parser.parse_args()


def main() -> None:
    """Generate Slurm artifacts and optionally submit the job array."""

    args = parse_args()
    artifacts = prepare_slurm_batch(
        args.config,
        submit=args.submit,
        output_dir=args.output_dir,
    )

    print("[slurm] Spec written to:", artifacts.spec_path)
    print("[slurm] Script written to:", artifacts.script_path)
    print("[slurm] Jobs in array:", artifacts.job_count)
    if artifacts.cost_manifest is not None:
        print("[slurm] Cost estimate:", artifacts.cost_manifest)
    if not args.submit:
        print("[slurm] Submit with: sbatch", artifacts.script_path)


if __name__ == "__main__":
    main()

