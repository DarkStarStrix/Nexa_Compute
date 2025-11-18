"""Utilities for preparing the evaluation environment.

This script verifies that required configuration files and directory
structures exist before running the scientific evaluation pipeline.
It can be invoked directly as a CLI entry point.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / ".env"
ENV_TEMPLATE = REPO_ROOT / "env" / "env.example"
EVAL_DIR = REPO_ROOT / "data" / "processed" / "evaluation"


@dataclass(frozen=True)
class EnvRequirement:
    """Description of a required environment variable."""

    key: str
    description: str


REQUIRED_ENV_VARS: List[EnvRequirement] = [
    EnvRequirement(
        key="OPENROUTER_API_KEY",
        description="API key used to access OpenRouter-hosted models.",
    ),
    EnvRequirement(
        key="OPENAI_API_KEY",
        description="Optional key for OpenAI endpoints (used for judge fallback).",
    ),
    EnvRequirement(
        key="NEXA_S3_PREFIX",
        description="Default S3 prefix for durable storage of evaluation artifacts.",
    ),
]


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create the directory tree required for evaluation artifacts."""

    for directory in paths:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_env_file(env_path: Path, template_path: Path) -> None:
    """Create or update the repository-level .env file with required keys."""

    if not env_path.exists():
        if template_path.exists():
            env_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            env_path.write_text("# NexaCompute environment variables\n", encoding="utf-8")

    existing_lines = env_path.read_text(encoding="utf-8").splitlines()
    present_keys = {line.split("=", 1)[0].strip() for line in existing_lines if "=" in line and not line.strip().startswith("#")}

    appended = False
    with env_path.open("a", encoding="utf-8") as handle:
        for requirement in REQUIRED_ENV_VARS:
            if requirement.key in present_keys:
                continue
            handle.write(
                f"\n# {requirement.description}\n{requirement.key}=\n"
            )
            appended = True

    if appended:
        print(f"[setup-env] Added missing keys to {env_path}")


def report_missing_environment() -> None:
    """Emit warnings for required keys that are not set in the current shell."""

    missing = [item.key for item in REQUIRED_ENV_VARS if not os.getenv(item.key)]
    if not missing:
        print("[setup-env] All required environment variables are present in the current shell.")
        return

    print("[setup-env] Missing environment variables detected:")
    for key in missing:
        print(f"  - {key}")
    print("Populate these values in your .env file and reload the shell before running evaluations.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Prepare the scientific evaluation environment.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect required files and directories without creating them.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for preparing evaluation environment assets."""

    args = parse_args()
    directories = [
        EVAL_DIR / "prompts",
        EVAL_DIR / "outputs",
        EVAL_DIR / "judgments",
        EVAL_DIR / "reports",
        EVAL_DIR / "reports" / "plots",
    ]

    if args.dry_run:
        print("[setup-env] Dry-run mode enabled; no changes will be written.")
        print(f"[setup-env] Would create: {[str(path) for path in directories]}")
        print(f"[setup-env] Would ensure env file exists at: {ENV_PATH}")
        report_missing_environment()
        return

    ensure_directories(directories)
    ensure_env_file(ENV_PATH, ENV_TEMPLATE)
    report_missing_environment()
    print("[setup-env] Environment preparation complete.")


if __name__ == "__main__":
    main()

