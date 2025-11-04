"""Nexa Distill package.

This package contains the modular distillation pipeline used to turn
scientific contexts into high-quality hypothesis–method pairs suitable for
supervised fine-tuning. The public modules expose collection, filtering,
inspection, regeneration, and packaging steps as described in
`docs/Overview_of_Project/Nexa_distill.md`.
"""

from importlib import resources
from pathlib import Path
from typing import Final


def package_path() -> Path:
    """Return the path to the installed Nexa Distill package."""

    return Path(resources.files(__package__))


PROMPTS_DIR: Final[Path] = package_path() / "prompts"
CONFIGS_DIR: Final[Path] = package_path() / "configs"

__all__ = ["PROMPTS_DIR", "CONFIGS_DIR", "package_path"]

