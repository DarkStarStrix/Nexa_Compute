"""Dataset augmentation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.data.dataset import SyntheticClassificationDataset  # type: ignore

Augmentor = Callable[[SyntheticClassificationDataset], SyntheticClassificationDataset]


def apply_augmentations(dataset: SyntheticClassificationDataset, augmentations: Dict[str, Augmentor]) -> SyntheticClassificationDataset:
    current = dataset
    for name, fn in augmentations.items():
        current = fn(current)
        print(f"[nexa-data] applied augmentation: {name}")
    return current
