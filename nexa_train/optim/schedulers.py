"""Custom scheduler utilities for Nexa training."""

from __future__ import annotations

import torch.optim as optim


def build_scheduler(name: str, optimizer: optim.Optimizer, **kwargs):
    name = name.lower()
    if name == "cosine_warmup":
        from math import cos, pi

        class CosineWarmup(optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, warmup_steps: int = 100, total_steps: int = 1000):
                self.warmup_steps = warmup_steps
                self.total_steps = total_steps
                super().__init__(optimizer)

            def get_lr(self):
                step = self.last_epoch + 1
                if step < self.warmup_steps:
                    scale = step / max(1, self.warmup_steps)
                else:
                    progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                    scale = 0.5 * (1 + cos(pi * progress))
                return [base_lr * scale for base_lr in self.base_lrs]

        return CosineWarmup(optimizer, **kwargs)
    raise ValueError(f"Unknown scheduler {name}")
