"""Distributed helpers (thin wrappers around torch.distributed)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..config.schema import DistributedConfig
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    config: DistributedConfig

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def _setup_process(rank: int, world_size: int, config: DistributedConfig, fn: Callable[[DistributedContext], None], args: Iterable[Any]) -> None:
    os.environ.setdefault("MASTER_ADDR", config.master_addr)
    os.environ.setdefault("MASTER_PORT", str(config.master_port))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend=config.backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        device_count = max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(rank % device_count)
    context = DistributedContext(rank=rank, world_size=world_size, local_rank=rank, config=config)
    fn(context, *args)
    dist.destroy_process_group()


def launch_distributed(config: DistributedConfig, fn: Callable[[DistributedContext], None], *args: Any) -> None:
    world_size = config.world_size
    if world_size <= 1:
        context = DistributedContext(rank=0, world_size=1, local_rank=0, config=config)
        fn(context, *args)
        return
    mp.spawn(_setup_process, args=(world_size, config, fn, args), nprocs=world_size, join=True)


@contextmanager
def maybe_distributed(context: DistributedContext | None):
    try:
        yield context
    finally:
        if context and context.is_distributed and dist.is_initialized():
            dist.barrier()
