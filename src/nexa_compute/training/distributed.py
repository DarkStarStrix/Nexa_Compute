"""Distributed helpers (thin wrappers around torch.distributed)."""

from __future__ import annotations

import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..config.schema import DistributedConfig
from ..utils.logging import get_logger
from ..utils.retry import RetryPolicy, retry_call

LOGGER = get_logger(__name__)
_DIST_INIT_POLICY = RetryPolicy(
    max_attempts=4,
    base_delay=2.0,
    max_delay=20.0,
    jitter=0.25,
    retry_exceptions=(RuntimeError, OSError),
)


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
    if rank != 0:
        _wait_for_master(config.master_addr, int(config.master_port))

    def _init_group() -> None:
        dist.init_process_group(backend=config.backend, rank=rank, world_size=world_size)

    retry_call(_init_group, policy=_DIST_INIT_POLICY)
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
    try:
        mp.spawn(_setup_process, args=(world_size, config, fn, args), nprocs=world_size, join=True)
    except Exception as exc:  # pragma: no cover - defensive; fallback path
        LOGGER.error(
            "distributed_launch_failed",
            extra={"extra_context": {"error": repr(exc), "world_size": world_size}},
        )
        LOGGER.info("falling_back_to_single_process")
        context = DistributedContext(rank=0, world_size=1, local_rank=0, config=config)
        fn(context, *args)


@contextmanager
def maybe_distributed(context: DistributedContext | None):
    try:
        yield context
    finally:
        if context and context.is_distributed and dist.is_initialized():
            dist.barrier()


def _wait_for_master(host: str, port: int, timeout: float = 30.0) -> None:
    """Best-effort health check ensuring master node is reachable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5.0):
                return
        except OSError:
            time.sleep(1.0)
    raise RuntimeError(f"Unable to reach distributed master at {host}:{port}")
