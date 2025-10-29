"""Training callback system."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch.nn as nn

from ..config.schema import CheckpointConfig
from ..utils.checkpoint import save_checkpoint, checkpoint_path
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    should_stop: bool = False


class Callback:
    def on_init(self, state: TrainerState, model: nn.Module) -> None:  # noqa: D401
        """Invoked after callbacks are registered."""

    def on_train_start(self, state: TrainerState, model: nn.Module) -> None:
        pass

    def on_epoch_start(self, state: TrainerState, model: nn.Module) -> None:
        pass

    def on_batch_end(self, state: TrainerState, model: nn.Module, metrics: Dict[str, float]) -> None:
        pass

    def on_validation_end(self, state: TrainerState, model: nn.Module, metrics: Dict[str, float]) -> None:
        pass

    def on_train_end(self, state: TrainerState, model: nn.Module) -> None:
        pass


class CallbackRegistry:
    def __init__(self, callbacks: Iterable[Callback] | None = None) -> None:
        self.callbacks: List[Callback] = list(callbacks or [])

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def dispatch(self, hook: str, state: TrainerState, model: nn.Module, metrics: Optional[Dict[str, float]] = None) -> None:
        for callback in self.callbacks:
            handler = getattr(callback, hook, None)
            if callable(handler):
                if metrics is not None:
                    handler(state, model, metrics)
                else:
                    handler(state, model)

    def on_init(self, state: TrainerState, model: nn.Module) -> None:
        for callback in self.callbacks:
            callback.on_init(state, model)


class LoggingCallback(Callback):
    def __init__(self, log_every_n_steps: int = 10, rank_filter: Optional[int] = None) -> None:
        self.log_every_n_steps = log_every_n_steps
        self.rank_filter = rank_filter

    def on_batch_end(self, state: TrainerState, model: nn.Module, metrics: Dict[str, float]) -> None:
        if not self._should_log(state):
            return
        if state.global_step % self.log_every_n_steps == 0:
            LOGGER.info(
                "train_step",
                extra={"extra_context": {"global_step": state.global_step, **metrics}},
            )

    def on_validation_end(self, state: TrainerState, model: nn.Module, metrics: Dict[str, float]) -> None:
        if not self._should_log(state):
            return
        LOGGER.info(
            "validation_epoch",
            extra={"extra_context": {"epoch": state.epoch, **metrics}},
        )

    def _should_log(self, state: TrainerState) -> bool:
        if self.rank_filter is None:
            return True
        return getattr(state, "rank", 0) == self.rank_filter


class CheckpointSaver(Callback):
    def __init__(self, config: CheckpointConfig, *, global_rank: int = 0) -> None:
        self.config = config
        self.best_score: Optional[float] = None
        self.global_rank = global_rank

    def on_validation_end(self, state: TrainerState, model: nn.Module, metrics: Dict[str, float]) -> None:
        if self.global_rank != 0:
            return
        monitor_value = metrics.get(self.config.monitor)
        if monitor_value is None:
            LOGGER.warning("Checkpoint monitor missing", extra={"extra_context": {"monitor": self.config.monitor}})
            return
        improved = False
        if self.config.mode == "min":
            improved = self.best_score is None or monitor_value < self.best_score
        else:
            improved = self.best_score is None or monitor_value > self.best_score
        if improved:
            self.best_score = monitor_value
            path = checkpoint_path(self.config, epoch=state.epoch)
            save_checkpoint({"model_state": model.state_dict(), "epoch": state.epoch}, path.parent, filename=path.name)
            LOGGER.info(
                "checkpoint_saved",
                extra={"extra_context": {"path": str(path), "monitor": self.config.monitor, "score": monitor_value}},
            )


class EarlyStopping(Callback):
    def __init__(self, monitor: str, patience: int = 5, mode: str = "min") -> None:
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_score: Optional[float] = None
        self.wait: int = 0

    def on_validation_end(self, state: TrainerState, model: nn.Module, metrics: Dict[str, float]) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            return
        if self.best_score is None:
            self.best_score = value
            self.wait = 0
            return
        improved = value < self.best_score if self.mode == "min" else value > self.best_score
        if improved:
            self.best_score = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                LOGGER.info(
                    "early_stop_triggered",
                    extra={"extra_context": {"epoch": state.epoch, "monitor": self.monitor}},
                )
                state.should_stop = True


class MLflowCallback(Callback):
    def __init__(self) -> None:
        self._mlflow = None
        with contextlib.suppress(ImportError):
            import mlflow  # type: ignore

            self._mlflow = mlflow

    def _log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._mlflow is None:
            return
        active = self._mlflow.active_run()
        if active is None:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def on_batch_end(self, state: TrainerState, model: nn.Module, metrics: Dict[str, float]) -> None:
        self._log(metrics, step=state.global_step)

    def on_validation_end(self, state: TrainerState, model: nn.Module, metrics: Dict[str, float]) -> None:
        self._log(metrics, step=state.epoch)
