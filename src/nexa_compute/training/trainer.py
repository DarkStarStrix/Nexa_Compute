"""Training loop implementation."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from ..config.schema import TrainingConfig
from ..utils.logging import get_logger
from ..utils.seed import seed_everything
from .callbacks import Callback, CallbackRegistry, TrainerState

if TYPE_CHECKING:
    from .distributed import DistributedContext

LOGGER = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        callbacks: Iterable[Callback] | None = None,
        *,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
        distributed_context: "DistributedContext" | None = None,
    ) -> None:
        self.config = config
        self.state = TrainerState()
        self.callbacks = CallbackRegistry(callbacks)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        seed_everything(config.training.distributed.seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mixed_precision = config.training.distributed.mixed_precision and self.device.type == "cuda"
        self.distributed_context = distributed_context
        if self.distributed_context is not None:
            setattr(self.state, "rank", self.distributed_context.rank)
        LOGGER.info(
            "trainer_init",
            extra={
                "extra_context": {
                    "device": str(self.device),
                    "mixed_precision": self.mixed_precision,
                    "epochs": config.training.epochs,
                }
            },
        )

    def fit(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> TrainerState:
        model.to(self.device)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.callbacks.on_init(self.state, model)
        self.callbacks.dispatch("on_train_start", self.state, model)

        accumulation_steps = max(1, self.config.training.distributed.gradient_accumulation_steps)
        global_step = 0
        for epoch in range(self.config.training.epochs):
            self.state.epoch = epoch
            self._set_epoch(train_loader, epoch)
            self.callbacks.dispatch("on_epoch_start", self.state, model)
            model.train()
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    outputs = model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler:
                        scheduler.step()
                    global_step += 1
                    self.state.global_step = global_step
                    metrics = {"loss": float(loss.detach().cpu())}
                    accuracy = self._compute_accuracy(outputs, targets)
                    if accuracy is not None:
                        metrics["accuracy"] = accuracy
                    self.callbacks.dispatch("on_batch_end", self.state, model, metrics)

                if self.config.training.max_steps and global_step >= self.config.training.max_steps:
                    break

            if val_loader:
                val_metrics = self.evaluate(model, val_loader)
                monitor_value = val_metrics.get(self.config.training.checkpoint.monitor)
                if monitor_value is not None:
                    self.state.best_metric = monitor_value
                self.callbacks.dispatch("on_validation_end", self.state, model, val_metrics)
            if self.state.should_stop:
                break

        self.callbacks.dispatch("on_train_end", self.state, model)
        return self.state

    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        model_to_eval = self._unwrap_model(model)
        model_to_eval.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model_to_eval(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += float(loss.detach().cpu()) * len(inputs)
                pred = outputs.argmax(dim=1)
                total_correct += int((pred == targets).sum().item())
                total_samples += len(inputs)
        if (
            self.distributed_context is not None
            and self.distributed_context.is_distributed
            and dist.is_available()
            and dist.is_initialized()
        ):
            stats = torch.tensor(
                [total_loss, float(total_correct), float(total_samples)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats)
            total_loss = float(stats[0].item())
            total_correct = int(stats[1].item())
            total_samples = int(stats[2].item())
        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        name = self.config.training.optimizer.name.lower()
        params = [p for p in model.parameters() if p.requires_grad]
        if name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.training.optimizer.lr,
                momentum=self.config.training.optimizer.args.get("momentum", 0.9),
                weight_decay=self.config.training.optimizer.weight_decay,
            )
        if name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.training.optimizer.lr,
                weight_decay=self.config.training.optimizer.weight_decay,
                eps=self.config.training.optimizer.eps,
            )
        return torch.optim.Adam(
            params,
            lr=self.config.training.optimizer.lr,
            weight_decay=self.config.training.optimizer.weight_decay,
            betas=tuple(self.config.training.optimizer.betas or (0.9, 0.999)),
            eps=self.config.training.optimizer.eps,
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        name = self.config.training.scheduler.name
        if not name:
            return None
        args = self.config.training.scheduler.args
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(args.get("t_max", self.config.training.epochs)),
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(args.get("step_size", 10)),
                gamma=float(args.get("gamma", 0.1)),
            )
        raise ValueError(f"Unknown scheduler '{name}'")

    def _compute_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> Optional[float]:
        if outputs.ndim < 2:
            return None
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct / max(1, targets.size(0))

    def _set_epoch(self, loader: DataLoader, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model
