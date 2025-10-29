"""Metric registry and helpers."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import torch
from sklearn import metrics as sk_metrics

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def precision(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1).cpu().numpy()
    y_true = targets.cpu().numpy()
    return float(sk_metrics.precision_score(y_true, preds, average="macro", zero_division=0))


def recall(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1).cpu().numpy()
    y_true = targets.cpu().numpy()
    return float(sk_metrics.recall_score(y_true, preds, average="macro", zero_division=0))


def f1(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1).cpu().numpy()
    y_true = targets.cpu().numpy()
    return float(sk_metrics.f1_score(y_true, preds, average="macro", zero_division=0))


def auroc(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()
    y_true = targets.cpu().numpy()
    if probabilities.shape[1] == 2:
        return float(sk_metrics.roc_auc_score(y_true, probabilities[:, 1]))
    return float(sk_metrics.roc_auc_score(y_true, probabilities, multi_class="ovo"))


def rmse(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.squeeze().detach().cpu().numpy()
    y_true = targets.squeeze().detach().cpu().numpy()
    return float(sk_metrics.mean_squared_error(y_true, preds, squared=False))


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: Dict[str, MetricFn] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
            "rmse": rmse,
        }

    def register(self, name: str, fn: MetricFn) -> None:
        self._metrics[name] = fn

    def compute(self, names: Iterable[str], outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name in names:
            if name not in self._metrics:
                raise KeyError(f"Metric '{name}' not registered")
            results[name] = self._metrics[name](outputs, targets)
        return results

    def available(self) -> List[str]:
        return list(self._metrics)


DEFAULT_REGISTRY = MetricRegistry()


def compute_metrics(names: Iterable[str], outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    return DEFAULT_REGISTRY.compute(names, outputs, targets)
