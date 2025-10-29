import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..config.schema import EvaluationConfig
from ..utils.logging import get_logger
from .metrics import compute_metrics

LOGGER = get_logger(__name__)


class Evaluator:
    def __init__(self, config: EvaluationConfig, device: Optional[str] = None) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        model.eval()
        outputs_list = []
        targets_list = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                outputs_list.append(outputs.detach().cpu())
                targets_list.append(targets.detach().cpu())
        outputs = torch.cat(outputs_list)
        targets = torch.cat(targets_list)
        metrics = compute_metrics(self.config.metrics, outputs, targets)
        LOGGER.info("evaluation_complete", extra={"extra_context": metrics})
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_predictions:
            self._save_predictions(outputs, targets, output_dir)
        if self.config.generate_confusion_matrix:
            self._save_confusion_matrix(outputs, targets, output_dir)
        if self.config.generate_calibration_plot:
            self._save_calibration_plot(outputs, targets, output_dir)
        return metrics

    def _save_predictions(self, outputs: torch.Tensor, targets: torch.Tensor, output_dir: Path) -> Path:
        probabilities = torch.softmax(outputs, dim=1).tolist()
        predictions = [
            {"target": int(target), "probs": probs}
            for target, probs in zip(targets.tolist(), probabilities)
        ]
        path = output_dir / "predictions.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(predictions, handle, indent=2)
        return path

    def _save_confusion_matrix(self, outputs: torch.Tensor, targets: torch.Tensor, output_dir: Path) -> Optional[Path]:
        try:
            from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            LOGGER.warning("Unable to generate confusion matrix plot; missing dependencies")
            return None

        preds = outputs.argmax(dim=1).numpy()
        y_true = targets.numpy()
        matrix = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(matrix)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix")
        path = output_dir / "confusion_matrix.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _save_calibration_plot(self, outputs: torch.Tensor, targets: torch.Tensor, output_dir: Path) -> Optional[Path]:
        try:
            from sklearn.calibration import calibration_curve
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            LOGGER.warning("Unable to generate calibration plot; missing dependencies")
            return None

        probabilities = torch.softmax(outputs, dim=1).numpy()
        y_true = targets.numpy()
        if probabilities.shape[1] > 2:
            predicted = probabilities.max(axis=1)
        else:
            predicted = probabilities[:, 1]
        prob_true, prob_pred = calibration_curve(y_true, predicted, n_bins=self.config.calibration_bins)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(prob_pred, prob_true, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        path = output_dir / "calibration_curve.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
