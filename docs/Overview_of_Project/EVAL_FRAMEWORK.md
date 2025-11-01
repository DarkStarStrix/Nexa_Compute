# Evaluation Framework

1. **Prediction Generation** — `nexa_eval.generate.generate_predictions` builds validation dataloaders and materialises softmax probabilities for each sample.
2. **Metric Computation** — `nexa_compute.evaluation.metrics.MetricRegistry` computes accuracy, precision, recall, F1, AUROC, RMSE.
3. **Artifact Logging** — `nexa_compute.evaluation.Evaluator` persist metrics, predictions, confusion matrices, and calibration plots to `artifacts/evaluation/`.
4. **Judging** — `nexa_eval.judge.judge_metrics` compares metrics against rubric thresholds for pass/fail decisions.
5. **Reporting** — `nexa_eval.analyze.evaluate_checkpoint` writes `eval_report.json` and optionally streams metrics to MLflow.

All evaluation steps are orchestrated via `orchestrate.py evaluate` or `scripts/cli.py evaluate`.
