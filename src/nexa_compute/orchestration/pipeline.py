"""Top-level pipeline wiring config → data → model → trainer → evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..config import TrainingConfig, load_config, save_run_config
from ..config.schema import TrainingConfig as TrainingSchema
from ..core.manifests import RunManifest, get_git_commit
from ..core.storage import generate_run_id
from ..data import DEFAULT_REGISTRY as DEFAULT_DATASET_REGISTRY
from ..data import DataPipeline, DatasetRegistry
from ..evaluation import Evaluator
from ..models import DEFAULT_MODEL_REGISTRY, ModelRegistry
from ..training import (
    Callback,
    CheckpointSaver,
    EarlyStopping,
    LoggingCallback,
    MLflowCallback,
    Trainer,
    launch_distributed,
)
from ..training.checkpoint import checkpoint_path, load_checkpoint
from ..training.distributed import DistributedContext
from ..core.logging import configure_logging, get_logger
from ..monitoring.tracing import configure_tracing, trace_span

LOGGER = get_logger(__name__)


@dataclass
class PipelineArtifacts:
    run_dir: Path
    checkpoint: Optional[Path]
    metrics: Dict[str, float]


class TrainingPipeline:
    def __init__(
        self,
        config: TrainingSchema,
        *,
        dataset_registry: DatasetRegistry | None = None,
        model_registry: ModelRegistry | None = None,
        callbacks: Optional[list[Callback]] = None,
    ) -> None:
        self.config = config
        self.dataset_registry = dataset_registry or DEFAULT_DATASET_REGISTRY
        self.model_registry = model_registry or DEFAULT_MODEL_REGISTRY
        configure_logging(
            level=config.training.logging.level,
            log_dir=config.training.logging.log_dir,
            json_logs=True,
        )
        # Configure tracing for the pipeline execution
        configure_tracing(service_name="nexa-training")

        self.callbacks = list(callbacks or [])
        self._mlflow_run_active = False
        self._mlflow_run_id: Optional[str] = None
        self._mlflow_enabled = self._mlflow_available() and bool(config.training.logging.mlflow_tracking_uri)

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        *,
        overrides: Optional[list[str]] = None,
        dataset_registry: DatasetRegistry | None = None,
        model_registry: ModelRegistry | None = None,
        callbacks: Optional[list[Callback]] = None,
    ) -> "TrainingPipeline":
        config = load_config(path, overrides=overrides)
        return cls(config, dataset_registry=dataset_registry, model_registry=model_registry, callbacks=callbacks)

    @trace_span("training_pipeline.run", attributes={"distributed": False})
    def run(
        self,
        *,
        enable_evaluation: bool = True,
        resume_from_checkpoint: bool | str | Path | None = None,
    ) -> PipelineArtifacts:
        """Execute the training pipeline.
        
        **Idempotency**: This method is idempotent when resuming from checkpoint.
        Running twice with the same checkpoint will produce identical results.
        Running without a checkpoint will create a new run each time.
        
        **Retry-safe**: Safe to retry after failure. Manifests track state and
        checkpoints enable resume from last successful point.
        """
        run_dir = self.config.output_directory()
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # V4: Run Manifest
        run_id = generate_run_id("train")
        manifest = RunManifest(
            run_id=run_id,
            monorepo_commit=get_git_commit(),
            config_snapshot=self.config.model_dump(),
            dataset_version=self.config.data.dataset_version,
            hardware_specs=self._get_hardware_specs(),
        )
        manifest_dir = Path("runs")
        manifest.save(manifest_dir)

        save_run_config(self.config, run_dir)
        LOGGER.info("run_started", extra={"extra_context": {"run_dir": str(run_dir), "run_id": run_id}})

        resume_path = self._resolve_resume_checkpoint(resume_from_checkpoint, run_dir)
        if resume_path:
            LOGGER.info("resume_checkpoint", extra={"extra_context": {"path": str(resume_path)}})
            manifest.resume_info = {"path": str(resume_path)}
            manifest.save(manifest_dir)

        try:
            if self._is_distributed():
                artifacts = self._run_distributed(
                    run_dir,
                    enable_evaluation=enable_evaluation,
                    resume_checkpoint=resume_path,
                )
            else:
                artifacts = self._run_single_process(
                    run_dir,
                    enable_evaluation=enable_evaluation,
                    resume_checkpoint=resume_path,
                )
            
            manifest.update_status("completed", artifacts.metrics)
            manifest.save(manifest_dir)
            return artifacts
            
        except Exception as e:
            manifest.update_status("failed")
            manifest.save(manifest_dir)
            raise e

    @trace_span("training_pipeline.single_process")
    def _run_single_process(
        self,
        run_dir: Path,
        *,
        enable_evaluation: bool,
        resume_checkpoint: Optional[Path],
    ) -> PipelineArtifacts:
        """Run training in single-process mode.
        
        **Idempotency**: Idempotent when resuming from checkpoint. Running with
        the same checkpoint produces identical results.
        """
        data_pipeline = DataPipeline(self.config.data, registry=self.dataset_registry)
        data_pipeline.materialize_metadata(run_dir)
        train_loader = data_pipeline.dataloader("train")
        val_loader = data_pipeline.dataloader("validation", batch_size=self.config.evaluation.batch_size)

        model = self.model_registry.build(self.config.model)
        mlflow_active = self._start_mlflow_run(run_dir)
        callbacks = self._build_callbacks(global_rank=0, mlflow_active=mlflow_active)
        trainer = Trainer(self.config, callbacks=callbacks)
        if resume_checkpoint:
            payload = self._load_checkpoint_payload(resume_checkpoint)
            if payload:
                trainer.resume_from_checkpoint(model, payload)
        trainer.fit(model, train_loader, val_loader)
        self._write_trainer_state(run_dir, trainer.state)

        checkpoint = self._latest_checkpoint(trainer.state.epoch)
        metrics: Dict[str, float] = {}
        if enable_evaluation:
            evaluator = Evaluator(self.config.evaluation)
            metrics = evaluator.evaluate(model, val_loader)
            self._write_metrics(run_dir, metrics)
            if mlflow_active:
                self._log_mlflow_metrics(metrics, step=trainer.state.epoch)
        LOGGER.info("run_completed", extra={"extra_context": metrics})
        if mlflow_active:
            self._end_mlflow_run()
        return PipelineArtifacts(run_dir=run_dir, checkpoint=checkpoint, metrics=metrics)

    @trace_span("training_pipeline.distributed")
    def _run_distributed(
        self,
        run_dir: Path,
        *,
        enable_evaluation: bool,
        resume_checkpoint: Optional[Path],
    ) -> PipelineArtifacts:
        mlflow_payload = self._mlflow_settings() if self._mlflow_enabled else None
        launch_distributed(
            self.config.training.distributed,
            self._distributed_worker,
            self.config.model_dump_json(),
            str(run_dir),
            enable_evaluation,
            mlflow_payload,
            str(resume_checkpoint) if resume_checkpoint else "",
        )
        metrics = self._load_metrics(run_dir)
        checkpoint = self._latest_checkpoint()
        LOGGER.info("run_completed", extra={"extra_context": metrics})
        return PipelineArtifacts(run_dir=run_dir, checkpoint=checkpoint, metrics=metrics)

    def _distributed_worker(
        self,
        context: DistributedContext,
        config_json: str,
        run_dir_str: str,
        enable_evaluation: bool,
        mlflow_payload: Optional[Dict[str, Any]],
        resume_checkpoint_path: str,
    ) -> None:
        config = TrainingConfig.model_validate_json(config_json)
        run_dir = Path(run_dir_str)
        data_pipeline = DataPipeline(config.data, registry=self.dataset_registry)
        if context.rank == 0:
            data_pipeline.materialize_metadata(run_dir)

        train_loader = data_pipeline.dataloader("train", distributed_context=context)
        val_loader = data_pipeline.dataloader(
            "validation",
            batch_size=config.evaluation.batch_size,
            distributed_context=context,
        )

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        model = self.model_registry.build(config.model)
        model.to(device)
        if context.is_distributed:
            if torch.cuda.is_available():
                model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
            else:
                model = nn.parallel.DistributedDataParallel(model)

        mlflow_active = False
        if mlflow_payload and context.rank == 0:
            mlflow_active = self._start_mlflow_run(run_dir, settings=mlflow_payload)

        callbacks = self._build_callbacks(global_rank=context.rank, mlflow_active=mlflow_active)
        trainer = Trainer(config, callbacks=callbacks, distributed_context=context)
        resume_checkpoint = Path(resume_checkpoint_path) if resume_checkpoint_path else None
        if resume_checkpoint:
            payload = self._load_checkpoint_payload(resume_checkpoint)
            if payload:
                trainer.resume_from_checkpoint(model, payload)
        trainer.fit(model, train_loader, val_loader)

        if context.rank == 0:
            self._write_trainer_state(run_dir, trainer.state)
            metrics: Dict[str, float] = {}
            if enable_evaluation:
                eval_pipeline = DataPipeline(config.data, registry=self.dataset_registry)
                eval_loader = eval_pipeline.dataloader("validation", batch_size=config.evaluation.batch_size)
                evaluator = Evaluator(config.evaluation)
                eval_model = Trainer._unwrap_model(model)
                eval_model.to(evaluator.device)
                metrics = evaluator.evaluate(eval_model, eval_loader)
                self._write_metrics(run_dir, metrics)
                if mlflow_active:
                    self._log_mlflow_metrics(metrics, step=trainer.state.epoch)
            if mlflow_active:
                self._end_mlflow_run()

    def _is_distributed(self) -> bool:
        cfg = self.config.training.distributed
        return cfg.world_size > 1 or cfg.num_nodes > 1

    def _build_callbacks(self, *, global_rank: int, mlflow_active: bool) -> list[Callback]:
        callbacks: list[Callback] = [
            LoggingCallback(
                log_every_n_steps=self.config.training.log_every_n_steps,
                rank_filter=0 if self._is_distributed() else None,
            ),
            CheckpointSaver(self.config.training.checkpoint, global_rank=global_rank),
            EarlyStopping(
                monitor=self.config.training.checkpoint.monitor,
                patience=self.config.training.scheduler.args.get("early_stop_patience", 5),
                mode=self.config.training.checkpoint.mode,
            ),
        ]
        if mlflow_active:
            callbacks.append(MLflowCallback())
        callbacks.extend(self.callbacks)
        return callbacks

    def _start_mlflow_run(self, run_dir: Path, settings: Optional[Dict[str, Any]] = None) -> bool:
        if not self._mlflow_enabled and not settings:
            return False
        try:
            import mlflow  # type: ignore
        except ImportError:
            LOGGER.warning("MLflow not available; skipping tracking")
            return False

        payload = settings or self._mlflow_settings()
        if payload is None:
            return False

        tracking_uri = payload.get("tracking_uri")
        experiment = payload.get("experiment")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment:
            mlflow.set_experiment(experiment)
        tags = {"run_dir": str(run_dir)}
        tags.update(self.config.experiment.tags)
        active_run = mlflow.start_run(run_name=self.config.experiment.name, tags=tags)
        flat_config = self._flatten_config(self.config.model_dump())
        mlflow.log_params(flat_config)
        self._mlflow_run_active = True
        self._mlflow_run_id = active_run.info.run_id
        return True

    def _log_mlflow_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None) -> None:
        if not self._mlflow_run_active:
            return
        try:
            import mlflow  # type: ignore
        except ImportError:
            return
        mlflow.log_metrics(metrics, step=step)

    def _end_mlflow_run(self) -> None:
        if not self._mlflow_run_active:
            return
        try:
            import mlflow  # type: ignore
        except ImportError:
            return
        mlflow.end_run()
        self._mlflow_run_active = False
        self._mlflow_run_id = None

    def _write_metrics(self, run_dir: Path, metrics: Dict[str, float]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    def _load_metrics(self, run_dir: Path) -> Dict[str, float]:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            return {}
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _latest_checkpoint(self, epoch: Optional[int] = None) -> Optional[Path]:
        candidate_epoch = epoch
        if candidate_epoch is None:
            state_path = self.config.output_directory() / "trainer_state.json"
            if state_path.exists():
                with state_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                    candidate_epoch = payload.get("epoch")
        if candidate_epoch is None:
            return None
        candidate = checkpoint_path(self.config.training.checkpoint, epoch=candidate_epoch)
        return candidate if candidate.exists() else None

    def _write_trainer_state(self, run_dir: Path, state) -> None:
        payload = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
        }
        with (run_dir / "trainer_state.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _resolve_resume_checkpoint(
        self,
        resume_flag: bool | str | Path | None,
        run_dir: Path,
    ) -> Optional[Path]:
        if not resume_flag:
            return None
        path: Optional[Path]
        if resume_flag is True:
            path = self._latest_checkpoint()
        else:
            candidate = Path(resume_flag)
            if not candidate.is_absolute():
                candidate = (run_dir / candidate).resolve()
            path = candidate
        if path and not path.exists():
            LOGGER.warning("resume_checkpoint_missing", extra={"extra_context": {"path": str(path)}})
            return None
        return path

    def _load_checkpoint_payload(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        try:
            return load_checkpoint(checkpoint_path)
        except FileNotFoundError:
            LOGGER.warning("resume_checkpoint_missing", extra={"extra_context": {"path": str(checkpoint_path)}})
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error(
                "resume_checkpoint_failed",
                extra={"extra_context": {"path": str(checkpoint_path), "error": repr(exc)}},
            )
        return None

    def _flatten_config(self, config_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        flattened: Dict[str, Any] = {}
        for key, value in config_dict.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict):
                flattened.update(self._flatten_config(value, prefix=full_key))
            elif isinstance(value, (list, tuple, set)):
                flattened[full_key] = json.dumps(list(value) if isinstance(value, set) else value)
            else:
                flattened[full_key] = value
        return flattened

    def _mlflow_settings(self) -> Optional[Dict[str, Any]]:
        if not self._mlflow_enabled:
            return None
        logging_cfg = self.config.training.logging
        return {
            "tracking_uri": logging_cfg.mlflow_tracking_uri,
            "experiment": logging_cfg.mlflow_experiment,
        }

    @staticmethod
    def _mlflow_available() -> bool:
        import importlib.util

        return importlib.util.find_spec("mlflow") is not None

    def _get_hardware_specs(self) -> Dict[str, Any]:
        """Collect hardware specifications for manifest."""
        specs: Dict[str, Any] = {}
        try:
            import torch
            if torch.cuda.is_available():
                specs["gpu_count"] = torch.cuda.device_count()
                specs["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception:
            pass
        try:
            import psutil
            specs["cpu_count"] = psutil.cpu_count()
            mem = psutil.virtual_memory()
            specs["memory_gb"] = mem.total / (1024**3)
        except Exception:
            pass
        return specs
