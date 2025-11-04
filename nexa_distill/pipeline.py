"""High-level orchestration helpers for the Nexa Distill workflow.

This module stitches together the core distillation stages described in
`docs/Overview_of_Project/Nexa_distill.md`. It exposes a lightweight
``DistillationPipeline`` class that can be used either interactively (e.g., in
notebooks or scripts) or scripted via CLI commands. The implementation is
purely organizational – it does not execute background jobs – allowing users to
trigger each stage manually per the compute plan while reusing shared
configuration defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import yaml

from . import CONFIGS_DIR
from .collect_teacher import run_collection
from .filter_pairs import run_filtering
from .regenerate_bad import run_regeneration
from .to_sft import run_packaging
from .utils import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DistillPaths:
    """Convenience container for key artifact locations."""

    raw_dataset: Path
    collected_dataset: Path
    filtered_dataset: Path
    regen_dataset: Path
    sft_jsonl: Path
    sft_parquet: Path


class DistillationPipeline:
    """Coordinate the distillation stages using shared configuration."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or CONFIGS_DIR / "distill_config.yaml"
        self._config: Dict[str, Dict] = self._load_config()
        self._defaults = self._config.get("defaults", {})
        storage_cfg = self._config.get("storage", {})
        self.paths = DistillPaths(
            raw_dataset=self._as_path(storage_cfg.get("raw_dataset")),
            collected_dataset=self._as_path(storage_cfg.get("collected_dataset")),
            filtered_dataset=self._as_path(storage_cfg.get("filtered_dataset")),
            regen_dataset=self._as_path(storage_cfg.get("regen_dataset")),
            sft_jsonl=self._as_path(storage_cfg.get("sft_jsonl")),
            sft_parquet=self._as_path(storage_cfg.get("sft_parquet")),
        )

    @property
    def config_path(self) -> Path:
        """Return the path to the active distillation configuration."""

        return self._config_path

    @property
    def collection_config(self) -> Dict:
        """Return configuration specific to the teacher collection stage."""

        return self._config.get("collection", {})

    @property
    def chunking_config(self) -> Dict:
        """Return chunking-related configuration helpers."""

        return self._config.get("chunking", {})

    def collect_teacher(
        self,
        *,
        api_key: Optional[str] = None,
        dry_run: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        """Generate teacher outputs for the raw dataset.

        Parameters
        ----------
        api_key:
            OpenAI API key used for generation. If ``None`` the client falls
            back to the ``OPENAI_API_KEY`` environment variable.
        dry_run:
            When ``True`` no external calls are made and placeholder outputs are
            written instead. Useful for validating pipeline wiring.
        max_samples:
            Optional cap on the number of rows processed from the source
            dataset.
        """

        collection_cfg = self.collection_config
        system_prompt = collection_cfg.get("system_prompt_path")
        system_prompt_path = self._resolve_optional_path(system_prompt)

        args = SimpleNamespace(
            src=self.paths.raw_dataset,
            dst=self.paths.collected_dataset,
            config=self.config_path,
            teacher=collection_cfg.get("teacher_model", "gpt-4o-mini"),
            system_prompt=system_prompt_path,
            prompt_column=self._defaults.get("prompt_column", "prompt_text"),
            context_column=self._defaults.get("context_column", "context"),
            task_type_column=self._defaults.get("task_type_column", "task_type"),
            max_samples=max_samples,
            batch_size=collection_cfg.get("batch_size", 8),
            dry_run=dry_run,
            api_key=api_key,
        )

        LOGGER.info(
            "Starting teacher collection",
            extra={
                "src": str(args.src),
                "dst": str(args.dst),
                "teacher": args.teacher,
                "dry_run": dry_run,
            },
        )
        run_collection(args)

    def filter_teacher(self, *, report_path: Path | None = None) -> None:
        """Apply heuristic filters to the collected dataset."""

        args = SimpleNamespace(
            src=self.paths.collected_dataset,
            dst=self.paths.filtered_dataset,
            config=CONFIGS_DIR / "filters.yaml",
            report=report_path,
        )

        LOGGER.info(
            "Running filter stage",
            extra={"src": str(args.src), "dst": str(args.dst), "config": str(args.config)},
        )
        run_filtering(args)

    def regenerate(
        self,
        *,
        annotations_path: Path,
        api_key: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        """Regenerate rejected samples based on inspector annotations."""

        args = SimpleNamespace(
            dataset=self.paths.filtered_dataset,
            annotations=annotations_path,
            dst=self.paths.regen_dataset,
            teacher=self.collection_config.get("regen_teacher_model", "o3-mini"),
            system_prompt=self._resolve_optional_path(
                self.collection_config.get("regen_system_prompt_path")
            ),
            prompt_column=self._defaults.get("prompt_column", "prompt_text"),
            context_column=self._defaults.get("context_column", "context"),
            task_type_column=self._defaults.get("task_type_column", "task_type"),
            api_key=api_key,
            dry_run=dry_run,
        )

        LOGGER.info(
            "Regeneration stage",
            extra={
                "annotations": str(annotations_path),
                "dst": str(self.paths.regen_dataset),
                "dry_run": dry_run,
            },
        )
        run_regeneration(args)

    def package_sft(self) -> None:
        """Package filtered (and optional regenerated) rows into SFT format."""

        regen_path = self.paths.regen_dataset if self.paths.regen_dataset.exists() else None
        args = SimpleNamespace(
            src=self.paths.filtered_dataset,
            regen=regen_path,
            dst_jsonl=self.paths.sft_jsonl,
            dst_parquet=self.paths.sft_parquet,
            prompt_column=self._defaults.get("prompt_column", "prompt_text"),
            context_column=self._defaults.get("context_column", "context"),
            task_type_column=self._defaults.get("task_type_column", "task_type"),
            output_column=self._defaults.get("output_column", "teacher_output"),
        )

        LOGGER.info(
            "Packaging SFT dataset",
            extra={
                "src": str(args.src),
                "dst_jsonl": str(args.dst_jsonl),
                "dst_parquet": str(args.dst_parquet),
            },
        )
        run_packaging(args)

    def plan_cli_commands(self) -> List[str]:
        """Return CLI equivalents for running each stage manually."""

        return [
            f"python -m nexa_distill.collect_teacher --src {self.paths.raw_dataset} --dst {self.paths.collected_dataset}",
            f"python -m nexa_distill.filter_pairs --src {self.paths.collected_dataset} --dst {self.paths.filtered_dataset}",
            "streamlit run nexa_distill/ui_inspect.py -- --src "
            f"{self.paths.filtered_dataset}",
            f"python -m nexa_distill.regenerate_bad --dataset {self.paths.filtered_dataset} --annotations <labels.jsonl> --dst {self.paths.regen_dataset}",
            f"python -m nexa_distill.to_sft --src {self.paths.filtered_dataset} --regen {self.paths.regen_dataset} --dst-jsonl {self.paths.sft_jsonl} --dst-parquet {self.paths.sft_parquet}",
        ]

    def _load_config(self) -> Dict[str, Dict]:
        if not self._config_path.exists():
            raise FileNotFoundError(f"Distillation config missing: {self._config_path}")
        with self._config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data

    @staticmethod
    def _as_path(value: Optional[str]) -> Path:
        if value is None:
            raise ValueError("Expected path string in distillation configuration")
        return Path(value).expanduser().resolve()

    @staticmethod
    def _resolve_optional_path(value: Optional[str]) -> Optional[Path]:
        if value in (None, "", "null"):
            return None
        return Path(value).expanduser().resolve()


