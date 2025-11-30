"""Configuration loading and path resolution for MS/MS pipeline."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""

    normalize_intensities: bool
    sort_mz: bool
    min_peaks: int
    max_precursor_mz: float
    filter_nonfinite: bool
    max_peaks: int = 4096
    max_input_peaks: int = 1_000_000


@dataclass
class QualityConfig:
    """Quality ranking configuration."""

    enable_ranking: bool
    ranker_model: str


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str
    log_file: Optional[Path] = None


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    dataset_name: str
    canonical_hdf5: List[str]
    output_root: Path
    max_shard_size_bytes: int
    schema_version: int
    max_spectra: Optional[int]
    preprocessing: PreprocessingConfig
    quality: QualityConfig
    logging: LoggingConfig
    random_seed: int
    config_path: Optional[Path] = None
    processing: "ProcessingConfig" = field(default_factory=lambda: ProcessingConfig())


@dataclass
class ProcessingConfig:
    """Execution-time processing overrides."""

    use_rust_batch: bool = False
    batch_size_override: Optional[int] = None
    num_workers_override: Optional[int] = None
    max_peaks: Optional[int] = None
    max_input_peaks: int = 1_000_000


def resolve_env_vars(text: str) -> str:
    """Resolve environment variable placeholders in string."""
    data_root = os.getenv("NEXA_DATA_ROOT", "data")
    scratch = os.getenv("NEXA_SCRATCH", "/tmp")

    text = text.replace("{NEXA_DATA_ROOT}", data_root)
    text = text.replace("{NEXA_SCRATCH}", scratch)
    return text


def resolve_path(path_str: str, config_dir: Optional[Path] = None) -> Path:
    """Resolve a path string, handling env vars and relative paths."""
    resolved = resolve_env_vars(path_str)

    if os.path.isabs(resolved):
        return Path(resolved)

    # If relative path, resolve relative to current working directory (project root)
    # Config files are in projects/msms_pipeline/configs/, but paths should be relative to project root
    project_root = Path.cwd()

    return (project_root / resolved).resolve()


def load_profile(profile_name: str) -> Optional[dict]:
    """Load a profile configuration.
    
    Args:
        profile_name: Name of profile to load
        
    Returns:
        Profile dictionary or None if not found
    """
    profile_dir = Path(__file__).parent / "profiles"
    profile_path = profile_dir / f"{profile_name}.yaml"
    
    if not profile_path.exists():
        return None
    
    with open(profile_path, "r") as f:
        return yaml.safe_load(f)


def load_config(path: Path, profile_name: Optional[str] = None) -> PipelineConfig:
    """Load and parse pipeline configuration from YAML file, optionally applying profile.

    Args:
        path: Path to config file
        profile_name: Optional profile name to apply
        
    Returns:
        PipelineConfig instance
    """
    config_dir = path.parent

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Apply profile if specified
    profile = None
    if profile_name:
        profile = load_profile(profile_name)
        if profile:
            # Profile values override config values
            if "shard_size_bytes" in profile:
                raw["max_shard_size_bytes"] = profile["shard_size_bytes"]
            if "logging_level" in profile:
                raw.setdefault("logging", {})["level"] = profile["logging_level"]

    canonical_hdf5 = [resolve_path(p, config_dir).as_posix() for p in raw["canonical_hdf5"]]
    output_root = resolve_path(raw["output_root"], config_dir)

    preprocessing_raw = raw["preprocessing"]
    preprocessing = PreprocessingConfig(
        normalize_intensities=preprocessing_raw["normalize_intensities"],
        sort_mz=preprocessing_raw["sort_mz"],
        min_peaks=preprocessing_raw["min_peaks"],
        max_precursor_mz=preprocessing_raw["max_precursor_mz"],
        filter_nonfinite=preprocessing_raw["filter_nonfinite"],
        max_peaks=preprocessing_raw.get("max_peaks", 4096),
        max_input_peaks=preprocessing_raw.get("max_input_peaks", 1_000_000),
    )

    quality = QualityConfig(
        enable_ranking=raw["quality"]["enable_ranking"],
        ranker_model=raw["quality"]["ranker_model"],
    )

    log_file = None
    if raw["logging"].get("log_file"):
        log_file = resolve_path(raw["logging"]["log_file"], config_dir)

    logging = LoggingConfig(
        level=raw["logging"]["level"],
        log_file=log_file,
    )

    processing_raw = raw.get("processing", {})
    processing = ProcessingConfig(
        use_rust_batch=processing_raw.get("use_rust_batch", False),
        batch_size_override=processing_raw.get("batch_size"),
        num_workers_override=processing_raw.get("num_workers"),
        max_peaks=processing_raw.get("max_peaks"),
        max_input_peaks=processing_raw.get(
            "max_input_peaks", preprocessing.max_input_peaks
        ),
    )

    # Apply profile shard size if specified
    max_shard_size = raw["max_shard_size_bytes"]
    if profile and "shard_size_bytes" in profile:
        max_shard_size = profile["shard_size_bytes"]

    return PipelineConfig(
        dataset_name=raw["dataset_name"],
        canonical_hdf5=canonical_hdf5,
        output_root=output_root,
        max_shard_size_bytes=max_shard_size,
        schema_version=raw["schema_version"],
        max_spectra=raw.get("max_spectra"),
        preprocessing=preprocessing,
        quality=quality,
        logging=logging,
        random_seed=raw["random_seed"],
        config_path=path,
        processing=processing,
    )

