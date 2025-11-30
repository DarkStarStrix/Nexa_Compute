"""CLI entrypoint for MS/MS pipeline."""

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Suppress RuntimeWarning about module import when run as -m
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

# Ensure unbuffered output for real-time progress bars
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from tqdm import tqdm

# Try to import questionary for arrow key navigation, fallback to Rich Prompt
try:
    import questionary
    QUESTIONARY_AVAILABLE = True
except ImportError:
    QUESTIONARY_AVAILABLE = False

from .checkpoint import CheckpointManager
from .config import LoggingConfig, load_config
from .crash_archive import CrashArchive
from .doctor import run_doctor_mode
from .hdf5_reader import HDF5SpectrumSource
from .hdf5_reader_dask import DaskHDF5SpectrumSource
from .manifest import build_dataset_manifest
from .memory_allocator import calculate_optimal_memory_allocation, get_system_memory_info
from .metrics import PipelineMetrics
from .observability import ObservabilityManager, DisplayMode, ObservabilityLevel
from .preflight import run_preflight_checks
from .processor import BatchProcessor
from .quality import QualityRanker
from .rebalance import rebalance_shards
from .shard_writer import ShardWriter
from .validate import generate_quality_report, test_determinism, validate_shards

try:
    import ctypes
    def trim_memory() -> None:
        """Release free memory back to the OS (Linux/glibc only)."""
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass  # Not on Linux or not glibc
except ImportError:
    def trim_memory() -> None:
        pass

console = Console()


def setup_logging(cfg: LoggingConfig) -> None:
    """Setup logging configuration."""
    level = getattr(logging, cfg.level.upper())
    handlers = [logging.StreamHandler()]

    if cfg.log_file:
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def cmd_build_shards(args: argparse.Namespace) -> None:
    """Build shards from HDF5 files with optimized parallel processing."""
    # Determine mode: local or cloud/production
    mode = getattr(args, 'mode', None) or "local"
    if mode not in ["local", "cloud", "production"]:
        console.print(f"[red]Error: Invalid mode '{mode}'. Use 'local' or 'cloud'/'production'[/red]")
        sys.exit(1)
    
    # Cloud/production mode uses different defaults
    is_production = mode in ["cloud", "production"]
    
    # Handle SSH connection for cloud mode
    ssh_config = getattr(args, 'ssh_config', None)
    if is_production and ssh_config:
        console.print(f"[cyan]Cloud mode: Connecting via SSH to {ssh_config['user']}@{ssh_config['host']}[/cyan]")
        # TODO: Implement SSH connection logic here
        # For now, we'll assume the pipeline runs on the remote host
    
    # Load config if provided, otherwise create from args
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            console.print(f"[yellow]Hint: Use --interactive to select a config file interactively[/yellow]")
            # Suggest some common paths
            common_configs = [
                "projects/msms_pipeline/configs/gems_test.yaml",
                "projects/msms_pipeline/configs/gems_local.yaml",
                "projects/msms_pipeline/configs/gems_production_48core.yaml",
            ]
            console.print(f"[dim]Common config files:[/dim]")
            for common in common_configs:
                if Path(common).exists():
                    console.print(f"[dim]  - {common} ✓[/dim]")
                else:
                    console.print(f"[dim]  - {common} (not found)[/dim]")
            sys.exit(1)
        cfg = load_config(config_path)
    else:
        # Create minimal config from CLI args
        from .config import PipelineConfig, PreprocessingConfig, QualityConfig, LoggingConfig
        
        input_hdf5 = getattr(args, 'input_hdf5', None)
        output_root = getattr(args, 'output_root', None)
        
        if not input_hdf5:
            console.print("[red]Error: --input-hdf5 required when --config is not provided[/red]")
            sys.exit(1)
        
        if not output_root:
            console.print("[red]Error: --output-root required when --config is not provided[/red]")
            sys.exit(1)
        
        cfg = PipelineConfig(
            dataset_name=getattr(args, 'dataset_name', None) or "cli_dataset",
            canonical_hdf5=input_hdf5,
            output_root=Path(output_root),
            max_shard_size_bytes=getattr(args, 'shard_size', None) or 1_000_000_000,
            schema_version=1,
            max_spectra=getattr(args, 'max_spectra', None),
            preprocessing=PreprocessingConfig(
                normalize_intensities=not getattr(args, 'no_normalize', False),
                sort_mz=not getattr(args, 'no_sort', False),
                min_peaks=getattr(args, 'min_peaks', None) or 1,
                max_precursor_mz=getattr(args, 'max_precursor_mz', None) or 2000.0,
                filter_nonfinite=not getattr(args, 'no_filter_nonfinite', False),
            ),
            quality=QualityConfig(
                enable_ranking=getattr(args, 'enable_quality_ranking', None),
                ranker_model=getattr(args, 'ranker_model', None) or "openai/gpt-4o-mini",
            ),
            logging=LoggingConfig(
                level=getattr(args, 'log_level', None) or "INFO",
                log_file=Path(args.log_file) if getattr(args, 'log_file', None) else None,
            ),
            random_seed=getattr(args, 'seed', None) or 42,
        )
    
    # Override config with CLI args if provided
    if getattr(args, 'max_spectra', None) is not None:
        cfg.max_spectra = args.max_spectra
    
    if getattr(args, 'output_root', None):
        cfg.output_root = Path(args.output_root)
    
    if getattr(args, 'input_hdf5', None):
        cfg.canonical_hdf5 = args.input_hdf5
    
    if getattr(args, 'shard_size', None):
        cfg.max_shard_size_bytes = args.shard_size
    
    if getattr(args, 'log_level', None):
        cfg.logging.level = args.log_level
    
    if getattr(args, 'log_file', None):
        cfg.logging.log_file = Path(args.log_file)
    
    if getattr(args, 'enable_quality_ranking', None) is not None:
        cfg.quality.enable_ranking = args.enable_quality_ranking
    
    if getattr(args, 'ranker_model', None):
        cfg.quality.ranker_model = args.ranker_model

    setup_logging(cfg.logging)

    # Get number of workers from args or environment (mode-aware defaults)
    if is_production:
        default_workers = min(os.cpu_count() or 48, 48)
        default_batch_size = 10000
    else:
        default_workers = min(os.cpu_count() or 4, 4)
        default_batch_size = 1000
    
    num_workers = getattr(args, 'num_workers', None) or int(
        os.getenv("NEXA_MSMS_NUM_WORKERS", str(default_workers))
    )
    batch_size = getattr(args, 'batch_size', None) or int(
        os.getenv("NEXA_MSMS_BATCH_SIZE", str(default_batch_size))
    )

    processing_cfg = getattr(cfg, "processing", None)
    if processing_cfg:
        if (
            processing_cfg.num_workers_override
            and getattr(args, "num_workers", None) is None
        ):
            num_workers = processing_cfg.num_workers_override
        if (
            processing_cfg.batch_size_override
            and getattr(args, "batch_size", None) is None
        ):
            batch_size = processing_cfg.batch_size_override

    # Calculate optimal memory allocation dynamically
    # Allow override via environment variable or args
    memory_limit_override = os.getenv("NEXA_MSMS_MEMORY_LIMIT", None)
    if memory_limit_override:
        memory_limit = memory_limit_override
        console.print(f"[yellow]Using memory limit override: {memory_limit} per worker[/yellow]")
    else:
        # Get system memory info for display
        total_gb, available_gb, used_gb = get_system_memory_info()
        if total_gb > 0:
            console.print(f"[cyan]System memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available, {used_gb:.1f}GB used[/cyan]")
        
        # Calculate optimal memory allocation
        # Production: more aggressive (20% system reserve, target 75% utilization)
        # Local: more conservative (25% system reserve, target 70% utilization)
        if is_production:
            memory_limit = calculate_optimal_memory_allocation(
                num_workers=num_workers,
                system_reserve_pct=20.0,  # Less reserve for production HPC systems
                target_utilization_pct=75.0,  # Higher utilization, but still below 80% threshold
                min_memory_gb=2.0,  # Higher minimum for production
                max_memory_gb=8.0,
            )
        else:
            memory_limit = calculate_optimal_memory_allocation(
                num_workers=num_workers,
                system_reserve_pct=25.0,  # More reserve for local development
                target_utilization_pct=60.0,  # Lower utilization to account for fragmentation (was 70%)
                min_memory_gb=1.0,
                max_memory_gb=4.0,  # Lower max for local machines
            )
        
        console.print(f"[green]Calculated optimal memory limit: {memory_limit} per worker[/green]")

    # Generate unique run_id for this execution (or use provided)
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Initialize observability manager (defaults if not set via interactive menu)
    # When using CLI args directly (not interactive), default to CLI mode with simple observability
    if not hasattr(args, 'display_mode') or args.display_mode is None:
        args.display_mode = 'cli'
    if not hasattr(args, 'observability') or args.observability is None:
        args.observability = 'simple'
    if not hasattr(args, 'dashboard_port') or args.dashboard_port is None:
        args.dashboard_port = 8080
    
    display_mode_str = args.display_mode
    observability_str = args.observability
    dashboard_port = args.dashboard_port
    
    display_mode = DisplayMode(display_mode_str)
    observability_level = ObservabilityLevel(observability_str) if display_mode in [DisplayMode.CLI, DisplayMode.TUI] else None
    
    obs_manager = ObservabilityManager(
        display_mode=display_mode,
        observability_level=observability_level,
        run_dir=cfg.output_root,
        run_id=run_id,
        dashboard_port=dashboard_port,
    )
    
    console.print(f"[bold]Building shards for {cfg.dataset_name}[/bold]")
    console.print(f"Input: {cfg.canonical_hdf5}")
    console.print(f"Output: {cfg.output_root}")
    console.print(f"Run ID: {run_id}")
    console.print(f"Workers: {num_workers}, Batch size: {batch_size}")
    if getattr(args, 'use_dask', True):
        console.print(f"Memory limit: {memory_limit} per worker")
    console.print(f"Display mode: {display_mode.value}")
    if observability_level:
        console.print(f"Observability level: {observability_level.value}")
    
    dashboard_url = obs_manager.get_dashboard_url()
    if dashboard_url:
        console.print(f"[cyan]Dashboard available at: {dashboard_url}[/cyan]")

    # Initialize checkpoint manager
    checkpoint_dir = cfg.output_root / "checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir, run_id) if not getattr(args, 'no_checkpoint', False) else None
    
    # Try to resume from checkpoint
    checkpoint = None
    processed_sample_ids = set()
    start_shard_index = 0
    start_processed = 0
    
    if checkpoint_manager and getattr(args, 'resume', False):
        checkpoint = checkpoint_manager.load()
        if checkpoint:
            processed_sample_ids = set(checkpoint["processed_sample_ids"])
            start_shard_index = checkpoint["shard_index"]
            start_processed = checkpoint["processed_samples"]
            console.print(f"[yellow]Resuming from checkpoint: {checkpoint['processed_samples']} samples processed[/yellow]")
    
    # Run preflight checks for safe execution (unless skipped)
    if not getattr(args, 'skip_preflight', False):
        console.print("\n[cyan]Running preflight checks...[/cyan]")
        hdf5_paths = [Path(p) for p in cfg.canonical_hdf5]
        preflight_ok, preflight_messages = run_preflight_checks(
            cfg.output_root,
            hdf5_paths,
            min_memory_gb=getattr(args, 'min_memory_gb', None) or 4.0,
            max_memory_gb=getattr(args, 'max_memory_gb', None) or 20.0,
            min_disk_gb=getattr(args, 'min_disk_gb', None) or 10.0,
        )
    else:
        preflight_ok = True
        preflight_messages = ["Preflight checks skipped"]
    
    for msg in preflight_messages:
        if "OK" in msg:
            console.print(f"  [green]✓[/green] {msg}")
        else:
            console.print(f"  [red]✗[/red] {msg}")
    
    if not preflight_ok:
        console.print("\n[red]Preflight checks failed. Aborting for safety.[/red]")
        sys.exit(1)
    
    console.print("[green]All preflight checks passed![/green]\n")
    
    metrics = PipelineMetrics()
    
    # Choose HDF5 reader based on flags and mode
    if getattr(args, 'use_dask', True):
        if is_production:
            # Production/Cloud mode: multiprocess Dask cluster for maximum throughput
            console.print("[cyan]Using Dask-optimized HDF5 reader (PRODUCTION MODE: multiprocess)[/cyan]")
            # Production settings: 48 workers (or specified), 1 thread each, processes=True
            prod_workers = num_workers or 48
            console.print(f"[bold]Production mode: {prod_workers} workers, multiprocess enabled[/bold]")
            source = DaskHDF5SpectrumSource(
                cfg.canonical_hdf5,
                max_spectra=cfg.max_spectra,
                num_workers=prod_workers,
                memory_limit=memory_limit,
                processes=True,  # Multiprocess for production
                threads_per_worker=1,
            )
        else:
            # Local mode: threads-only Dask cluster (safe for M-series Macs)
            console.print("[cyan]Using Dask-optimized HDF5 reader (LOCAL MODE: threads-only)[/cyan]")
            # Safe settings for M4 MacBook: 4 workers, 1 thread each, 3GB per worker
            # NOTE: Dask spill configuration is critical here
            import dask.config
            # Start spilling to disk at 60% memory usage to avoid the 80% pause threshold
            dask.config.set({
                "distributed.worker.memory.target": 0.60,  # Spill to disk sooner
                "distributed.worker.memory.spill": 0.70,   # Spill more aggressively
                "distributed.worker.memory.pause": 0.85,   # Pause threshold
                "distributed.worker.memory.terminate": 0.95, # Kill worker threshold
            })
            
            # If using threads-only (processes=False), the "worker" is the main process.
            # Dask's default memory management confuses the main process memory with worker memory.
            # We should relax the limit or use system memory as the limit.
            # Local mode implies processes=False
            if not is_production: 
                # Set memory limit to 0 (unlimited) or full system memory to prevent Dask from panic
                # We are managing memory ourselves with adaptive GC.
                memory_limit = 0  
            
            safe_workers = min(num_workers or 4, 4)  # Cap at 4 for M-series
            source = DaskHDF5SpectrumSource(
                cfg.canonical_hdf5,
                max_spectra=cfg.max_spectra,
                num_workers=safe_workers,
                memory_limit=memory_limit,
                processes=False,  # Threads-only for local
                threads_per_worker=1,
            )
    else:
        source = HDF5SpectrumSource(
            cfg.canonical_hdf5,
            max_spectra=cfg.max_spectra,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    writer = ShardWriter(
        output_dir=cfg.output_root,
        max_size=cfg.max_shard_size_bytes,
        dataset_name=cfg.dataset_name,
        schema_version=cfg.schema_version,
        split=args.split,
        metrics=metrics,
        run_id=run_id,
    )
    
    # Set shard index if resuming
    if checkpoint:
        writer.shard_index = start_shard_index

    use_rust_batch = False
    if processing_cfg:
        use_rust_batch = processing_cfg.use_rust_batch
    if getattr(args, "use_rust_batch", None) is not None:
        use_rust_batch = args.use_rust_batch

    max_peaks_override = processing_cfg.max_peaks if processing_cfg else None
    max_input_peaks = (
        processing_cfg.max_input_peaks if processing_cfg else cfg.preprocessing.max_input_peaks
    )

    processor = BatchProcessor(
        preprocessing=cfg.preprocessing,
        metrics=metrics,
        batch_size=batch_size,
        num_workers=num_workers,
        use_rust_batch=use_rust_batch,
        max_peaks=max_peaks_override,
        max_input_peaks=max_input_peaks,
    )

    quality_ranker = None
    if cfg.quality.enable_ranking:
        try:
            quality_ranker = QualityRanker(
                model_id=cfg.quality.ranker_model,
                dry_run=False,
            )
        except ImportError:
            console.print("[yellow]Quality ranking not available, skipping[/yellow]")

    # Estimate total if possible
    total_estimate = cfg.max_spectra
    if not total_estimate and hasattr(source, "__len__"):
        try:
            total_estimate = len(source)
        except Exception:
            pass

    if not total_estimate:
        # Try to estimate from HDF5 file size (rough estimate)
        try:
            hdf5_size = sum(os.path.getsize(p) for p in cfg.canonical_hdf5 if os.path.exists(p))
            # Rough estimate: 900 bytes per spectrum (based on GeMS_C.9 ~850B avg)
            total_estimate = int(hdf5_size / 900) if hdf5_size > 0 else None
        except:
            total_estimate = None

    # Initialize observability and start components
    with obs_manager:
        # Use tqdm for detailed progress tracking if enabled
        pbar = None
        progress_refresh_interval = 0.5
        last_progress_refresh = [time.time()]
        last_metrics_export = [time.time()]
        metrics_export_interval = 1.0  # Limit JSON writes to once per second

        def update_progress_display(batch_len: int = 0, force: bool = False) -> None:
            if not pbar:
                return
            now = time.time()
            if not force and now - last_progress_refresh[0] < progress_refresh_interval:
                return
            pbar.set_postfix_str(
                f"written={metrics.samples_written} "
                f"shards={metrics.shards_written} "
                f"errors={sum(metrics.integrity_errors.values())} "
                f"batch={batch_len}"
            )
            pbar.refresh()
            sys.stdout.flush()
            last_progress_refresh[0] = now

        if obs_manager.should_use_tqdm():
            # Ensure tqdm shows up properly with real-time updates
            pbar = tqdm(
                total=total_estimate,
                desc="Processing spectra",
                unit="spectra",
                ncols=120,
                file=sys.stdout,  # Use stdout for better visibility
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                dynamic_ncols=True,
                mininterval=0.05,
                maxinterval=0.2,
                miniters=1,
                smoothing=0.1,
            )
        elif obs_manager.should_use_minimal_output():
            # Minimal output for args mode - just log progress periodically
            console.print("[cyan]Starting processing...[/cyan]")

    batch = []
    processed_count = start_processed
    last_sample_id = None
    # Checkpoint interval (mode-aware defaults)
    if is_production:
        default_checkpoint_interval = 10000  # Less frequent for production
    else:
        default_checkpoint_interval = 5000  # More frequent for local safety
    
    checkpoint_interval = args.checkpoint_interval or default_checkpoint_interval

    try:
        for sample_id, raw_record in source.iter_spectra():
            # Skip if already processed (resume from checkpoint)
            if sample_id in processed_sample_ids:
                continue
            
            batch.append((sample_id, raw_record))
            
            # Update progress
            processed_count += 1
            last_sample_id = sample_id
            
            # Update progress bar if using tqdm
            if pbar:
                pbar.update(1)
                update_progress_display(len(batch))
            elif obs_manager.should_use_minimal_output():
                # Minimal output for args mode - log periodically
                if processed_count % 1000 == 0:
                    console.print(f"[cyan]Processed {processed_count} spectra...[/cyan]")

            if len(batch) >= batch_size:
                # Process batch in parallel with retry logic
                max_retries = 3
                cleaned_batch = None
                
                for attempt in range(max_retries):
                    try:
                        cleaned_batch = processor.process_batch(batch)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logging.error(f"Failed to process batch after {max_retries} attempts: {e}")
                            raise
                        logging.warning(f"Batch processing attempt {attempt + 1} failed: {e}, retrying...")

                # Apply quality ranking if enabled
                if quality_ranker and cleaned_batch:
                    ranked_batch = []
                    for sample_id, cleaned in cleaned_batch:
                        try:
                            ranked = quality_ranker.rank_spectrum(cleaned)
                            if ranked:
                                ranked_batch.append(ranked)
                        except Exception as e:
                            logging.warning(f"Quality ranking failed for {sample_id}: {e}")
                            ranked_batch.append(cleaned)
                    cleaned_batch = [(r["sample_id"], r) for r in ranked_batch]

                # Write batch efficiently (maintain order for determinism)
                # Each shard is validated before writing - failed shards are rejected
                if cleaned_batch:
                    records_to_write = [cleaned for _, cleaned in cleaned_batch]
                    if not args.dry_run:
                        try:
                            writer.add_batch(records_to_write)
                            # Mark samples as processed only after successful write
                            for record in records_to_write:
                                processed_sample_ids.add(record["sample_id"])
                        except ValueError as e:
                            # Shard validation failed - samples are rejected
                            logging.error(f"Shard validation failed: {e}")
                            # Don't mark samples as processed if shard was rejected
                    else:
                        # Dry-run: just track metrics without writing
                        for record in records_to_write:
                            metrics.record_sample_written()
                            processed_sample_ids.add(record["sample_id"])

                # Update postfix with current stats
                update_progress_display(len(batch), force=True)
                
                # Memory cleanup after batch to reduce fragmentation
                # Clear large objects to free memory immediately
                if 'cleaned_batch' in locals():
                    del cleaned_batch
                if 'records_to_write' in locals():
                    del records_to_write
                batch.clear()  # Clear batch list to free memory
                # Adaptive Garbage Collection
                # Only GC if we have processed enough batches AND memory pressure is high
                # This prevents "stop-the-world" pauses when memory is healthy
                batch_num = (processed_count // batch_size) if batch_size > 0 else 0
                if batch_num > 0 and batch_num % 5 == 0:
                    should_gc = True
                    if PSUTIL_AVAILABLE:
                        try:
                            # Check memory usage of current process
                            process = psutil.Process(os.getpid())
                            mem_info = process.memory_info()
                            mem_gb = mem_info.rss / (1024**3)
                            
                            # Only GC if we are using > 2GB or > 70% of system RAM
                            # This threshold can be tuned
                            sys_mem = psutil.virtual_memory()
                            if mem_gb < 2.0 and sys_mem.percent < 70:
                                should_gc = False
                        except Exception:
                            pass
                    
                    if should_gc:
                        gc.collect()
                        trim_memory()  # Force return of memory to OS
                
                # Export metrics for TUI/dashboard consumption (rate limited)
                now = time.time()
                if now - last_metrics_export[0] >= metrics_export_interval:
                    metrics.export_json(cfg.output_root / "metrics.json")
                    if hasattr(metrics, 'export_timeseries_json'):
                        metrics.export_timeseries_json(cfg.output_root / "resource_timeseries.json")
                    last_metrics_export[0] = now

                # Save checkpoint periodically (every 5k samples for safety)
                if checkpoint_manager and processed_count % checkpoint_interval == 0:
                    checkpoint_manager.save(
                        processed_count,
                        last_sample_id or "",
                        writer.shard_index,
                        processed_sample_ids,
                        metrics.get_metrics_dict(),
                    )

                batch = []

        # Process remaining batch
        if batch:
            max_retries = 3
            cleaned_batch = None
            
            for attempt in range(max_retries):
                try:
                    cleaned_batch = processor.process_batch(batch)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Failed to process final batch after {max_retries} attempts: {e}")
                        raise
                    logging.warning(f"Final batch processing attempt {attempt + 1} failed: {e}, retrying...")

            if quality_ranker and cleaned_batch:
                ranked_batch = []
                for sample_id, cleaned in cleaned_batch:
                    try:
                        ranked = quality_ranker.rank_spectrum(cleaned)
                        if ranked:
                            ranked_batch.append(ranked)
                    except Exception as e:
                        logging.warning(f"Quality ranking failed for {sample_id}: {e}")
                        ranked_batch.append(cleaned)
                cleaned_batch = [(r["sample_id"], r) for r in ranked_batch]

            if cleaned_batch:
                records_to_write = [cleaned for _, cleaned in cleaned_batch]
                if not args.dry_run:
                    try:
                        writer.add_batch(records_to_write)
                        for record in records_to_write:
                            processed_sample_ids.add(record["sample_id"])
                    except ValueError as e:
                        logging.error(f"Final shard validation failed: {e}")
                else:
                    # Dry-run: just track metrics without writing
                    for record in records_to_write:
                        metrics.record_sample_written()
                        processed_sample_ids.add(record["sample_id"])

            if pbar:
                pbar.set_postfix_str(
                    f"written={metrics.samples_written}, "
                    f"shards={metrics.shards_written}, "
                    f"errors={sum(metrics.integrity_errors.values())}"
                )
                pbar.refresh()
                sys.stdout.flush()
            
            # Export metrics for TUI/dashboard consumption (rate limited)
            now = time.time()
            if now - last_metrics_export[0] >= metrics_export_interval:
                metrics.export_json(cfg.output_root / "metrics.json")
                if hasattr(metrics, 'export_timeseries_json'):
                    metrics.export_timeseries_json(cfg.output_root / "resource_timeseries.json")
                last_metrics_export[0] = now

        # Final checkpoint
        if checkpoint_manager:
            checkpoint_manager.save(
                processed_count,
                last_sample_id or "",
                writer.shard_index,
                processed_sample_ids,
                metrics.get_metrics_dict(),
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Saving checkpoint...[/yellow]")
        if checkpoint_manager:
            checkpoint_manager.save(
                processed_count,
                last_sample_id or "",
                writer.shard_index,
                processed_sample_ids,
                metrics.get_metrics_dict(),
            )
        raise
    except Exception as e:
        console.print(f"\n[red]Error occurred. Saving checkpoint...[/red]")
        if checkpoint_manager:
            checkpoint_manager.save(
                processed_count,
                last_sample_id or "",
                writer.shard_index,
                processed_sample_ids,
                metrics.get_metrics_dict(),
            )
        raise
    finally:
        if pbar:
            pbar.close()
        # Close Dask client if using Dask
        if args.use_dask and hasattr(source, 'close'):
            source.close()

    writer.close()
    
    # Clear checkpoint on successful completion
    if checkpoint_manager:
        checkpoint_manager.clear()
    
    # Build dataset manifest (skip in dry-run mode)
    if not args.dry_run:
        # Optional rebalancing step
        if getattr(args, "rebalance", False):
            console.print("\n[cyan]Rebalancing shards to 2GB targets...[/cyan]")
            rebalance_shards(
                input_dir=cfg.output_root,
                output_dir=cfg.output_root,
                target_shard_size=2_000_000_000,
                dataset_name=cfg.dataset_name,
                delete_originals=True
            )

        console.print("\n[cyan]Building dataset manifest...[/cyan]")
        build_dataset_manifest(cfg.dataset_name, cfg.output_root, cfg, run_id=run_id)
    else:
        console.print("\n[yellow]Skipping manifest build (dry-run mode)[/yellow]")

    # Always generate quality report with validation and training readiness checks
    # (skip in dry-run if no shards were written)
    if not args.dry_run or metrics.shards_written > 0:
        console.print("\n[cyan]Generating quality report with validation checks...[/cyan]")
        quality_report = generate_quality_report(cfg.output_root, metrics, cfg, run_id=run_id)
        if not args.dry_run:
            report_path = cfg.output_root / "quality_report.json"
            with open(report_path, "w") as f:
                json.dump(quality_report, f, indent=2)
            console.print(f"\n[green]Quality report saved to {report_path}[/green]")
        else:
            console.print("\n[yellow]Quality report (dry-run, not saved):[/yellow]")
            console.print(json.dumps(quality_report, indent=2))
    
    # Display training readiness status
    training_status = quality_report["stages"]["training_readiness"]["status"]
    training_info = quality_report["stages"]["training_readiness"]
    if training_status == "PASS":
        console.print(f"[green]Training readiness: PASS[/green] (checked {training_info['samples_checked']} samples)")
    else:
        console.print(f"[yellow]Training readiness: {training_status}[/yellow] (failures: {training_info['nan_batches']})")
    metrics.print_summary()

    if quality_report["overall_status"] != "PASS":
        console.print(f"[yellow]Overall status: {quality_report['overall_status']}[/yellow]")
        if quality_report["recommendations"]:
            console.print("Recommendations:")
            for rec in quality_report["recommendations"]:
                console.print(f"  - {rec}")


def cmd_validate_shards(args: argparse.Namespace) -> None:
    """Validate shards."""
    console.print(f"[bold]Validating shards in {args.output}[/bold]")

    results = validate_shards(args.output, full=args.full)

    console.print("\nValidation Results:")
    console.print(f"  Shards checked: {results['shards_checked']}")
    console.print(f"  Schema mismatches: {results['schema_mismatches']}")
    console.print(f"  Duplicates: {len(results['duplicates'])}")
    console.print(f"  Checksum failures: {results['checksum_failures']}")
    console.print(f"  Sample validation failures: {results['sample_validation_failures']}")
    console.print(f"  Training readiness failures: {results['training_readiness_failures']}")

    if results["duplicates"]:
        console.print(f"[red]ERROR: Found duplicate sample_ids[/red]")
        for dup in results["duplicates"][:10]:
            console.print(f"  - {dup}")

    if results["checksum_failures"]:
        console.print(f"[red]ERROR: Found checksum mismatches[/red]")

    if results["schema_mismatches"]:
        console.print(f"[red]ERROR: Found schema mismatches[/red]")

    if all(
        v == 0
        for k, v in results.items()
        if k not in ["shards_checked", "duplicates"]
    ) and not results["duplicates"]:
        console.print("[green]All validations passed![/green]")


def cmd_test_determinism(args: argparse.Namespace) -> None:
    """Test determinism."""
    import tempfile

    temp_dir = Path(tempfile.mkdtemp()) if not args.temp_dir else args.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Testing determinism with config {args.config}[/bold]")
    console.print(f"Temp directory: {temp_dir}")

    result = test_determinism(args.config, temp_dir)

    if result:
        console.print("[green]Determinism test PASSED[/green]")
    else:
        console.print("[red]Determinism test FAILED[/red]")


def cmd_quality_report(args: argparse.Namespace) -> None:
    """Generate quality report."""
    from .config import load_config

    console.print(f"[bold]Generating quality report for {args.shards_dir}[/bold]")

    config_path = args.shards_dir.parent / "config.yaml"
    if not config_path.exists():
        console.print("[yellow]Config file not found, using defaults[/yellow]")
        cfg = None
    else:
        cfg = load_config(config_path)

    metrics = PipelineMetrics()
    metrics.export_json(args.shards_dir / "metrics.json")

    if cfg:
        report = generate_quality_report(args.shards_dir, metrics, cfg)
    else:
        report = {"error": "Config file required for full quality report"}

    if args.output:
        output_path = args.output
    else:
        output_path = args.shards_dir / "quality_report.json"

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"[green]Quality report saved to {output_path}[/green]")


def interactive_menu() -> argparse.Namespace:
    """Interactive menu-driven CLI using Rich with arrow key navigation."""
    console.clear()
    console.print("\n[bold cyan]🧪 MS/MS Data Pipeline - Interactive Mode[/bold cyan]\n")
    
    # Step 1: Mode selection (Local or Cloud) - Arrow key navigation
    console.print("[bold]Step 1: Select Pipeline Mode[/bold]")
    if QUESTIONARY_AVAILABLE:
        mode_choice = questionary.select(
            "Pipeline Mode:",
            choices=[
                questionary.Choice("Local (MacBook/laptop)", "local"),
                questionary.Choice("Cloud (HPC/remote)", "cloud"),
                questionary.Choice("Production (48-core)", "production"),
            ],
            default="local",
        ).ask()
    else:
        mode_choice = Prompt.ask(
            "[cyan]Pipeline Mode[/cyan]",
            choices=["local", "cloud", "production"],
            default="local"
        )
    
    # Handle cloud mode SSH setup
    ssh_config = None
    if mode_choice in ["cloud", "production"]:
        console.print("\n[bold yellow]Cloud Mode Selected - SSH Configuration Required[/bold yellow]")
        if QUESTIONARY_AVAILABLE:
            use_ssh = questionary.confirm("Connect via SSH?", default=True).ask()
        else:
            use_ssh = Confirm.ask("Connect via SSH?", default=True)
        
        if use_ssh:
            ssh_config = {
                "host": Prompt.ask("[cyan]SSH Host[/cyan]", default=""),
                "user": Prompt.ask("[cyan]SSH User[/cyan]", default=""),
                "port": IntPrompt.ask("[cyan]SSH Port[/cyan]", default=22),
                "key_path": Prompt.ask("[cyan]SSH Key Path[/cyan] (press Enter to skip)", default=""),
            }
            if not ssh_config["key_path"]:
                ssh_config["key_path"] = None
            
            console.print(f"[green]✓ SSH configured: {ssh_config['user']}@{ssh_config['host']}:{ssh_config['port']}[/green]")
        else:
            console.print("[yellow]No SSH - assuming local cloud instance[/yellow]")
    
    # Step 2: Command selection - Arrow key navigation
    console.print("\n[bold]Step 2: Select Command[/bold]")
    if QUESTIONARY_AVAILABLE:
        command_choice = questionary.select(
            "Command:",
            choices=[
                questionary.Choice("Build Shards (process HDF5 → Parquet)", "build-shards"),
                questionary.Choice("Validate Shards (check data integrity)", "validate-shards"),
                questionary.Choice("Test Determinism (verify reproducibility)", "test-determinism"),
                questionary.Choice("Quality Report (generate metrics)", "quality-report"),
            ],
            default="build-shards",
        ).ask()
    else:
        command_choice = Prompt.ask(
            "[cyan]Command[/cyan]",
            choices=["build-shards", "validate-shards", "test-determinism", "quality-report"],
            default="build-shards"
        )
    
    # Create args namespace
    args = argparse.Namespace()
    args.command = command_choice
    args.mode = mode_choice
    args.ssh_config = ssh_config
    # Set defaults for observability (will be overridden in menu if build-shards)
    args.display_mode = "cli"
    args.observability = "simple"
    args.dashboard_port = 8080
    
    # Step 3: Configuration mode (Simple or Granular) - Arrow key navigation
    if command_choice == "build-shards":
        console.print("\n[bold]Step 3: Configuration Mode[/bold]")
        if QUESTIONARY_AVAILABLE:
            config_mode = questionary.select(
                "Configuration Mode:",
                choices=[
                    questionary.Choice("Simple (sensible defaults, minimal questions)", "simple"),
                    questionary.Choice("Granular (full control, all options)", "granular"),
                ],
                default="simple",
            ).ask()
        else:
            config_mode = Prompt.ask(
                "[cyan]Configuration Mode[/cyan]",
                choices=["simple", "granular"],
                default="simple"
            )
        args.config_mode = config_mode
        
        # Step 4: Display mode and observability selection
        console.print("\n[bold]Step 4: Display & Observability[/bold]")
        if QUESTIONARY_AVAILABLE:
            display_mode_choice = questionary.select(
                "Display Mode:",
                choices=[
                    questionary.Choice("Args (raw args, minimal output)", "args"),
                    questionary.Choice("CLI (CLI mode with observability)", "cli"),
                    questionary.Choice("TUI (Golang TUI with observability)", "tui"),
                ],
                default="cli",
            ).ask()
        else:
            display_mode_choice = Prompt.ask(
                "[cyan]Display Mode[/cyan]",
                choices=["args", "cli", "tui"],
                default="cli"
            )
        args.display_mode = display_mode_choice
        
        # Observability level (only for cli/tui modes)
        if display_mode_choice in ["cli", "tui"]:
            if QUESTIONARY_AVAILABLE:
                observability_choice = questionary.select(
                    "Observability Level:",
                    choices=[
                        questionary.Choice("Simple (tqdm progress bar + logs)", "simple"),
                        questionary.Choice("Granular (FastAPI dashboard + tqdm)", "granular"),
                    ],
                    default="simple",
                ).ask()
            else:
                observability_choice = Prompt.ask(
                    "[cyan]Observability Level[/cyan]",
                    choices=["simple", "granular"],
                    default="simple"
                )
            args.observability = observability_choice
            
            if observability_choice == "granular":
                dashboard_port = IntPrompt.ask(
                    "[cyan]Dashboard port[/cyan]",
                    default=8080
                )
                args.dashboard_port = dashboard_port
        else:
            args.observability = None
            args.dashboard_port = 8080
        
        if config_mode == "simple":
            args = interactive_build_shards_simple(args)
        else:
            args = interactive_build_shards_granular(args)
            
        # Ask for rebalancing in interactive mode (both simple and granular)
        if not args.dry_run:
            if QUESTIONARY_AVAILABLE:
                args.rebalance = questionary.confirm("Rebalance shards to 2GB chunks after processing?", default=True).ask()
            else:
                args.rebalance = Confirm.ask("Rebalance shards to 2GB chunks after processing?", default=True)
            
    elif command_choice == "validate-shards":
        args = interactive_validate_shards(args)
    elif command_choice == "test-determinism":
        args = interactive_test_determinism(args)
    elif command_choice == "quality-report":
        args = interactive_quality_report(args)
    
    return args


def interactive_build_shards_simple(args: argparse.Namespace) -> argparse.Namespace:
    """Simple interactive configuration with sensible defaults."""
    console.print("\n[bold]📦 Simple Configuration Mode[/bold]")
    console.print("[dim]Using sensible defaults - minimal configuration required[/dim]\n")
    
    is_production = args.mode in ["cloud", "production"]
    
    # Config file
    if QUESTIONARY_AVAILABLE:
        use_config = questionary.confirm("Use existing config file?", default=True).ask()
    else:
        use_config = Confirm.ask("Use existing config file?", default=True)
    
    if use_config:
        if is_production:
            default_config = "projects/msms_pipeline/configs/gems_production_48core.yaml"
        else:
            default_config = "projects/msms_pipeline/configs/gems_local.yaml"
        
        config_path = Prompt.ask(
            "[cyan]Config file path[/cyan]",
            default=default_config
        )
        args.config = Path(config_path)
        
        # Only ask for key overrides
        if QUESTIONARY_AVAILABLE:
            override = questionary.confirm("\nOverride any settings?", default=False).ask()
        else:
            override = Confirm.ask("\nOverride any settings?", default=False)
        if override:
            max_spectra = Prompt.ask(
                "[cyan]Max spectra[/cyan] (press Enter for all)",
                default=""
            )
            args.max_spectra = int(max_spectra) if max_spectra else None
            
            if is_production:
                args.use_dask = Confirm.ask("Use Dask (multiprocess)?", default=True)
                if args.use_dask:
                    args.num_workers = IntPrompt.ask(
                        "[cyan]Number of workers[/cyan]",
                        default=48
                    )
            else:
                args.use_dask = Confirm.ask("Use Dask (threads-only)?", default=True)
                if args.use_dask:
                    args.num_workers = IntPrompt.ask(
                        "[cyan]Number of workers[/cyan]",
                        default=4
                    )
        else:
            args.max_spectra = None
            args.use_dask = True
            args.num_workers = None
    else:
        # Minimal CLI-only setup
        args.config = None
        args.input_hdf5 = [Prompt.ask(
            "[cyan]Input HDF5 file[/cyan]",
            default="data/raw/GeMS_C.9.hdf5"
        )]
        args.output_root = Prompt.ask(
            "[cyan]Output directory[/cyan]",
            default="data/shards/my_dataset"
        )
        args.dataset_name = Prompt.ask(
            "[cyan]Dataset name[/cyan]",
            default="simple_dataset"
        )
        args.max_spectra = None
        args.use_dask = True
        
        if is_production:
            args.num_workers = 48
            args.shard_size = 2_000_000_000
        else:
            args.num_workers = 4
            args.shard_size = 1_000_000_000
    
    # Set defaults for simple mode
    args.split = "train"
    args.batch_size = None  # Use config defaults
    args.resume = False
    args.no_checkpoint = False
    args.checkpoint_interval = None
    args.run_id = None
    args.skip_preflight = False
    args.dry_run = False
    
    # Preprocessing defaults
    args.no_normalize = False
    args.no_sort = False
    args.no_filter_nonfinite = False
    args.min_peaks = None
    args.max_precursor_mz = None
    
    # Quality defaults
    args.enable_quality_ranking = None
    args.disable_quality_ranking = False
    args.ranker_model = None
    
    # Logging defaults
    args.log_level = None
    args.log_file = None
    args.seed = None
    
    # Preflight defaults
    args.min_memory_gb = None
    args.max_memory_gb = None
    args.min_disk_gb = None
    
    # Summary
    console.print("\n[bold green]✓ Simple Configuration Complete![/bold green]\n")
    summary_text = f"[cyan]Mode:[/cyan] {args.mode}\n"
    summary_text += f"[cyan]Config:[/cyan] {args.config or 'CLI args'}\n"
    if args.max_spectra:
        summary_text += f"[cyan]Max spectra:[/cyan] {args.max_spectra}\n"
    else:
        summary_text += f"[cyan]Max spectra:[/cyan] All\n"
    summary_text += f"[cyan]Use Dask:[/cyan] {args.use_dask}\n"
    if args.use_dask:
        summary_text += f"[cyan]Workers:[/cyan] {args.num_workers or 'Auto'}\n"
    
    summary = Panel.fit(summary_text, title="Configuration Summary", border_style="cyan")
    console.print(summary)
    
    proceed = Confirm.ask("\n[cyan]Proceed with this configuration?[/cyan]", default=True)
    if not proceed:
        console.print("[yellow]Cancelled.[/yellow]")
        sys.exit(0)
    
    return args


def interactive_build_shards_granular(args: argparse.Namespace) -> argparse.Namespace:
    """Interactive configuration for build-shards command with full control."""
    console.print("\n[bold]📦 Granular Configuration Mode[/bold]")
    console.print("[dim]Full control over all pipeline options[/dim]\n")
    
    is_production = args.mode in ["cloud", "production"]
    
    # Config file options
    if QUESTIONARY_AVAILABLE:
        config_option = questionary.select(
            "Configuration source:",
            choices=[
                questionary.Choice("Use existing config file", "existing"),
                questionary.Choice("Create custom config file", "custom"),
                questionary.Choice("CLI arguments only (no config)", "cli"),
            ],
            default="existing",
        ).ask()
    else:
        config_choice = Prompt.ask(
            "[cyan]Config option[/cyan]",
            choices=["existing", "custom", "cli"],
            default="existing"
        )
        config_option = config_choice
    
    if config_option == "custom":
        # Create custom config interactively
        args = create_custom_config_interactive(args)
        return args
    elif config_option == "existing":
        use_config = True
    else:
        use_config = False
    
    # Config file or manual entry
    if QUESTIONARY_AVAILABLE:
        use_config_confirm = questionary.confirm("Use config file?", default=use_config).ask()
    else:
        use_config_confirm = Confirm.ask("Use config file?", default=use_config)
    
    if use_config:
        config_path = Prompt.ask(
            "[cyan]Config file path[/cyan]",
            default="projects/msms_pipeline/configs/gems_full_safe.yaml"
        )
        args.config = Path(config_path)
        
        # Allow overrides
        override = Confirm.ask("\nOverride config with CLI options?", default=False)
        if not override:
            return args
    else:
        args.config = None
        args.input_hdf5 = Prompt.ask(
            "[cyan]Input HDF5 file(s)[/cyan] (space-separated for multiple)",
            default="data/raw/GeMS_C.9.hdf5"
        ).split()
        args.output_root = Prompt.ask(
            "[cyan]Output root directory[/cyan]",
            default="data/shards/my_dataset"
        )
        args.dataset_name = Prompt.ask(
            "[cyan]Dataset name[/cyan]",
            default="cli_dataset"
        )
    
    # Processing limits
    console.print("\n[bold]Processing Limits:[/bold]")
    max_spectra = Prompt.ask(
        "[cyan]Max spectra[/cyan] (press Enter for all)",
        default=""
    )
    args.max_spectra = int(max_spectra) if max_spectra else None
    
    args.split = Prompt.ask(
        "[cyan]Split[/cyan]",
        choices=["train", "val", "test"],
        default="train"
    )
    
    # Mode is already set from main menu, use it
    is_production = args.mode in ["cloud", "production"]
    
    # Performance options
    console.print("\n[bold]Performance Options:[/bold]")
    args.use_dask = Confirm.ask("Use Dask for parallel HDF5 reading?", default=True)
    
    if args.use_dask:
        if is_production:
            default_workers = str(min(os.cpu_count() or 48, 48))
            console.print("[yellow]Production mode: Using multiprocess Dask (bypasses GIL)[/yellow]")
        else:
            default_workers = str(min(os.cpu_count() or 4, 4))
            console.print("[yellow]Local mode: Using threads-only Dask (safe for MacBook)[/yellow]")
        
        num_workers = Prompt.ask(
            "[cyan]Number of workers[/cyan]",
            default=default_workers
        )
        args.num_workers = int(num_workers)
    else:
        args.num_workers = None
    
    batch_size = Prompt.ask(
        "[cyan]Batch size[/cyan]",
        default="1000"
    )
    args.batch_size = int(batch_size)
    
    shard_size = Prompt.ask(
        "[cyan]Shard size (bytes)[/cyan]",
        default="1000000000"
    )
    args.shard_size = int(shard_size) if shard_size else None
    
    # Preprocessing options
    console.print("\n[bold]Preprocessing Options:[/bold]")
    args.no_normalize = not Confirm.ask("Normalize intensities?", default=True)
    args.no_sort = not Confirm.ask("Sort m/z values?", default=True)
    args.no_filter_nonfinite = not Confirm.ask("Filter non-finite values?", default=True)
    
    min_peaks = Prompt.ask(
        "[cyan]Minimum peaks[/cyan]",
        default="1"
    )
    args.min_peaks = int(min_peaks) if min_peaks else None
    
    max_precursor = Prompt.ask(
        "[cyan]Max precursor m/z[/cyan]",
        default="2000.0"
    )
    args.max_precursor_mz = float(max_precursor) if max_precursor else None
    
    # Quality ranking
    console.print("\n[bold]Quality Options:[/bold]")
    args.enable_quality_ranking = Confirm.ask("Enable quality ranking?", default=False)
    if args.enable_quality_ranking:
        args.ranker_model = Prompt.ask(
            "[cyan]Ranker model[/cyan]",
            default="openai/gpt-4o-mini"
        )
    else:
        args.ranker_model = None
    args.disable_quality_ranking = False
    
    # Checkpointing
    console.print("\n[bold]Checkpointing:[/bold]")
    args.resume = Confirm.ask("Resume from checkpoint?", default=False)
    args.no_checkpoint = not Confirm.ask("Enable checkpointing?", default=True)
    
    if not args.no_checkpoint:
        checkpoint_interval = Prompt.ask(
            "[cyan]Checkpoint interval (samples)[/cyan]",
            default="5000"
        )
        args.checkpoint_interval = int(checkpoint_interval) if checkpoint_interval else None
    else:
        args.checkpoint_interval = None
    
    args.run_id = Prompt.ask(
        "[cyan]Run ID[/cyan] (press Enter for auto-generated)",
        default=""
    )
    args.run_id = args.run_id if args.run_id else None
    
    # Preflight checks
    console.print("\n[bold]Preflight Checks:[/bold]")
    args.skip_preflight = not Confirm.ask("Run preflight checks?", default=True)
    
    if not args.skip_preflight:
        args.min_memory_gb = float(Prompt.ask(
            "[cyan]Min memory (GB)[/cyan]",
            default="4.0"
        ))
        args.max_memory_gb = float(Prompt.ask(
            "[cyan]Max memory (GB)[/cyan]",
            default="20.0"
        ))
        args.min_disk_gb = float(Prompt.ask(
            "[cyan]Min disk space (GB)[/cyan]",
            default="10.0"
        ))
    else:
        args.min_memory_gb = None
        args.max_memory_gb = None
        args.min_disk_gb = None
    
    # Logging
    console.print("\n[bold]Logging:[/bold]")
    args.log_level = Prompt.ask(
        "[cyan]Log level[/cyan]",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO"
    )
    
    log_file = Prompt.ask(
        "[cyan]Log file[/cyan] (press Enter to skip)",
        default=""
    )
    args.log_file = log_file if log_file else None
    
    # Other options
    args.seed = int(Prompt.ask(
        "[cyan]Random seed[/cyan]",
        default="42"
    ))
    
    if QUESTIONARY_AVAILABLE:
        args.dry_run = questionary.confirm("Dry run (simulate without writing)?", default=False).ask()
    else:
        args.dry_run = Confirm.ask("Dry run (simulate without writing)?", default=False)
    args.use_rust = False  # Not commonly used
    
    # Summary
    console.print("\n[bold green]✓ Configuration complete![/bold green]\n")
    summary = Panel.fit(
        f"[cyan]Command:[/cyan] {args.command}\n"
        f"[cyan]Config:[/cyan] {args.config or 'CLI args only'}\n"
        f"[cyan]Max spectra:[/cyan] {args.max_spectra or 'All'}\n"
        f"[cyan]Use Dask:[/cyan] {args.use_dask}\n"
        f"[cyan]Workers:[/cyan] {args.num_workers or 'Auto'}\n"
        f"[cyan]Dry run:[/cyan] {args.dry_run}",
        title="Configuration Summary",
        border_style="cyan"
    )
    console.print(summary)
    
    proceed = Confirm.ask("\n[cyan]Proceed with this configuration?[/cyan]", default=True)
    if not proceed:
        console.print("[yellow]Cancelled.[/yellow]")
        sys.exit(0)
    
    return args


def create_custom_config_interactive(args: argparse.Namespace) -> argparse.Namespace:
    """Interactively create a custom configuration file."""
    console.print("\n[bold]📝 Create Custom Configuration File[/bold]\n")
    
    is_production = args.mode in ["cloud", "production"]
    
    # Get config file path
    if QUESTIONARY_AVAILABLE:
        config_path = questionary.text(
            "Config file path:",
            default=f"projects/msms_pipeline/configs/custom_{args.mode}.yaml"
        ).ask()
    else:
        config_path = Prompt.ask(
            "[cyan]Config file path[/cyan]",
            default=f"projects/msms_pipeline/configs/custom_{args.mode}.yaml"
        )
    
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Dataset info
    console.print("\n[bold]Dataset Information:[/bold]")
    dataset_name = Prompt.ask("[cyan]Dataset name[/cyan]", default=f"custom_{args.mode}")
    
    # Input files
    console.print("\n[bold]Input Files:[/bold]")
    input_files = []
    while True:
        hdf5_file = Prompt.ask(
            "[cyan]HDF5 file path[/cyan] (press Enter to finish)",
            default=""
        )
        if not hdf5_file:
            break
        input_files.append(hdf5_file)
    
    if not input_files:
        input_files = ["data/raw/GeMS_C.9.hdf5"]
    
    # Output
    output_root = Prompt.ask(
        "[cyan]Output root directory[/cyan]",
        default=f"data/shards/{dataset_name}"
    )
    
    # Shard size
    if QUESTIONARY_AVAILABLE:
        shard_size_gb = questionary.select(
            "Shard size:",
            choices=[
                questionary.Choice("1 GB (local/safe)", 1),
                questionary.Choice("2 GB (production)", 2),
                questionary.Choice("Custom size", "custom"),
            ],
            default=2 if is_production else 1,
        ).ask()
    else:
        shard_size_choice = Prompt.ask(
            "[cyan]Shard size[/cyan]",
            choices=["1", "2", "custom"],
            default="2" if is_production else "1"
        )
        shard_size_gb = int(shard_size_choice) if shard_size_choice != "custom" else "custom"
    
    if shard_size_gb == "custom":
        shard_size_bytes = int(Prompt.ask("[cyan]Shard size (bytes)[/cyan]", default="2000000000"))
    else:
        shard_size_bytes = shard_size_gb * 1_000_000_000
    
    # Processing limits
    console.print("\n[bold]Processing Limits:[/bold]")
    max_spectra_input = Prompt.ask(
        "[cyan]Max spectra[/cyan] (press Enter for all)",
        default=""
    )
    max_spectra = int(max_spectra_input) if max_spectra_input else None
    
    # Preprocessing
    console.print("\n[bold]Preprocessing Options:[/bold]")
    normalize = Confirm.ask("Normalize intensities?", default=True)
    sort_mz = Confirm.ask("Sort m/z values?", default=True)
    min_peaks = int(Prompt.ask("[cyan]Minimum peaks[/cyan]", default="1"))
    max_precursor_mz = float(Prompt.ask("[cyan]Max precursor m/z[/cyan]", default="2000.0"))
    filter_nonfinite = Confirm.ask("Filter non-finite values?", default=True)
    
    # Performance
    console.print("\n[bold]Performance Options:[/bold]")
    use_dask = Confirm.ask("Use Dask?", default=True)
    if use_dask:
        if is_production:
            num_workers = int(Prompt.ask("[cyan]Number of workers[/cyan]", default="48"))
        else:
            num_workers = int(Prompt.ask("[cyan]Number of workers[/cyan]", default="4"))
    else:
        num_workers = None
    
    batch_size = int(Prompt.ask("[cyan]Batch size[/cyan]", default="10000" if is_production else "1000"))
    
    # Quality
    console.print("\n[bold]Quality Options:[/bold]")
    enable_ranking = Confirm.ask("Enable quality ranking?", default=False)
    ranker_model = Prompt.ask(
        "[cyan]Ranker model[/cyan]",
        default="openai/gpt-4o-mini"
    ) if enable_ranking else "openai/gpt-4o-mini"
    
    # Logging
    console.print("\n[bold]Logging:[/bold]")
    log_level = Prompt.ask(
        "[cyan]Log level[/cyan]",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="DEBUG" if not is_production else "INFO"
    )
    log_file = Prompt.ask(
        "[cyan]Log file path[/cyan] (press Enter for default)",
        default=""
    )
    if not log_file:
        log_file = f"logs/msms/{dataset_name}_pipeline.log"
    
    # Build YAML config
    import yaml
    
    config_data = {
        "dataset_name": dataset_name,
        "canonical_hdf5": input_files,
        "output_root": output_root,
        "max_shard_size_bytes": shard_size_bytes,
        "schema_version": 1,
        "max_spectra": max_spectra,
        "preprocessing": {
            "normalize_intensities": normalize,
            "sort_mz": sort_mz,
            "min_peaks": min_peaks,
            "max_precursor_mz": max_precursor_mz,
            "filter_nonfinite": filter_nonfinite,
        },
        "quality": {
            "enable_ranking": enable_ranking,
            "ranker_model": ranker_model,
        },
        "logging": {
            "level": log_level,
            "log_file": log_file,
        },
        "random_seed": 42,
    }
    
    # Add mode-specific settings
    if is_production:
        config_data["production"] = {
            "num_workers": num_workers or 48,
            "processes": True,
            "threads_per_worker": 1,
            "memory_limit_per_worker": "3GB",
            "batch_size": batch_size,
            "parallel_writers": 2,
        }
    else:
        config_data["local"] = {
            "num_workers": num_workers or 4,
            "processes": False,
            "threads_per_worker": 1,
            "memory_limit_per_worker": "3GB",
            "batch_size": batch_size,
        }
    
    # Write config file
    with open(config_file, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    
    console.print(f"\n[green]✓ Custom config created: {config_file}[/green]")
    
    # Use the created config
    args.config = config_file
    args.use_dask = use_dask
    args.num_workers = num_workers
    args.batch_size = batch_size
    args.max_spectra = max_spectra
    
    return args


def interactive_validate_shards(args: argparse.Namespace) -> argparse.Namespace:
    """Interactive configuration for validate-shards command."""
    console.print("\n[bold]✅ Validate Shards Configuration[/bold]\n")
    
    args.output = Path(Prompt.ask(
        "[cyan]Shards directory[/cyan]",
        default="data/shards/gems_test"
    ))
    
    args.full = Confirm.ask("Full validation (slower but thorough)?", default=False)
    
    return args


def interactive_test_determinism(args: argparse.Namespace) -> argparse.Namespace:
    """Interactive configuration for test-determinism command."""
    console.print("\n[bold]🔄 Test Determinism Configuration[/bold]\n")
    
    args.config = Path(Prompt.ask(
        "[cyan]Config file path[/cyan]",
        default="projects/msms_pipeline/configs/gems_test.yaml"
    ))
    
    temp_dir = Prompt.ask(
        "[cyan]Temporary directory[/cyan] (press Enter for default)",
        default=""
    )
    args.temp_dir = Path(temp_dir) if temp_dir else None
    
    return args


def interactive_quality_report(args: argparse.Namespace) -> argparse.Namespace:
    """Interactive configuration for quality-report command."""
    console.print("\n[bold]📊 Quality Report Configuration[/bold]\n")
    
    args.shards_dir = Path(Prompt.ask(
        "[cyan]Shards directory[/cyan]",
        default="data/shards/gems_test"
    ))
    
    output = Prompt.ask(
        "[cyan]Output file[/cyan] (press Enter to print to console)",
        default=""
    )
    args.output = Path(output) if output else None
    
    return args


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run doctor mode validation."""
    config_path = args.config if hasattr(args, 'config') and args.config else None
    all_passed = run_doctor_mode(config_path)
    
    if not all_passed:
        console.print("\n[red]Doctor mode failed. Please fix the issues above before running the pipeline.[/red]")
        sys.exit(1)
    else:
        console.print("\n[green]All doctor checks passed! System is ready for pipeline execution.[/green]")


def cmd_crash(args: argparse.Namespace) -> None:
    """Handle crash archive commands."""
    archive = CrashArchive()
    
    if args.crash_command == "list":
        crashes = archive.list_crashes()
        if not crashes:
            console.print("[yellow]No crash dumps found.[/yellow]")
            return
        
        table = Table(title="Crash Dumps")
        table.add_column("Run ID", style="cyan")
        table.add_column("Archived At", style="green")
        table.add_column("Path", style="white")
        
        for crash in crashes:
            table.add_row(
                crash["run_id"],
                crash.get("archived_at", "Unknown"),
                crash["path"],
            )
        
        console.print(table)
    
    elif args.crash_command == "inspect":
        try:
            info = archive.inspect_crash(args.run_id)
            console.print(f"\n[bold]Crash Dump: {args.run_id}[/bold]")
            console.print(f"Path: {info['path']}")
            
            if "archived_at" in info:
                console.print(f"Archived At: {info['archived_at']}")
            
            if "manifest" in info:
                console.print(f"\n[bold]Manifest:[/bold]")
                console.print(f"  Status: {info['manifest'].get('status', 'unknown')}")
                console.print(f"  Shards: {info['manifest'].get('num_shards', 0)}")
            
            if "metrics" in info:
                console.print(f"\n[bold]Metrics:[/bold]")
                console.print(f"  Samples Written: {info['metrics'].get('samples_written', 0)}")
                console.print(f"  Integrity Errors: {info['metrics'].get('integrity_error_count', 0)}")
            
            if "log_files" in info:
                console.print(f"\n[bold]Log Files:[/bold]")
                for log_file in info["log_files"]:
                    console.print(f"  {log_file}")
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    
    elif args.crash_command == "archive":
        console.print(f"[cyan]Archiving crash for run: {args.run_id}[/cyan]")
        crash_dir = archive.archive_crash(
            args.run_id,
            remote_host=getattr(args, 'remote_host', None),
            remote_user=getattr(args, 'remote_user', None),
            remote_run_dir=getattr(args, 'remote_run_dir', None),
        )
        console.print(f"[green]Crash archived to: {crash_dir}[/green]")


def cmd_monitor(args: argparse.Namespace) -> None:
    """Handle monitor commands."""
    if args.monitor_command == "serve":
        from .dashboard import serve
        port = getattr(args, 'port', 8080)
        host = getattr(args, 'host', '0.0.0.0')
        console.print(f"[cyan]Starting dashboard server on {host}:{port}[/cyan]")
        console.print(f"[green]Dashboard available at http://{host}:{port}[/green]")
        serve(host=host, port=port)


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="MS/MS Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive menu mode
  python -m nexa_data.msms.cli --interactive

  # Direct command with args
  python -m nexa_data.msms.cli build-shards --config config.yaml

  # Mix: interactive mode for specific command
  python -m nexa_data.msms.cli build-shards --interactive
        """
    )
    
    # Add --interactive flag at top level
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive menu mode"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=False)

    build_parser = subparsers.add_parser(
        "build-shards",
        help="Build shards from HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended)
  python -m nexa_data.msms.cli build-shards --config projects/msms_pipeline/configs/gems_full.yaml

  # Using CLI args only (no config file)
  python -m nexa_data.msms.cli build-shards \\
    --input-hdf5 data/raw/GeMS_C.9.hdf5 \\
    --output-root data/shards/my_dataset \\
    --dataset-name my_dataset \\
    --max-spectra 10000 \\
    --use-dask

  # Override config with CLI args
  python -m nexa_data.msms.cli build-shards \\
    --config projects/msms_pipeline/configs/gems_full.yaml \\
    --max-spectra 50000 \\
    --num-workers 8 \\
    --batch-size 2000 \\
    --shard-size 2000000000 \\
    --use-dask \\
    --enable-quality-ranking

  # Resume from checkpoint
  python -m nexa_data.msms.cli build-shards \\
    --config projects/msms_pipeline/configs/gems_full.yaml \\
    --resume

  # Dry run (simulate without writing)
  python -m nexa_data.msms.cli build-shards \\
    --config projects/msms_pipeline/configs/gems_full.yaml \\
    --dry-run \\
    --max-spectra 100
        """
    )
    
    # Config file (optional if other args provided)
    build_parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to YAML config file (optional if --input-hdf5 and --output-root provided)"
    )
    
    # Input/Output
    build_parser.add_argument(
        "--input-hdf5", type=str, nargs="+", default=None,
        help="Input HDF5 file(s) (required if --config not provided)"
    )
    build_parser.add_argument(
        "--output-root", type=str, default=None,
        help="Output root directory (required if --config not provided)"
    )
    build_parser.add_argument(
        "--dataset-name", type=str, default=None,
        help="Dataset name (default: 'cli_dataset' or from config)"
    )
    
    # Processing limits
    build_parser.add_argument(
        "--max-spectra", type=int, default=None,
        help="Maximum number of spectra to process (default: all)"
    )
    build_parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train",
        help="Data split name (default: train)"
    )
    
    # Performance tuning
    build_parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count, max 8)"
    )
    build_parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for processing (default: 1000)"
    )
    build_parser.add_argument(
        "--shard-size", type=int, default=None,
        help="Maximum shard size in bytes (default: from config or 1GB)"
    )
    
    # Processing options
    build_parser.add_argument(
        "--use-dask", action="store_true",
        help="Use Dask for parallel HDF5 reading (faster for large files)"
    )
    build_parser.add_argument(
        "--mode", choices=["local", "cloud", "production"], default="local",
        help="Pipeline mode: 'local' for MacBook (threads-only), 'cloud'/'production' for HPC (multiprocess)"
    )
    
    # SSH configuration for cloud mode
    build_parser.add_argument(
        "--ssh-host", type=str, default=None,
        help="SSH host for cloud mode (required if --mode cloud/production)"
    )
    build_parser.add_argument(
        "--ssh-user", type=str, default=None,
        help="SSH user for cloud mode"
    )
    build_parser.add_argument(
        "--ssh-port", type=int, default=22,
        help="SSH port for cloud mode (default: 22)"
    )
    build_parser.add_argument(
        "--ssh-key", type=str, default=None,
        help="SSH key path for cloud mode"
    )
    build_parser.add_argument(
        "--use-rust", action="store_true",
        help="Use Rust extension for HDF5 reading (requires compiled extension)"
    )
    build_parser.add_argument(
        "--use-rust-batch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Process spectra batches via Rust extension (use --no-use-rust-batch to force Python)",
    )
    build_parser.add_argument(
        "--no-normalize", action="store_true",
        help="Disable intensity normalization"
    )
    build_parser.add_argument(
        "--no-sort", action="store_true",
        help="Disable m/z sorting"
    )
    build_parser.add_argument(
        "--no-filter-nonfinite", action="store_true",
        help="Disable filtering of non-finite values"
    )
    build_parser.add_argument(
        "--min-peaks", type=int, default=None,
        help="Minimum number of peaks required (default: 1)"
    )
    build_parser.add_argument(
        "--max-precursor-mz", type=float, default=None,
        help="Maximum precursor m/z value (default: 2000.0)"
    )
    
    # Quality ranking
    build_parser.add_argument(
        "--enable-quality-ranking", action="store_true", default=None,
        help="Enable LLM-based quality ranking (slow, requires API key)"
    )
    build_parser.add_argument(
        "--disable-quality-ranking", action="store_true",
        help="Disable quality ranking (overrides config)"
    )
    build_parser.add_argument(
        "--ranker-model", type=str, default=None,
        help="Quality ranker model ID (default: openai/gpt-4o-mini)"
    )
    
    # Checkpointing
    build_parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available"
    )
    build_parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpointing"
    )
    build_parser.add_argument(
        "--checkpoint-interval", type=int, default=None,
        help="Checkpoint every N samples (default: 5000)"
    )
    build_parser.add_argument(
        "--run-id", type=str, default=None,
        help="Custom run ID (default: auto-generated timestamp)"
    )
    
    # Preflight checks
    build_parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip preflight checks (memory, disk space, HDF5 accessibility)"
    )
    build_parser.add_argument(
        "--min-memory-gb", type=float, default=None,
        help="Minimum required memory in GB (default: 4.0)"
    )
    build_parser.add_argument(
        "--max-memory-gb", type=float, default=None,
        help="Maximum allowed memory in GB (default: 20.0)"
    )
    build_parser.add_argument(
        "--min-disk-gb", type=float, default=None,
        help="Minimum required disk space in GB (default: 10.0)"
    )
    
    # Logging
    build_parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
        help="Logging level (default: from config)"
    )
    build_parser.add_argument(
        "--log-file", type=str, default=None,
        help="Log file path (default: from config)"
    )
    
    # Other options
    build_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: 42)"
    )
    build_parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate processing without writing shards (for testing)"
    )
    

    build_parser.add_argument(
        "--rebalance", action="store_true",
        help="Automatically rebalance shards to 2GB chunks after processing"
    )

    validate_parser = subparsers.add_parser("validate-shards", help="Validate shards")
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.add_argument("--full", action="store_true")

    determinism_parser = subparsers.add_parser("test-determinism", help="Test determinism")
    determinism_parser.add_argument("--config", type=Path, required=True)
    determinism_parser.add_argument("--temp-dir", type=Path, default=None)

    report_parser = subparsers.add_parser("quality-report", help="Generate quality report")
    report_parser.add_argument("--shards-dir", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, default=None)

    doctor_parser = subparsers.add_parser("doctor", help="Run dry-run doctor mode validation")
    doctor_parser.add_argument("--config", type=Path, default=None, help="Path to config file")

    crash_parser = subparsers.add_parser("crash", help="Crash archive management")
    crash_subparsers = crash_parser.add_subparsers(dest="crash_command", help="Crash command")
    
    crash_list_parser = crash_subparsers.add_parser("list", help="List all crash dumps")
    
    crash_inspect_parser = crash_subparsers.add_parser("inspect", help="Inspect a crash dump")
    crash_inspect_parser.add_argument("run_id", type=str, help="Run ID to inspect")
    
    crash_archive_parser = crash_subparsers.add_parser("archive", help="Archive a crash")
    crash_archive_parser.add_argument("run_id", type=str, help="Run ID to archive")
    crash_archive_parser.add_argument("--remote-host", type=str, help="Remote hostname")
    crash_archive_parser.add_argument("--remote-user", type=str, help="Remote username")
    crash_archive_parser.add_argument("--remote-run-dir", type=Path, help="Remote run directory")

    rebalance_parser = subparsers.add_parser("rebalance", help="Rebalance shards")
    rebalance_parser.add_argument("--input-dir", type=Path, required=True, help="Input directory containing shards")
    rebalance_parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: same as input)")
    rebalance_parser.add_argument("--target-size", type=int, default=2_000_000_000, help="Target shard size in bytes (default: 2GB)")
    rebalance_parser.add_argument("--delete-originals", action="store_true", help="Delete original shards after rebalancing")

    monitor_parser = subparsers.add_parser("monitor", help="Pipeline monitoring dashboard")
    monitor_subparsers = monitor_parser.add_subparsers(dest="monitor_command", help="Monitor command")
    
    monitor_serve_parser = monitor_subparsers.add_parser("serve", help="Start FastAPI dashboard server")
    monitor_serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    monitor_serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")

    args = parser.parse_args()
    
    # Handle interactive mode
    if args.interactive:
        # Always use full interactive menu (mode selection first)
        args = interactive_menu()
    
    # Execute command
    if args.command == "build-shards":
        cmd_build_shards(args)
    elif args.command == "validate-shards":
        cmd_validate_shards(args)
    elif args.command == "test-determinism":
        cmd_test_determinism(args)
    elif args.command == "quality-report":
        cmd_quality_report(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "crash":
        cmd_crash(args)
    elif args.command == "rebalance":
        from .rebalance import rebalance_shards
        output_dir = args.output_dir or args.input_dir
        rebalance_shards(
            input_dir=args.input_dir,
            output_dir=output_dir,
            target_shard_size=args.target_size,
            dataset_name="rebalanced",
            delete_originals=args.delete_originals
        )
    elif args.command == "monitor":
        cmd_monitor(args)
    elif not args.command and not args.interactive:
        # No command and not interactive - show help
        parser.print_help()
        console.print("\n[yellow]Tip: Use --interactive for menu mode[/yellow]")


def _main():
    """Entry point to avoid RuntimeWarning when run as module."""
    main()


if __name__ == "__main__":
    _main()

