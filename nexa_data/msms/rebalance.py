"""Shard rebalancing utility."""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .metrics import PipelineMetrics
from .shard_writer import ShardWriter

logger = logging.getLogger(__name__)


def rebalance_shards(
    input_dir: Path,
    output_dir: Path,
    target_shard_size: int = 2_000_000_000,  # 2GB
    dataset_name: str = "rebalanced_dataset",
    delete_originals: bool = False,
) -> None:
    """Rebalance shards into larger files.

    Args:
        input_dir: Directory containing existing shards
        output_dir: Directory to write rebalanced shards (can be same as input)
        target_shard_size: Target size for new shards in bytes
        dataset_name: Name for the dataset in new shards
        delete_originals: Whether to delete original shards after successful rebalance
    """
    input_files = sorted(list(input_dir.rglob("*.parquet")))
    if not input_files:
        logger.warning(f"No parquet files found in {input_dir}")
        return

    logger.info(f"Rebalancing {len(input_files)} shards into {target_shard_size / 1e9:.1f}GB chunks")
    
    # Create temporary output directory if writing to same location
    temp_output = output_dir / "rebalanced_temp"
    temp_output.mkdir(parents=True, exist_ok=True)

    writer = ShardWriter(
        output_dir=temp_output.parent,  # ShardWriter appends split name, so we pass parent
        max_size=target_shard_size,
        dataset_name=dataset_name,
        schema_version=1,
        split=temp_output.name, # Use temp name as split
        metrics=None,
        run_id="rebalance",
    )
    
    # ShardWriter creates a subdirectory for the split, so we need to adjust
    # We want to write to temp_output directly effectively.
    # Let's just use ShardWriter normally and move files later.
    
    try:
        for file_path in tqdm(input_files, desc="Rebalancing shards"):
            try:
                # Iterate in batches to avoid OOM
                parquet_file = pq.ParquetFile(file_path)
                for record_batch in parquet_file.iter_batches(batch_size=10000):
                    batch = record_batch.to_pylist()
                    writer.add_batch(batch)
                
            except Exception as e:
                logger.error(f"Failed to read shard {file_path}: {e}")
                continue
        
        writer.close()
        
        # If successful, move files
        logger.info("Moving rebalanced shards to final destination...")
        
        # Move from temp_output (split name) to output_dir
        # ShardWriter wrote to: temp_output.parent / temp_output.name
        source_dir = temp_output.parent / temp_output.name
        
        final_shards = list(source_dir.glob("*.parquet"))
        for shard in final_shards:
            shutil.move(str(shard), output_dir / shard.name)
            # Move manifest if exists
            manifest = shard.with_suffix(".manifest.json")
            if manifest.exists():
                shutil.move(str(manifest), output_dir / manifest.name)
        
        # Clean up temp dir
        if source_dir.exists():
            shutil.rmtree(source_dir)
            
        if delete_originals:
            logger.info("Deleting original shards...")
            for f in input_files:
                f.unlink()
                manifest = f.with_suffix(".manifest.json")
                if manifest.exists():
                    manifest.unlink()
                    
        logger.info("Rebalancing complete.")

    except Exception as e:
        logger.error(f"Rebalancing failed: {e}")
        # Cleanup temp
        if 'source_dir' in locals() and source_dir.exists():
            shutil.rmtree(source_dir)
        raise

