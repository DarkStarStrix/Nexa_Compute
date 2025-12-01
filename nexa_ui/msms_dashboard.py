"""Streamlit dashboard and CLI for inspecting MS/MS data shards."""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import altair as alt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Only import streamlit if we're running the UI
STREAMLIT_AVAILABLE = True
try:
    import streamlit as st
    st.set_page_config(
        page_title="MS/MS Dashboard",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARDS_ROOT = PROJECT_ROOT / "data" / "shards"
EVAL_RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "eval"
RESULTS_ROOT = PROJECT_ROOT / "results" / "evaluation"


def list_datasets() -> list[str]:
    """List available datasets."""
    if not SHARDS_ROOT.exists():
        return []
    return [d.name for d in SHARDS_ROOT.iterdir() if d.is_dir()]


def list_run_ids(dataset: str) -> list[str]:
    """List available run IDs for a dataset."""
    run_ids = set()
    
    for split in ["train", "val", "test", "shards"]:
        split_dir = SHARDS_ROOT / dataset / split
        if split_dir.exists():
            for manifest_path in split_dir.glob("*.manifest.json"):
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                        if "run_id" in manifest:
                            run_ids.add(manifest["run_id"])
                except:
                    pass
    
    return sorted(list(run_ids), reverse=True)  # Most recent first


def list_shards(dataset: str, split: str, run_id: Optional[str] = None) -> list[Path]:
    """List shard files for a dataset and split, optionally filtered by run_id."""
    split_dir = SHARDS_ROOT / dataset / split
    if not split_dir.exists():
        return []
    all_shards = sorted(split_dir.glob("*.parquet"))
    
    if run_id:
        # Filter by run_id in filename or manifest
        filtered = []
        for shard_path in all_shards:
            if run_id in shard_path.name:
                filtered.append(shard_path)
            else:
                # Check manifest
                manifest_path = shard_path.with_suffix(".manifest.json")
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                            if manifest.get("run_id") == run_id:
                                filtered.append(shard_path)
                    except:
                        pass
        return filtered
    
    return all_shards


def load_shard(shard_path: Path) -> pd.DataFrame:
    """Load a shard file."""
    table = pq.read_table(shard_path)
    return table.to_pandas()


def load_shard_table(shard_path: Path) -> pa.Table:
    """Load shard as Arrow table for schema inspection."""
    return pq.read_table(shard_path)


def get_shard_statistics(shard_path: Path) -> dict:
    """Get detailed statistics about a shard."""
    df = load_shard(shard_path)
    table = load_shard_table(shard_path)
    
    stats = {
        "num_samples": len(df),
        "schema": {col: str(table.schema.field(col).type) for col in table.column_names},
        "file_size_bytes": shard_path.stat().st_size,
    }
    
    # Calculate statistics for array columns
    if len(df) > 0:
        peak_counts = []
        mz_ranges = []
        intensity_ranges = []
        precursor_mzs = []
        charges = []
        
        for _, row in df.iterrows():
            mzs = np.array(row["mzs"]) if isinstance(row["mzs"], list) else row["mzs"]
            ints = np.array(row["ints"]) if isinstance(row["ints"], list) else row["ints"]
            
            peak_counts.append(len(mzs))
            if len(mzs) > 0:
                mz_ranges.append((mzs.min(), mzs.max()))
                intensity_ranges.append((ints.min(), ints.max()))
            precursor_mzs.append(row["precursor_mz"])
            charges.append(row["charge"])
        
        stats["peak_counts"] = {
            "min": int(np.min(peak_counts)) if peak_counts else 0,
            "max": int(np.max(peak_counts)) if peak_counts else 0,
            "mean": float(np.mean(peak_counts)) if peak_counts else 0.0,
            "median": float(np.median(peak_counts)) if peak_counts else 0.0,
            "all": peak_counts  # Store for distribution plots
        }
        
        if mz_ranges:
            stats["mz_ranges"] = {
                "min": float(np.min([r[0] for r in mz_ranges])),
                "max": float(np.max([r[1] for r in mz_ranges])),
            }
        
        if intensity_ranges:
            stats["intensity_ranges"] = {
                "min": float(np.min([r[0] for r in intensity_ranges])),
                "max": float(np.max([r[1] for r in intensity_ranges])),
            }
            
            # Calculate intensity quantiles for the shard
            all_ints = np.concatenate([
                np.array(row["ints"]) if isinstance(row["ints"], list) else row["ints"] 
                for _, row in df.iterrows() if len(row["ints"]) > 0
            ])
            if len(all_ints) > 0:
                stats["intensity_quantiles"] = {
                    "q05": float(np.percentile(all_ints, 5)),
                    "q25": float(np.percentile(all_ints, 25)),
                    "median": float(np.median(all_ints)),
                    "q75": float(np.percentile(all_ints, 75)),
                    "q95": float(np.percentile(all_ints, 95)),
                }
        
        stats["precursor_mz"] = {
            "min": float(np.min(precursor_mzs)) if precursor_mzs else 0.0,
            "max": float(np.max(precursor_mzs)) if precursor_mzs else 0.0,
            "mean": float(np.mean(precursor_mzs)) if precursor_mzs else 0.0,
            "all": precursor_mzs # Store for distribution plots
        }
        
        stats["charge_distribution"] = dict(Counter(charges))
        
        if "adduct" in df.columns:
            stats["adduct_distribution"] = dict(df["adduct"].value_counts())

        # Metadata completeness
        metadata_fields = ["adduct", "instrument_type", "smiles", "inchikey", "formula"]
        metadata_stats = {}
        for field in metadata_fields:
            if field in df.columns:
                non_null = df[field].notna().sum()
                metadata_stats[field] = {
                    "present": int(non_null),
                    "missing": int(len(df) - non_null),
                    "completeness": float(non_null / len(df)) if len(df) > 0 else 0.0,
                }
        stats["metadata_completeness"] = metadata_stats
    
    return stats


def load_quality_report(dataset: str) -> dict:
    """Load quality report for a dataset."""
    # Check root
    report_path = SHARDS_ROOT / dataset / "quality_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
            
    # Check logs dir
    logs_path = SHARDS_ROOT / dataset / "logs"
    if logs_path.exists():
        # Look for any quality report
        reports = list(logs_path.glob("*quality_report.json"))
        if reports:
            # Return the first one found, or maybe we should allow selection?
            # For now, just return the most recent or first
            with open(reports[0]) as f:
                return json.load(f)
                
    return {}

def load_metrics(dataset: str) -> dict:
    """Load metrics.json for a dataset."""
    metrics_path = SHARDS_ROOT / dataset / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f)

def load_dataset_manifest(dataset: str) -> dict:
    """Load dataset manifest."""
    manifest_path = SHARDS_ROOT / dataset / "dataset_manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def list_evaluation_datasets() -> list[str]:
    """List datasets that have evaluation results."""
    eval_datasets = set()
    
    # Check multiple possible locations
    eval_paths = [
        EVAL_RESULTS_ROOT,
        RESULTS_ROOT,
    ]
    
    for eval_path in eval_paths:
        if eval_path.exists():
            for item in eval_path.iterdir():
                if item.is_dir():
                    # Check if directory has evaluation files
                    if any(item.glob("*.json")) or any(item.glob("*.parquet")):
                        eval_datasets.add(item.name)
    
    return sorted(list(eval_datasets))


def load_evaluation_results(dataset: str, checkpoint: Optional[str] = None) -> dict:
    """Load evaluation results for a dataset, optionally filtered by checkpoint."""
    results = {}
    
    # Check multiple possible locations
    eval_paths = [
        EVAL_RESULTS_ROOT / dataset,
        RESULTS_ROOT / dataset,
        SHARDS_ROOT / dataset / "eval",
    ]
    
    for eval_path in eval_paths:
        if eval_path.exists():
            # Look for JSON files
            for json_file in eval_path.glob("*.json"):
                if checkpoint and checkpoint not in json_file.name:
                    continue
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        results[json_file.stem] = data
                except:
                    pass
            
            # Look for parquet files
            for parquet_file in eval_path.glob("*.parquet"):
                if checkpoint and checkpoint not in parquet_file.name:
                    continue
                try:
                    df = pd.read_parquet(parquet_file)
                    results[parquet_file.stem] = df.to_dict("records")
                except:
                    pass
    
    return results


def get_adoption_tier(metrics: dict) -> Tuple[str, dict]:
    """Determine adoption tier based on metrics."""
    tier1_thresholds = {
        "top10_accuracy": 0.05,
        "tanimoto_at1": 0.50,
        "hitrate_at1": 0.35,
        "peaks_explained": 0.60,
        "intensity_explained": 0.60,
        "formula_match": 0.90,
    }
    
    tier2_thresholds = {
        "top10_accuracy": 0.10,
        "top1_tanimoto": 0.55,
        "mces_at1": 9.0,
        "hitrate_at1": 0.50,
        "intensity_explained": 0.70,
        "candidate_reduction": 10.0,
        "class_accuracy": 0.80,
    }
    
    tier3_thresholds = {
        "top1_accuracy": 0.08,
        "top10_accuracy": 0.20,
        "tanimoto_at1": 0.60,
        "mces_at1": 8.0,
        "hitrate_at1": 0.60,
        "hitrate_at20": 0.95,
        "intensity_explained": 0.80,
        "candidate_reduction": 30.0,
    }
    
    tier_scores = {"Tier 1": 0, "Tier 2": 0, "Tier 3": 0}
    tier_details = {"Tier 1": [], "Tier 2": [], "Tier 3": []}
    
    # Check Tier 3
    for metric, threshold in tier3_thresholds.items():
        value = metrics.get(metric)
        if value is not None:
            if metric == "mces_at1":  # Lower is better
                if value <= threshold:
                    tier_scores["Tier 3"] += 1
                    tier_details["Tier 3"].append(f"{metric}: {value:.3f} ≤ {threshold}")
            else:  # Higher is better
                if value >= threshold:
                    tier_scores["Tier 3"] += 1
                    tier_details["Tier 3"].append(f"{metric}: {value:.3f} ≥ {threshold}")
    
    # Check Tier 2
    for metric, threshold in tier2_thresholds.items():
        value = metrics.get(metric)
        if value is not None:
            if metric == "mces_at1":  # Lower is better
                if value <= threshold:
                    tier_scores["Tier 2"] += 1
                    tier_details["Tier 2"].append(f"{metric}: {value:.3f} ≤ {threshold}")
            else:  # Higher is better
                if value >= threshold:
                    tier_scores["Tier 2"] += 1
                    tier_details["Tier 2"].append(f"{metric}: {value:.3f} ≥ {threshold}")
    
    # Check Tier 1
    for metric, threshold in tier1_thresholds.items():
        value = metrics.get(metric)
        if value is not None:
            if value >= threshold:
                tier_scores["Tier 1"] += 1
                tier_details["Tier 1"].append(f"{metric}: {value:.3f} ≥ {threshold}")
    
    # Determine current tier
    if tier_scores["Tier 3"] >= len(tier3_thresholds) * 0.8:
        current_tier = "Tier 3: Viral"
    elif tier_scores["Tier 2"] >= len(tier2_thresholds) * 0.8:
        current_tier = "Tier 2: Turning Point"
    elif tier_scores["Tier 1"] >= len(tier1_thresholds) * 0.8:
        current_tier = "Tier 1: Taken Seriously"
    else:
        current_tier = "Below Tier 1"
    
    return current_tier, {
        "scores": tier_scores,
        "details": tier_details,
        "thresholds": {
            "Tier 1": tier1_thresholds,
            "Tier 2": tier2_thresholds,
            "Tier 3": tier3_thresholds,
        }
    }


def plot_spectrum(mzs: list[float], ints: list[float], title: str = "Spectrum") -> alt.Chart:
    """Plot a mass spectrum."""
    df = pd.DataFrame({"mz": mzs, "intensity": ints})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("mz:Q", title="m/z"),
            y=alt.Y("intensity:Q", title="Intensity"),
            tooltip=["mz:Q", "intensity:Q"],
        )
        .properties(title=title, width=600, height=300)
    )
    return chart


# CLI Commands
def cmd_list_datasets() -> None:
    """List all available datasets."""
    datasets = list_datasets()
    if not datasets:
        print("No datasets found.")
        return
    
    print("Available datasets:")
    for dataset in datasets:
        print(f"  • {dataset}")


def cmd_list_shards(dataset: str, split: str = "train", run_id: Optional[str] = None) -> None:
    """List shards for a dataset."""
    shards = list_shards(dataset, split, run_id=run_id)
    if not shards:
        print(f"No shards found for dataset '{dataset}' (split: {split})" + (f", run: {run_id}" if run_id else ""))
        return
    
    print(f"Shards for '{dataset}' (split: {split})" + (f", run: {run_id}" if run_id else "") + ":")
    for shard_path in shards:
        size_mb = shard_path.stat().st_size / (1024 * 1024)
        print(f"  • {shard_path.name} ({size_mb:.2f} MB)")


def cmd_show_stats(dataset: str, split: str = "train", shard: Optional[str] = None) -> None:
    """Show statistics for a dataset or specific shard."""
    if shard:
        shard_path = SHARDS_ROOT / dataset / split / shard
        if not shard_path.exists():
            print(f"Error: Shard '{shard_path}' not found.")
            sys.exit(1)
        
        stats = get_shard_statistics(shard_path)
        print(f"\nStatistics for shard: {shard_path.name}")
        print("=" * 70)
        print(f"Number of samples: {stats['num_samples']}")
        print(f"File size: {stats['file_size_bytes']:,} bytes ({stats['file_size_bytes'] / (1024*1024):.2f} MB)")
        print(f"\nSchema: {stats['schema']}")
        
        if 'peak_counts' in stats:
            print(f"\nPeak counts:")
            print(f"  Min: {stats['peak_counts']['min']}")
            print(f"  Max: {stats['peak_counts']['max']}")
            print(f"  Mean: {stats['peak_counts']['mean']:.2f}")
            print(f"  Median: {stats['peak_counts']['median']:.2f}")
        
        if 'precursor_mz' in stats:
            print(f"\nPrecursor m/z:")
            print(f"  Min: {stats['precursor_mz']['min']:.2f}")
            print(f"  Max: {stats['precursor_mz']['max']:.2f}")
            print(f"  Mean: {stats['precursor_mz']['mean']:.2f}")
        
        if 'metadata_completeness' in stats:
            print(f"\nMetadata completeness:")
            for field, data in stats['metadata_completeness'].items():
                print(f"  {field}: {data['completeness']:.1%} ({data['present']}/{data['present'] + data['missing']})")
    else:
        # Show dataset-level stats
        manifest = load_dataset_manifest(dataset)
        quality_report = load_quality_report(dataset)
        
        print(f"\nDataset: {dataset}")
        print("=" * 70)
        
        if manifest:
            total_samples = sum(shard.get("num_samples", 0) for shard in manifest.get("shards", []))
            print(f"Total shards: {len(manifest.get('shards', []))}")
            print(f"Total samples: {total_samples:,}")
        
        if quality_report:
            print(f"\nQuality Report:")
            overall_status = quality_report.get("overall_status", "UNKNOWN")
            print(f"  Overall status: {overall_status}")
            
            if "stages" in quality_report:
                stages = quality_report["stages"]
                canon = stages.get("canonicalization", {})
                if canon:
                    print(f"  Integrity error rate: {canon.get('integrity_error_rate', 0):.4%}")
                    print(f"  Attrition rate: {canon.get('attrition_rate', 0):.4%}")
                
                train = stages.get("training_readiness", {})
                if train:
                    print(f"  Training readiness: {train.get('status', 'UNKNOWN')}")
                    print(f"  NaN batches: {train.get('nan_batches', 0)}")
                    print(f"  Inf batches: {train.get('inf_batches', 0)}")


def cmd_show_quality_report(dataset: str, output: Optional[str] = None) -> None:
    """Show quality report for a dataset."""
    quality_report = load_quality_report(dataset)
    if not quality_report:
        print(f"No quality report found for dataset '{dataset}'.")
        sys.exit(1)
    
    if output:
        output_path = Path(output)
        with open(output_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        print(f"Quality report saved to {output_path}")
    else:
        print(json.dumps(quality_report, indent=2))


def cmd_export_sample(dataset: str, sample_id: str, split: str = "train", output: Optional[str] = None) -> None:
    """Export a specific sample to JSON."""
    shards = list_shards(dataset, split)
    
    sample_found = False
    for shard_path in shards:
        df = load_shard(shard_path)
        if sample_id in df['sample_id'].values:
            sample_row = df[df['sample_id'] == sample_id].iloc[0]
            
            # Convert to dict, handling numpy arrays
            sample_dict = {
                'sample_id': sample_row['sample_id'],
                'mzs': sample_row['mzs'].tolist() if hasattr(sample_row['mzs'], 'tolist') else list(sample_row['mzs']),
                'ints': sample_row['ints'].tolist() if hasattr(sample_row['ints'], 'tolist') else list(sample_row['ints']),
                'precursor_mz': float(sample_row['precursor_mz']),
                'charge': int(sample_row['charge']),
                'collision_energy': float(sample_row['collision_energy']),
                'adduct': sample_row.get('adduct'),
                'instrument_type': sample_row.get('instrument_type'),
                'smiles': sample_row.get('smiles'),
                'inchikey': sample_row.get('inchikey'),
                'formula': sample_row.get('formula'),
            }
            
            if output:
                output_path = Path(output)
                with open(output_path, 'w') as f:
                    json.dump(sample_dict, f, indent=2)
                print(f"Sample exported to {output_path}")
            else:
                print(json.dumps(sample_dict, indent=2))
            
            sample_found = True
            break
    
    if not sample_found:
        print(f"Sample '{sample_id}' not found in dataset '{dataset}' (split: {split})")
        sys.exit(1)


def cmd_list_runs(dataset: str) -> None:
    """List all run IDs for a dataset."""
    run_ids = list_run_ids(dataset)
    if not run_ids:
        print(f"No run IDs found for dataset '{dataset}'.")
        return
    
    print(f"Run IDs for '{dataset}':")
    for run_id in run_ids:
        print(f"  • {run_id}")


def run_streamlit_ui() -> None:
    """Launch the Streamlit UI."""
    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit is not installed. Install it with: pip install streamlit")
        sys.exit(1)
    
    import subprocess
    script_path = Path(__file__).resolve()
    subprocess.run(["streamlit", "run", str(script_path), "--", "--ui"])


def setup_cli() -> argparse.ArgumentParser:
    """Setup CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="MS/MS Data Inspector - CLI and Streamlit UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Streamlit UI (default)
  python -m nexa_ui.msms_dashboard

  # List all datasets
  python -m nexa_ui.msms_dashboard list-datasets

  # List shards for a dataset
  python -m nexa_ui.msms_dashboard list-shards --dataset gems_full

  # Show statistics
  python -m nexa_ui.msms_dashboard stats --dataset gems_full

  # Show quality report
  python -m nexa_ui.msms_dashboard quality --dataset gems_full

  # Export a sample
  python -m nexa_ui.msms_dashboard export-sample --dataset gems_full --sample-id sample_001
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List datasets
    subparsers.add_parser('list-datasets', help='List all available datasets')
    
    # List shards
    list_shards_parser = subparsers.add_parser('list-shards', help='List shards for a dataset')
    list_shards_parser.add_argument('--dataset', required=True, help='Dataset name')
    list_shards_parser.add_argument('--split', default='train', choices=['train', 'val', 'test'], help='Data split')
    list_shards_parser.add_argument('--run-id', help='Filter by run ID')
    
    # Show stats
    stats_parser = subparsers.add_parser('stats', help='Show statistics for a dataset or shard')
    stats_parser.add_argument('--dataset', required=True, help='Dataset name')
    stats_parser.add_argument('--split', default='train', choices=['train', 'val', 'test'], help='Data split')
    stats_parser.add_argument('--shard', help='Specific shard filename (optional)')
    
    # Quality report
    quality_parser = subparsers.add_parser('quality', help='Show quality report')
    quality_parser.add_argument('--dataset', required=True, help='Dataset name')
    quality_parser.add_argument('--output', help='Save report to JSON file')
    
    # Export sample
    export_parser = subparsers.add_parser('export-sample', help='Export a specific sample to JSON')
    export_parser.add_argument('--dataset', required=True, help='Dataset name')
    export_parser.add_argument('--sample-id', required=True, help='Sample ID to export')
    export_parser.add_argument('--split', default='train', choices=['train', 'val', 'test'], help='Data split')
    export_parser.add_argument('--output', help='Output JSON file path')
    
    # List runs
    list_runs_parser = subparsers.add_parser('list-runs', help='List all run IDs for a dataset')
    list_runs_parser.add_argument('--dataset', required=True, help='Dataset name')
    
    # UI mode
    ui_parser = subparsers.add_parser('ui', help='Launch Streamlit UI (default if no command)')
    
    return parser


def main_cli() -> None:
    """Main CLI entry point."""
    parser = setup_cli()
    args = parser.parse_args()
    
    # If no command provided, default to UI
    if not args.command or args.command == 'ui':
        if '--ui' in sys.argv or args.command == 'ui':
            run_streamlit_ui()
        else:
            # Check if we're being run by Streamlit
            if 'streamlit' in sys.modules or 'STREAMLIT_SERVER' in os.environ:
                main()  # Run Streamlit main
            else:
                run_streamlit_ui()
        return
    
    # Execute CLI commands
    if args.command == 'list-datasets':
        cmd_list_datasets()
    elif args.command == 'list-shards':
        cmd_list_shards(args.dataset, args.split, getattr(args, 'run_id', None))
    elif args.command == 'stats':
        cmd_show_stats(args.dataset, args.split, getattr(args, 'shard', None))
    elif args.command == 'quality':
        cmd_show_quality_report(args.dataset, getattr(args, 'output', None))
    elif args.command == 'export-sample':
        cmd_export_sample(args.dataset, args.sample_id, args.split, getattr(args, 'output', None))
    elif args.command == 'list-runs':
        cmd_list_runs(args.dataset)
    else:
        parser.print_help()
        sys.exit(1)


def list_evaluation_datasets() -> list[str]:
    """List datasets that have evaluation results."""
    eval_datasets = set()
    
    # Check multiple possible locations
    eval_paths = [
        EVAL_RESULTS_ROOT,
        RESULTS_ROOT,
    ]
    
    for eval_path in eval_paths:
        if eval_path.exists():
            for item in eval_path.iterdir():
                if item.is_dir():
                    # Check if directory has evaluation files
                    if any(item.glob("*.json")) or any(item.glob("*.parquet")):
                        eval_datasets.add(item.name)
    
    return sorted(list(eval_datasets))


def main():
    """Main dashboard application."""
    st.title("🧪 MS/MS Dashboard")
    st.markdown("**Comprehensive analysis and evaluation platform for MS/MS spectrum-to-structure models**")

    datasets = list_datasets()
    eval_datasets = list_evaluation_datasets()
    
    # Load evaluation results from all available sources
    all_eval_results = {}
    for eval_dataset in eval_datasets:
        results = load_evaluation_results(eval_dataset)
        all_eval_results.update(results)
    
    # If no datasets but we have eval results, use eval datasets
    if not datasets and eval_datasets:
        datasets = eval_datasets
        st.info(f"📊 Showing evaluation results. No data shards found in {SHARDS_ROOT}.")

    with st.sidebar:
        st.header("Dataset Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", help="Refresh to see newly processed shards"):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Cache", help="Clear all cached data"):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        if datasets:
            selected_dataset = st.selectbox("Dataset", datasets)
        else:
            selected_dataset = None
            st.warning(f"No datasets found in {SHARDS_ROOT}. Run the pipeline first or check evaluation results.")

        # Run ID selection (only if dataset exists in shards)
        if selected_dataset and SHARDS_ROOT.joinpath(selected_dataset).exists():
            run_ids = list_run_ids(selected_dataset)
            if run_ids:
                selected_run_id = st.selectbox("Run ID", ["All"] + run_ids, index=0)
                if selected_run_id == "All":
                    selected_run_id = None
            else:
                selected_run_id = None
                st.info("No run IDs found")
        else:
            selected_run_id = None

        splits = ["train", "val", "test", "shards"]
        selected_split = st.selectbox("Split", splits)

        if selected_dataset and SHARDS_ROOT.joinpath(selected_dataset).exists():
            shards = list_shards(selected_dataset, selected_split, run_id=selected_run_id)
            if shards:
                selected_shard_name = st.selectbox("Shard", [s.name for s in shards], index=0)
                selected_shard_path = next(s for s in shards if s.name == selected_shard_name)
            else:
                st.warning(f"No shards found for {selected_split} split" + (f" (run: {selected_run_id})" if selected_run_id else ""))
                selected_shard_path = None
        else:
            selected_shard_path = None

    # Load global data
    quality_report = {}
    metrics = {}
    manifest = {}
    eval_results = {}
    
    if selected_dataset:
        quality_report = load_quality_report(selected_dataset)
        metrics = load_metrics(selected_dataset)
        manifest = load_dataset_manifest(selected_dataset)
        eval_results = load_evaluation_results(selected_dataset)
    
    # If no eval results for selected dataset, try loading from all eval datasets
    if not eval_results:
        for eval_dataset in eval_datasets:
            results = load_evaluation_results(eval_dataset)
            if results:
                eval_results.update(results)
                if not selected_dataset:
                    selected_dataset = eval_dataset
                break  # Use first available dataset

    # ========== MAIN TABS: DATA INSPECTOR AND EVALUATION ==========
    main_tabs = st.tabs([
        "📊 MS/MS Data Inspector",
        "🎯 MS/MS Evaluation"
    ])

    # ========== MS/MS DATA INSPECTOR SECTION ==========
    with main_tabs[0]:
        st.header("📊 MS/MS Data Inspector")
        st.markdown("Inspect and analyze MS/MS data shards, quality metrics, and sample statistics")
        
        data_tabs = st.tabs([
            "Overview",
            "Data Quality",
            "Data Statistics",
            "Shards & Samples"
        ])
        
        with data_tabs[0]:
            st.subheader("Dataset Overview")
            
            if not datasets and not eval_datasets:
                st.warning(f"No datasets found in {SHARDS_ROOT}. Run the pipeline first or check evaluation results.")
            else:
                # Dataset Information
                if manifest:
                    st.subheader("Dataset Information")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Dataset", manifest.get("dataset", "N/A"))
                    with col2:
                        st.metric("Version", manifest.get("version", "N/A"))
                    with col3:
                        st.metric("Total Shards", len(manifest.get("shards", [])))
                    with col4:
                        total_samples = sum(shard.get("num_samples", 0) for shard in manifest.get("shards", []))
                        st.metric("Total Samples", f"{total_samples:,}")
                    
                    if manifest.get("shards"):
                        st.write("**Shard Summary:**")
                        shard_summary = []
                        for shard in manifest["shards"]:
                            shard_summary.append({
                                "Split": shard.get("split", "N/A"),
                                "Shard Index": shard.get("shard_index", "N/A"),
                                "Samples": shard.get("num_samples", 0),
                                "Size (MB)": f"{shard.get('file_size_bytes', 0) / (1024*1024):.2f}",
                            })
                        shard_df = pd.DataFrame(shard_summary)
                        st.dataframe(shard_df, use_container_width=True, hide_index=True)

                # Quality Status
                if quality_report:
                    st.subheader("Quality Status")
                    status = quality_report.get("overall_status", "UNKNOWN")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if status == "PASS":
                            st.success(f"✅ {status}")
                        elif status == "WARN":
                            st.warning(f"⚠️ {status}")
                        else:
                            st.error(f"❌ {status}")
                    with col2:
                        if quality_report.get("stages"):
                            stages = quality_report["stages"]
                            canon = stages.get("canonicalization", {})
                            if canon:
                                st.write(f"**Integrity Error Rate:** {canon.get('integrity_error_rate', 0):.4%}")
                                st.write(f"**Attrition Rate:** {canon.get('attrition_rate', 0):.4%}")

                # Pipeline Metrics Summary
                if metrics:
                    st.subheader("Pipeline Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        throughput = metrics.get("samples_per_second", 0)
                        st.metric("Throughput", f"{throughput:.1f} spectra/sec")
                    with col2:
                        elapsed = metrics.get("elapsed_seconds", 0)
                        st.metric("Total Runtime", f"{elapsed:.1f}s")
                    with col3:
                        total_spectra = metrics.get("total_spectra", 0)
                        st.metric("Total Spectra", f"{total_spectra:,}")

        with data_tabs[1]:
            st.subheader("Data Quality")
            
            # Error Rate Overview
            st.subheader("Error Rate Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            total_spectra = metrics.get("total_spectra", 0)
            invalid_spectra = metrics.get("integrity_error_count", 0)
            if invalid_spectra == 0 and quality_report:
                 invalid_spectra = quality_report.get("stages", {}).get("canonicalization", {}).get("integrity_errors", 0)
                 
            error_rate = metrics.get("integrity_error_rate", 0.0)
            if error_rate == 0 and quality_report:
                error_rate = quality_report.get("stages", {}).get("canonicalization", {}).get("integrity_error_rate", 0.0)

            status = "UNKNOWN"
            if quality_report:
                status = quality_report.get("overall_status", "UNKNOWN")
            
            with col1:
                st.metric("Total Spectra", f"{total_spectra:,}")
            with col2:
                st.metric("Invalid Spectra", f"{invalid_spectra:,}")
            with col3:
                st.metric("Error Rate", f"{error_rate:.4%}")
            with col4:
                duplicates_count = 0
                if quality_report:
                    duplicates_count = quality_report.get("stages", {}).get("shard_construction", {}).get("duplicates", 0)
                st.metric("Duplicates", duplicates_count)

            # Validation Failure Types
            st.subheader("Validation Failure Types")
            failures = metrics.get("integrity_errors", {})
            if not failures and quality_report:
                 failures = quality_report.get("stages", {}).get("canonicalization", {}).get("attrition_reasons", {})
            
            if failures:
                col1, col2 = st.columns([2, 1])
                with col1:
                    failure_df = pd.DataFrame([
                        {"Error Type": k.replace("_", " ").title(), "Count": v} for k, v in failures.items()
                    ])
                    chart = (
                        alt.Chart(failure_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Error Type", sort="-y", title=""),
                            y=alt.Y("Count", title="Count"),
                            color=alt.Color("Error Type", legend=None, scale=alt.Scale(scheme="category20"))
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart, use_container_width=True)
                with col2:
                    st.write("**Summary:**")
                    for k, v in failures.items():
                        st.write(f"- {k.replace('_', ' ').title()}: {v}")
            else:
                st.info("No validation failures recorded.")

        with data_tabs[2]:
            st.subheader("Data Statistics")
            
            if selected_shard_path:
                stats = get_shard_statistics(selected_shard_path)
                
                # Peak Statistics
                st.subheader("Peak Statistics")
                if "peak_counts" in stats and "all" in stats["peak_counts"]:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        peak_data = pd.DataFrame({"peaks": stats["peak_counts"]["all"]})
                        peak_chart = (
                            alt.Chart(peak_data)
                            .mark_bar()
                            .encode(
                                x=alt.X("peaks:Q", bin=alt.Bin(maxbins=50), title="Number of Peaks"),
                                y=alt.Y("count()", title="Frequency")
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(peak_chart, use_container_width=True)
                    with col2:
                        st.metric("Mean", f"{stats['peak_counts']['mean']:.1f}")
                        st.metric("Median", f"{stats['peak_counts']['median']:.1f}")
                        st.metric("Min", f"{stats['peak_counts']['min']}")
                        st.metric("Max", f"{stats['peak_counts']['max']}")
                
                # Precursor m/z Distribution
                st.subheader("Precursor m/z Distribution")
                if "precursor_mz" in stats and "all" in stats["precursor_mz"]:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        prec_data = pd.DataFrame({"mz": stats["precursor_mz"]["all"]})
                        prec_chart = (
                            alt.Chart(prec_data)
                            .mark_bar()
                            .encode(
                                x=alt.X("mz:Q", bin=alt.Bin(maxbins=50), title="Precursor m/z"),
                                y=alt.Y("count()", title="Frequency")
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(prec_chart, use_container_width=True)
                    with col2:
                        st.metric("Mean", f"{stats['precursor_mz']['mean']:.2f}")
                        st.metric("Min", f"{stats['precursor_mz']['min']:.2f}")
                        st.metric("Max", f"{stats['precursor_mz']['max']:.2f}")

                # Intensity & Distributions
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Intensity Quantiles")
                    if "intensity_quantiles" in stats:
                        q_data = pd.DataFrame([
                            {"Quantile": k, "Value": f"{v:.2e}"} for k, v in stats["intensity_quantiles"].items()
                        ])
                        st.dataframe(q_data.set_index("Quantile").T, use_container_width=True)
                
                with col2:
                    st.subheader("m/z Range")
                    if "mz_ranges" in stats:
                        st.metric("Min m/z", f"{stats['mz_ranges']['min']:.4f}")
                        st.metric("Max m/z", f"{stats['mz_ranges']['max']:.4f}")

                # Charge & Adduct Distributions
                st.subheader("Charge & Adduct Distributions")
                col1, col2 = st.columns(2)
                with col1:
                    if "charge_distribution" in stats:
                        charge_df = pd.DataFrame([
                            {"Charge": str(k), "Count": v} for k, v in stats["charge_distribution"].items()
                        ])
                        charge_chart = (
                            alt.Chart(charge_df)
                            .mark_arc(innerRadius=50)
                            .encode(
                                theta="Count",
                                color="Charge:N",
                                tooltip=["Charge", "Count"]
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(charge_chart, use_container_width=True)
                
                with col2:
                    if "adduct_distribution" in stats:
                        adduct_df = pd.DataFrame([
                            {"Adduct": k, "Count": v} for k, v in stats["adduct_distribution"].items()
                        ])
                        adduct_chart = (
                            alt.Chart(adduct_df)
                            .mark_bar()
                            .encode(
                                y=alt.Y("Adduct", sort="-x", title=""),
                                x=alt.X("Count", title="Count"),
                                color=alt.Color("Adduct", legend=None)
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(adduct_chart, use_container_width=True)
            else:
                st.info("Select a shard in the sidebar to view data statistics.")

        with data_tabs[3]:
            st.subheader("Shards & Samples")
            
            if selected_shard_path:
                manifest_path = selected_shard_path.with_suffix(".manifest.json")
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        shard_manifest = json.load(f)

                    st.subheader(f"Shard: {selected_shard_path.name}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Samples", shard_manifest.get("num_samples", 0))
                with col2:
                    file_size_mb = shard_manifest.get("file_size_bytes", 0) / (1024 * 1024)
                    st.metric("Size", f"{file_size_mb:.2f} MB")
                with col3:
                    st.metric("Schema Version", shard_manifest.get("schema_version", "N/A"))
                with col4:
                    checksum = shard_manifest.get("checksum", "N/A")
                    st.metric("Checksum", checksum[:12] + "..." if len(checksum) > 12 else checksum)
                
                    with st.expander("View Shard Manifest"):
                        st.json(shard_manifest)

                # Sample Inspector
                st.subheader("Sample Inspector")
                df = load_shard(selected_shard_path)
                st.write(f"**Total samples:** {len(df)}")

                sample_id_search = st.text_input("🔍 Search by sample_id", key="sample_search")
                if sample_id_search:
                    df = df[df["sample_id"].str.contains(sample_id_search, case=False, na=False)]

                if len(df) > 0:
                    selected_sample = st.selectbox(
                        "Select sample",
                        df["sample_id"].tolist(),
                        index=0,
                    )

                    sample_row = df[df["sample_id"] == selected_sample].iloc[0]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Metadata:**")
                        st.write(f"- Sample ID: `{sample_row['sample_id']}`")
                        st.write(f"- Precursor m/z: {sample_row['precursor_mz']:.4f}")
                        st.write(f"- Charge: {sample_row['charge']}")
                        st.write(f"- Collision Energy: {sample_row['collision_energy']:.2f}")
                        st.write(f"- Adduct: {sample_row.get('adduct', 'N/A')}")
                        st.write(f"- Instrument: {sample_row.get('instrument_type', 'N/A')}")

                    with col2:
                        st.write("**Molecular Information:**")
                        if pd.notna(sample_row.get("smiles")):
                            st.write(f"- SMILES: `{sample_row['smiles']}`")
                        if pd.notna(sample_row.get("inchikey")):
                            st.write(f"- InChIKey: `{sample_row['inchikey']}`")
                        if pd.notna(sample_row.get("formula")):
                            st.write(f"- Formula: `{sample_row['formula']}`")

                    mzs = sample_row["mzs"]
                    ints = sample_row["ints"]

                    if isinstance(mzs, list):
                        mzs = np.array(mzs)
                    if isinstance(ints, list):
                        ints = np.array(ints)

                    st.subheader("Spectrum Visualization")
                    chart = plot_spectrum(mzs.tolist(), ints.tolist(), f"Spectrum: {selected_sample}")
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No samples found. Try a different search term.")
            else:
                st.info("Select a shard in the sidebar to view shard details and samples.")

    # ========== MS/MS EVALUATION SECTION ==========
    with main_tabs[1]:
        st.header("🎯 MS/MS Evaluation")
        st.markdown("Evaluate model performance with ML metrics, domain-specific metrics, and competitive analysis")
        
        eval_tabs = st.tabs([
            "ML Metrics",
            "Domain Metrics",
            "Competitive Analysis",
            "Adoption Tiers"
        ])
        
        with eval_tabs[0]:
            st.subheader("ML Evaluation Metrics")
            
            if not eval_results:
                st.info("No evaluation results found. Run model evaluation to see ML metrics.")
                st.write("**Expected locations:**")
                st.write(f"- `{EVAL_RESULTS_ROOT}/<dataset>/`")
                st.write(f"- `{RESULTS_ROOT}/<dataset>/`")
                st.write(f"- `{SHARDS_ROOT}/<dataset>/eval/`")
            else:
                # Model/Checkpoint selection
                if len(eval_results) > 1:
                    selected_eval = st.selectbox("Select Evaluation Run", list(eval_results.keys()))
                    eval_data = eval_results[selected_eval]
                else:
                    eval_data = list(eval_results.values())[0]
            
            # Evaluation Run Info
            if isinstance(eval_data, dict):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model", eval_data.get("model_id", "N/A"))
                with col2:
                    st.metric("Checkpoint", eval_data.get("checkpoint", "N/A"))
                with col3:
                    st.metric("Dataset", eval_data.get("dataset", "N/A"))
                with col4:
                    timestamp = eval_data.get("timestamp", "N/A")
                    if timestamp != "N/A":
                        st.metric("Timestamp", timestamp.split("T")[0])
            
            # Training Metrics
            st.subheader("Training Metrics")
            training_metrics = eval_data.get("training", {}) if isinstance(eval_data, dict) else {}
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Loss", f"{training_metrics.get('total_loss', 0):.4f}")
            with col2:
                st.metric("Perplexity", f"{training_metrics.get('perplexity', 0):.2f}")
            with col3:
                st.metric("Valid SMILES", f"{training_metrics.get('valid_smiles_pct', 0):.2%}")
            with col4:
                st.metric("Formula Consistent", f"{training_metrics.get('formula_consistent_pct', 0):.2%}")
            
            # Generation Metrics
            st.subheader("Generation Metrics")
            gen_metrics = eval_data.get("generation", {}) if isinstance(eval_data, dict) else {}
            
            col1, col2 = st.columns(2)
            with col1:
                topk_data = []
                for k in [1, 10]:
                    key = f"top{k}_accuracy"
                    value = gen_metrics.get(key, 0)
                    topk_data.append({"k": f"Top-{k}", "Accuracy": value})
                if topk_data:
                    topk_df = pd.DataFrame(topk_data)
                    chart = (
                        alt.Chart(topk_df)
                        .mark_bar()
                        .encode(
                            x="k:N",
                            y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color("k:N", legend=None, scale=alt.Scale(scheme="category10"))
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart, use_container_width=True)
            
            with col2:
                sim_data = []
                for k in [1, 10]:
                    tanimoto_key = f"tanimoto_at{k}"
                    mces_key = f"mces_at{k}"
                    tanimoto = gen_metrics.get(tanimoto_key, 0)
                    mces = gen_metrics.get(mces_key, 0)
                    sim_data.append({
                        "k": f"Top-{k}",
                        "Tanimoto": tanimoto,
                        "MCES": mces
                    })
                if sim_data:
                    sim_df = pd.DataFrame(sim_data)
                    chart = (
                        alt.Chart(sim_df.melt(id_vars=["k"], var_name="Metric", value_name="Value"))
                        .mark_bar()
                        .encode(
                            x="k:N",
                            y="Value:Q",
                            color="Metric:N",
                            column="Metric:N"
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart, use_container_width=True)
            
            # Retrieval & Calibration
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Retrieval Metrics")
                retrieval_metrics = eval_data.get("retrieval", {}) if isinstance(eval_data, dict) else {}
                col1a, col1b, col1c, col1d = st.columns(4)
                with col1a:
                    st.metric("HitRate@1", f"{retrieval_metrics.get('hitrate_at1', 0):.2%}")
                with col1b:
                    st.metric("HitRate@5", f"{retrieval_metrics.get('hitrate_at5', 0):.2%}")
                with col1c:
                    st.metric("HitRate@20", f"{retrieval_metrics.get('hitrate_at20', 0):.2%}")
                with col1d:
                    st.metric("MRR", f"{retrieval_metrics.get('mrr', 0):.3f}")
            
            with col2:
                st.subheader("Calibration")
                calib_metrics = eval_data.get("calibration", {}) if isinstance(eval_data, dict) else {}
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("ECE (SMILES)", f"{calib_metrics.get('ece_smiles', 0):.4f}")
                with col2b:
                    st.metric("PICP@90", f"{calib_metrics.get('picp_90', 0):.2%}")
            
            # Generalization
            st.subheader("Generalization")
            gen_slices = eval_data.get("generalization", {}) if isinstance(eval_data, dict) else {}
            if gen_slices:
                slice_type = st.selectbox("Slice By", ["instrument_type", "collision_energy", "ionization_mode", "chemical_class"], key="gen_slice")
                if slice_type in gen_slices:
                    slice_data = gen_slices[slice_type]
                    if isinstance(slice_data, dict):
                        slice_df = pd.DataFrame([
                            {"Category": k, "Top-1 Accuracy": v.get("top1_accuracy", 0), 
                             "Tanimoto@1": v.get("tanimoto_at1", 0)}
                            for k, v in slice_data.items()
                        ])
                        if not slice_df.empty:
                            chart = (
                                alt.Chart(slice_df.melt(id_vars=["Category"], var_name="Metric", value_name="Value"))
                                .mark_bar()
                                .encode(
                                    x="Category:N",
                                    y="Value:Q",
                                    color="Metric:N",
                                    column="Metric:N"
                                )
                                .properties(height=300)
                            )
                            st.altair_chart(chart, use_container_width=True)

        with eval_tabs[1]:
            st.subheader("Domain-Specific Evaluation")
            
            if not eval_results:
                st.info("No evaluation results found. Run model evaluation to see domain-specific metrics.")
                st.write("**Expected locations:**")
                st.write(f"- `{EVAL_RESULTS_ROOT}/<dataset>/`")
                st.write(f"- `{RESULTS_ROOT}/<dataset>/`")
            else:
                if len(eval_results) > 1:
                    selected_eval = st.selectbox("Select Evaluation Run", list(eval_results.keys()), key="domain_eval")
                    eval_data = eval_results[selected_eval]
                else:
                    eval_data = list(eval_results.values())[0]
                
                domain_metrics = eval_data.get("domain", {}) if isinstance(eval_data, dict) else {}
                
                # 2.1 Precursor Mass & Formula Consistency
                st.subheader("Mass & Formula Consistency")
                col1, col2, col3 = st.columns(3)
                
                mass_metrics = domain_metrics.get("mass_formula", {})
                with col1:
                    mass_error = mass_metrics.get("mass_error_ppm", 0)
                    st.metric("Mass Error (ppm)", f"{mass_error:.2f}", 
                             delta="✓" if abs(mass_error) < 5 else "✗")
                with col2:
                    formula_acc = mass_metrics.get("formula_accuracy", 0)
                    st.metric("Formula Accuracy", f"{formula_acc:.2%}")
                with col3:
                    exact_match = mass_metrics.get("exact_formula_match", 0)
                    st.metric("Exact Formula Match", f"{exact_match:.2%}")
                
                # 2.2 Fragmentation Explainability
                st.subheader("Fragmentation Explainability")
                frag_metrics = domain_metrics.get("fragmentation", {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    peaks_explained = frag_metrics.get("peaks_explained_pct", 0)
                    st.metric("Peaks Explained %", f"{peaks_explained:.2%}")
                with col2:
                    intensity_explained = frag_metrics.get("intensity_explained_pct", 0)
                    st.metric("Intensity Explained %", f"{intensity_explained:.2%}")
                with col3:
                    unexplained_high = frag_metrics.get("unexplained_high_intensity_peaks", 0)
                    st.metric("Unexplained High-Intensity Peaks", f"{unexplained_high:.1f}")
                
                # 2.3 Adduct & Ion Mode Validity
                st.subheader("Adduct & Ion Mode Validity")
                adduct_metrics = domain_metrics.get("adduct", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    adduct_compatible = adduct_metrics.get("adduct_compatible_pct", 0)
                    st.metric("Adduct-Compatible Predictions", f"{adduct_compatible:.2%}")
                with col2:
                    charge_violations = adduct_metrics.get("charge_violations_pct", 0)
                    st.metric("Charge/Isotope Violations", f"{charge_violations:.2%}")
                
                # 2.4 Chemical Class and Scaffold Correctness
                st.subheader("Chemical Class & Scaffold Correctness")
                class_metrics = domain_metrics.get("chemical_class", {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    superclass_acc = class_metrics.get("superclass_accuracy", 0)
                    st.metric("Superclass Accuracy", f"{superclass_acc:.2%}")
                with col2:
                    class_acc = class_metrics.get("class_accuracy", 0)
                    st.metric("Class Accuracy", f"{class_acc:.2%}")
                with col3:
                    scaffold_match = class_metrics.get("scaffold_match_rate", 0)
                    st.metric("Scaffold Match Rate", f"{scaffold_match:.2%}")
                with col4:
                    fg_recall = class_metrics.get("functional_group_recall", 0)
                    st.metric("Functional Group Recall", f"{fg_recall:.2%}")
                
                # 2.5 Candidate Reduction / Workflow Gains
                st.subheader("Workflow Gains")
                workflow_metrics = domain_metrics.get("workflow", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    reduction_factor = workflow_metrics.get("candidate_reduction_factor", 0)
                    st.metric("Candidate Reduction Factor", f"{reduction_factor:.1f}×")
                with col2:
                    annotation_rate = workflow_metrics.get("annotation_rate_improvement", 0)
                    st.metric("Annotation Rate Improvement", f"{annotation_rate:.1f}×")
                
                # 2.6 Robustness Across Spectral Conditions
                st.subheader("Robustness Across Conditions")
                robustness = domain_metrics.get("robustness", {})
                
                if robustness:
                    robustness_df = pd.DataFrame([
                        {
                            "Condition": k,
                            "Peaks Explained": v.get("peaks_explained_pct", 0),
                            "Intensity Explained": v.get("intensity_explained_pct", 0),
                            "Class Accuracy": v.get("class_accuracy", 0)
                        }
                        for k, v in robustness.items()
                    ])
                    if not robustness_df.empty:
                        chart = (
                            alt.Chart(robustness_df.melt(id_vars=["Condition"], var_name="Metric", value_name="Value"))
                            .mark_bar()
                            .encode(
                                x="Condition:N",
                                y="Value:Q",
                                color="Metric:N",
                                column="Metric:N"
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(chart, use_container_width=True)

        with eval_tabs[2]:
            st.subheader("Competitive Analysis")
            
            competitive_models = {
            "MSNovelist": {
                "top1_accuracy": 0.07,  # Conservative estimate
                "top1_tanimoto": None,
                "mces_at1": None,
                "notes": "Top-1 ~7–39% in constrained settings"
            },
            "Spec2Mol": {
                "top1_accuracy": None,
                "top1_tanimoto": None,
                "mces_at1": None,
                "notes": "Focused on similarity; MW/formula errors reported"
            },
            "Test-Time Tuned LM (NPLIB1)": {
                "top1_accuracy": 0.168,
                "top1_tanimoto": 0.62,
                "mces_at1": 6.5,
                "notes": "NPLIB1 dataset"
            },
            "Test-Time Tuned LM (MassSpecGym)": {
                "top1_accuracy": 0.028,
                "top1_tanimoto": 0.45,
                "mces_at1": 11.9,
                "notes": "MassSpecGym dataset"
            },
            "DIFFMS/MADGEN/MS-BART": {
                "top1_accuracy": 0.02,  # Range 2–4%
                "top1_tanimoto": None,
                "mces_at1": None,
                "notes": "Range of 2–4% Top-1"
            },
            "MIST/MS2Query/Spec2Vec": {
                "top1_accuracy": None,
                "hitrate_at1": 0.40,  # ~40–60%
                "notes": "Embedding & retrieval models"
            }
            }
            
            if eval_results:
                if len(eval_results) > 1:
                    selected_eval = st.selectbox("Select Evaluation Run", list(eval_results.keys()), key="comp_eval")
                    eval_data = eval_results[selected_eval]
                else:
                    eval_data = list(eval_results.values())[0]
                
                gen_metrics = eval_data.get("generation", {}) if isinstance(eval_data, dict) else {}
                retrieval_metrics = eval_data.get("retrieval", {}) if isinstance(eval_data, dict) else {}
                
                # Add current model
                current_model = {
                    "Current Model": {
                        "top1_accuracy": gen_metrics.get("top1_accuracy", 0),
                        "top1_tanimoto": gen_metrics.get("tanimoto_at1", 0),
                        "mces_at1": gen_metrics.get("mces_at1", 0),
                        "hitrate_at1": retrieval_metrics.get("hitrate_at1", 0),
                        "notes": "Your model"
                    }
                }
                competitive_models.update(current_model)
            
            # Create comparison dataframe
            comp_data = []
            for model_name, metrics in competitive_models.items():
                comp_data.append({
                    "Model": model_name,
                    "Top-1 Accuracy": metrics.get("top1_accuracy", None),
                    "Tanimoto@1": metrics.get("top1_tanimoto", None),
                    "MCES@1": metrics.get("mces_at1", None),
                    "HitRate@1": metrics.get("hitrate_at1", None),
                    "Notes": metrics.get("notes", "")
                })
            
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True)
            
            # Visualization
            if eval_results:
                chart_data = []
                for model_name, metrics in competitive_models.items():
                    if metrics.get("top1_accuracy") is not None:
                        chart_data.append({
                            "Model": model_name,
                            "Top-1 Accuracy": metrics["top1_accuracy"]
                        })
                
                if chart_data:
                    chart_df = pd.DataFrame(chart_data)
                    chart = (
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Model:N", sort="-y"),
                            y=alt.Y("Top-1 Accuracy:Q", scale=alt.Scale(domain=[0, 0.25])),
                            color=alt.Color("Model:N", legend=None)
                        )
                        .properties(height=400)
                    )
                    st.altair_chart(chart, use_container_width=True)

        with eval_tabs[3]:
            st.subheader("Adoption Tiers")
            
            if not eval_results:
                st.info("No evaluation results found. Run model evaluation to see adoption tier status.")
                st.write("**Expected locations:**")
                st.write(f"- `{EVAL_RESULTS_ROOT}/<dataset>/`")
                st.write(f"- `{RESULTS_ROOT}/<dataset>/`")
            else:
                if len(eval_results) > 1:
                    selected_eval = st.selectbox("Select Evaluation Run", list(eval_results.keys()), key="tier_eval")
                    eval_data = eval_results[selected_eval]
                else:
                    eval_data = list(eval_results.values())[0]
                
                # Aggregate metrics
                all_metrics = {}
                if isinstance(eval_data, dict):
                    gen_metrics = eval_data.get("generation", {})
                    retrieval_metrics = eval_data.get("retrieval", {})
                    domain_metrics = eval_data.get("domain", {})
                    frag_metrics = domain_metrics.get("fragmentation", {})
                    mass_metrics = domain_metrics.get("mass_formula", {})
                    class_metrics = domain_metrics.get("chemical_class", {})
                    workflow_metrics = domain_metrics.get("workflow", {})
                    
                    all_metrics = {
                        "top1_accuracy": gen_metrics.get("top1_accuracy", 0),
                        "top10_accuracy": gen_metrics.get("top10_accuracy", 0),
                        "tanimoto_at1": gen_metrics.get("tanimoto_at1", 0),
                        "mces_at1": gen_metrics.get("mces_at1", 0),
                        "hitrate_at1": retrieval_metrics.get("hitrate_at1", 0),
                        "hitrate_at20": retrieval_metrics.get("hitrate_at20", 0),
                        "peaks_explained": frag_metrics.get("peaks_explained_pct", 0),
                        "intensity_explained": frag_metrics.get("intensity_explained_pct", 0),
                        "formula_match": mass_metrics.get("formula_accuracy", 0),
                        "candidate_reduction": workflow_metrics.get("candidate_reduction_factor", 0),
                        "class_accuracy": class_metrics.get("class_accuracy", 0),
                    }
                
                current_tier, tier_info = get_adoption_tier(all_metrics)
                
                st.subheader("Current Tier Status")
                tier_colors = {
                    "Tier 3: Viral": "🟢",
                    "Tier 2: Turning Point": "🟡",
                    "Tier 1: Taken Seriously": "🟠",
                    "Below Tier 1": "🔴"
                }
                st.markdown(f"### {tier_colors.get(current_tier, '⚪')} {current_tier}")
                
                # Tier thresholds display
                st.subheader("Tier Thresholds")
                
                for tier_name in ["Tier 1", "Tier 2", "Tier 3"]:
                    with st.expander(f"{tier_name} Thresholds"):
                        thresholds = tier_info["thresholds"][tier_name]
                        threshold_df = pd.DataFrame([
                            {"Metric": k, "Threshold": v, 
                             "Current": all_metrics.get(k.replace("_at1", "_at1").replace("_at20", "_at20"), None)}
                            for k, v in thresholds.items()
                        ])
                        st.dataframe(threshold_df, use_container_width=True)
                        
                        # Show which thresholds are met
                        met_count = tier_info["scores"][tier_name]
                        total_count = len(thresholds)
                        st.progress(met_count / total_count if total_count > 0 else 0)
                        st.write(f"**{met_count}/{total_count} thresholds met**")
                        
                        if tier_info["details"][tier_name]:
                            st.write("**Met thresholds:**")
                            for detail in tier_info["details"][tier_name]:
                                st.write(f"- {detail}")



if __name__ == "__main__":
    # Check if we're being run by Streamlit directly
    if len(sys.argv) > 1 and sys.argv[1] == '--ui':
        # Streamlit is calling us with --ui flag
        main()
    elif len(sys.argv) > 1:
        # CLI mode - parse arguments
        main_cli()
    else:
        # Default: launch Streamlit UI
        run_streamlit_ui()
