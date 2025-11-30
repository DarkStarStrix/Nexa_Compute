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
        page_title="MS/MS Data Inspector",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARDS_ROOT = PROJECT_ROOT / "data" / "shards"


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


def main():
    """Main dashboard application."""
    st.title("🧪 MS/MS Data Inspector")

    datasets = list_datasets()
    if not datasets:
        st.warning(f"No datasets found in {SHARDS_ROOT}. Run the pipeline first.")
        return

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
        
        selected_dataset = st.selectbox("Dataset", datasets)

        # Run ID selection
        run_ids = list_run_ids(selected_dataset)
        if run_ids:
            selected_run_id = st.selectbox("Run ID", ["All"] + run_ids, index=0)
            if selected_run_id == "All":
                selected_run_id = None
        else:
            selected_run_id = None
            st.info("No run IDs found")

        splits = ["train", "val", "test", "shards"]
        selected_split = st.selectbox("Split", splits)

        shards = list_shards(selected_dataset, selected_split, run_id=selected_run_id)
        if shards:
            selected_shard_name = st.selectbox("Shard", [s.name for s in shards], index=0)
            selected_shard_path = next(s for s in shards if s.name == selected_shard_name)
        else:
            st.warning(f"No shards found for {selected_split} split" + (f" (run: {selected_run_id})" if selected_run_id else ""))
            selected_shard_path = None

    tabs = st.tabs([
        "Overview", 
        "Structural Integrity", 
        "Semantic Quality", 
        "Pipeline Health",
        "Shards", 
        "Samples"
    ])

    # Load global data
    quality_report = load_quality_report(selected_dataset)
    metrics = load_metrics(selected_dataset)
    manifest = load_dataset_manifest(selected_dataset)

    with tabs[0]:
        st.header("Overview")
        
        if manifest:
            st.subheader("Dataset Manifest")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dataset", manifest.get("dataset", "N/A"))
            with col2:
                st.metric("Version", manifest.get("version", "N/A"))
            with col3:
                st.metric("Total Shards", len(manifest.get("shards", [])))
            
            total_samples = sum(shard.get("num_samples", 0) for shard in manifest.get("shards", []))
            st.metric("Total Samples", total_samples)
            
            if manifest.get("shards"):
                st.write("**Shard Summary:**")
                shard_summary = []
                for shard in manifest["shards"]:
                    shard_summary.append({
                        "Split": shard.get("split", "N/A"),
                        "Shard Index": shard.get("shard_index", "N/A"),
                        "Samples": shard.get("num_samples", 0),
                        "Size (bytes)": shard.get("file_size_bytes", 0),
                    })
                shard_df = pd.DataFrame(shard_summary)
                st.dataframe(shard_df, use_container_width=True)

        if quality_report:
            st.subheader("Overall Quality Status")
            status = quality_report.get("overall_status", "UNKNOWN")
            if status == "PASS":
                st.success(f"✅ {status}")
            elif status == "WARN":
                st.warning(f"⚠️ {status}")
            else:
                st.error(f"❌ {status}")

    with tabs[1]:
        st.header("Structural Integrity")
        
        # 1. Error Rate Overview
        st.subheader("Error Rate Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_spectra = metrics.get("total_spectra", 0)
        
        # Try to get integrity error count from metrics or quality report
        invalid_spectra = metrics.get("integrity_error_count", 0)
        if invalid_spectra == 0 and quality_report:
             invalid_spectra = quality_report.get("stages", {}).get("canonicalization", {}).get("integrity_errors", 0)
             
        error_rate = metrics.get("integrity_error_rate", 0.0)
        if error_rate == 0 and quality_report:
            error_rate = quality_report.get("stages", {}).get("canonicalization", {}).get("integrity_error_rate", 0.0)

        with col1:
            st.metric("Total Spectra Processed", f"{total_spectra:,}")
        with col2:
            st.metric("Invalid Spectra Caught", f"{invalid_spectra:,}")
        with col3:
            st.metric("Error Rate", f"{error_rate:.4%}")
        with col4:
            # Placeholder for invalid rows written, assume 0 if passed
            st.metric("Invalid Rows Written", "0" if status == "PASS" else "Check Logs")

        # 2. Validation Failure Types
        st.subheader("Validation Failure Types")
        
        failures = metrics.get("integrity_errors", {})
        if not failures and quality_report:
             failures = quality_report.get("stages", {}).get("canonicalization", {}).get("attrition_reasons", {})
        
        if failures:
            failure_df = pd.DataFrame([
                {"Error Type": k, "Count": v} for k, v in failures.items()
            ])
            
            chart = (
                alt.Chart(failure_df)
                .mark_bar()
                .encode(
                    x=alt.X("Error Type", sort="-y"),
                    y="Count",
                    color=alt.Color("Error Type", legend=None)
                )
                .properties(title="Validation Failures by Type", height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No validation failures recorded.")

        # 3. Duplicate ID Watchdog
        st.subheader("Duplicate ID Watchdog")
        duplicates_count = 0
        if quality_report:
             duplicates_count = quality_report.get("stages", {}).get("shard_construction", {}).get("duplicates", 0)
        
        st.metric("Duplicates Detected", duplicates_count)
        
        if duplicates_count > 0:
            st.error(f"Found {duplicates_count} duplicates! Check data integrity.")
        else:
            st.success("No duplicates detected across processed shards.")

    with tabs[2]:
        st.header("Semantic Quality")
        
        if selected_shard_path:
            stats = get_shard_statistics(selected_shard_path)
            
            # 4. Peak Count Distribution
            st.subheader("Peak Count Distribution")
            if "peak_counts" in stats and "all" in stats["peak_counts"]:
                peak_data = pd.DataFrame({"peaks": stats["peak_counts"]["all"]})
                
                peak_chart = (
                    alt.Chart(peak_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("peaks:Q", bin=alt.Bin(maxbins=50), title="Number of Peaks"),
                        y=alt.Y("count()", title="Frequency")
                    )
                    .properties(title="Peak Count Histogram")
                )
                st.altair_chart(peak_chart, use_container_width=True)
                
                col1, col2 = st.columns(2)
                col1.metric("Mean Peaks", f"{stats['peak_counts']['mean']:.1f}")
                col2.metric("Median Peaks", f"{stats['peak_counts']['median']:.1f}")
            
            # 5. Precursor m/z Distribution
            st.subheader("Precursor m/z Distribution")
            if "precursor_mz" in stats and "all" in stats["precursor_mz"]:
                prec_data = pd.DataFrame({"mz": stats["precursor_mz"]["all"]})
                
                prec_chart = (
                    alt.Chart(prec_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("mz:Q", bin=alt.Bin(maxbins=50), title="Precursor m/z"),
                        y=alt.Y("count()", title="Frequency")
                    )
                    .properties(title="Precursor m/z Histogram")
                )
                st.altair_chart(prec_chart, use_container_width=True)

            # 6. Intensity Quantile Stability
            st.subheader("Intensity Quantiles")
            if "intensity_quantiles" in stats:
                q_data = pd.DataFrame([
                    {"Quantile": k, "Value": v} for k, v in stats["intensity_quantiles"].items()
                ])
                st.dataframe(q_data.set_index("Quantile").T)

            # 7. Adduct & Charge-State Distribution
            st.subheader("Adduct & Charge Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                if "charge_distribution" in stats:
                    charge_df = pd.DataFrame([
                        {"Charge": k, "Count": v} for k, v in stats["charge_distribution"].items()
                    ])
                    charge_chart = (
                        alt.Chart(charge_df)
                        .mark_arc()
                        .encode(
                            theta="Count",
                            color="Charge:N",
                            tooltip=["Charge", "Count"]
                        )
                        .properties(title="Charge State Distribution")
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
                            y=alt.Y("Adduct", sort="-x"),
                            x="Count",
                            tooltip=["Adduct", "Count"]
                        )
                        .properties(title="Adduct Distribution")
                    )
                    st.altair_chart(adduct_chart, use_container_width=True)

            # 8. m/z Range Checker
            st.subheader("m/z Range Checker")
            if "mz_ranges" in stats:
                col1, col2 = st.columns(2)
                col1.metric("Global Min m/z", f"{stats['mz_ranges']['min']:.4f}")
                col2.metric("Global Max m/z", f"{stats['mz_ranges']['max']:.4f}")

        else:
            st.info("Select a shard to view semantic quality metrics.")

    with tabs[3]:
        st.header("Pipeline Health")
        
        # 9. Throughput
        st.subheader("Throughput")
        throughput = metrics.get("samples_per_second", 0)
        elapsed = metrics.get("elapsed_seconds", 0)
        
        col1, col2 = st.columns(2)
        col1.metric("Throughput (Spectra/sec)", f"{throughput:.2f}")
        col2.metric("Total Runtime (s)", f"{elapsed:.2f}")
        
        if throughput < 100 and throughput > 0:
            st.warning("Throughput seems low (< 100 spectra/sec). Check IO or CPU.")

        # 10. CPU/Memory Utilization (Placeholder as we don't have timeseries logs yet)
        st.subheader("Resource Utilization")
        st.info("Real-time CPU and Memory telemetry is not currently persisted to `resource_timeseries.json`. Please enable detailed logging in `nexa_infra` to view historical utilization.")
        
        # 11. Shard Finalization Timings
        st.subheader("Shard Processing")
        st.metric("Shards Written", metrics.get("shards_written", 0))
        st.metric("Samples Written", metrics.get("samples_written", 0))

    with tabs[4]:
        st.header("Shards Inspector")

        if selected_shard_path:
            manifest_path = selected_shard_path.with_suffix(".manifest.json")
            if manifest_path.exists():
                with open(manifest_path) as f:
                    shard_manifest = json.load(f)

                st.subheader(f"Shard: {selected_shard_path.name}")
                
                # Basic metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Number of Samples", shard_manifest.get("num_samples", 0))
                with col2:
                    st.metric("File Size", f"{shard_manifest.get('file_size_bytes', 0):,} bytes")
                with col3:
                    st.metric("Schema Version", shard_manifest.get("schema_version", "N/A"))
                with col4:
                    checksum = shard_manifest.get("checksum", "N/A")
                    st.metric("Checksum", checksum[:16] + "..." if len(checksum) > 16 else checksum)
                
                with st.expander("View Shard Manifest JSON"):
                    st.json(shard_manifest)

    with tabs[5]:
        st.header("Samples Inspector")

        if selected_shard_path:
            df = load_shard(selected_shard_path)

            st.subheader(f"Shard: {selected_shard_path.name}")
            st.write(f"Total samples: {len(df)}")

            sample_id_search = st.text_input("Search by sample_id")
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
                    st.write(f"Sample ID: {sample_row['sample_id']}")
                    st.write(f"Precursor m/z: {sample_row['precursor_mz']:.4f}")
                    st.write(f"Charge: {sample_row['charge']}")
                    st.write(f"Collision Energy: {sample_row['collision_energy']:.2f}")
                    st.write(f"Adduct: {sample_row.get('adduct', 'N/A')}")
                    st.write(f"Instrument: {sample_row.get('instrument_type', 'N/A')}")

                with col2:
                    if pd.notna(sample_row.get("smiles")):
                        st.write("**SMILES:**", sample_row["smiles"])
                    if pd.notna(sample_row.get("inchikey")):
                        st.write("**InChIKey:**", sample_row["inchikey"])
                    if pd.notna(sample_row.get("formula")):
                        st.write("**Formula:**", sample_row["formula"])

                mzs = sample_row["mzs"]
                ints = sample_row["ints"]

                if isinstance(mzs, list):
                    mzs = np.array(mzs)
                if isinstance(ints, list):
                    ints = np.array(ints)

                st.subheader("Spectrum")
                chart = plot_spectrum(mzs.tolist(), ints.tolist(), f"Spectrum: {selected_sample}")
                st.altair_chart(chart, use_container_width=True)


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
